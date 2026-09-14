package dev.neuroforge.imaging

import android.graphics.Bitmap
import android.util.Log
import dev.neuroforge.core.TileGrid
import dev.neuroforge.core.formatBytes
import dev.neuroforge.runtime.LiteRtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlin.coroutines.coroutineContext

/** Progress of one super-resolution sweep. */
data class UpscaleProgress(
  val pass: Int,
  val tilesDone: Int,
  val tilesTotal: Int,
  val millisPerTile: Double,
) {
  val fraction: Float get() = if (tilesTotal > 0) tilesDone.toFloat() / tilesTotal else 0f
}

/**
 * Runs a fixed-input super-resolution graph over a whole image, tile by tile.
 *
 * This is what turns a 512² or 1024² generated image into a 4K one. The network only ever
 * sees its compiled input shape, which is exactly what an NPU wants — one compilation,
 * replayed dozens of times — and memory stays bounded by the accumulator rather than by any
 * single activation.
 *
 * Results are accumulated with the feathered weights from [TileGrid] and normalised at the
 * end, so seams between tiles cross-fade instead of showing as a grid.
 */
class TiledUpscaler(
  private val session: LiteRtSession,
  private val normalization: Normalization = Normalization.UNIT,
) {

  /**
   * @param onProgress called on the calling coroutine's dispatcher after each tile.
   * @throws OutOfMemoryError deliberately not caught — see [checkBudget], which turns the
   *   predictable case into a clear message before any allocation happens.
   */
  suspend fun upscale(
    source: Bitmap,
    grid: TileGrid,
    passIndex: Int = 0,
    onProgress: (UpscaleProgress) -> Unit = {},
  ): Bitmap = withContext(Dispatchers.Default) {
    require(source.width == grid.srcWidth && source.height == grid.srcHeight) {
      "grid was planned for ${grid.srcWidth}x${grid.srcHeight} but got ${source.width}x${source.height}"
    }
    checkBudget(grid)

    val dstW = grid.dstWidth
    val dstH = grid.dstHeight

    // One weighted accumulator for colour, one for the weights themselves. Dividing at the
    // end is what makes the blend correct for any overlap pattern, including the wider
    // overlap the last tile in a row inherits when it is pulled back to the image edge.
    val acc = FloatArray(dstW * dstH * 3)
    val wsum = FloatArray(dstW * dstH)

    val tileW = grid.tiles.first().srcW
    val tileH = grid.tiles.first().srcH
    val inBuf = FloatArray(tileW * tileH * 3)
    val pixelBuf = IntArray(tileW * tileH)

    var totalMillis = 0.0
    grid.tiles.forEachIndexed { index, tile ->
      coroutineContext.ensureActive()

      ImageTensor.readRegion(
        source, tile.srcX, tile.srcY, tile.srcW, tile.srcH,
        inBuf, pixelBuf, normalization,
      )
      session.writeInput(0, inBuf)
      totalMillis += session.run().toDouble()
      val out = session.readOutput(0)

      val outW = tile.srcW * grid.scale
      val outH = tile.srcH * grid.scale
      val expected = outW * outH * 3
      check(out.size >= expected) {
        "upscaler returned ${out.size} floats, expected $expected for a ${outW}x$outH tile — " +
          "the graph's scale factor probably is not ${grid.scale}"
      }

      val baseX = tile.srcX * grid.scale
      val baseY = tile.srcY * grid.scale
      for (ly in 0 until outH) {
        val dstRow = (baseY + ly) * dstW
        val srcRow = ly * outW
        for (lx in 0 until outW) {
          val w = grid.weightAt(tile, lx, ly)
          if (w <= 0f) continue
          val di = dstRow + baseX + lx
          val si = (srcRow + lx) * 3
          val da = di * 3
          acc[da] += w * out[si]
          acc[da + 1] += w * out[si + 1]
          acc[da + 2] += w * out[si + 2]
          wsum[di] += w
        }
      }

      onProgress(
        UpscaleProgress(
          pass = passIndex,
          tilesDone = index + 1,
          tilesTotal = grid.tiles.size,
          millisPerTile = totalMillis / (index + 1),
        )
      )
    }

    Log.i(
      TAG,
      "pass $passIndex: ${grid.tiles.size} tiles on ${session.accelerator}, " +
        "%.1f ms/tile, %.1f s total".format(totalMillis / grid.tiles.size, totalMillis / 1000),
    )

    normalizeToBitmap(acc, wsum, dstW, dstH)
  }

  /** Divides out the accumulated weights and packs the result into a bitmap. */
  private fun normalizeToBitmap(acc: FloatArray, wsum: FloatArray, width: Int, height: Int): Bitmap {
    val pixels = IntArray(width * height)
    for (i in pixels.indices) {
      val w = wsum[i]
      val a = i * 3
      if (w <= 0f) {
        // Cannot happen with a grid that covers the image, but a black pixel is a far
        // better failure than a NaN propagating into the PNG encoder.
        pixels[i] = 0xFF000000.toInt()
        continue
      }
      val inv = 1f / w
      val r = normalization.toPixel(acc[a] * inv)
      val g = normalization.toPixel(acc[a + 1] * inv)
      val b = normalization.toPixel(acc[a + 2] * inv)
      pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
    val bitmap = androidx.core.graphics.createBitmap(width, height)
    bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
    return bitmap
  }

  /**
   * Fails early and legibly when the plan cannot fit.
   *
   * A 4096² accumulator is ~256 MiB and the output bitmap another ~64 MiB. On a 12 GB phone
   * with `largeHeap` that is fine; on a smaller device it is not, and an explicit message
   * naming the number beats an `OutOfMemoryError` from somewhere inside a pixel loop.
   */
  private fun checkBudget(grid: TileGrid) {
    val needed = grid.accumulatorBytes() + grid.dstWidth.toLong() * grid.dstHeight * 4
    val max = Runtime.getRuntime().maxMemory()
    val used = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
    val headroom = max - used
    if (needed > headroom) {
      error(
        "Not enough heap for a ${grid.dstWidth}x${grid.dstHeight} result: " +
          "needs ${formatBytes(needed)}, ${formatBytes(headroom)} free. " +
          "Pick a smaller target or the FAST upscale strategy."
      )
    }
  }

  private companion object {
    const val TAG = "TiledUpscaler"
  }
}

package dev.neuroforge.core

import kotlin.math.ceil
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.roundToInt

/** A named output size. */
data class OutputTarget(val label: String, val width: Int, val height: Int) {
  val megapixels: Float get() = width.toLong().toFloat() * height / 1_000_000f

  companion object {
    val UHD_4K = OutputTarget("4K UHD 16:9", 3840, 2160)
    val DCI_4K = OutputTarget("4K DCI 17:9", 4096, 2160)
    val SQUARE_4K = OutputTarget("4K square", 4096, 4096)
    val QHD = OutputTarget("2K square", 2048, 2048)
    val NATIVE = OutputTarget("Generator native", 512, 512)

    val presets = listOf(NATIVE, QHD, SQUARE_4K, UHD_4K, DCI_4K)
  }
}

/**
 * How aggressively to spend super-resolution passes.
 *
 * Each pass multiplies by the network's scale factor and costs a full tile sweep, so the
 * trade is real: [FAST] resamples the generator output up to the size one pass can finish
 * from, [QUALITY] never resamples upward and lets the network do all the enlarging.
 */
enum class UpscaleStrategy {
  /**
   * At most one network pass. Anything the pass cannot cover is made up with a Lanczos
   * resample *before* it, so the network sees a smooth enlargement and sharpens it. This is
   * the workflow tiled-SR pipelines use in practice and it is several times cheaper.
   */
  FAST,

  /**
   * Never enlarge by resampling. Run as many network passes as it takes to reach or exceed
   * the target, then resample *down* to land exactly on it. Downsampling is information-
   * preserving, so this is the sharper result — and on a ×4 network reaching 4K from 512
   * it means two passes and roughly five times the tiles.
   */
  QUALITY,
}

/** One super-resolution sweep in the plan. */
data class UpscalePass(val index: Int, val inputWidth: Int, val inputHeight: Int, val grid: TileGrid) {
  val outputWidth: Int get() = grid.dstWidth
  val outputHeight: Int get() = grid.dstHeight
  val tileCount: Int get() = grid.tiles.size
}

/**
 * The full route from the generator's fixed output size to a 4K image.
 *
 * Nothing on a phone generates 4096² directly — the attention cost of a diffusion model grows
 * quadratically in token count and the activations alone would exhaust the RAM budget several
 * times over. What does work is generating at the size the model was trained for and then
 * enlarging with a convolutional network that runs tile by tile in bounded memory. This class
 * works out that route and, importantly, what it will cost, *before* the user commits to it.
 */
data class RenderPlan(
  val baseWidth: Int,
  val baseHeight: Int,
  val target: OutputTarget,
  val strategy: UpscaleStrategy,
  val preScale: Float,
  val passes: List<UpscalePass>,
  val finalWidth: Int,
  val finalHeight: Int,
) {

  /** Total number of super-resolution model invocations. */
  val totalTiles: Int get() = passes.sumOf { it.tileCount }

  /** Peak accumulator size across passes — the number that decides whether this OOMs. */
  val peakAccumulatorBytes: Long get() = passes.maxOfOrNull { it.grid.accumulatorBytes() } ?: 0L

  /** True when the final step crops rather than just scaling, i.e. content is lost. */
  val cropsContent: Boolean
    get() {
      val produced = passes.lastOrNull()?.let { it.outputWidth.toFloat() / it.outputHeight }
        ?: (baseWidth.toFloat() / baseHeight)
      val wanted = target.width.toFloat() / target.height
      return kotlin.math.abs(produced - wanted) > 0.01f
    }

  /**
   * Estimated wall-clock for the upscale stage, from a *measured* per-tile time.
   *
   * Takes the measurement rather than baking in a constant: per-tile cost depends on which
   * accelerator the graph actually landed on, and a hardcoded guess would be fiction.
   */
  fun estimatedUpscaleMillis(measuredMillisPerTile: Double): Double =
    totalTiles * measuredMillisPerTile

  fun describe(): String = buildString {
    appendLine("${baseWidth}x$baseHeight -> ${target.label} (${finalWidth}x$finalHeight)")
    if (preScale != 1.0f) {
      appendLine("  pre-resample x%.3f (Lanczos)".format(preScale))
    }
    passes.forEach { p ->
      appendLine(
        "  pass ${p.index + 1}: ${p.inputWidth}x${p.inputHeight} -> " +
          "${p.outputWidth}x${p.outputHeight}, ${p.tileCount} tiles"
      )
    }
    appendLine("  total tiles: $totalTiles, peak accumulator: ${peakAccumulatorBytes / (1L shl 20)} MiB")
    if (cropsContent) appendLine("  NOTE: aspect differs from the generator's — the last step crops.")
  }

  companion object {

    /**
     * Works out how to get from a [baseWidth]x[baseHeight] generator output to [target].
     *
     * @param networkScale the upscaler's own factor, 4 for a ×4 ESRGAN-class network.
     * @param tileSize the upscaler's fixed input size.
     * @param maxPasses hard ceiling on network sweeps; two is already ~5x the tiles of one.
     */
    fun of(
      baseWidth: Int,
      baseHeight: Int,
      target: OutputTarget,
      strategy: UpscaleStrategy = UpscaleStrategy.FAST,
      networkScale: Int = 4,
      tileSize: Int = 256,
      maxPasses: Int = 2,
    ): RenderPlan {
      require(baseWidth > 0 && baseHeight > 0) { "base must be non-empty" }
      require(networkScale >= 1) { "networkScale must be at least 1" }

      // How much enlargement is needed to cover the target on its tighter axis.
      val needed = max(
        target.width.toFloat() / baseWidth,
        target.height.toFloat() / baseHeight,
      )

      if (needed <= 1.0f || networkScale == 1) {
        // Already big enough: no network pass, just resample to the target.
        return RenderPlan(
          baseWidth, baseHeight, target, strategy,
          preScale = 1.0f, passes = emptyList(),
          finalWidth = target.width, finalHeight = target.height,
        )
      }

      val passCount = when (strategy) {
        // Enough passes to reach the target without ever enlarging by resampling.
        UpscaleStrategy.QUALITY ->
          ceil(ln(needed.toDouble()) / ln(networkScale.toDouble())).toInt().coerceIn(1, maxPasses)
        // One pass, with a resample making up whatever it cannot cover.
        UpscaleStrategy.FAST -> 1
      }

      val networkTotal = pow(networkScale, passCount)
      // Resample before the passes so the chain lands exactly on the target.
      val preScale = needed / networkTotal

      var w = max(1, (baseWidth * preScale).roundToInt())
      var h = max(1, (baseHeight * preScale).roundToInt())

      val passes = (0 until passCount).map { i ->
        val grid = TileGrid.forTarget(w, h, networkScale, tileSize)
        val pass = UpscalePass(i, w, h, grid)
        w = grid.dstWidth
        h = grid.dstHeight
        pass
      }

      return RenderPlan(
        baseWidth, baseHeight, target, strategy,
        preScale = preScale, passes = passes,
        finalWidth = target.width, finalHeight = target.height,
      )
    }

    private fun pow(base: Int, exp: Int): Int {
      var r = 1
      repeat(exp) { r *= base }
      return r
    }
  }
}

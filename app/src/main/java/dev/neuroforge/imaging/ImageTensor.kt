package dev.neuroforge.imaging

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import androidx.core.graphics.createBitmap
import dev.neuroforge.core.InputRange

/** How pixel values map onto tensor values. */
enum class Normalization(val scale: Float, val bias: Float) {
  /** `[0, 255]` → `[0, 1]`. The common convention for vision networks. */
  UNIT(1f / 255f, 0f),

  /** `[0, 255]` → `[-1, 1]`. VAE decoders and most classifiers. */
  SIGNED(2f / 255f, -1f),

  /**
   * `[0, 255]` float, passed through unchanged.
   *
   * Rarer, and the one that fails invisibly if assumed wrong: the ESRGAN graph this app
   * uses wants this, and fed `[0, 1]` it returns values spanning roughly -5 to 10 — noise,
   * not a slightly worse image.
   */
  BYTE(1f, 0f);

  fun toTensor(byteValue: Int): Float = byteValue * scale + bias
  fun toPixel(tensorValue: Float): Int =
    (((tensorValue - bias) / scale) + 0.5f).toInt().coerceIn(0, 255)

  companion object {
    /** The convention a catalogue entry declares. */
    fun of(range: InputRange): Normalization = when (range) {
      InputRange.UNIT -> UNIT
      InputRange.SIGNED -> SIGNED
      InputRange.BYTE -> BYTE
    }
  }
}

/**
 * Conversions between [Bitmap]s and the flat float tensors LiteRT wants.
 *
 * Everything here is NHWC with interleaved RGB, which is the layout LiteRT's vision graphs
 * declare and the layout `Bitmap.getPixels` already produces — so no transpose is needed on
 * the hot path. A tiled 4K run does this 25 times on tiles and once on the full frame, so
 * the inner loops avoid per-pixel allocation and `Color.red`-style helper calls.
 */
object ImageTensor {

  /**
   * Reads a sub-rectangle of [bitmap] into a caller-owned NHWC float array.
   *
   * Takes the destination array so the tiled upscaler can reuse one buffer across every
   * tile rather than allocating a 256×256×3 array per invocation.
   */
  fun readRegion(
    bitmap: Bitmap,
    srcX: Int,
    srcY: Int,
    width: Int,
    height: Int,
    out: FloatArray,
    pixels: IntArray = IntArray(width * height),
    normalization: Normalization = Normalization.UNIT,
  ): FloatArray {
    require(out.size >= width * height * 3) {
      "output buffer too small: ${out.size} < ${width * height * 3}"
    }
    bitmap.getPixels(pixels, 0, width, srcX, srcY, width, height)
    for (i in 0 until width * height) {
      val p = pixels[i]
      val o = i * 3
      out[o] = normalization.toTensor((p shr 16) and 0xFF)
      out[o + 1] = normalization.toTensor((p shr 8) and 0xFF)
      out[o + 2] = normalization.toTensor(p and 0xFF)
    }
    return out
  }

  /** Whole-bitmap convenience over [readRegion]. */
  fun fromBitmap(
    bitmap: Bitmap,
    normalization: Normalization = Normalization.UNIT,
  ): FloatArray = readRegion(
    bitmap, 0, 0, bitmap.width, bitmap.height,
    FloatArray(bitmap.width * bitmap.height * 3),
    normalization = normalization,
  )

  /** Builds an ARGB_8888 bitmap from an NHWC float tensor. */
  fun toBitmap(
    tensor: FloatArray,
    width: Int,
    height: Int,
    normalization: Normalization = Normalization.UNIT,
  ): Bitmap {
    require(tensor.size >= width * height * 3) {
      "tensor too small for ${width}x$height: ${tensor.size}"
    }
    val pixels = IntArray(width * height)
    for (i in pixels.indices) {
      val o = i * 3
      val r = normalization.toPixel(tensor[o])
      val g = normalization.toPixel(tensor[o + 1])
      val b = normalization.toPixel(tensor[o + 2])
      pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
    val bitmap = createBitmap(width, height)
    bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
    return bitmap
  }

  /**
   * High-quality resample.
   *
   * Uses bilinear filtering with `isFilterBitmap`, which is what the platform offers; it is
   * not a true Lanczos kernel, so enlargements are softer than a desktop tool would give.
   * That softness is acceptable here precisely because a super-resolution pass runs *after*
   * it and re-synthesises the detail — but it is the reason
   * [dev.neuroforge.core.UpscaleStrategy.QUALITY] exists for people who would rather not
   * resample upward at all.
   */
  fun resample(source: Bitmap, width: Int, height: Int): Bitmap {
    if (source.width == width && source.height == height) return source
    val out = createBitmap(width, height)
    val canvas = Canvas(out)
    val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG).apply {
      isDither = true
    }
    canvas.drawBitmap(source, Rect(0, 0, source.width, source.height), Rect(0, 0, width, height), paint)
    return out
  }

  /** Centre-crops [source] to the given aspect ratio, then resamples onto the exact target. */
  fun cropAndFit(source: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
    val targetAspect = targetWidth.toFloat() / targetHeight
    val sourceAspect = source.width.toFloat() / source.height

    val cropW: Int
    val cropH: Int
    if (sourceAspect > targetAspect) {
      cropH = source.height
      cropW = (source.height * targetAspect).toInt().coerceAtMost(source.width)
    } else {
      cropW = source.width
      cropH = (source.width / targetAspect).toInt().coerceAtMost(source.height)
    }
    val x = (source.width - cropW) / 2
    val y = (source.height - cropH) / 2
    val cropped = Bitmap.createBitmap(source, x, y, cropW, cropH)
    return resample(cropped, targetWidth, targetHeight)
  }
}

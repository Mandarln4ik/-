package dev.neuroforge.core

/**
 * Channel ordering of an image-shaped tensor.
 *
 * Worth being explicit about rather than assuming: PyTorch models are CHW natively, Android
 * bitmaps are HWC, and a converter may or may not have inserted the transpose. Getting it
 * wrong does not crash — it produces an image whose colour channels are smeared across the
 * frame, which looks like a broken model rather than a layout bug.
 */
enum class TensorLayout { HWC, CHW }

/** Rearranges a CHW tensor into HWC in a fresh array. */
fun chwToHwc(src: FloatArray, channels: Int, height: Int, width: Int): FloatArray {
  require(src.size >= channels * height * width) {
    "tensor has ${src.size} elements, expected ${channels * height * width}"
  }
  val out = FloatArray(channels * height * width)
  val plane = height * width
  for (c in 0 until channels) {
    val base = c * plane
    for (i in 0 until plane) out[i * channels + c] = src[base + i]
  }
  return out
}

/** Rearranges an HWC tensor into CHW in a fresh array. */
fun hwcToChw(src: FloatArray, channels: Int, height: Int, width: Int): FloatArray {
  require(src.size >= channels * height * width) {
    "tensor has ${src.size} elements, expected ${channels * height * width}"
  }
  val out = FloatArray(channels * height * width)
  val plane = height * width
  for (c in 0 until channels) {
    val base = c * plane
    for (i in 0 until plane) out[base + i] = src[i * channels + c]
  }
  return out
}

/** Returns [src] as HWC, converting only when [layout] says it is not already. */
fun asHwc(src: FloatArray, layout: TensorLayout, channels: Int, height: Int, width: Int): FloatArray =
  when (layout) {
    TensorLayout.HWC -> src
    TensorLayout.CHW -> chwToHwc(src, channels, height, width)
  }

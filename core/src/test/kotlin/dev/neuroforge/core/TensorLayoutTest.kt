package dev.neuroforge.core

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

class TensorLayoutTest {

  @Test
  fun `chw to hwc interleaves the planes`() {
    // 3 channels, 1x2 image: [R0 R1][G0 G1][B0 B1] -> [R0 G0 B0][R1 G1 B1]
    val chw = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
    assertContentEquals(
      floatArrayOf(1f, 3f, 5f, 2f, 4f, 6f),
      chwToHwc(chw, channels = 3, height = 1, width = 2),
    )
  }

  @Test
  fun `hwc to chw is the inverse`() {
    val chw = FloatArray(3 * 4 * 5) { it.toFloat() }
    val round = hwcToChw(chwToHwc(chw, 3, 4, 5), 3, 4, 5)
    assertContentEquals(chw, round)
  }

  @Test
  fun `asHwc does not copy a tensor that is already hwc`() {
    val src = FloatArray(3 * 2 * 2)
    assertSame(src, asHwc(src, TensorLayout.HWC, 3, 2, 2))
  }

  @Test
  fun `asHwc converts a chw tensor`() {
    val chw = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
    assertContentEquals(
      floatArrayOf(1f, 3f, 5f, 2f, 4f, 6f),
      asHwc(chw, TensorLayout.CHW, 3, 1, 2),
    )
  }

  @Test
  fun `a short tensor is rejected`() {
    assertTrue(runCatching { chwToHwc(FloatArray(5), 3, 2, 2) }.isFailure)
  }
}

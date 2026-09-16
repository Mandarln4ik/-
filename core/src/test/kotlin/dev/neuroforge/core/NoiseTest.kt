package dev.neuroforge.core

import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class NoiseTest {

  @Test
  fun `same seed gives the same stream`() {
    val a = latentNoise(seed = 7, channels = 4, height = 8, width = 8)
    val b = latentNoise(seed = 7, channels = 4, height = 8, width = 8)
    assertContentEquals(a, b, "a seed that does not reproduce is not a seed")
  }

  @Test
  fun `different seeds diverge`() {
    val a = latentNoise(seed = 7, channels = 4, height = 8, width = 8)
    val b = latentNoise(seed = 8, channels = 4, height = 8, width = 8)
    assertTrue(a.indices.any { abs(a[it] - b[it]) > 1e-6f })
  }

  @Test
  fun `noise is shaped like a standard normal`() {
    val n = 200_000
    val xs = Pcg32(1234).gaussians(n)
    val mean = xs.sum().toDouble() / n
    val variance = xs.sumOf { (it - mean) * (it - mean) } / (n - 1)
    val sd = sqrt(variance)

    // Standard error of the mean at this n is ~0.0022, so 0.02 is a wide but meaningful bound.
    assertTrue(abs(mean) < 0.02, "mean drifted: $mean")
    assertTrue(abs(sd - 1.0) < 0.02, "sd off: $sd")
  }

  @Test
  fun `uniforms stay strictly inside the open unit interval`() {
    // ln(0) in Box-Muller would produce infinities that poison the whole latent.
    val rng = Pcg32(99)
    repeat(100_000) {
      val u = rng.nextFloat()
      assertTrue(u > 0f && u < 1f, "uniform out of range: $u")
    }
  }

  @Test
  fun `gaussians are finite`() {
    val xs = Pcg32(5).gaussians(50_000)
    assertTrue(xs.all { it.isFinite() })
  }

  @Test
  fun `latent length matches the requested shape`() {
    assertEquals(4 * 64 * 64, latentNoise(1, 4, 64, 64).size)
  }

  @Test
  fun `token noise is drawn in the shape the transformer iterates on`() {
    // The DiT never sees a C x H x W grid; it sees (tokens, packedChannels). The two hold
    // the same number of values for the same image, so drawing the wrong one produces a
    // valid run with a seed that no longer means anything.
    val tokens = tokenNoise(seed = 3, tokens = Bonsai.TOKENS, packedChannels = Bonsai.PACKED_CHANNELS)
    assertEquals(Bonsai.TOKENS * Bonsai.PACKED_CHANNELS, tokens.size)
    assertEquals(latentNoise(3, 32, 64, 64).size, tokens.size)
    assertTrue(tokens.contentEquals(tokenNoise(3, Bonsai.TOKENS, Bonsai.PACKED_CHANNELS)))
    assertTrue(!tokens.contentEquals(tokenNoise(4, Bonsai.TOKENS, Bonsai.PACKED_CHANNELS)))
  }
}

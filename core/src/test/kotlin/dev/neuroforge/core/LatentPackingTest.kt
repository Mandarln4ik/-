package dev.neuroforge.core

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LatentPackingTest {

  @Test
  fun `patchify then unpatchify is the identity`() {
    // The property that matters: a scrambled index order still round-trips if both halves
    // are wrong in the same way, so this is necessary but not sufficient - see the
    // hand-checked layout test below.
    val spec = LatentSpec.FLUX
    val h = 8
    val w = 8
    val latent = FloatArray(spec.channels * h * w) { it * 0.01f }
    val tokens = LatentPacking.patchify(latent, spec, h, w)
    assertContentEquals(latent, LatentPacking.unpatchify(tokens, spec, h, w))
  }

  @Test
  fun `token tensor has the shape the transformer declares`() {
    val spec = LatentSpec.FLUX
    val tokens = LatentPacking.patchify(FloatArray(spec.channels * 64 * 64), spec, 64, 64)
    assertEquals(spec.tokenCount(512, 512) * spec.tokenDim(), tokens.size)
    // 512px image, f8 VAE, 2x2 patches -> 32x32 = 1024 tokens of 16*4 = 64 components.
    assertEquals(1024, spec.tokenCount(512, 512))
    assertEquals(64, spec.tokenDim())
  }

  @Test
  fun `patch components are laid out channel-major then row then column`() {
    // One channel, 2x2 latent, patch 2: exactly one token whose four components must be the
    // patch read in raster order. Hand-checked so a transposed patch cannot hide behind the
    // round trip.
    val spec = LatentSpec(channels = 1, vaeScale = 8, patch = 2)
    val latent = floatArrayOf(
      1f, 2f,
      3f, 4f,
    )
    val tokens = LatentPacking.patchify(latent, spec, 2, 2)
    assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), tokens)
  }

  @Test
  fun `tokens are ordered row-major across the latent grid`() {
    val spec = LatentSpec(channels = 1, vaeScale = 8, patch = 2)
    // 2x4 latent -> one row of two tokens: left patch then right patch.
    val latent = floatArrayOf(
      1f, 2f, 5f, 6f,
      3f, 4f, 7f, 8f,
    )
    val tokens = LatentPacking.patchify(latent, spec, 2, 4)
    assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f), tokens)
  }

  @Test
  fun `channels are separated within a token`() {
    val spec = LatentSpec(channels = 2, vaeScale = 8, patch = 2)
    val latent = floatArrayOf(
      1f, 2f, 3f, 4f, // channel 0, 2x2
      9f, 8f, 7f, 6f, // channel 1, 2x2
    )
    val tokens = LatentPacking.patchify(latent, spec, 2, 2)
    assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f, 9f, 8f, 7f, 6f), tokens)
  }

  @Test
  fun `patch size one passes the latent through untouched`() {
    val spec = LatentSpec.SD15
    val latent = FloatArray(4 * 8 * 8) { it.toFloat() }
    assertContentEquals(latent, LatentPacking.patchify(latent, spec, 8, 8))
  }

  @Test
  fun `a latent that does not divide by the patch size is rejected`() {
    val spec = LatentSpec.FLUX
    val e = runCatching { LatentPacking.patchify(FloatArray(16 * 7 * 8), spec, 7, 8) }
    assertTrue(e.isFailure, "an odd latent height must be rejected, not silently truncated")
  }

  @Test
  fun `latent geometry matches the known model families`() {
    assertEquals(64, LatentSpec.SD15.latentWidth(512))
    assertEquals(4 * 64 * 64, LatentSpec.SD15.latentElements(512, 512))
    assertEquals(16 * 64 * 64, LatentSpec.FLUX.latentElements(512, 512))
  }
}

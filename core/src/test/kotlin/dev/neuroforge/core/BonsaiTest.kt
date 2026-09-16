package dev.neuroforge.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Pins the tensor contract against the model's own reference host loop.
 *
 * Each of these once had a plausible wrong answer in this codebase, and none of the wrong
 * answers would have thrown: the graphs accept any tensor of the right shape. So the tests
 * check values, not shapes.
 */
class BonsaiTest {

  @Test
  fun `geometry matches generate dot py`() {
    assertEquals(256, Bonsai.SEQ)
    assertEquals(1024, Bonsai.TOKENS)
    assertEquals(32, Bonsai.LAT_GRID)
    assertEquals(128, Bonsai.PACKED_CHANNELS)
    assertEquals(Bonsai.TOKENS, Bonsai.LAT_GRID * Bonsai.LAT_GRID)
    assertEquals(Bonsai.PACKED_CHANNELS, Bonsai.latent.tokenDim())
    assertEquals(Bonsai.TOKENS, Bonsai.latent.tokenCount(Bonsai.IMAGE_SIZE, Bonsai.IMAGE_SIZE))
    // The packed latent and the VAE's input hold the same numbers in a different order.
    assertEquals(
      Bonsai.TOKENS * Bonsai.PACKED_CHANNELS,
      Bonsai.latent.latentElements(Bonsai.IMAGE_SIZE, Bonsai.IMAGE_SIZE),
    )
  }

  @Test
  fun `image position ids are zero, h, w, zero in row-major order`() {
    val ids = Bonsai.imagePositionIds()
    assertEquals(Bonsai.TOKENS * 4, ids.size)
    // np.meshgrid(..., indexing="ij") then reshape(TOKENS, 4): token h*32+w carries (h, w).
    for (h in 0 until Bonsai.LAT_GRID) {
      for (w in 0 until Bonsai.LAT_GRID) {
        val base = (h * Bonsai.LAT_GRID + w) * 4
        assertEquals(0f, ids[base], "batch slot at ($h,$w)")
        assertEquals(h.toFloat(), ids[base + 1], "h slot at ($h,$w)")
        assertEquals(w.toFloat(), ids[base + 2], "w slot at ($h,$w)")
        assertEquals(0f, ids[base + 3], "seq slot at ($h,$w)")
      }
    }
  }

  @Test
  fun `text position ids advance only on the sequence axis`() {
    val ids = Bonsai.textPositionIds()
    assertEquals(Bonsai.SEQ * 4, ids.size)
    for (i in 0 until Bonsai.SEQ) {
      assertEquals(0f, ids[i * 4])
      assertEquals(0f, ids[i * 4 + 1])
      assertEquals(0f, ids[i * 4 + 2])
      assertEquals(i.toFloat(), ids[i * 4 + 3])
    }
  }

  @Test
  fun `denormalize applies the affine per packed channel`() {
    // Two tokens, four packed channels: the vectors index the channel, not the token.
    val tokens = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
    val scale = floatArrayOf(2f, 0.5f, -1f, 10f)
    val shift = floatArrayOf(0f, 1f, 2f, 3f)
    val out = Bonsai.denormalize(tokens, scale, shift)
    assertTrue(out === tokens, "should write in place")
    assertEquals(floatArrayOf(2f, 2f, -1f, 43f, 10f, 4f, -5f, 83f).toList(), out.toList())
  }

  @Test
  fun `denormalize rejects a latent that is not a whole number of tokens`() {
    val e = runCatching {
      Bonsai.denormalize(FloatArray(5), FloatArray(4) { 1f }, FloatArray(4))
    }.exceptionOrNull()
    assertTrue(e is IllegalArgumentException, "got $e")
  }

  @Test
  fun `unpatchify places packed channel c times four plus i times two plus j`() {
    // The reference states the mapping outright: packed channel m = c*4 + i*2 + j lands at
    // z[c, 2h+i, 2w+j] for token (h, w). This reimplements that indexing directly and
    // checks LatentPacking agrees, so the two derivations have to fail together to hide a
    // scrambled image.
    val grid = 4
    val side = grid * 2
    val channels = 3
    val width = channels * 4
    val tokens = FloatArray(grid * grid * width) { it.toFloat() }

    val actual = LatentPacking.unpatchify(
      tokens, LatentSpec(channels = channels, vaeScale = 8, patch = 2), side, side,
    )

    for (h in 0 until grid) {
      for (w in 0 until grid) {
        for (c in 0 until channels) {
          for (i in 0 until 2) {
            for (j in 0 until 2) {
              val m = c * 4 + i * 2 + j
              val packed = tokens[(h * grid + w) * width + m]
              val unpacked = actual[c * side * side + (2 * h + i) * side + (2 * w + j)]
              assertEquals(packed, unpacked, "c=$c h=$h w=$w i=$i j=$j")
            }
          }
        }
      }
    }
  }

  @Test
  fun `the chat template wraps the prompt the way the encoder was trained to read it`() {
    // apply_chat_template([{user, prompt}], add_generation_prompt=True,
    // enable_thinking=False) on the Qwen3 template the pruned text encoder ships with.
    assertEquals(
      "<|im_start|>user\na red fox<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
      Bonsai.chatPrompt("a red fox"),
    )
  }

  @Test
  fun `the chat template is not the bare prompt`() {
    // The regression this guards: encoding the raw sentence is the obvious thing to do and
    // silently shifts every embedding the DiT is conditioned on.
    assertTrue(Bonsai.chatPrompt("x") != "x")
    assertTrue(Bonsai.chatPrompt("x").contains("<|im_start|>"))
  }

  @Test
  fun `the full latent decode matches a numpy run of the reference`() {
    // The strongest check in this file: it runs the reference's `unpatchify()` at its real
    // shape and compares against values computed by numpy from generate.py's own code.
    //
    // The inputs vary along both axes, so a transposed read, a wrong stride or the affine
    // applied after the unfold all move the answer. The step sizes between probes are
    // deliberately uneven — 64 is a row, 4096 is a channel — so an off-by-one in either
    // direction shows up.
    val tokens = FloatArray(Bonsai.TOKENS * Bonsai.PACKED_CHANNELS) { i ->
      val t = i / Bonsai.PACKED_CHANNELS
      val m = i % Bonsai.PACKED_CHANNELS
      ((t * 131 + m * 7) % 1000) / 1000.0f
    }
    val scale = FloatArray(Bonsai.PACKED_CHANNELS) { 1.0f + (it % 17) / 100.0f }
    val shift = FloatArray(Bonsai.PACKED_CHANNELS) { (it % 11) / 10.0f - 0.5f }

    val out = Bonsai.toVaeLatent(tokens, scale, shift)

    assertEquals(32 * 64 * 64, out.size)
    val expected = mapOf(
      0 to -0.500000f, 1 to -0.392930f, 63 to -0.331320f, 64 to -0.285720f,
      65 to -0.178370f, 4095 to -0.164980f, 4096 to -0.070880f, 65535 to 0.808480f,
      70000 to -0.081640f, 131071 to 1.074160f,
    )
    expected.forEach { (i, want) -> assertEquals(want, out[i], 1e-5f, "element $i") }

    // Aggregates too: the probes above could all agree while a whole region is wrong.
    var sum = 0.0
    out.forEach { sum += it }
    assertEquals(69083.09f, sum.toFloat(), 1f)
    assertEquals(1.648850f, out.max(), 1e-5f)
    assertEquals(-0.500000f, out.min(), 1e-5f)
  }

  @Test
  fun `decoding applies the affine before the unfold, not after`() {
    // Both orders run and both produce a correctly shaped latent. Only one is right, and
    // the wrong one scatters each channel's correction across the image.
    // A 2x2 grid, not 1x1: with a single token the unfold is the identity and the two
    // orders agree by accident, which would make this test pass while proving nothing.
    val spec = LatentSpec(channels = 2, vaeScale = 8, patch = 2)
    val side = 4
    val width = spec.tokenDim()
    val tokens = FloatArray(4 * width) { (it + 1).toFloat() }
    val scale = FloatArray(width) { if (it == 0) 10f else 1f }
    val shift = FloatArray(width)

    val correct = Bonsai.toVaeLatent(tokens.copyOf(), scale, shift, spec, latentSide = side)
    // Packed channel 0 is c=0, i=0, j=0, so token (0,0)'s scaled value lands at z[0][0][0].
    assertEquals(10f, correct[0])

    val wrongOrder = Bonsai.denormalize(
      LatentPacking.unpatchify(tokens.copyOf(), spec, side, side), scale, shift,
    )
    assertTrue(!wrongOrder.contentEquals(correct), "the two orders must be distinguishable")
  }
}

package dev.neuroforge.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class GraphBindingTest {

  @Test
  fun `arguments are placed by element count, whatever order the buffers arrive in`() {
    // The DiT's five inputs, as the reference feeds them, against a runtime that hands the
    // buffers back in a different order.
    val expected = intArrayOf(131072, 655360, 1, 4096, 1024)
    val declared = intArrayOf(1, 4096, 131072, 1024, 655360)
    val binding = GraphBinding.bind(declared, expected)!!
    assertEquals(GraphBinding.Basis.ELEMENT_COUNT, binding.basis)
    expected.indices.forEach { arg ->
      assertEquals(expected[arg], declared[binding[arg]], "argument $arg")
    }
  }

  @Test
  fun `an already-ordered graph binds to the identity`() {
    val counts = intArrayOf(131072, 655360, 1, 4096, 1024)
    val binding = GraphBinding.bind(counts, counts)!!
    assertEquals(GraphBinding.Basis.ELEMENT_COUNT, binding.basis)
    counts.indices.forEach { assertEquals(it, binding[it]) }
  }

  @Test
  fun `two arguments of the same length fall back to declared order`() {
    // The text encoder: input_ids and attention_mask are both (1, 256), so no count-based
    // rule can separate them and the declared order is all there is.
    val binding = GraphBinding.bind(intArrayOf(256, 256), intArrayOf(256, 256))!!
    assertEquals(GraphBinding.Basis.POSITIONAL, binding.basis)
    assertEquals(0, binding[0])
    assertEquals(1, binding[1])
  }

  @Test
  fun `a graph with a different arity does not bind`() {
    assertNull(GraphBinding.bind(intArrayOf(1, 2, 3), intArrayOf(1, 2)))
  }

  @Test
  fun `a length the graph has no buffer for does not bind`() {
    // Better to fail here than to run: a wrong binding produces a plausible image of noise.
    assertNull(GraphBinding.bind(intArrayOf(1, 4096, 1024), intArrayOf(1, 4096, 2048)))
  }

  @Test
  fun `ambiguous lengths that also disagree positionally do not bind`() {
    assertNull(GraphBinding.bind(intArrayOf(256, 512), intArrayOf(512, 512)))
  }

  @Test
  fun `the mismatch message names both sides`() {
    val message = GraphBinding.describeMismatch(
      "DiT", intArrayOf(7, 9), mapOf("latent" to 131072, "sigma" to 1),
    )
    listOf("DiT", "[7, 9]", "latent=131072", "sigma=1").forEach {
      kotlin.test.assertTrue(message.contains(it), "missing '$it' in: $message")
    }
  }
}

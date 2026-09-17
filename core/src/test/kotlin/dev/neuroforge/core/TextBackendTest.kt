package dev.neuroforge.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class TextBackendTest {

  @Test
  fun `only litert reaches the npu`() {
    // Not a ranking of the frameworks: a statement about the artifacts that ship. The
    // public ExecuTorch AAR carries XNNPACK and no vendor backend, and llama.cpp has no
    // MediaTek path at all. Claiming otherwise in the picker would be the same mistake as
    // claiming the diffusion model runs on the APU.
    assertTrue(TextBackend.LITERT_LM.reachesNpu)
    assertTrue(!TextBackend.EXECUTORCH.reachesNpu)
    assertTrue(!TextBackend.LLAMA_CPP.reachesNpu)
    assertEquals(
      listOf(TextBackend.LITERT_LM),
      TextBackend.entries.filter { it.reachesNpu },
    )
  }

  @Test
  fun `each backend owns one file format`() {
    val extensions = TextBackend.entries.map { it.extension }
    assertEquals(extensions.size, extensions.toSet().size, "two backends claim one format")
    assertEquals(".litertlm", TextBackend.LITERT_LM.extension)
    assertEquals(".pte", TextBackend.EXECUTORCH.extension)
    assertEquals(".gguf", TextBackend.LLAMA_CPP.extension)
  }

  @Test
  fun `the file decides the backend, not the setting`() {
    // A .gguf cannot be opened by LiteRT-LM whatever is selected, so routing by extension
    // is the only thing that can work; the setting decides what gets offered.
    assertEquals(TextBackend.LLAMA_CPP, backendFor("qwen3-0.6b-q4_k_m.gguf"))
    assertEquals(TextBackend.LITERT_LM, backendFor("Gemma3-1B-IT_q4_ekv1280_mt6991.litertlm"))
    assertEquals(TextBackend.EXECUTORCH, backendFor("llama3_2-1b.pte"))
  }

  @Test
  fun `case does not matter and unknown formats route nowhere`() {
    assertEquals(TextBackend.LLAMA_CPP, backendFor("MODEL.GGUF"))
    assertNull(backendFor("model.tflite"))
    assertNull(backendFor("readme.md"))
    assertNull(backendFor(""))
  }

  @Test
  fun `a stored choice survives, and an unreadable one falls back to the npu path`() {
    // Same failure this app already shipped once with the accelerator policy: a rename
    // between versions leaves a string in preferences that no longer names anything, and
    // reading it during construction turned that into a launch crash.
    TextBackend.entries.forEach {
      assertEquals(it, TextBackend.fromStoredName(it.name))
    }
    assertEquals(TextBackend.LITERT_LM, TextBackend.fromStoredName(null))
    assertEquals(TextBackend.LITERT_LM, TextBackend.fromStoredName(""))
    assertEquals(TextBackend.LITERT_LM, TextBackend.fromStoredName("REMOVED_LATER"))
  }

  @Test
  fun `every backend says what it trades away`() {
    TextBackend.entries.forEach { backend ->
      assertTrue(backend.summary.isNotBlank(), "${backend.name} has no summary")
      // A CPU-only backend must say so where the choice is made, not in a doc somewhere.
      if (!backend.reachesNpu) {
        assertTrue(
          backend.summary.contains("CPU"),
          "${backend.name} does not admit it is CPU-only: ${backend.summary}",
        )
      }
    }
  }
}

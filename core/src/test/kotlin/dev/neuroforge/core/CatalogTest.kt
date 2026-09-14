package dev.neuroforge.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CatalogTest {

  @Test
  fun `model ids are unique`() {
    val ids = ModelCatalog.all.map { it.id }
    assertEquals(ids.size, ids.toSet().size, "duplicate model id in the catalogue")
  }

  @Test
  fun `the text-to-4K pipeline has one graph per stage`() {
    val roles = ModelCatalog.textTo4kPipeline.map { it.role }
    assertEquals(
      listOf(ModelRole.TEXT_ENCODER, ModelRole.DENOISER, ModelRole.VAE_DECODER, ModelRole.UPSCALER),
      roles,
    )
  }

  @Test
  fun `the upscaler is the stage that targets the NPU`() {
    assertEquals(Accel.NPU, ModelCatalog.REAL_ESRGAN_X4.accelerators.first())
    // The int4 DiT is deliberately not an NPU target - blockwise int4 with dynamic attention
    // is not what an APU accelerates, and claiming otherwise would mislead.
    assertTrue(Accel.NPU !in ModelCatalog.BONSAI_DIT.accelerators)
  }

  @Test
  fun `models needing conversion are not advertised as ready to run`() {
    assertTrue(!ModelCatalog.REAL_ESRGAN_X4.isReadyToRun)
    assertTrue(ModelCatalog.BONSAI_DIT.isReadyToRun)
  }

  @Test
  fun `download urls point at the file inside its repo`() {
    val f = ModelCatalog.BONSAI_DIT.files.single()
    assertEquals(
      "https://huggingface.co/litert-community/Bonsai-Image-ternary-4B/resolve/main/dit_int4.tflite?download=true",
      f.downloadUrl(),
    )
  }

  @Test
  fun `byId round-trips`() {
    ModelCatalog.all.forEach { assertEquals(it, ModelCatalog.byId(it.id)) }
  }

  @Test
  fun `formatBytes is readable at every scale`() {
    assertEquals("512 B", formatBytes(512))
    assertEquals("1.0 KiB", formatBytes(1024))
    assertEquals("2.1 GiB", formatBytes(2_265_524_224L))
  }
}

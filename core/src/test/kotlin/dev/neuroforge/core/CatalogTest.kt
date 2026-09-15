package dev.neuroforge.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class CatalogTest {

  @Test
  fun `model ids are unique`() {
    val ids = ModelCatalog.all.map { it.id }
    assertEquals(ids.size, ids.toSet().size, "duplicate model id in the catalogue")
  }

  @Test
  fun `every model downloads and runs without a desktop step`() {
    // The whole point of this revision: no entry may need a conversion the phone cannot do.
    val needingConversion = ModelCatalog.all.filterNot { it.isReadyToRun }
    assertTrue(
      needingConversion.isEmpty(),
      "these still need a PC: ${needingConversion.map { it.id }}",
    )
  }

  @Test
  fun `the text-to-4K pipeline has one graph per stage`() {
    assertEquals(
      listOf(ModelRole.TEXT_ENCODER, ModelRole.DENOISER, ModelRole.VAE_DECODER, ModelRole.UPSCALER),
      ModelCatalog.textTo4kPipeline.map { it.role },
    )
  }

  @Test
  fun `the upscaler is the stage that targets the NPU`() {
    assertEquals(Accel.NPU, ModelCatalog.UPSCALER_ESRGAN_X4.accelerators.first())
    // The int4 DiT is deliberately not an NPU target: blockwise int4 with dynamic attention
    // is not what an APU accelerates, and claiming otherwise would mislead.
    assertTrue(Accel.NPU !in ModelCatalog.BONSAI_DIT.accelerators)
  }

  @Test
  fun `the upscaler geometry matches the graph that was measured`() {
    // Established by running esrgan_int8.tflite: input [1,128,128,3], output [1,512,512,3].
    val spec = ModelCatalog.UPSCALER_ESRGAN_X4
    assertEquals(128, spec.tileSize)
    assertEquals(4, spec.scale)
    assertContentEqualsInt(intArrayOf(1, 128, 128, 3), spec.inputShape!!)
  }

  @Test
  fun `the upscaler expects byte-range input`() {
    // Fed [0,1] this graph returns roughly -5..10 - noise, not a slightly worse image.
    // Encoding that in the catalogue is what stops it being rediscovered the hard way.
    assertEquals(InputRange.BYTE, ModelCatalog.UPSCALER_ESRGAN_X4.inputRange)
  }

  @Test
  fun `the upscaler archive member is named so the model is extracted, not its sidecar`() {
    val file = ModelCatalog.UPSCALER_ESRGAN_X4.files.single()
    assertEquals("esrgan_int8.tflite", file.archiveMember)
    assertTrue(file.isArchive)
    assertTrue(!Tar.isAppleDoubleSidecar(file.archiveMember!!))
  }

  @Test
  fun `the upscaler download is pinned by checksum`() {
    // Verified by downloading the asset and hashing it, so a corrupted or swapped file
    // is caught before it reaches the model loader.
    val file = ModelCatalog.UPSCALER_ESRGAN_X4.files.single()
    assertEquals(4_157_799L, file.sizeBytes)
    assertEquals(64, file.sha256?.length, "sha256 must be pinned for a direct download")
  }

  @Test
  fun `hugging face urls point at the file inside its repo`() {
    assertEquals(
      "https://huggingface.co/litert-community/Bonsai-Image-ternary-4B/resolve/main/dit_int4.tflite?download=true",
      ModelCatalog.BONSAI_DIT.files.single().downloadUrl(),
    )
  }

  @Test
  fun `direct urls are used verbatim`() {
    val url = ModelCatalog.UPSCALER_ESRGAN_X4.files.single().downloadUrl()
    assertTrue(url.startsWith("https://github.com/"), url)
  }

  @Test
  fun `byId round-trips`() {
    ModelCatalog.all.forEach { assertEquals(it, ModelCatalog.byId(it.id)) }
  }

  @Test
  fun `auto policy prefers the npu where the model allows it`() {
    assertEquals(
      listOf(Accel.NPU, Accel.GPU, Accel.CPU),
      AcceleratorPolicy.AUTO.resolveFor(ModelCatalog.UPSCALER_ESRGAN_X4),
    )
  }

  @Test
  fun `auto policy falls back to what a cpu-only model supports`() {
    assertEquals(
      listOf(Accel.CPU),
      AcceleratorPolicy.AUTO.resolveFor(ModelCatalog.BONSAI_DIT),
    )
  }

  @Test
  fun `strict npu resolves to nothing for a model that cannot use it`() {
    // Empty means "refuse", not "run somewhere else". A strict mode that silently fell back
    // would answer the one question the mode exists to answer with a lie.
    assertTrue(AcceleratorPolicy.NPU_STRICT.resolveFor(ModelCatalog.BONSAI_DIT).isEmpty())
    assertEquals(
      listOf(Accel.NPU),
      AcceleratorPolicy.NPU_STRICT.resolveFor(ModelCatalog.UPSCALER_ESRGAN_X4),
    )
  }

  @Test
  fun `gpu policy never silently uses the npu`() {
    val resolved = AcceleratorPolicy.GPU_FIRST.resolveFor(ModelCatalog.UPSCALER_ESRGAN_X4)
    assertTrue(Accel.NPU !in resolved, "GPU policy resolved to $resolved")
  }

  @Test
  fun `npu capable list contains the benchmark and the upscaler`() {
    val ids = ModelCatalog.npuCapable.map { it.id }
    assertTrue(ModelCatalog.BENCHMARK_MOBILENET.id in ids)
    assertTrue(ModelCatalog.UPSCALER_ESRGAN_X4.id in ids)
  }

  @Test
  fun `a stored policy name round-trips`() {
    AcceleratorPolicy.entries.forEach {
      assertEquals(it, AcceleratorPolicy.fromStoredName(it.name))
    }
  }

  @Test
  fun `an unreadable stored policy falls back instead of failing`() {
    // This runs while the view model is being constructed, so anything that throws here
    // is a launch crash rather than a wrong setting.
    assertEquals(AcceleratorPolicy.AUTO, AcceleratorPolicy.fromStoredName(null))
    assertEquals(AcceleratorPolicy.AUTO, AcceleratorPolicy.fromStoredName(""))
    assertEquals(AcceleratorPolicy.AUTO, AcceleratorPolicy.fromStoredName("REMOVED_IN_A_LATER_VERSION"))
  }

  @Test
  fun `the exact expected file wins when it is present`() {
    val files = listOf("model_int8.tflite", "model.tflite", "readme.md")
    assertEquals("model.tflite", chooseModelFile(files, "model.tflite", listOf("int8")))
  }

  @Test
  fun `a renamed file is found by its hints`() {
    // The failure this exists for: the catalogued name 404'd on device because it was a
    // guess. Given the real listing, the int8 variant still has to be picked out.
    val files = listOf(
      "mobilenet_v3_small.tflite",
      "mobilenet_v3_small_int8_dynamic.tflite",
      "config.json",
    )
    assertEquals(
      "mobilenet_v3_small_int8_dynamic.tflite",
      chooseModelFile(files, "mobilenet_v3_small_int8_channelwise.tflite", listOf("int8")),
    )
  }

  @Test
  fun `more hint matches beat fewer`() {
    val files = listOf("dit_fp16.tflite", "dit_int4.tflite", "encoder_int4.tflite")
    assertEquals(
      "dit_int4.tflite",
      chooseModelFile(files, "missing.tflite", listOf("dit", "int4")),
    )
  }

  @Test
  fun `ties break toward the shorter name`() {
    val files = listOf("model_int8_experimental_v2.tflite", "model_int8.tflite")
    assertEquals("model_int8.tflite", chooseModelFile(files, "nope.tflite", listOf("int8")))
  }

  @Test
  fun `nothing matching the hints returns null rather than a wrong file`() {
    // Downloading the wrong graph is worse than failing: it fails later, inside the loader.
    val files = listOf("model_fp32.tflite")
    assertNull(chooseModelFile(files, "model_int8.tflite", listOf("int8")))
  }

  @Test
  fun `a listing with no models at all returns null`() {
    assertNull(chooseModelFile(listOf("README.md", "config.json"), "m.tflite", listOf("int8")))
  }

  @Test
  fun `with no hints any model is acceptable`() {
    assertEquals("only.tflite", chooseModelFile(listOf("only.tflite"), "other.tflite", emptyList()))
  }

  @Test
  fun `formatBytes is readable at every scale`() {
    assertEquals("512 B", formatBytes(512))
    assertEquals("1.0 KiB", formatBytes(1024))
    assertEquals("2.1 GiB", formatBytes(2_265_524_224L))
  }

  private fun assertContentEqualsInt(expected: IntArray, actual: IntArray) {
    assertEquals(expected.toList(), actual.toList())
  }
}

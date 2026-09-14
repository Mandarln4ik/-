package dev.neuroforge.core

/** What a graph does inside the pipeline. */
enum class ModelRole { TEXT_ENCODER, DENOISER, VAE_DECODER, UPSCALER, BENCHMARK }

/**
 * Which accelerator a graph should be *tried* on first.
 *
 * This is a preference, not a promise: [Accel] ordering is handed to LiteRT, which falls
 * back down the list when a graph has an op the target cannot take.
 */
enum class Accel { NPU, GPU, CPU }

/**
 * Where a model artifact actually comes from — the difference between "download and run"
 * and "you have to build this yourself first".
 *
 * Stated per-file and surfaced in the UI on purpose. A catalogue that quietly lists a URL
 * that has to be produced by a conversion script wastes the first hour of anyone's evening.
 */
enum class Provenance {
  /** Published as a ready-to-run LiteRT graph by its authors; download and go. */
  PUBLISHED_LITERT,

  /** Published upstream in another format (PyTorch/ONNX); needs the conversion recipe in docs/MODELS.md. */
  NEEDS_CONVERSION,

  /** Supplied by the user and side-loaded into the app's model directory. */
  USER_SUPPLIED,
}

/**
 * One downloadable file belonging to a model.
 *
 * @param sha256 lowercase hex digest, or null when it has not been pinned. Null means the
 *   downloader can only check the length — say so rather than pretending to verify.
 */
data class ModelFile(
  val fileName: String,
  val repoId: String,
  val repoPath: String,
  val sizeBytes: Long,
  val sha256: String? = null,
  val provenance: Provenance = Provenance.PUBLISHED_LITERT,
) {
  /** Hugging Face resolve URL for this file. */
  fun downloadUrl(endpoint: String = HF_ENDPOINT): String =
    "$endpoint/$repoId/resolve/main/$repoPath?download=true"

  companion object {
    const val HF_ENDPOINT = "https://huggingface.co"
  }
}

/**
 * A graph plus everything needed to run it.
 *
 * @param inputShape NHWC or flat, whatever the graph declares; used to size buffers and to
 *   sanity-check a downloaded file before the first run.
 * @param accelerators preference order handed to LiteRT.
 */
data class ModelSpec(
  val id: String,
  val displayName: String,
  val role: ModelRole,
  val files: List<ModelFile>,
  val accelerators: List<Accel>,
  val inputShape: IntArray? = null,
  val notes: String = "",
) {
  val totalBytes: Long get() = files.sumOf { it.sizeBytes }

  /** True when every file is ready to run as published. */
  val isReadyToRun: Boolean
    get() = files.all { it.provenance == Provenance.PUBLISHED_LITERT }

  override fun equals(other: Any?): Boolean = this === other || (other is ModelSpec && id == other.id)
  override fun hashCode(): Int = id.hashCode()
}

/**
 * The models this app knows how to run.
 *
 * ## Why the split between the generator and the upscaler
 *
 * The NPU is not a general-purpose accelerator. A MediaTek APU is at its best on a graph
 * with **static shapes and int8 convolutions**, compiled once and replayed — which describes
 * a super-resolution network exactly, and describes a 4-billion-parameter diffusion
 * transformer with int4 blockwise weights and dynamic attention not at all. So the pipeline
 * splits along that line:
 *
 *  - the **generator** runs on CPU/XNNPACK, where blockwise-int4 kernels actually exist;
 *  - the **upscaler** runs on the NPU, tile after tile, and that is where the silicon pays.
 *
 * That split is also what makes 4K reachable at all: nothing generates 4096² directly on a
 * phone, but generating 1024² and running 64 NPU tiles through a ×4 network does.
 */
object ModelCatalog {

  /**
   * Bonsai Image 4B — the only text-to-image model with *verified* LiteRT graphs today.
   *
   * A ternary-weight diffusion transformer on the FLUX.2-klein-4B architecture, step-distilled
   * to 4 sampling steps at 512×512. The ternary weights land losslessly in an int4 block-32
   * container, so 3.88 B parameters fit in 2.11 GiB. Peak memory is ~2.9 GiB because the three
   * graphs load sequentially and are freed between stages — which fits 12 GB of RAM with room
   * to spare, and is the reason this rather than an fp16 model.
   */
  val BONSAI_TEXT_ENCODER = ModelSpec(
    id = "bonsai.text_encoder",
    displayName = "Bonsai text encoder (Qwen3-4B, pruned)",
    role = ModelRole.TEXT_ENCODER,
    accelerators = listOf(Accel.CPU),
    files = listOf(
      ModelFile(
        fileName = "bonsai_text_encoder_int4.tflite",
        repoId = "litert-community/Bonsai-Image-ternary-4B",
        repoPath = "text_encoder_int4.tflite",
        sizeBytes = 1_803_886_592L,
      ),
      // The vocabulary travels with the weights. Ids produced by a different tokenizer do
      // not fail - they quietly encode a different prompt - so this is not optional.
      ModelFile(
        fileName = "tokenizer.json",
        repoId = "litert-community/Bonsai-Image-ternary-4B",
        repoPath = "tokenizer.json",
        sizeBytes = 0L,
      ),
    ),
    notes = "Only hidden layers 9/18/27 are read, so the top 9 layers and the LM head are pruned away.",
  )

  val BONSAI_DIT = ModelSpec(
    id = "bonsai.dit",
    displayName = "Bonsai DiT 3.88B (ternary, int4 block-32)",
    role = ModelRole.DENOISER,
    accelerators = listOf(Accel.CPU),
    files = listOf(
      ModelFile(
        fileName = "bonsai_dit_int4.tflite",
        repoId = "litert-community/Bonsai-Image-ternary-4B",
        repoPath = "dit_int4.tflite",
        sizeBytes = 2_265_524_224L,
      )
    ),
    notes = "Needs XNNPACK explicitly attached; without it the runtime falls back to reference " +
      "kernels that are orders of magnitude slower on blockwise-int4 weights.",
  )

  val BONSAI_VAE = ModelSpec(
    id = "bonsai.vae_decoder",
    displayName = "Bonsai VAE decoder (fp32)",
    role = ModelRole.VAE_DECODER,
    accelerators = listOf(Accel.GPU, Accel.CPU),
    files = listOf(
      ModelFile(
        fileName = "bonsai_vae_decoder.tflite",
        repoId = "litert-community/Bonsai-Image-ternary-4B",
        repoPath = "vae_decoder.tflite",
        sizeBytes = 204_010_496L,
      )
    ),
    notes = "Convolutional and fixed-shape — the one generator stage worth trying on GPU.",
  )

  /**
   * The ×4 super-resolution stage: 512² → 2048², or 1024² → 4096².
   *
   * Real-ESRGAN is convolution-only with a static 256×256×3 input, which is the shape an APU
   * compiles cleanly. It is published as PyTorch/ONNX rather than LiteRT, so it has to be
   * converted and int8-quantised once — `docs/MODELS.md` has the recipe. That is a one-off
   * on a desktop, not something the phone does.
   */
  val REAL_ESRGAN_X4 = ModelSpec(
    id = "upscaler.real_esrgan_x4",
    displayName = "Real-ESRGAN ×4 (int8, 256px tiles)",
    role = ModelRole.UPSCALER,
    accelerators = listOf(Accel.NPU, Accel.GPU, Accel.CPU),
    inputShape = intArrayOf(1, 256, 256, 3),
    files = listOf(
      ModelFile(
        fileName = "real_esrgan_x4_int8_256.tflite",
        repoId = "ai-forever/Real-ESRGAN",
        repoPath = "RealESRGAN_x4.pth",
        sizeBytes = 67_040_989L,
        provenance = Provenance.NEEDS_CONVERSION,
      )
    ),
    notes = "The NPU workhorse. One compiled graph replayed over every tile.",
  )

  /**
   * A deliberately tiny classifier used only to prove the NPU path end to end.
   *
   * Worth having separately from the real models: when a 2 GiB download fails to run you
   * want to already know whether the NPU works at all on this device, and MobileNetV2 int8
   * answers that in a few megabytes and a few milliseconds.
   */
  val MOBILENET_V2_INT8 = ModelSpec(
    id = "bench.mobilenet_v2_int8",
    displayName = "MobileNetV2 (int8) — NPU smoke test",
    role = ModelRole.BENCHMARK,
    accelerators = listOf(Accel.NPU, Accel.GPU, Accel.CPU),
    inputShape = intArrayOf(1, 224, 224, 3),
    files = listOf(
      ModelFile(
        fileName = "mobilenet_v2_int8.tflite",
        repoId = "litert-community/mobilenet_v2",
        repoPath = "mobilenet_v2_int8.tflite",
        sizeBytes = 3_577_760L,
        provenance = Provenance.NEEDS_CONVERSION,
      )
    ),
    notes = "Runs on every accelerator, so it is the honest way to compare NPU vs GPU vs CPU.",
  )

  /** Everything, for the model-manager screen. */
  val all: List<ModelSpec> = listOf(
    MOBILENET_V2_INT8,
    BONSAI_TEXT_ENCODER,
    BONSAI_DIT,
    BONSAI_VAE,
    REAL_ESRGAN_X4,
  )

  /** The graphs a full text-to-4K run needs, in the order the pipeline loads them. */
  val textTo4kPipeline: List<ModelSpec> =
    listOf(BONSAI_TEXT_ENCODER, BONSAI_DIT, BONSAI_VAE, REAL_ESRGAN_X4)

  fun byId(id: String): ModelSpec? = all.firstOrNull { it.id == id }
}

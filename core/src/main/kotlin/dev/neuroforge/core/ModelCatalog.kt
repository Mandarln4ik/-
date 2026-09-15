package dev.neuroforge.core

/** What a graph does inside the pipeline. */
enum class ModelRole { TEXT_ENCODER, DENOISER, VAE_DECODER, UPSCALER, BENCHMARK }

/** An inference target. */
enum class Accel { NPU, GPU, CPU }

/**
 * How the caller should choose an accelerator.
 *
 * [NPU_STRICT] exists because a silent fallback is the thing that makes "NPU accelerated"
 * meaningless: without it you cannot tell a graph that ran on the APU from one that quietly
 * ran on the CPU three times slower. In strict mode a graph the NPU will not take fails
 * loudly, which is the answer you actually wanted.
 */
enum class AcceleratorPolicy(val label: String, val order: List<Accel>, val strict: Boolean) {
  AUTO("Auto (NPU → GPU → CPU)", listOf(Accel.NPU, Accel.GPU, Accel.CPU), strict = false),
  NPU_STRICT("NPU only (fail if unsupported)", listOf(Accel.NPU), strict = true),
  GPU_FIRST("GPU → CPU", listOf(Accel.GPU, Accel.CPU), strict = false),
  CPU_ONLY("CPU only", listOf(Accel.CPU), strict = true);

  /**
   * Narrows this policy to what [spec] can actually use.
   *
   * A policy is a preference, not a claim that the model supports the target: asking for
   * the NPU on a graph whose weights are blockwise int4 cannot work, and pretending
   * otherwise just moves the failure later.
   */
  fun resolveFor(spec: ModelSpec): List<Accel> {
    val allowed = order.filter { it in spec.accelerators }
    return when {
      allowed.isNotEmpty() -> allowed
      strict -> emptyList()
      else -> spec.accelerators
    }
  }
}

/** How pixel values map onto the tensor values a graph expects. */
enum class InputRange {
  /** `[0, 1]` — the common convention for vision networks. */
  UNIT,

  /** `[-1, 1]` — VAE decoders and most classifiers. */
  SIGNED,

  /** `[0, 255]` float. Rarer, and the one that silently produces garbage if assumed wrong. */
  BYTE,
}

/**
 * Where a model artifact comes from — the difference between "download and run" and
 * "you have to build this yourself first".
 *
 * Stated per-file and surfaced in the UI, because a catalogue that lists a URL which has
 * to be produced by a desktop conversion script wastes the first hour of someone's evening.
 */
enum class Provenance {
  /** A ready-to-run LiteRT graph; download and go. */
  PUBLISHED_LITERT,

  /** Published upstream in another format; needs the recipe in docs/MODELS.md. */
  NEEDS_CONVERSION,

  /** Supplied by the user and side-loaded into the app's model directory. */
  USER_SUPPLIED,
}

/** Where a file is fetched from. */
sealed interface ModelSource {
  /** A file inside a Hugging Face repository. */
  data class HuggingFace(val repoId: String, val path: String) : ModelSource

  /** Any direct URL — a GitHub release asset, a mirror. */
  data class Direct(val url: String, val origin: String) : ModelSource
}

/**
 * One downloadable file belonging to a model.
 *
 * @param archiveMember when the download is a `.tar.gz`, the entry to extract from it.
 *   Unpacking on device rather than asking for a desktop step is the point.
 * @param sha256 lowercase hex digest of the *downloaded* bytes (the archive, when there is
 *   one), or null when it has not been pinned. Null means only the length can be checked —
 *   reported as such rather than as a verified file.
 */
data class ModelFile(
  val fileName: String,
  val source: ModelSource,
  val sizeBytes: Long,
  val sha256: String? = null,
  val archiveMember: String? = null,
  val provenance: Provenance = Provenance.PUBLISHED_LITERT,
) {
  val isArchive: Boolean get() = archiveMember != null

  fun downloadUrl(huggingFaceEndpoint: String = HF_ENDPOINT): String = when (source) {
    is ModelSource.HuggingFace ->
      "$huggingFaceEndpoint/${source.repoId}/resolve/main/${source.path}?download=true"
    is ModelSource.Direct -> source.url
  }

  /** Human-readable origin, for the model list. */
  fun originLabel(): String = when (source) {
    is ModelSource.HuggingFace -> source.repoId
    is ModelSource.Direct -> source.origin
  }

  companion object {
    const val HF_ENDPOINT = "https://huggingface.co"
  }
}

/**
 * A graph plus everything needed to run it.
 *
 * @param tileSize fixed input side for a tiled model; also the shape the NPU compiles once.
 * @param scale output enlargement factor, 1 for everything that is not an upscaler.
 */
data class ModelSpec(
  val id: String,
  val displayName: String,
  val role: ModelRole,
  val files: List<ModelFile>,
  val accelerators: List<Accel>,
  val inputShape: IntArray? = null,
  val inputRange: InputRange = InputRange.UNIT,
  val tileSize: Int? = null,
  val scale: Int = 1,
  val notes: String = "",
) {
  val totalBytes: Long get() = files.sumOf { it.sizeBytes }

  /** True when every file is ready to run as published. */
  val isReadyToRun: Boolean
    get() = files.all { it.provenance == Provenance.PUBLISHED_LITERT }

  /** True when the NPU is a target this model can plausibly use. */
  val npuCapable: Boolean get() = Accel.NPU in accelerators

  override fun equals(other: Any?): Boolean = this === other || (other is ModelSpec && id == other.id)
  override fun hashCode(): Int = id.hashCode()
}

/**
 * The models this app knows how to run.
 *
 * ## Why the generator is not on the NPU
 *
 * A MediaTek APU is at its best on a graph with **static shapes and integer convolutions**,
 * compiled once and replayed. That describes a super-resolution network exactly, and
 * describes a 4-billion-parameter diffusion transformer with blockwise-int4 weights and
 * dynamic attention not at all. So the pipeline splits along that line: the generator runs
 * on CPU/XNNPACK, where blockwise-int4 kernels exist, and the upscaler runs on the NPU,
 * tile after tile. No public diffusion model ships as an NPU-compatible LiteRT graph today;
 * claiming the generator runs on the APU would be claiming something that cannot be done.
 *
 * ## Everything here downloads and runs
 *
 * No entry requires a desktop conversion step. Where a model is published only as a
 * `.tar.gz` release asset, the app unpacks it.
 */
object ModelCatalog {

  /**
   * MobileNetV3-Small, int8 — the smoke test that answers "does the NPU work here".
   *
   * Worth having separately from the real models: when a multi-gigabyte graph fails to run
   * you want to already know whether the NPU works at all, and this answers that in a few
   * megabytes and a few milliseconds. It is also the only model every accelerator will
   * accept, which is what makes an NPU-vs-GPU-vs-CPU comparison meaningful.
   */
  val BENCHMARK_MOBILENET = ModelSpec(
    id = "bench.mobilenet_v3_small_int8",
    displayName = "MobileNetV3-Small (int8) — NPU smoke test",
    role = ModelRole.BENCHMARK,
    accelerators = listOf(Accel.NPU, Accel.GPU, Accel.CPU),
    inputShape = intArrayOf(1, 224, 224, 3),
    inputRange = InputRange.UNIT,
    files = listOf(
      ModelFile(
        fileName = "mobilenet_v3_small_int8.tflite",
        source = ModelSource.HuggingFace(
          repoId = "litert-community/MobileNet-v3-small",
          path = "mobilenet_v3_small_int8_channelwise.tflite",
        ),
        sizeBytes = 0L,
      )
    ),
    notes = "Channelwise int8 weights and int8 activations — the shape an APU compiles cleanly.",
  )

  /**
   * ESRGAN ×4 — the upscaler, and the stage the NPU actually earns its place on.
   *
   * Convolution-only with a fixed 128×128×3 input and a 512×512×3 output, so the graph is
   * compiled once and replayed over every tile. Published as a `.tar.gz` release asset,
   * which the app unpacks rather than sending anyone to a desktop.
   *
   * The I/O is float32 in **[0, 255]**, not the usual [0, 1]. That was established by
   * running the graph: fed [0, 1] it returns values spanning roughly -5 to 10, which is
   * not a subtly worse image but noise.
   */
  val UPSCALER_ESRGAN_X4 = ModelSpec(
    id = "upscaler.esrgan_x4_int8",
    displayName = "ESRGAN ×4 (int8 weights, 128px tiles)",
    role = ModelRole.UPSCALER,
    accelerators = listOf(Accel.NPU, Accel.GPU, Accel.CPU),
    inputShape = intArrayOf(1, 128, 128, 3),
    inputRange = InputRange.BYTE,
    tileSize = 128,
    scale = 4,
    files = listOf(
      ModelFile(
        fileName = "esrgan_int8.tflite",
        source = ModelSource.Direct(
          url = "https://github.com/margaretmz/esrgan-e2e-tflite-tutorial/releases/download/" +
            "v0.1.0/esrgan_int8.tar.gz",
          origin = "margaretmz/esrgan-e2e-tflite-tutorial",
        ),
        sizeBytes = 4_157_799L,
        sha256 = "7cb6e5f27bf43b896b0ae659d4f6dc600982e5b5fc5d0f12713f70f83bfe508b",
        archiveMember = "esrgan_int8.tflite",
      )
    ),
    notes = "The NPU workhorse: one compiled graph replayed over every tile. Input is " +
      "float32 in [0, 255].",
  )

  /**
   * Bonsai Image 4B — the only text-to-image model with verified LiteRT graphs today.
   *
   * A ternary-weight diffusion transformer on the FLUX.2-klein-4B architecture, step-
   * distilled to 4 sampling steps at 512×512. The ternary weights land losslessly in an
   * int4 block-32 container, so 3.88 B parameters fit in 2.11 GiB, and peak memory is
   * ~2.9 GiB because the graphs load sequentially and are freed between stages.
   */
  val BONSAI_TEXT_ENCODER = ModelSpec(
    id = "bonsai.text_encoder",
    displayName = "Bonsai text encoder (Qwen3-4B, pruned)",
    role = ModelRole.TEXT_ENCODER,
    accelerators = listOf(Accel.CPU),
    files = listOf(
      ModelFile(
        fileName = "bonsai_text_encoder_int4.tflite",
        source = ModelSource.HuggingFace(
          "litert-community/Bonsai-Image-ternary-4B", "text_encoder_int4.tflite",
        ),
        sizeBytes = 1_803_886_592L,
      ),
      // The vocabulary travels with the weights. Ids from a different tokenizer do not
      // fail - they quietly encode a different prompt - so this is not optional.
      ModelFile(
        fileName = "tokenizer.json",
        source = ModelSource.HuggingFace(
          "litert-community/Bonsai-Image-ternary-4B", "tokenizer.json",
        ),
        sizeBytes = 0L,
      ),
    ),
    notes = "Only hidden layers 9/18/27 are read, so the top 9 layers and the LM head prune away.",
  )

  val BONSAI_DIT = ModelSpec(
    id = "bonsai.dit",
    displayName = "Bonsai DiT 3.88B (ternary, int4 block-32)",
    role = ModelRole.DENOISER,
    accelerators = listOf(Accel.CPU),
    files = listOf(
      ModelFile(
        fileName = "bonsai_dit_int4.tflite",
        source = ModelSource.HuggingFace(
          "litert-community/Bonsai-Image-ternary-4B", "dit_int4.tflite",
        ),
        sizeBytes = 2_265_524_224L,
      )
    ),
    notes = "CPU by design: blockwise int4 with dynamic attention is not what an APU " +
      "accelerates. Needs XNNPACK explicitly attached.",
  )

  val BONSAI_VAE = ModelSpec(
    id = "bonsai.vae_decoder",
    displayName = "Bonsai VAE decoder (fp32)",
    role = ModelRole.VAE_DECODER,
    accelerators = listOf(Accel.GPU, Accel.CPU),
    inputRange = InputRange.SIGNED,
    files = listOf(
      ModelFile(
        fileName = "bonsai_vae_decoder.tflite",
        source = ModelSource.HuggingFace(
          "litert-community/Bonsai-Image-ternary-4B", "vae_decoder.tflite",
        ),
        sizeBytes = 204_010_496L,
      )
    ),
    notes = "Convolutional and fixed-shape — the one generator stage worth trying on GPU.",
  )

  /** Everything, for the model-manager screen. */
  val all: List<ModelSpec> = listOf(
    BENCHMARK_MOBILENET,
    UPSCALER_ESRGAN_X4,
    BONSAI_TEXT_ENCODER,
    BONSAI_DIT,
    BONSAI_VAE,
  )

  /** The graphs a full text-to-4K run needs, in the order the pipeline loads them. */
  val textTo4kPipeline: List<ModelSpec> =
    listOf(BONSAI_TEXT_ENCODER, BONSAI_DIT, BONSAI_VAE, UPSCALER_ESRGAN_X4)

  /** Models that can target the NPU at all. */
  val npuCapable: List<ModelSpec> = all.filter { it.npuCapable }

  fun byId(id: String): ModelSpec? = all.firstOrNull { it.id == id }
}

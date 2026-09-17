package dev.neuroforge.core

/** What a graph does inside the pipeline. */
enum class ModelRole { TEXT_ENCODER, TEXT_CHAT, DENOISER, VAE_DECODER, UPSCALER, AUDIO, BENCHMARK }

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

  companion object {
    /**
     * Restores a policy from whatever was persisted, falling back to [AUTO].
     *
     * A stored name can outlive the constant it referred to — a rename between app
     * versions leaves an unreadable string in preferences, and the app must not fail to
     * start over one. Pure and unit-tested because it runs during construction, where a
     * defect shows up as a launch crash rather than a bad reading.
     */
    fun fromStoredName(name: String?): AcceleratorPolicy =
      entries.firstOrNull { it.name == name } ?: AUTO
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
  /**
   * A file inside a Hugging Face repository.
   *
   * @param path the expected file name. Repositories rename and re-quantise files without
   *   warning, and a name that is merely plausible produces a 404 at the worst moment — on
   *   someone's phone, at the point they press Download. So [hints] exists: when [path] is
   *   not there, the repository is listed and the file is picked by these instead.
   * @param hints lowercase substrings that *rank* the candidates — typically the
   *   quantisation recipe. A file matching none of them is still a candidate.
   * @param requires lowercase substrings a candidate *must* contain. This is for telling
   *   apart the different models in one repository: the Bonsai release holds a DiT, a text
   *   encoder and a VAE, and picking the wrong one is worse than reporting nothing found,
   *   because it fails later and somewhere else. Leave it empty in a single-model
   *   repository, where whatever is published is by definition the file wanted.
   * @param gated true when the repository requires accepting a licence and an access token.
   *   Worth stating in the catalogue rather than discovering as an HTTP 401 halfway through
   *   a download: the remedy is a person visiting a web page, which no retry will achieve.
   */
  data class HuggingFace(
    val repoId: String,
    val path: String,
    val hints: List<String> = emptyList(),
    val requires: List<String> = emptyList(),
    val gated: Boolean = false,
  ) : ModelSource {
    /**
     * The file extension to search for when [path] turns out not to exist.
     *
     * Derived rather than declared: a `.litertlm` engine bundle, a `.json` tokenizer and a
     * `.tflite` graph all go through the same resolution path, and defaulting the search to
     * `.tflite` meant a renamed `.litertlm` could never be found — the listing was fetched
     * and then filtered down to nothing.
     */
    val extension: String get() = "." + path.substringAfterLast('.', "tflite")
  }

  /** Any direct URL — a GitHub release asset, a mirror. */
  data class Direct(val url: String, val origin: String) : ModelSource
}

/**
 * Picks the right model file from a repository listing.
 *
 * Kept pure and tested because the alternative — a hardcoded file name — is what broke:
 * the name was a plausible guess from a search result, could not be verified offline, and
 * 404'd on the device. Scoring a real listing removes the guess.
 *
 * Preference order: the exact expected name; then the candidate matching the most [hints],
 * breaking ties toward the shortest name so `model_int8.tflite` wins over
 * `model_int8_experimental_v2.tflite`.
 */
fun chooseModelFile(
  available: List<String>,
  expected: String,
  hints: List<String>,
  extension: String = ".tflite",
  requires: List<String> = emptyList(),
): String? {
  val candidates = available
    .filter { it.endsWith(extension, ignoreCase = true) }
    // A required substring identifies *which* model, so a candidate missing one is a
    // different graph, not a differently-quantised version of this one.
    .filter { name -> requires.all { name.contains(it, ignoreCase = true) } }
  if (candidates.isEmpty()) return null
  candidates.firstOrNull { it.equals(expected, ignoreCase = true) }?.let { return it }

  // Hints only rank. A repository that publishes a q8 build where the hints ask for int4
  // still has exactly one file worth downloading, and returning null there would turn a
  // usable model into "not found".
  return candidates
    .map { name ->
      val lower = name.lowercase()
      name to hints.count { lower.contains(it.lowercase()) }
    }
    .minWithOrNull(
      compareByDescending<Pair<String, Int>> { it.second }.thenBy { it.first.length }
        .thenBy { it.first },
    )
    ?.first
}

/**
 * Puts this device's SoC at the front of a file entry's ranking hints.
 *
 * Some repositories publish one bundle per accelerator — `..._mt6991.litertlm`,
 * `..._sm8750.litertlm`, `..._Google_Tensor_G5.litertlm`. Those are graphs already compiled
 * for one specific NPU, and on the matching phone one of them is worth more than any
 * quantisation preference: it is the difference between a model that reaches the APU and
 * one that falls back to the CPU. So the SoC outranks everything else when it is known.
 *
 * A repository with no per-SoC build is unaffected: an unmatched hint only fails to add to
 * a candidate's score.
 *
 * @param socModel `Build.SOC_MODEL`, e.g. "MT6991". Blank, null and the "unknown" the
 *   platform reports for a device that will not say are all ignored.
 */
fun hintsForDevice(declared: List<String>, socModel: String?): List<String> {
  val soc = socModel?.trim()?.lowercase().orEmpty()
  if (soc.isEmpty() || soc == "unknown" || soc in declared.map { it.lowercase() }) return declared
  return listOf(soc) + declared
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

  /**
   * True when this entry's bytes were actually inspected rather than inferred.
   *
   * Only a pinned [sha256] means someone downloaded the file and hashed it. Without one,
   * [sizeBytes] is a figure read off a model card — useful for a progress bar, and no basis
   * for rejecting a download. Treating a guessed size as a gate turns a slightly stale
   * catalogue into "nothing downloads", which is exactly what happened.
   */
  val verified: Boolean get() = sha256 != null

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
  /**
   * Whether this model can be shown an image.
   *
   * A property of the weights, not of the runtime: LiteRT-LM can carry a vision tower, but
   * a text-only Gemma 3 1B bundle running on it still cannot see anything. Attaching a
   * photo to a chat with a model that has no vision tower produces either an error from the
   * engine or, worse, a confident answer about an image it never received — so the app
   * checks this before sending rather than finding out afterwards.
   */
  val vision: Boolean = false,
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

  /** The converted Bonsai release: three graphs, the tokenizer and the latent statistics. */
  private const val BONSAI_REPO = "litert-community/Bonsai-Image-ternary-4B"

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
    displayName = "MobileNetV3-Small (int8 weights) — NPU smoke test",
    role = ModelRole.BENCHMARK,
    accelerators = listOf(Accel.NPU, Accel.GPU, Accel.CPU),
    inputShape = intArrayOf(1, 224, 224, 3),
    inputRange = InputRange.UNIT,
    files = listOf(
      ModelFile(
        fileName = "mobilenet_v3_small_wi8.tflite",
        source = ModelSource.HuggingFace(
          repoId = "litert-community/MobileNet-v3-small",
          // Verified against the real repository listing by CI, not guessed. Note the
          // naming: `wi8` (weight int8), not `int8` - an earlier hint list looking for
          // "int8" would have missed this file too.
          path = "mobilenet_v3_small_weight_only_wi8_afp32.tflite",
          hints = listOf("wi8", "weight_only"),
        ),
        sizeBytes = 0L,
      )
    ),
    notes = "Weight-only int8 with fp32 activations. That is what the repository publishes; " +
      "a fully integer graph would suit an APU better, so if the NPU refuses this one, that " +
      "is the reason rather than a broken NPU path.",
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
        fileName = "textenc_int4.tflite",
        source = ModelSource.HuggingFace(
          BONSAI_REPO, "textenc_int4.tflite",
          requires = listOf("textenc"), hints = listOf("int4"),
        ),
        // Deliberately 0: the README says "1.68 GiB", which is a rounded figure and not a
        // size. Only a number someone actually measured belongs here, and CI now prints the
        // Content-Length of every catalogued file so it can be pinned rather than estimated.
        sizeBytes = 0L,
      ),
      // Latent BatchNorm statistics and the graph file names. The reference host loop reads
      // this before anything else, and without the two 128-wide vectors in it the decoded
      // image comes out washed out rather than wrong in any way that raises an error.
      ModelFile(
        fileName = "pipeline_meta.json",
        source = ModelSource.HuggingFace(
          BONSAI_REPO, "pipeline_meta.json",
          requires = listOf("pipeline_meta"),
        ),
        sizeBytes = 0L,
      ),
      // The vocabulary lives in a `tokenizer/` directory in that repository - CI's listing
      // showed it - so the path has a prefix and the local name does not.
      ModelFile(
        fileName = "tokenizer.json",
        source = ModelSource.HuggingFace(
          BONSAI_REPO, "tokenizer/tokenizer.json",
          requires = listOf("tokenizer.json"),
        ),
        sizeBytes = 0L,
      ),
      // Carries the pad token and the chat template. Both are things this app would
      // otherwise have to assume, and assuming either one shifts every embedding.
      ModelFile(
        fileName = "tokenizer_config.json",
        source = ModelSource.HuggingFace(
          BONSAI_REPO, "tokenizer/tokenizer_config.json",
          requires = listOf("tokenizer_config"),
        ),
        sizeBytes = 0L,
      ),
    ),
    notes = "Only hidden layers 9/18/27 are read, so the top 9 layers and the LM head prune " +
      "away. Takes 256 prompt tokens wrapped in the Qwen3 chat template.",
  )

  val BONSAI_DIT = ModelSpec(
    id = "bonsai.dit",
    displayName = "Bonsai DiT 3.88B (ternary, int4 block-32)",
    role = ModelRole.DENOISER,
    accelerators = listOf(Accel.CPU),
    files = listOf(
      ModelFile(
        fileName = "dit_int4b32.tflite",
        source = ModelSource.HuggingFace(
          BONSAI_REPO, "dit_int4b32.tflite",
          // Not `dit_gpu_*`: that is a separate GPU-targeted export of the same weights.
          requires = listOf("dit"), hints = listOf("int4b32"),
        ),
        sizeBytes = 0L,
      )
    ),
    notes = "Not an NPU target: blockwise int4 with dynamic attention is not what an APU " +
      "accelerates. The repository also publishes dit_gpu_int4b32.tflite, a GPU-targeted " +
      "export, so GPU is worth trying even though the NPU is not. Needs XNNPACK attached.",
  )

  val BONSAI_VAE = ModelSpec(
    id = "bonsai.vae_decoder",
    displayName = "Bonsai VAE decoder (fp32)",
    role = ModelRole.VAE_DECODER,
    accelerators = listOf(Accel.GPU, Accel.CPU),
    inputRange = InputRange.SIGNED,
    files = listOf(
      ModelFile(
        fileName = "vae_dec_fp32.tflite",
        source = ModelSource.HuggingFace(
          BONSAI_REPO, "vae_dec_fp32.tflite",
          requires = listOf("vae"), hints = listOf("dec", "fp32"),
        ),
        sizeBytes = 0L,
      )
    ),
    notes = "Convolutional and fixed-shape — the one generator stage worth trying on GPU.",
  )

  /**
   * Gemma 3 1B, instruction-tuned — a text model small enough to be worth running here.
   *
   * LiteRT-LM is a different runtime from the CompiledModel API the vision graphs use: it
   * owns the KV cache and the decode loop, which is why a chat cannot simply be built out
   * of the same pieces. It also exposes `Backend.NPU`, so this is the one place where a
   * generative model really can reach the APU.
   *
   * **This repository is gated.** CI found the file exactly where the catalogue says it is,
   * and a download of it still answers 401: Google requires accepting the Gemma licence,
   * which is a person visiting a web page and not something a retry can fix. So the app
   * says that, and Settings takes a Hugging Face token for anyone who has accepted it.
   *
   * It stays in the catalogue because of what else the listing showed: this repository
   * publishes per-SoC NPU builds — `..._mt6991.litertlm`, `..._sm8750.litertlm` and so on.
   * Those are graphs already compiled for one accelerator, which is the closest thing to a
   * real answer to "run the text model on the NPU" that exists today. The device's own SoC
   * is added to the ranking hints at download time, so a phone reporting MT6991 gets the
   * MT6991 build rather than the generic one.
   */
  val LLM_GEMMA3_1B = ModelSpec(
    id = "llm.gemma3_1b_it",
    displayName = "Gemma 3 1B Instruct",
    role = ModelRole.TEXT_CHAT,
    accelerators = listOf(Accel.NPU, Accel.GPU, Accel.CPU),
    files = listOf(
      ModelFile(
        fileName = "gemma3_1b_it.litertlm",
        source = ModelSource.HuggingFace(
          repoId = "litert-community/Gemma3-1B-IT",
          path = "gemma3-1b-it-int4.litertlm",
          // Ranked, not required: an int4 build is the one worth having on an APU, but a
          // repository that only publishes q8 still has exactly one usable file. The
          // device's SoC is prepended to this list at download time and outranks the rest,
          // because a graph compiled for this exact accelerator beats a better recipe.
          hints = listOf("int4", "q4", "ekv1280"),
          gated = true,
        ),
        sizeBytes = 0L,
      )
    ),
    notes = "Needs a Hugging Face token: Google gates this repository behind the Gemma " +
      "licence. Add one in Settings after accepting it on the model page. In exchange it " +
      "publishes per-SoC NPU builds — on a MediaTek MT6991 the app picks the MT6991 graph.",
  )

  val LLM_QWEN3_06B = ModelSpec(
    id = "llm.qwen3_0_6b",
    displayName = "Qwen3 0.6B",
    role = ModelRole.TEXT_CHAT,
    accelerators = listOf(Accel.NPU, Accel.GPU, Accel.CPU),
    files = listOf(
      ModelFile(
        fileName = "qwen3_0_6b_mixed_int4.litertlm",
        source = ModelSource.HuggingFace(
          repoId = "litert-community/Qwen3-0.6B",
          // Resolved from the repository listing by CI, not guessed.
          path = "qwen3_0_6b_mixed_int4.litertlm",
          hints = listOf("int4", "q4"),
        ),
        sizeBytes = 0L,
      )
    ),
    notes = "The smaller of the two, ungated, and 475 MiB — a better first try if memory " +
      "is tight or you would rather not sign anything.",
  )

  /**
   * Llama 3.2 1B as an ExecuTorch `.pte`.
   *
   * ExecuTorch is only worth offering if something can be loaded into it, and
   * `executorch-community` is where PyTorch's own exports land. This repository name came
   * out of the hub's own search rather than a guess: the first attempt here pointed at a
   * repository that answered 401, which is Hugging Face's reply for "private" and for
   * "does not exist" alike, and the probe now prints what does exist instead.
   *
   * SpinQuant is a 4-bit post-training quantisation, so this is about the size of the int4
   * LiteRT bundles and runs on the same phone.
   *
   * Two files, because a `.pte` is the graph alone and ExecuTorch takes the tokenizer as a
   * separate path.
   *
   * CPU, and marked as such. The public ExecuTorch runtime carries XNNPACK and no vendor
   * backend, so claiming an NPU target here would be the claim this app exists to avoid.
   */
  val LLM_LLAMA32_1B_PTE = ModelSpec(
    id = "llm.llama32_1b_executorch",
    displayName = "Llama 3.2 1B Instruct (ExecuTorch)",
    role = ModelRole.TEXT_CHAT,
    accelerators = listOf(Accel.CPU),
    files = listOf(
      ModelFile(
        fileName = "llama32_1b_spinquant.pte",
        source = ModelSource.HuggingFace(
          repoId = "executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET",
          path = "Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte",
          hints = listOf("spinquant", "int4", "1b"),
        ),
        sizeBytes = 0L,
      ),
      ModelFile(
        fileName = "tokenizer.model",
        source = ModelSource.HuggingFace(
          repoId = "executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET",
          path = "tokenizer.model",
          requires = listOf("tokenizer"),
        ),
        sizeBytes = 0L,
      ),
    ),
    notes = "Runs on ExecuTorch, which is CPU-only as published. Pick it to compare a " +
      "PyTorch export against the LiteRT path, not to go faster.",
  )

  /**
   * SmolLM2 135M as an ExecuTorch `.pte`.
   *
   * The smallest thing in the catalogue by an order of magnitude in parameters, and here as
   * a "does this backend work at all" model rather than as a useful assistant — at 135M the
   * answers show it.
   *
   * Not, however, a small download: this export is float, so 135M parameters come to 518
   * MiB — larger than the 469 MiB Qwen 0.5B GGUF, which has four times the parameters at
   * four bits each. What it is, is fast: eight times fewer parameters than the SpinQuant
   * Llama above and half the bytes to fetch, which is the reason to reach for it when the
   * question is whether ExecuTorch works at all rather than what it says.
   */
  val LLM_SMOLLM2_135M_PTE = ModelSpec(
    id = "llm.smollm2_135m_executorch",
    displayName = "SmolLM2 135M (ExecuTorch)",
    role = ModelRole.TEXT_CHAT,
    accelerators = listOf(Accel.CPU),
    files = listOf(
      ModelFile(
        fileName = "smollm2_135m.pte",
        source = ModelSource.HuggingFace(
          // The repository publishes exactly one graph, under the generic name.
          //
          // sizeBytes stays 0 here and everywhere else without a checksum: CI measured
          // 542,848,176 bytes, but a size the catalogue has not hashed is not allowed to
          // become a gate, and the downloader takes the real length from Content-Length
          // anyway. See CatalogTest.
          repoId = "executorch-community/SmolLM2-135M",
          path = "model.pte",
          hints = listOf("xnnpack", "8da4w", "q8"),
        ),
        sizeBytes = 0L,
      ),
      ModelFile(
        fileName = "tokenizer.json",
        source = ModelSource.HuggingFace(
          repoId = "executorch-community/SmolLM2-135M",
          path = "tokenizer.json",
          requires = listOf("tokenizer"),
        ),
        sizeBytes = 0L,
      ),
    ),
    notes = "135M parameters, but exported as float, so still a 518 MiB download. Quick to " +
      "run rather than quick to fetch, and far too small to be useful for anything but " +
      "checking the backend works.",
  )


  /**
   * Gemma 3 1B as GGUF, so the same model can be run on two different runtimes.
   *
   * That is most of why it is here. [LLM_GEMMA3_1B] is the same weights as a `.litertlm`
   * bundle that reaches the APU; this is the llama.cpp path on the CPU. Switching the
   * backend in Settings and asking both the same question is the only honest way to see
   * what the NPU is actually buying, and it needs the model held constant.
   *
   * The conversion is ggml's own rather than a community re-quantisation, and it is not
   * behind the Gemma licence gate the way Google's own repository is.
   */
  val LLM_GEMMA3_1B_GGUF = ModelSpec(
    id = "llm.gemma3_1b_it_gguf",
    displayName = "Gemma 3 1B Instruct (GGUF)",
    role = ModelRole.TEXT_CHAT,
    accelerators = listOf(Accel.CPU),
    files = listOf(
      ModelFile(
        fileName = "gemma3_1b_it_q4_k_m.gguf",
        source = ModelSource.HuggingFace(
          repoId = "ggml-org/gemma-3-1b-it-GGUF",
          path = "gemma-3-1b-it-Q4_K_M.gguf",
          // A GGUF repository publishes a dozen quantisations of one model, so the ranking
          // has to separate them rather than just find the family. Three overlapping hints
          // score Q4_K_M above Q4_K_S above Q4_0, which is the order worth having on a
          // phone: K-quants at 4 bits are the usual quality-per-byte sweet spot.
          hints = listOf("q4_k_m", "q4_k", "q4"),
        ),
        sizeBytes = 0L,
      )
    ),
    notes = "The same weights as the LiteRT Gemma 3 1B entry, in llama.cpp's format. Run " +
      "both and compare: this one is CPU, the other reaches the APU.",
  )

  /**
   * Qwen 2.5 0.5B as GGUF, from Qwen's own repository.
   *
   * The smallest thing here that still holds a conversation. It is a ChatML model where
   * [LLM_GEMMA3_1B_GGUF] is not, so between the two of them llama.cpp's chat templating is
   * exercised on both of the families it is most likely to meet.
   */
  val LLM_QWEN25_05B_GGUF = ModelSpec(
    id = "llm.qwen25_0_5b_gguf",
    displayName = "Qwen2.5 0.5B Instruct (GGUF)",
    role = ModelRole.TEXT_CHAT,
    accelerators = listOf(Accel.CPU),
    files = listOf(
      ModelFile(
        fileName = "qwen25_0_5b_instruct_q4_k_m.gguf",
        source = ModelSource.HuggingFace(
          repoId = "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
          path = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
          hints = listOf("q4_k_m", "q4_k", "q4"),
        ),
        sizeBytes = 0L,
      )
    ),
    notes = "469 MiB and ungated. Small enough to be the first thing to try on the " +
      "llama.cpp backend, and small enough that its answers show it.",
  )

  /**
   * Gemma 3n E2B — the vision model.
   *
   * The only thing in this catalogue that can be shown a photo. LiteRT-LM carries the
   * vision tower and starts a separate vision executor for it; everything else here is
   * text-only, which is why [ModelSpec.vision] is a per-model flag rather than a backend
   * one.
   *
   * E2B rather than E4B: "effective 2B" is what it costs to run, and the larger sibling
   * does not fit alongside a vision tower on a phone with other apps alive.
   */
  val LLM_GEMMA3N_E2B = ModelSpec(
    id = "llm.gemma3n_e2b",
    displayName = "Gemma 3n E2B (vision)",
    role = ModelRole.TEXT_CHAT,
    accelerators = listOf(Accel.NPU, Accel.GPU, Accel.CPU),
    vision = true,
    files = listOf(
      ModelFile(
        fileName = "gemma3n_e2b_it.litertlm",
        source = ModelSource.HuggingFace(
          repoId = "google/gemma-3n-E2B-it-litert-lm",
          path = "gemma-3n-E2B-it-int4.litertlm",
          hints = listOf("int4", "q4"),
          gated = true,
        ),
        sizeBytes = 0L,
      )
    ),
    notes = "The one model here that can look at a photo. Needs a Hugging Face token: " +
      "Google gates it behind the Gemma licence. Attaching an image to any other model " +
      "is refused rather than silently ignored.",
  )

  /** Everything, for the model-manager screen. */
  val all: List<ModelSpec> = listOf(
    BENCHMARK_MOBILENET,
    LLM_GEMMA3_1B,
    LLM_QWEN3_06B,
    LLM_LLAMA32_1B_PTE,
    LLM_SMOLLM2_135M_PTE,
    LLM_GEMMA3_1B_GGUF,
    LLM_QWEN25_05B_GGUF,
    LLM_GEMMA3N_E2B,
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

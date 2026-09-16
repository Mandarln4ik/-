package dev.neuroforge.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import dev.neuroforge.core.Accel
import dev.neuroforge.core.AcceleratorPolicy
import dev.neuroforge.core.Bonsai
import dev.neuroforge.core.FlowMatchEulerScheduler
import dev.neuroforge.core.GraphBinding
import dev.neuroforge.core.LatentPacking
import dev.neuroforge.core.LatentSpec
import dev.neuroforge.core.ModelCatalog
import dev.neuroforge.core.ModelSpec
import dev.neuroforge.core.OutputTarget
import dev.neuroforge.core.Pcg32
import dev.neuroforge.core.RenderPlan
import dev.neuroforge.core.Scheduler
import dev.neuroforge.core.TensorLayout
import dev.neuroforge.core.asHwc
import dev.neuroforge.core.tokenNoise
import dev.neuroforge.core.UpscaleStrategy
import dev.neuroforge.imaging.ImageTensor
import dev.neuroforge.imaging.Normalization
import dev.neuroforge.imaging.TiledUpscaler
import dev.neuroforge.models.ModelRepository
import dev.neuroforge.runtime.LiteRtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlin.coroutines.coroutineContext

/** What the user asked for. */
data class GenerationRequest(
  val prompt: String,
  val seed: Long = System.currentTimeMillis(),
  val steps: Int = Bonsai.DEFAULT_STEPS,
  val target: OutputTarget = OutputTarget.SQUARE_4K,
  val strategy: UpscaleStrategy = UpscaleStrategy.FAST,
  val upscale: Boolean = true,
  val policy: AcceleratorPolicy = AcceleratorPolicy.AUTO,
)

/** Where the run currently is. */
sealed interface GenerationStage {
  data object LoadingTextEncoder : GenerationStage
  data object Encoding : GenerationStage
  data class Denoising(val step: Int, val total: Int) : GenerationStage
  data object Decoding : GenerationStage
  data class Upscaling(val pass: Int, val passes: Int, val tile: Int, val tiles: Int) : GenerationStage
  data object Finishing : GenerationStage
}

data class GenerationProgress(
  val stage: GenerationStage,
  val fraction: Float,
  val detail: String,
  val elapsedMillis: Long,
)

/** Timings and the accelerator each stage actually landed on. */
data class StageTiming(val name: String, val accelerator: Accel, val millis: Long)

data class GenerationResult(
  val bitmap: Bitmap,
  val request: GenerationRequest,
  val plan: RenderPlan,
  val timings: List<StageTiming>,
) {
  val totalMillis: Long get() = timings.sumOf { it.millis }
}

/**
 * Runs a prompt all the way to a 4K image.
 *
 * ## The shape of the run
 *
 * ```
 *   prompt ─▶ chat template ─▶ 256 ids + mask ─▶ [ text encoder ] ─▶ embeddings
 *                                                                        │
 *   seed ──▶ noise (1024 × 128) ────────▶ [ DiT × steps ] ◀───────────────┘
 *                                                │
 *                             latent affine + unpatchify
 *                                                ▼
 *                                   (1, 32, 64, 64) ─▶ [ VAE ] ─▶ 512² image
 *                                                                        │
 *                                            ┌───────────────────────────┘
 *                                            ▼
 *                              resample ──▶ [ ×4 SR over N tiles ] ──▶ 4K image
 * ```
 *
 * ## Every number here was read, not inferred
 *
 * An earlier version of this file assumed the FLUX family's defaults — 16 latent channels,
 * 77 prompt tokens, a time-shift of 3.0, a `timestep` scaled by 1000, a three-input DiT —
 * and every one of them was wrong for this model. None of them would have thrown. The
 * graphs are fixed-shape and fully float; hand them a correctly-shaped tensor of wrong
 * numbers and they return a correctly-shaped image of noise. So the contract now comes
 * from [Bonsai], which mirrors the model's own reference host loop, and the pieces that
 * cannot be known until load time — how many elements each input buffer holds — are
 * checked rather than assumed.
 *
 * ## Why the graphs are loaded one at a time
 *
 * Together the three generator graphs are close to 4 GiB of weights. Holding all of them
 * resident would push peak memory past what the OS will give a single app even on a 12 GB
 * phone, so each stage is loaded, run, and closed before the next one opens. The measured
 * peak for the reference pipeline is ~2.9 GiB, and that number only holds because of this.
 *
 * It costs real time — the DiT alone is 2.1 GiB to read off UFS — which is why the progress
 * reporting names the stage that is loading rather than showing an unexplained stall.
 */
class GenerationPipeline(
  private val context: Context,
  private val repository: ModelRepository,
) {

  /**
   * The I/O contract each converted graph must satisfy.
   *
   * Written down rather than discovered, because a mismatch here does not crash — it
   * produces a plausible-looking wrong image. What the pipeline *can* check at load time
   * (input arity and element counts) it does check, and fails with a message naming the
   * mismatch.
   */
  data class GraphContract(
    val latent: LatentSpec = LatentSpec.BONSAI,
    val imageSize: Int = Bonsai.IMAGE_SIZE,
    val promptTokens: Int = Bonsai.SEQ,
    /**
     * Channel order of the VAE decoder's output.
     *
     * The reference reads it as `(1, 3, 512, 512)` and transposes to HWC before saving, so
     * CHW is not a guess here. It stays a setting because a re-export could insert the
     * transpose into the graph, and guessing wrong does not fail — it produces an image
     * with the colour channels smeared across the frame.
     */
    val vaeOutputLayout: TensorLayout = TensorLayout.CHW,
  ) {
    val latentSide: Int get() = imageSize / latent.vaeScale
    val tokenCount: Int get() = latent.tokenCount(imageSize, imageSize)
    val packedWidth: Int get() = latent.tokenDim()
    val packedElements: Int get() = tokenCount * packedWidth
  }

  suspend fun generate(
    request: GenerationRequest,
    contract: GraphContract = GraphContract(),
    onProgress: (GenerationProgress) -> Unit = {},
  ): GenerationResult = withContext(Dispatchers.Default) {
    val started = SystemClock.elapsedRealtime()
    val timings = mutableListOf<StageTiming>()
    fun report(stage: GenerationStage, fraction: Float, detail: String) =
      onProgress(GenerationProgress(stage, fraction, detail, SystemClock.elapsedRealtime() - started))

    val encoderSpec = ModelCatalog.BONSAI_TEXT_ENCODER
    requireReady(encoderSpec, ModelCatalog.BONSAI_DIT, ModelCatalog.BONSAI_VAE)

    // Latent normalisation travels with the checkpoint, so it is read before anything runs:
    // a missing file is better discovered now than after two gigabytes of weights have been
    // paged in and four denoising steps have gone by.
    val meta = PipelineMeta.read(repository.fileFor(encoderSpec, metaFileOf(encoderSpec)))
    require(meta.latentBnScale.size == contract.packedWidth) {
      "pipeline_meta.json describes a ${meta.latentBnScale.size}-wide latent but this " +
        "pipeline packs ${contract.packedWidth}. The metadata and the graphs are from " +
        "different releases."
    }

    // ---- 1. Text ---------------------------------------------------------------------
    report(GenerationStage.LoadingTextEncoder, 0f, "Loading text encoder (1.7 GiB)")
    val embeddings = stage(encoderSpec, timings, "text encoder", request.policy) { session ->
      coroutineContext.ensureActive()
      report(GenerationStage.Encoding, 0.05f, "Encoding prompt on ${session.accelerator}")

      // The encoder was trained on chat-formatted text and converted at a fixed length; the
      // mask is what stops it attending to the padding as though it were prompt.
      val encoded = PromptTokenizer.forModel(context, repository, encoderSpec)
        .encode(Bonsai.chatPrompt(request.prompt), contract.promptTokens)

      check(session.inputCount == 2) {
        "the text encoder declares ${session.inputCount} inputs; this pipeline feeds " +
          "(input_ids, attention_mask)"
      }
      // Both inputs are (1, 256), so their lengths cannot tell them apart and the declared
      // order is all there is to go on. This is the one place in the pipeline where that is
      // true, and the reference implementation flags it for the same reason.
      session.writeInput(0, encoded.ids)
      session.writeInput(1, encoded.mask)
      session.run()
      session.readOutput(0)
    }
    Log.i(TAG, "prompt embeddings: ${embeddings.size} elements")

    // ---- 2. Denoise ------------------------------------------------------------------
    // The latent lives in token space for the whole loop. The reference never packs or
    // unpacks between steps; it draws (1024, 128) noise and unpatchifies once at the end.
    val latent = tokenNoise(request.seed, contract.tokenCount, contract.packedWidth)
    val scheduler: Scheduler = FlowMatchEulerScheduler.forImage(contract.tokenCount, request.steps)
    for (i in latent.indices) latent[i] *= scheduler.initialNoiseScale
    val rng = Pcg32(request.seed xor SAMPLER_STREAM)

    val imageIds = Bonsai.imagePositionIds(contract.latentSide / contract.latent.patch)
    val textIds = Bonsai.textPositionIds(contract.promptTokens)

    stage(ModelCatalog.BONSAI_DIT, timings, "diffusion transformer", request.policy) { session ->
      // Argument order, in the order the model's forward() declares it.
      val wanted = intArrayOf(
        contract.packedElements, embeddings.size, 1, imageIds.size, textIds.size,
      )
      val declared = IntArray(session.inputCount) { session.inputElementCount(it) }
      val binding = GraphBinding.bind(declared, wanted) ?: error(
        GraphBinding.describeMismatch(
          "The diffusion transformer", declared,
          mapOf(
            "latent" to wanted[0], "embeddings" to wanted[1], "sigma" to wanted[2],
            "img_ids" to wanted[3], "txt_ids" to wanted[4],
          ),
        )
      )
      Log.i(TAG, "DiT inputs bound by ${binding.basis}: ${binding.slots.toList()}")

      // Three of the five inputs never change during the loop, so they are written once.
      session.writeInput(binding[ARG_EMBEDDINGS], embeddings)
      session.writeInput(binding[ARG_IMG_IDS], imageIds)
      session.writeInput(binding[ARG_TXT_IDS], textIds)

      scheduler.steps.forEach { step ->
        coroutineContext.ensureActive()
        report(
          GenerationStage.Denoising(step.index + 1, scheduler.steps.size),
          0.1f + 0.5f * (step.index.toFloat() / scheduler.steps.size),
          "Step ${step.index + 1}/${scheduler.steps.size} on ${session.accelerator}",
        )
        session.writeInput(binding[ARG_LATENT], latent)
        session.writeInput(binding[ARG_SIGMA], floatArrayOf(step.timestep))
        session.run()
        scheduler.step(step.index, latent, session.readOutput(0), rng)
      }
      Unit
    }

    // ---- 3. Decode -------------------------------------------------------------------
    report(GenerationStage.Decoding, 0.62f, "Decoding latent")
    // Affine first, on the packed axis, then the patch unfold — the only order in which a
    // 128-wide vector lines up with anything.
    Bonsai.denormalize(latent, meta.latentBnScale, meta.latentBnShift)
    val vaeLatent = LatentPacking.unpatchify(
      latent, contract.latent, contract.latentSide, contract.latentSide,
    )

    val baseImage = stage(ModelCatalog.BONSAI_VAE, timings, "VAE decoder", request.policy) { session ->
      session.writeInput(0, vaeLatent)
      session.run()
      val pixels = asHwc(
        session.readOutput(0),
        contract.vaeOutputLayout,
        channels = 3,
        height = contract.imageSize,
        width = contract.imageSize,
      )
      // The VAE emits [-1, 1]; the rest of the imaging path works in [0, 1].
      ImageTensor.toBitmap(pixels, contract.imageSize, contract.imageSize, Normalization.SIGNED)
    }

    // ---- 4. Upscale to 4K ------------------------------------------------------------
    val upscaler = ModelCatalog.UPSCALER_ESRGAN_X4
    val plan = RenderPlan.of(
      baseWidth = baseImage.width,
      baseHeight = baseImage.height,
      target = request.target,
      strategy = request.strategy,
      // Geometry comes from the graph that will actually run, not from a default: this
      // upscaler takes 128px tiles and emits x4, and a mismatch here would only surface
      // as a shape error on the first tile.
      networkScale = upscaler.scale,
      tileSize = upscaler.tileSize ?: 256,
    )
    Log.i(TAG, plan.describe())

    if (!request.upscale || plan.passes.isEmpty()) {
      report(GenerationStage.Finishing, 1f, "Done")
      val fitted = ImageTensor.cropAndFit(baseImage, plan.finalWidth, plan.finalHeight)
      return@withContext GenerationResult(fitted, request, plan, timings)
    }

    requireReady(upscaler)
    var current = baseImage
    stage(upscaler, timings, "upscaler", request.policy) { session ->
      val tiled = TiledUpscaler(session, Normalization.of(upscaler.inputRange))
      plan.passes.forEach { pass ->
        coroutineContext.ensureActive()
        current = ImageTensor.resample(current, pass.inputWidth, pass.inputHeight)
        current = tiled.upscale(current, pass.grid, pass.index) { p ->
          report(
            GenerationStage.Upscaling(pass.index + 1, plan.passes.size, p.tilesDone, p.tilesTotal),
            0.65f + 0.33f * ((pass.index + p.fraction) / plan.passes.size),
            "Pass ${pass.index + 1}/${plan.passes.size}, tile ${p.tilesDone}/${p.tilesTotal} " +
              "on ${session.accelerator} (%.0f ms/tile)".format(p.millisPerTile),
          )
        }
      }
      Unit
    }

    report(GenerationStage.Finishing, 1f, "Fitting to ${request.target.label}")
    val finalImage = ImageTensor.cropAndFit(current, plan.finalWidth, plan.finalHeight)
    GenerationResult(finalImage, request, plan, timings)
  }

  /**
   * Loads one graph, runs [block], and closes it before returning.
   *
   * `use` rather than a field: the next stage's weights cannot be read into memory until
   * this stage's are out of it.
   */
  private inline fun <T> stage(
    spec: ModelSpec,
    timings: MutableList<StageTiming>,
    label: String,
    policy: AcceleratorPolicy = AcceleratorPolicy.AUTO,
    block: (LiteRtSession) -> T,
  ): T {
    val file = repository.fileFor(spec, spec.files.first())
    val targets = policy.resolveFor(spec)
    check(targets.isNotEmpty()) {
      "$label supports ${spec.accelerators.joinToString("/")}, which the " +
        "'${policy.label}' policy excludes. Change the policy in Settings."
    }
    val t0 = SystemClock.elapsedRealtime()
    return LiteRtSession.load(context, spec.displayName, file, targets, policy.strict)
      .use { session ->
      val result = block(session)
      timings += StageTiming(label, session.accelerator, SystemClock.elapsedRealtime() - t0)
      result
    }
  }

  private fun metaFileOf(spec: ModelSpec) =
    spec.files.firstOrNull { it.fileName == "pipeline_meta.json" }
      ?: error("${spec.id} has no pipeline_meta.json entry in the catalogue")

  private fun requireReady(vararg specs: ModelSpec) {
    val missing = specs.filterNot { repository.isReady(it) }
    if (missing.isNotEmpty()) {
      error("Missing weights for: ${missing.joinToString { it.displayName }}. Download them first.")
    }
  }

  private companion object {
    const val TAG = "GenerationPipeline"

    // The DiT's arguments, in the order its forward() declares them:
    // dit(lat, embeds, sigma, img_ids, txt_ids).
    const val ARG_LATENT = 0
    const val ARG_EMBEDDINGS = 1
    const val ARG_SIGMA = 2
    const val ARG_IMG_IDS = 3
    const val ARG_TXT_IDS = 4

    /**
     * Keeps the ancestral sampler's noise on a different stream from the initial latent, so
     * two runs that share a seed but differ in step count still start from the same image
     * rather than diverging from step zero.
     */
    const val SAMPLER_STREAM = 0x5EED0000L
  }
}

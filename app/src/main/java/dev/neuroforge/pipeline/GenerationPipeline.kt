package dev.neuroforge.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import dev.neuroforge.core.Accel
import dev.neuroforge.core.AcceleratorPolicy
import dev.neuroforge.core.FlowMatchEulerScheduler
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
import dev.neuroforge.core.UpscaleStrategy
import dev.neuroforge.core.latentNoise
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
  val steps: Int = 4,
  val guidance: Float = 1.0f,
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
 *   prompt ──▶ text encoder ──▶ embeddings
 *                                  │
 *   seed ──▶ latent noise ──▶ [ DiT × steps ] ──▶ latent ──▶ VAE ──▶ 512² image
 *                                                                        │
 *                                            ┌───────────────────────────┘
 *                                            ▼
 *                              resample ──▶ [ ×4 SR over N tiles ] ──▶ 4K image
 * ```
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
   * produces a plausible-looking wrong image. The pipeline validates what it can (buffer
   * counts and element counts) at load time and fails with a message naming the mismatch.
   */
  data class GraphContract(
    val latent: LatentSpec = LatentSpec.FLUX,
    val imageSize: Int = 512,
    val maxPromptTokens: Int = 77,
    val embeddingDim: Int = 2560,
    /**
     * Channel order of the VAE decoder's output.
     *
     * Declared rather than guessed. The upstream decoder is CHW like every PyTorch vision
     * model; whether the converted graph still is depends on whether the conversion inserted
     * a transpose. Guessing wrong does not fail — it produces an image with the colour
     * channels smeared across the frame, which reads as a broken model rather than a
     * one-line layout bug, so it is a setting with a stated default.
     */
    val vaeOutputLayout: TensorLayout = TensorLayout.CHW,
  ) {
    val latentSide: Int get() = imageSize / latent.vaeScale
    val tokenCount: Int get() = latent.tokenCount(imageSize, imageSize)
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

    requireReady(ModelCatalog.BONSAI_TEXT_ENCODER, ModelCatalog.BONSAI_DIT, ModelCatalog.BONSAI_VAE)

    // ---- 1. Text ---------------------------------------------------------------------
    report(GenerationStage.LoadingTextEncoder, 0f, "Loading text encoder (1.7 GiB)")
    val embeddings = stage(ModelCatalog.BONSAI_TEXT_ENCODER, timings, "text encoder", request.policy) { session ->
      coroutineContext.ensureActive()
      report(GenerationStage.Encoding, 0.05f, "Encoding prompt on ${session.accelerator}")
      val ids = PromptTokenizer.forModel(context, repository, ModelCatalog.BONSAI_TEXT_ENCODER)
        .encodePadded(request.prompt, contract.maxPromptTokens)
      session.writeInput(0, ids)
      session.run()
      session.readOutput(0)
    }

    // ---- 2. Denoise ------------------------------------------------------------------
    val scheduler: Scheduler = FlowMatchEulerScheduler(numInferenceSteps = request.steps)
    val latentSide = contract.latentSide
    val latent = latentNoise(
      seed = request.seed,
      channels = contract.latent.channels,
      height = latentSide,
      width = latentSide,
    )
    for (i in latent.indices) latent[i] *= scheduler.initialNoiseScale
    val rng = Pcg32(request.seed xor SAMPLER_STREAM)

    stage(ModelCatalog.BONSAI_DIT, timings, "diffusion transformer", request.policy) { session ->
      check(session.inputCount >= 3) {
        "the diffusion graph declares ${session.inputCount} inputs; this pipeline feeds " +
          "(packed latent, prompt embeddings, timestep) and needs at least 3"
      }
      session.writeInput(1, embeddings)

      scheduler.steps.forEach { step ->
        coroutineContext.ensureActive()
        report(
          GenerationStage.Denoising(step.index + 1, scheduler.steps.size),
          0.1f + 0.5f * (step.index.toFloat() / scheduler.steps.size),
          "Step ${step.index + 1}/${scheduler.steps.size} on ${session.accelerator}",
        )

        val tokens = LatentPacking.patchify(latent, contract.latent, latentSide, latentSide)
        session.writeInput(0, tokens)
        session.writeInput(2, floatArrayOf(step.timestep))
        session.run()

        val velocity = LatentPacking.unpatchify(
          session.readOutput(0), contract.latent, latentSide, latentSide,
        )
        scheduler.step(step.index, latent, velocity, rng)
      }
      Unit
    }

    // ---- 3. Decode -------------------------------------------------------------------
    report(GenerationStage.Decoding, 0.62f, "Decoding latent")
    val baseImage = stage(ModelCatalog.BONSAI_VAE, timings, "VAE decoder", request.policy) { session ->
      session.writeInput(0, latent)
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

  private fun requireReady(vararg specs: ModelSpec) {
    val missing = specs.filterNot { repository.isReady(it) }
    if (missing.isNotEmpty()) {
      error("Missing weights for: ${missing.joinToString { it.displayName }}. Download them first.")
    }
  }

  private companion object {
    const val TAG = "GenerationPipeline"

    /**
     * Keeps the ancestral sampler's noise on a different stream from the initial latent, so
     * two runs that share a seed but differ in step count still start from the same image
     * rather than diverging from step zero.
     */
    const val SAMPLER_STREAM = 0x5EED0000L
  }
}

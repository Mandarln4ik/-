package dev.neuroforge.core

import kotlin.math.max
import kotlin.math.sqrt

/**
 * One denoising step's worth of state handed to the model.
 *
 * @param index zero-based position in the schedule.
 * @param timestep the value the model's `timestep` input expects, in the model's own units.
 * @param sigma the noise level at this step.
 */
data class DiffusionStep(val index: Int, val timestep: Float, val sigma: Float)

/**
 * A sampler: turns a model's per-step prediction into the next latent.
 *
 * Implementations are pure and allocation-light — [step] writes into [latents] in place,
 * because on a phone a 4×64×64 latent reallocated 20 times per image is 20 needless GC
 * pressure spikes on the exact thread the UI is waiting on.
 */
sealed interface Scheduler {

  /** The steps to run, in order. */
  val steps: List<DiffusionStep>

  /**
   * Scale factor the initial pure-noise latent is multiplied by.
   *
   * Flow-matching models start at sigma = 1 and want unit-variance noise; the eps-prediction
   * models start at sigma ≈ 14.6 and want the noise scaled up to match.
   */
  val initialNoiseScale: Float

  /**
   * Conditioning scale applied to the latent *before* it is handed to the model.
   *
   * Karras-style eps samplers divide by `sqrt(sigma² + 1)`; flow-matching models take the
   * latent as-is. Returning 1 means "no scaling".
   */
  fun scaleModelInput(stepIndex: Int): Float = 1.0f

  /**
   * Advances [latents] from step [stepIndex] to the next one, in place.
   *
   * @param modelOutput the network's raw prediction for this step (velocity or epsilon,
   *   depending on the implementation).
   * @param rng source of the extra noise ancestral samplers inject; unused by deterministic
   *   samplers.
   */
  fun step(stepIndex: Int, latents: FloatArray, modelOutput: FloatArray, rng: Pcg32)
}

/**
 * The time-shift a FLUX.2-klein schedule uses, as its reference implementation computes it.
 *
 * The shift is not a taste knob on these models: it is derived from how many tokens the
 * image occupies and how many steps are being taken, by interpolating two empirical fits
 * (at 200 and 10 steps) and exponentiating. Hardcoding a plausible value instead — this
 * code used 3.0 — puts every sigma in the schedule slightly wrong, which shows up as an
 * image that is soft and badly composed rather than as an error.
 *
 * Mirrors `flowmatch_sigmas` in the model's own `generate.py`, which in turn mirrors
 * `compute_empirical_mu` in diffusers.
 */
fun empiricalShift(tokens: Int, steps: Int): Float {
  val m200 = 0.00016927 * tokens + 0.45666666
  val m10 = 8.73809524e-05 * tokens + 1.89833333
  val a = (m200 - m10) / 190.0
  val mu = a * steps + (m200 - 200.0 * a)
  return kotlin.math.exp(mu).toFloat()
}

/**
 * Flow-matching Euler sampler — the one FLUX-family and SD3-family models are trained for,
 * and the one [dev.neuroforge.core.ModelCatalog.BONSAI_IMAGE_4B] needs.
 *
 * The model predicts a velocity `v`, and the update is a plain Euler step along it:
 * `x ← x + (σ_next − σ) · v`. The schedule is a linear ramp in sigma from 1 down to
 * `1/steps`, bent by [shift]: larger shift spends more of the budget at high noise, which
 * is what lets a 4-step schedule still resolve global structure.
 *
 * Mirrors `FlowMatchEulerDiscreteScheduler` from HuggingFace diffusers, in the form the
 * Bonsai reference host loop uses it.
 */
class FlowMatchEulerScheduler(
  numInferenceSteps: Int,
  private val shift: Float,
) : Scheduler {

  /** `sigmas[i]` for each step, with a trailing 0 so the last step lands on a clean latent. */
  private val sigmas: FloatArray

  override val steps: List<DiffusionStep>
  override val initialNoiseScale: Float get() = 1.0f

  init {
    require(numInferenceSteps > 0) { "numInferenceSteps must be positive, was $numInferenceSteps" }
    // `np.linspace(1.0, 1.0 / steps, steps)`. The lower endpoint is 1/steps — it moves with
    // the step count and is not the 1/numTrainTimesteps the SD-family schedulers use. With
    // the SD convention a 4-step run would start its ramp at 0.001 instead of 0.25 and
    // spend three of its four steps at essentially the same noise level.
    val raw = FloatArray(numInferenceSteps) { i ->
      val lo = 1.0f / numInferenceSteps
      if (numInferenceSteps == 1) 1.0f
      else 1.0f + (lo - 1.0f) * i / (numInferenceSteps - 1).toFloat()
    }
    sigmas = FloatArray(numInferenceSteps + 1)
    for (i in raw.indices) {
      // `exp(mu) / (exp(mu) + (1/s - 1))`, spelled without the reciprocal so s = 0 is safe.
      val s = raw[i]
      sigmas[i] = shift * s / (1.0f + (shift - 1.0f) * s)
    }
    sigmas[numInferenceSteps] = 0.0f
    steps = (0 until numInferenceSteps).map { i ->
      // The reference implementation states it outright: "timestep == sigma". Multiplying
      // by numTrainTimesteps, as the SD-family convention would, feeds the graph a number
      // three orders of magnitude off.
      DiffusionStep(index = i, timestep = sigmas[i], sigma = sigmas[i])
    }
  }

  override fun step(stepIndex: Int, latents: FloatArray, modelOutput: FloatArray, rng: Pcg32) {
    require(latents.size == modelOutput.size) {
      "latent/model-output size mismatch: ${latents.size} vs ${modelOutput.size}"
    }
    val dt = sigmas[stepIndex + 1] - sigmas[stepIndex]
    for (i in latents.indices) latents[i] += dt * modelOutput[i]
  }

  companion object {
    /**
     * The schedule for an image of [tokens] tokens run for [steps] steps.
     *
     * Preferred over the constructor: the shift is a function of the geometry, not a taste
     * knob, and every caller that picked its own number picked it wrong.
     */
    fun forImage(tokens: Int, steps: Int): FlowMatchEulerScheduler =
      FlowMatchEulerScheduler(steps, empiricalShift(tokens, steps))
  }
}

/**
 * Euler-Ancestral sampler for epsilon-prediction models — the Stable-Diffusion-1.x/2.x family.
 *
 * Ancestral means each step removes a little more noise than a plain Euler step would and
 * injects a fresh draw to make up the difference; that extra stochasticity is what keeps
 * short schedules from looking flat. The sigma ladder comes from the model's
 * `scaled_linear` beta schedule, so it matches what the network saw in training.
 *
 * Mirrors `EulerAncestralDiscreteScheduler` from HuggingFace diffusers.
 */
class EulerAncestralScheduler(
  numInferenceSteps: Int,
  numTrainTimesteps: Int = 1000,
  betaStart: Float = 0.00085f,
  betaEnd: Float = 0.012f,
) : Scheduler {

  private val sigmas: FloatArray
  override val steps: List<DiffusionStep>
  override val initialNoiseScale: Float

  init {
    require(numInferenceSteps > 0) { "numInferenceSteps must be positive, was $numInferenceSteps" }

    // "scaled_linear": betas are a linear ramp in sqrt-space, squared back.
    val sigmasFull = FloatArray(numTrainTimesteps)
    var alphaCumprod = 1.0
    val sqrtStart = sqrt(betaStart.toDouble())
    val sqrtEnd = sqrt(betaEnd.toDouble())
    for (t in 0 until numTrainTimesteps) {
      val r = if (numTrainTimesteps == 1) 0.0 else t.toDouble() / (numTrainTimesteps - 1)
      val beta = (sqrtStart + (sqrtEnd - sqrtStart) * r).let { it * it }
      alphaCumprod *= (1.0 - beta)
      sigmasFull[t] = sqrt((1.0 - alphaCumprod) / alphaCumprod).toFloat()
    }

    // Walk the training ladder backwards at `numInferenceSteps` evenly spaced points,
    // interpolating between the integer timesteps the ladder is defined on.
    sigmas = FloatArray(numInferenceSteps + 1)
    val timesteps = FloatArray(numInferenceSteps)
    for (i in 0 until numInferenceSteps) {
      val t = if (numInferenceSteps == 1) 0.0f
      else (numTrainTimesteps - 1).toFloat() * (numInferenceSteps - 1 - i) / (numInferenceSteps - 1)
      timesteps[i] = t
      val lo = t.toInt().coerceIn(0, numTrainTimesteps - 1)
      val hi = (lo + 1).coerceAtMost(numTrainTimesteps - 1)
      val frac = t - lo
      sigmas[i] = sigmasFull[lo] + (sigmasFull[hi] - sigmasFull[lo]) * frac
    }
    sigmas[numInferenceSteps] = 0.0f

    initialNoiseScale = sigmas[0]
    steps = (0 until numInferenceSteps).map { i ->
      DiffusionStep(index = i, timestep = timesteps[i], sigma = sigmas[i])
    }
  }

  /** Karras preconditioning: the network expects a unit-variance input. */
  override fun scaleModelInput(stepIndex: Int): Float {
    val sigma = sigmas[stepIndex]
    return 1.0f / sqrt(sigma * sigma + 1.0f)
  }

  override fun step(stepIndex: Int, latents: FloatArray, modelOutput: FloatArray, rng: Pcg32) {
    require(latents.size == modelOutput.size) {
      "latent/model-output size mismatch: ${latents.size} vs ${modelOutput.size}"
    }
    val sigma = sigmas[stepIndex]
    val sigmaNext = sigmas[stepIndex + 1]

    // Split the jump into a deterministic part (down to sigmaDown) and a fresh noise part
    // (sigmaUp), chosen so the total variance lands exactly on sigmaNext.
    val sigmaUp = if (sigma <= 0f) 0f else {
      val v = sigmaNext * sigmaNext * (sigma * sigma - sigmaNext * sigmaNext) / (sigma * sigma)
      sqrt(max(v, 0.0f))
    }
    val sigmaDown = sqrt(max(sigmaNext * sigmaNext - sigmaUp * sigmaUp, 0.0f))
    val dt = sigmaDown - sigma

    // For eps-prediction the derivative *is* the predicted noise.
    for (i in latents.indices) latents[i] += modelOutput[i] * dt
    if (sigmaUp > 0f) {
      for (i in latents.indices) latents[i] += rng.nextGaussian() * sigmaUp
    }
  }
}

/**
 * Classifier-free guidance: push the conditioned prediction away from the unconditioned one.
 *
 * Writes into [cond] in place and returns it, because the conditioned batch is already a
 * throwaway buffer owned by the caller and a UNet-sized copy per step is not free.
 */
fun applyGuidance(cond: FloatArray, uncond: FloatArray, scale: Float): FloatArray {
  require(cond.size == uncond.size) {
    "guidance batch mismatch: ${cond.size} vs ${uncond.size}"
  }
  if (scale == 1.0f) return cond
  for (i in cond.indices) cond[i] = uncond[i] + scale * (cond[i] - uncond[i])
  return cond
}

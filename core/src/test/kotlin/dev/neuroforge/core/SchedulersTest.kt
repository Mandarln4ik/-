package dev.neuroforge.core

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SchedulersTest {

  @Test
  fun `flow match sigmas decrease monotonically`() {
    val s = FlowMatchEulerScheduler(numInferenceSteps = 4, shift = 3.0f)
    val sigmas = s.steps.map { it.sigma }
    assertEquals(4, sigmas.size)
    for (i in 1 until sigmas.size) {
      assertTrue(sigmas[i] < sigmas[i - 1], "sigma rose at step $i: $sigmas")
    }
    assertTrue(sigmas.first() <= 1.0f + 1e-6f, "first sigma should start at 1: ${sigmas.first()}")
  }

  @Test
  fun `flow match shift pushes budget toward high noise`() {
    val plain = FlowMatchEulerScheduler(8, shift = 1.0f).steps.map { it.sigma }
    val shifted = FlowMatchEulerScheduler(8, shift = 3.0f).steps.map { it.sigma }
    // With shift > 1 every interior sigma sits higher: more steps are spent on structure.
    for (i in 1 until plain.size) {
      assertTrue(shifted[i] > plain[i], "shift did not raise sigma at $i")
    }
  }

  @Test
  fun `flow match step is a plain euler move along the velocity`() {
    val s = FlowMatchEulerScheduler(numInferenceSteps = 2, shift = 1.0f)
    val latents = floatArrayOf(1f, 2f, 3f)
    val before = latents.copyOf()
    val velocity = floatArrayOf(0.5f, -0.5f, 1f)

    s.step(0, latents, velocity, Pcg32(0))

    val dt = s.steps[1].sigma - s.steps[0].sigma
    for (i in latents.indices) {
      assertEquals(before[i] + dt * velocity[i], latents[i], 1e-5f)
    }
  }

  @Test
  fun `flow match final step lands on sigma zero`() {
    // The last step must drive the latent all the way to the clean manifold, i.e. its dt
    // has to be exactly -sigma_last. If the trailing zero were missing the image would
    // come out of the VAE still carrying noise.
    val s = FlowMatchEulerScheduler(numInferenceSteps = 3, shift = 2.0f)
    val last = s.steps.last()
    val latents = floatArrayOf(0f)
    s.step(2, latents, floatArrayOf(1f), Pcg32(0))
    assertEquals(-last.sigma, latents[0], 1e-6f)
  }

  @Test
  fun `euler ancestral starts near the stable diffusion sigma max`() {
    val s = EulerAncestralScheduler(numInferenceSteps = 20)
    // SD 1.x's scaled_linear schedule tops out around 14.6.
    assertTrue(
      abs(s.initialNoiseScale - 14.6f) < 0.6f,
      "sigma_max off the known SD value: ${s.initialNoiseScale}",
    )
  }

  @Test
  fun `euler ancestral sigmas decrease and input scaling shrinks with noise`() {
    val s = EulerAncestralScheduler(numInferenceSteps = 10)
    val sigmas = s.steps.map { it.sigma }
    for (i in 1 until sigmas.size) assertTrue(sigmas[i] < sigmas[i - 1])
    // scale = 1/sqrt(sigma^2+1): smallest at the noisiest step, approaching 1 at the end.
    assertTrue(s.scaleModelInput(0) < s.scaleModelInput(9))
    assertTrue(s.scaleModelInput(9) <= 1.0f)
  }

  @Test
  fun `euler ancestral is reproducible for a fixed rng seed`() {
    fun run(): FloatArray {
      val s = EulerAncestralScheduler(numInferenceSteps = 5)
      val x = FloatArray(16) { it * 0.1f }
      val rng = Pcg32(42)
      for (i in 0 until 5) s.step(i, x, FloatArray(16) { 0.01f * it }, rng)
      return x
    }
    assertTrue(run().contentEquals(run()))
  }

  @Test
  fun `guidance interpolates away from the unconditioned prediction`() {
    val cond = floatArrayOf(2f, 4f)
    val uncond = floatArrayOf(1f, 1f)
    val out = applyGuidance(cond, uncond, scale = 3f)
    // uncond + 3*(cond - uncond)
    assertEquals(1f + 3f * 1f, out[0], 1e-6f)
    assertEquals(1f + 3f * 3f, out[1], 1e-6f)
  }

  @Test
  fun `guidance scale of one is a no-op`() {
    val cond = floatArrayOf(2f, 4f)
    val out = applyGuidance(cond, floatArrayOf(1f, 1f), scale = 1f)
    assertEquals(2f, out[0], 1e-6f)
    assertEquals(4f, out[1], 1e-6f)
  }

  @Test
  fun `single step schedules are valid`() {
    // A 1-step run is a legitimate configuration for a step-distilled model; it must not
    // divide by zero while building the linspace.
    val flow = FlowMatchEulerScheduler(numInferenceSteps = 1)
    assertEquals(1, flow.steps.size)
    val x = floatArrayOf(5f)
    flow.step(0, x, floatArrayOf(1f), Pcg32(0))
    assertTrue(x[0].isFinite())

    val euler = EulerAncestralScheduler(numInferenceSteps = 1)
    assertEquals(1, euler.steps.size)
  }
}

package dev.neuroforge.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RenderPlanTest {

  @Test
  fun `fast strategy reaches 4K square in one network pass`() {
    val plan = RenderPlan.of(512, 512, OutputTarget.SQUARE_4K, UpscaleStrategy.FAST)
    assertEquals(1, plan.passes.size)
    assertEquals(4096, plan.passes.single().outputWidth)
    assertEquals(4096, plan.passes.single().outputHeight)
    // 512 needs x8; one x4 pass covers half of it, so the pre-resample is x2.
    assertEquals(2.0f, plan.preScale, 1e-4f)
  }

  @Test
  fun `quality strategy never enlarges by resampling`() {
    val plan = RenderPlan.of(512, 512, OutputTarget.SQUARE_4K, UpscaleStrategy.QUALITY)
    assertEquals(2, plan.passes.size)
    assertTrue(plan.preScale <= 1.0f, "quality mode pre-upscaled by ${plan.preScale}")
    assertTrue(
      plan.totalTiles > RenderPlan.of(512, 512, OutputTarget.SQUARE_4K, UpscaleStrategy.FAST).totalTiles,
      "quality mode should cost more tiles, not fewer",
    )
  }

  @Test
  fun `plan reports the tile count the upscaler will actually run`() {
    val plan = RenderPlan.of(512, 512, OutputTarget.SQUARE_4K, UpscaleStrategy.FAST)
    // 1024x1024 input, 256px tiles with 32px overlap -> stride 224 -> 5 origins per axis.
    assertEquals(25, plan.totalTiles)
  }

  @Test
  fun `target already covered needs no network pass`() {
    val plan = RenderPlan.of(4096, 4096, OutputTarget.QHD, UpscaleStrategy.FAST)
    assertTrue(plan.passes.isEmpty())
    assertEquals(2048, plan.finalWidth)
  }

  @Test
  fun `widescreen target from a square generator is flagged as cropping`() {
    // A 512x512 generator cannot fill 16:9 without losing content; the plan has to say so
    // rather than silently handing back a cropped image.
    val plan = RenderPlan.of(512, 512, OutputTarget.UHD_4K, UpscaleStrategy.FAST)
    assertTrue(plan.cropsContent)
  }

  @Test
  fun `peak accumulator for 4K stays within a phone's budget`() {
    val plan = RenderPlan.of(512, 512, OutputTarget.SQUARE_4K, UpscaleStrategy.FAST)
    val mib = plan.peakAccumulatorBytes / (1L shl 20)
    // 4096*4096*4 floats = 256 MiB. Large but allocatable; if this ever regresses past
    // ~512 MiB the tiled design has stopped paying for itself.
    assertTrue(mib in 1..512, "peak accumulator ballooned to $mib MiB")
  }

  @Test
  fun `estimated time scales with the measured per-tile cost`() {
    val plan = RenderPlan.of(512, 512, OutputTarget.SQUARE_4K, UpscaleStrategy.FAST)
    assertEquals(plan.totalTiles * 12.5, plan.estimatedUpscaleMillis(12.5), 1e-6)
  }

  @Test
  fun `describe mentions every pass`() {
    val text = RenderPlan.of(512, 512, OutputTarget.SQUARE_4K, UpscaleStrategy.QUALITY).describe()
    assertTrue(text.contains("pass 1") && text.contains("pass 2"), text)
  }
}

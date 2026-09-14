package dev.neuroforge.core

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TileGridTest {

  @Test
  fun `tiles cover every source pixel`() {
    val grid = TileGrid(srcWidth = 1024, srcHeight = 768, tileSize = 256, overlap = 32, scale = 4)
    val covered = Array(grid.srcHeight) { BooleanArray(grid.srcWidth) }
    grid.tiles.forEach { t ->
      for (y in t.srcY until t.srcBottom) for (x in t.srcX until t.srcRight) covered[y][x] = true
    }
    val missing = (0 until grid.srcHeight).sumOf { y ->
      (0 until grid.srcWidth).count { x -> !covered[y][x] }
    }
    assertEquals(0, missing, "$missing source pixels were never fed to the upscaler")
  }

  @Test
  fun `tiles stay inside the source`() {
    val grid = TileGrid(1000, 600, tileSize = 256, overlap = 32, scale = 4)
    grid.tiles.forEach { t ->
      assertTrue(t.srcX >= 0 && t.srcY >= 0, "tile origin outside image: $t")
      assertTrue(t.srcRight <= grid.srcWidth, "tile overruns width: $t")
      assertTrue(t.srcBottom <= grid.srcHeight, "tile overruns height: $t")
    }
  }

  @Test
  fun `every tile has the model's fixed input size`() {
    // An NPU compiles one shape and replays it; a ragged edge tile would force a recompile.
    val grid = TileGrid(1000, 600, tileSize = 256, overlap = 32, scale = 4)
    grid.tiles.forEach { t ->
      assertEquals(256, t.srcW, "ragged tile width: $t")
      assertEquals(256, t.srcH, "ragged tile height: $t")
    }
  }

  @Test
  fun `image smaller than one tile yields a single untapered tile`() {
    val grid = TileGrid(100, 80, tileSize = 256, overlap = 32, scale = 4)
    assertEquals(1, grid.tiles.size)
    val t = grid.tiles.single()
    assertEquals(100, t.srcW)
    assertEquals(80, t.srcH)
    assertTrue(!t.taperLeft && !t.taperTop && !t.taperRight && !t.taperBottom)
  }

  @Test
  fun `border edges are never feathered`() {
    val grid = TileGrid(1024, 1024, tileSize = 256, overlap = 32, scale = 4)
    val leftmost = grid.tiles.filter { it.srcX == 0 }
    val rightmost = grid.tiles.filter { it.srcRight == grid.srcWidth }
    assertTrue(leftmost.isNotEmpty() && rightmost.isNotEmpty())
    assertTrue(leftmost.none { it.taperLeft }, "left image border was tapered — frame would darken")
    assertTrue(rightmost.none { it.taperRight }, "right image border was tapered")
  }

  /**
   * The property that actually matters: feathered, normalised blending must reconstruct the
   * input exactly when the "network" is the identity. If this holds, any visible seam in a
   * real run comes from the network, not from the stitching.
   */
  @Test
  fun `blending a flat field reproduces it exactly`() {
    val grid = TileGrid(512, 512, tileSize = 128, overlap = 16, scale = 1)
    val (acc, wsum) = blendIdentity(grid) { _, _ -> 0.625f }
    for (i in acc.indices) {
      assertTrue(wsum[i] > 0f, "pixel $i got no weight at all")
      assertEquals(0.625f, acc[i] / wsum[i], 1e-4f, "flat field was not preserved at $i")
    }
  }

  @Test
  fun `blending a gradient reproduces it exactly`() {
    // A gradient catches errors a flat field cannot: a mis-indexed tile placement shifts
    // content sideways, which a constant image would hide completely.
    val grid = TileGrid(300, 200, tileSize = 128, overlap = 16, scale = 1)
    val expected = { x: Int, y: Int -> (x * 0.003f + y * 0.005f) }
    val (acc, wsum) = blendIdentity(grid, expected)
    for (y in 0 until grid.dstHeight) {
      for (x in 0 until grid.dstWidth) {
        val i = y * grid.dstWidth + x
        assertEquals(expected(x, y), acc[i] / wsum[i], 1e-3f, "gradient broken at ($x,$y)")
      }
    }
  }

  @Test
  fun `feather weights across an interior seam sum to one`() {
    val grid = TileGrid(512, 256, tileSize = 128, overlap = 16, scale = 1)
    // Two horizontally adjacent interior tiles from the same row.
    val row = grid.tiles.filter { it.srcY == 0 }.sortedBy { it.srcX }
    val a = row[0]
    val b = row[1]
    val shared = a.srcRight - b.srcX
    assertTrue(shared > 0, "expected the first two tiles to overlap")

    // Sample the tiles' own vertical centre, away from the vertical ramps, so what is
    // under test is purely the horizontal cross-fade.
    val y = a.srcY * grid.scale + a.srcH * grid.scale / 2
    for (dx in 0 until shared * grid.scale) {
      val globalX = b.srcX * grid.scale + dx
      val wa = grid.weightAt(a, globalX - a.srcX * grid.scale, y - a.srcY * grid.scale)
      val wb = grid.weightAt(b, globalX - b.srcX * grid.scale, y - b.srcY * grid.scale)
      assertTrue(wa + wb > 0.99f, "seam weights dip to ${wa + wb} at dx=$dx — visible dark band")
    }
  }

  @Test
  fun `weights are always positive so normalisation never divides by zero`() {
    val grid = TileGrid(640, 480, tileSize = 256, overlap = 32, scale = 4)
    grid.tiles.forEach { t ->
      for (y in intArrayOf(0, t.srcH * grid.scale - 1)) {
        for (x in intArrayOf(0, t.srcW * grid.scale - 1)) {
          assertTrue(grid.weightAt(t, x, y) > 0f, "zero weight in corner of $t")
        }
      }
    }
  }

  @Test
  fun `output dimensions scale exactly`() {
    val grid = TileGrid(1024, 1024, tileSize = 256, overlap = 32, scale = 4)
    assertEquals(4096, grid.dstWidth)
    assertEquals(4096, grid.dstHeight)
  }

  @Test
  fun `rejects an overlap that would stall the walk`() {
    // overlap >= tileSize/2 means the stride could stop making progress.
    val e = runCatching { TileGrid(512, 512, tileSize = 128, overlap = 64, scale = 4) }
    assertTrue(e.isFailure, "a non-advancing overlap must be rejected, not loop forever")
  }

  @Test
  fun `a coordinate outside the tile contributes nothing`() {
    // Regression: the ramp formula evaluated past an edge used to go negative, which would
    // subtract signal from the accumulator rather than simply not adding any.
    val grid = TileGrid(512, 256, tileSize = 128, overlap = 16, scale = 1)
    val t = grid.tiles.first()
    assertEquals(0f, grid.weightAt(t, -1, 0))
    assertEquals(0f, grid.weightAt(t, 0, t.srcH * grid.scale))
    assertEquals(0f, grid.weightAt(t, t.srcW * grid.scale, 0))
  }

  /** Runs the grid with an identity "network" and returns the weighted accumulator and weights. */
  private fun blendIdentity(
    grid: TileGrid,
    source: (Int, Int) -> Float,
  ): Pair<FloatArray, FloatArray> {
    val acc = FloatArray(grid.dstWidth * grid.dstHeight)
    val wsum = FloatArray(grid.dstWidth * grid.dstHeight)
    grid.tiles.forEach { t ->
      val tw = t.srcW * grid.scale
      val th = t.srcH * grid.scale
      for (ly in 0 until th) {
        for (lx in 0 until tw) {
          val gx = t.srcX * grid.scale + lx
          val gy = t.srcY * grid.scale + ly
          val w = grid.weightAt(t, lx, ly)
          val i = gy * grid.dstWidth + gx
          acc[i] += w * source(gx, gy)
          wsum[i] += w
        }
      }
    }
    return acc to wsum
  }
}

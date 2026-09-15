package dev.neuroforge.core

import kotlin.math.max
import kotlin.math.min

/**
 * One tile of a tiled super-resolution pass, in source-image coordinates.
 *
 * @param srcX left edge in the source image.
 * @param srcY top edge in the source image.
 * @param srcW width fed to the model; equals the grid's tile size except when the whole
 *   image is narrower than one tile.
 * @param srcH height fed to the model.
 * @param taperLeft whether this tile's left edge is an interior seam that must be feathered.
 *   An edge lying on the true image border is never feathered — there is nothing to blend
 *   against there, and tapering it would darken the frame.
 */
data class Tile(
  val srcX: Int,
  val srcY: Int,
  val srcW: Int,
  val srcH: Int,
  val taperLeft: Boolean,
  val taperTop: Boolean,
  val taperRight: Boolean,
  val taperBottom: Boolean,
) {
  val srcRight: Int get() = srcX + srcW
  val srcBottom: Int get() = srcY + srcH
}

/**
 * Plans how to walk a large image through a fixed-input-size super-resolution model.
 *
 * This is the piece that makes 4K output possible on a phone. A ×4 SR network cannot be
 * handed a whole 1024×1024 image — a naive run allocates the full 4096×4096×feature-channel
 * activation and dies — and an NPU wants one fixed input shape it can compile once and reuse.
 * So the image is cut into fixed [tileSize] tiles that overlap by [overlap] source pixels,
 * each tile is upscaled on its own, and the results are feathered back together.
 *
 * The overlap is what keeps the seams invisible: a network's output near a tile border is
 * computed from a truncated receptive field and is subtly wrong, so neighbouring tiles
 * cross-fade over the overlap region instead of butting up against each other.
 *
 * Tiles are clamped to stay inside the image rather than padded, so the last tile in a row
 * may overlap its predecessor by more than [overlap]. That is harmless: [weightAt] feathers
 * whatever overlap actually exists and the accumulator normalises by the summed weight.
 */
class TileGrid(
  val srcWidth: Int,
  val srcHeight: Int,
  val tileSize: Int,
  val overlap: Int,
  val scale: Int,
) {

  init {
    require(srcWidth > 0 && srcHeight > 0) { "source must be non-empty, was ${srcWidth}x$srcHeight" }
    require(tileSize > 0) { "tileSize must be positive, was $tileSize" }
    require(scale >= 1) { "scale must be at least 1, was $scale" }
    require(overlap >= 0) { "overlap must be non-negative, was $overlap" }
    require(overlap * 2 < tileSize) {
      "overlap ($overlap) must leave forward progress: 2*overlap must be < tileSize ($tileSize)"
    }
  }

  /** Output dimensions. */
  val dstWidth: Int get() = srcWidth * scale
  val dstHeight: Int get() = srcHeight * scale

  /** Feather width in *destination* pixels. */
  val rampPixels: Int get() = overlap * scale

  /** The tiles to run, in row-major order. */
  val tiles: List<Tile> by lazy { plan() }

  private fun plan(): List<Tile> {
    val xs = axisOrigins(srcWidth)
    val ys = axisOrigins(srcHeight)
    val tileW = min(tileSize, srcWidth)
    val tileH = min(tileSize, srcHeight)

    val out = ArrayList<Tile>(xs.size * ys.size)
    for (y in ys.indices) {
      for (x in xs.indices) {
        out += Tile(
          srcX = xs[x],
          srcY = ys[y],
          srcW = tileW,
          srcH = tileH,
          taperLeft = x > 0,
          taperTop = y > 0,
          taperRight = x < xs.size - 1,
          taperBottom = y < ys.size - 1,
        )
      }
    }
    return out
  }

  /**
   * Tile origins along one axis: step by `tileSize - overlap`, and pull the final tile back
   * so it ends exactly on the image edge instead of hanging over it.
   */
  private fun axisOrigins(extent: Int): IntArray {
    if (extent <= tileSize) return intArrayOf(0)
    val stride = tileSize - overlap
    val origins = ArrayList<Int>()
    var pos = 0
    while (true) {
      if (pos + tileSize >= extent) {
        origins += extent - tileSize
        break
      }
      origins += pos
      pos += stride
    }
    return origins.toIntArray()
  }

  /**
   * Blend weight for a pixel inside a tile's *upscaled* output.
   *
   * [localX] and [localY] are offsets into the tile's output, i.e. in `[0, srcW*scale)`.
   * The weight is a separable linear ramp: it rises from ~0 to 1 across [rampPixels] on any
   * edge that is an interior seam, and stays 1 on edges that sit on the image border.
   *
   * Two tiles feathering across a shared seam of exactly [rampPixels] contribute weights
   * that sum to 1, so the cross-fade is energy-preserving before normalisation even happens.
   */
  fun weightAt(tile: Tile, localX: Int, localY: Int): Float {
    val w = tile.srcW * scale
    val h = tile.srcH * scale
    // Outside the tile this tile contributes nothing. Saying so explicitly matters: the ramp
    // formula evaluated past an edge goes negative, and a negative weight in the accumulator
    // would subtract real signal instead of simply not adding any.
    if (localX !in 0 until w || localY !in 0 until h) return 0.0f
    val fx = axisWeight(localX, w, tile.taperLeft, tile.taperRight)
    val fy = axisWeight(localY, h, tile.taperTop, tile.taperBottom)
    // Floor at a small positive value: a pixel covered by exactly one tile whose ramp reads
    // 0 there must still contribute, or normalisation divides by zero and the seam goes black.
    return max(fx * fy, MIN_WEIGHT)
  }

  private fun axisWeight(pos: Int, extent: Int, taperLow: Boolean, taperHigh: Boolean): Float {
    val ramp = rampPixels
    if (ramp <= 0) return 1.0f
    var w = 1.0f
    if (taperLow && pos < ramp) w = min(w, (pos + 0.5f) / ramp)
    val fromHigh = extent - 1 - pos
    if (taperHigh && fromHigh < ramp) w = min(w, (fromHigh + 0.5f) / ramp)
    return w
  }

  /** Bytes an RGB float accumulator for the full output needs — the app's OOM guard rail. */
  fun accumulatorBytes(): Long =
    dstWidth.toLong() * dstHeight.toLong() * (3L * Float.SIZE_BYTES + Float.SIZE_BYTES)

  override fun toString(): String =
    "TileGrid(${srcWidth}x$srcHeight -> ${dstWidth}x$dstHeight, " +
      "${tiles.size} tiles of $tileSize px, overlap $overlap, scale x$scale)"

  companion object {
    private const val MIN_WEIGHT = 1e-4f

    /**
     * Picks a tile size and overlap for a target output resolution.
     *
     * Bigger tiles mean fewer model invocations and fewer seams, but each tile's activations
     * have to fit; 256 is the usual sweet spot for a ×4 ESRGAN-class network on a phone, and
     * an overlap of an eighth of the tile is enough to hide the receptive-field truncation.
     */
    fun forTarget(srcWidth: Int, srcHeight: Int, scale: Int, tileSize: Int = 256): TileGrid =
      TileGrid(
        srcWidth = srcWidth,
        srcHeight = srcHeight,
        tileSize = tileSize,
        overlap = max(8, tileSize / 8),
        scale = scale,
      )
  }
}

package dev.neuroforge.core

/**
 * Geometry of a diffusion model's latent space.
 *
 * @param channels latent channels the VAE encodes to (4 for SD 1.x, 16 for the FLUX family).
 * @param vaeScale how much smaller the latent is than the image; 8 for every common VAE,
 *   32 for the deep-compression autoencoders that make 4K generation tractable at all.
 * @param patch side of the square patch a DiT packs into one token. 1 means the model
 *   consumes the latent grid directly, as a UNet does.
 */
data class LatentSpec(
  val channels: Int,
  val vaeScale: Int = 8,
  val patch: Int = 2,
) {
  fun latentWidth(imageWidth: Int): Int = imageWidth / vaeScale
  fun latentHeight(imageHeight: Int): Int = imageHeight / vaeScale

  /** Number of tokens a DiT sees for an image of this size. */
  fun tokenCount(imageWidth: Int, imageHeight: Int): Int =
    (latentWidth(imageWidth) / patch) * (latentHeight(imageHeight) / patch)

  /** Width of one token vector. */
  fun tokenDim(): Int = channels * patch * patch

  /** Elements in the unpacked latent tensor. */
  fun latentElements(imageWidth: Int, imageHeight: Int): Int =
    channels * latentHeight(imageHeight) * latentWidth(imageWidth)

  companion object {
    /** Stable Diffusion 1.x/2.x: a UNet over a 4-channel f8 latent, no patching. */
    val SD15 = LatentSpec(channels = 4, vaeScale = 8, patch = 1)

    /** FLUX-family DiT: 16-channel f8 latent packed into 2x2 patches. */
    val FLUX = LatentSpec(channels = 16, vaeScale = 8, patch = 2)
  }
}

/**
 * Packing between a latent *image* and the flat token sequence a diffusion transformer takes.
 *
 * A DiT does not consume a CHW grid; it consumes a sequence, where each token carries a
 * `patch × patch` neighbourhood of every channel. Reference pipelines do this with a pair of
 * `rearrange` calls, and getting the index order wrong produces an image that is subtly
 * scrambled rather than obviously broken — which is exactly the kind of bug that survives a
 * visual once-over. Hence: explicit, and unit-tested against a round trip.
 *
 * Layout matches `einops.rearrange(x, "c (h p1) (w p2) -> (h w) (c p1 p2)")`, the form the
 * FLUX reference implementation uses.
 */
object LatentPacking {

  /** Latent `[C][H][W]` (flattened CHW) → tokens `[H/p * W/p][C*p*p]` (flattened). */
  fun patchify(latent: FloatArray, spec: LatentSpec, latentH: Int, latentW: Int): FloatArray {
    val p = spec.patch
    require(latentH % p == 0 && latentW % p == 0) {
      "latent ${latentW}x$latentH is not divisible by patch size $p"
    }
    require(latent.size == spec.channels * latentH * latentW) {
      "latent has ${latent.size} elements, expected ${spec.channels * latentH * latentW}"
    }
    if (p == 1) return latent.copyOf()

    val gh = latentH / p
    val gw = latentW / p
    val dim = spec.tokenDim()
    val out = FloatArray(gh * gw * dim)

    for (c in 0 until spec.channels) {
      val cBase = c * latentH * latentW
      for (py in 0 until p) {
        for (px in 0 until p) {
          val comp = (c * p + py) * p + px
          for (gy in 0 until gh) {
            val srcRow = cBase + (gy * p + py) * latentW
            val dstRow = gy * gw
            for (gx in 0 until gw) {
              out[(dstRow + gx) * dim + comp] = latent[srcRow + gx * p + px]
            }
          }
        }
      }
    }
    return out
  }

  /** The exact inverse of [patchify]. */
  fun unpatchify(tokens: FloatArray, spec: LatentSpec, latentH: Int, latentW: Int): FloatArray {
    val p = spec.patch
    require(latentH % p == 0 && latentW % p == 0) {
      "latent ${latentW}x$latentH is not divisible by patch size $p"
    }
    if (p == 1) return tokens.copyOf()

    val gh = latentH / p
    val gw = latentW / p
    val dim = spec.tokenDim()
    require(tokens.size == gh * gw * dim) {
      "token tensor has ${tokens.size} elements, expected ${gh * gw * dim}"
    }
    val out = FloatArray(spec.channels * latentH * latentW)

    for (c in 0 until spec.channels) {
      val cBase = c * latentH * latentW
      for (py in 0 until p) {
        for (px in 0 until p) {
          val comp = (c * p + py) * p + px
          for (gy in 0 until gh) {
            val dstRow = cBase + (gy * p + py) * latentW
            val srcRow = gy * gw
            for (gx in 0 until gw) {
              out[dstRow + gx * p + px] = tokens[(srcRow + gx) * dim + comp]
            }
          }
        }
      }
    }
    return out
  }
}

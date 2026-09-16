package dev.neuroforge.core

import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * PCG32 — a small, fast, statistically solid counter-based PRNG.
 *
 * The diffusion loop needs a noise stream that is *bit-reproducible*: the same seed has to
 * yield the same image on this phone, on a desktop reference run, and across app versions,
 * otherwise a reported seed is worthless and comparing an on-device result against a host
 * run can't tell a conversion bug from a different random draw. [java.util.Random] is
 * specified well enough for that, but it is a 48-bit LCG whose low-order bits are weak and
 * whose Gaussian generator (`nextGaussian`) has never been pinned down across JDKs the way
 * the raw `next(bits)` stream is. PCG32 is defined here in full, so the stream is ours.
 *
 * Reference: O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good
 * Algorithms for Random Number Generation" (2014).
 */
class Pcg32(seed: Long, sequence: Long = DEFAULT_SEQUENCE) {

  private var state: ULong = 0uL
  private val inc: ULong = (sequence.toULong() shl 1) or 1uL

  init {
    // Standard PCG seeding: step once, add the seed, step again.
    nextUInt()
    state += seed.toULong()
    nextUInt()
  }

  /** Next raw 32 bits. */
  fun nextUInt(): UInt {
    val old = state
    state = old * MULTIPLIER + inc
    val xorShifted = (((old shr 18) xor old) shr 27).toUInt()
    val rot = (old shr 59).toInt()
    // Rotate right by `rot` within 32 bits, spelled out rather than via UInt.rotateRight
    // so the routine reads as the published algorithm and needs no stdlib opt-in.
    return (xorShifted shr rot) or (xorShifted shl ((32 - rot) and 31))
  }

  /** Uniform in `(0, 1)` — never exactly 0, so `ln()` in Box–Muller is always finite. */
  fun nextFloat(): Float {
    // 24 significant bits, shifted off zero by half an ulp.
    val bits = nextUInt() shr 8 // [0, 2^24)
    return (bits.toFloat() + 0.5f) / 16777216.0f
  }

  /**
   * Standard normal sample via Box–Muller.
   *
   * Box–Muller produces two independent normals per pair of uniforms; the second is cached
   * so a run of [gaussians] costs one transcendental pair per two values rather than per value.
   */
  fun nextGaussian(): Float {
    cached?.let { cached = null; return it }
    val u1 = nextFloat()
    val u2 = nextFloat()
    val r = sqrt(-2.0f * ln(u1))
    val theta = TWO_PI * u2
    cached = r * kotlin.math.sin(theta)
    return r * cos(theta)
  }

  private var cached: Float? = null

  /** Fills a fresh array of [n] standard normals. */
  fun gaussians(n: Int): FloatArray = FloatArray(n) { nextGaussian() }

  private companion object {
    const val MULTIPLIER: ULong = 6364136223846793005uL
    const val DEFAULT_SEQUENCE: Long = 0x14057B7EF767814FL
    const val TWO_PI = 6.2831853071795864769f
  }
}

/**
 * Initial latent noise for a diffusion run.
 *
 * Laid out NCHW and filled in that order, which is the order the reference Python pipelines
 * fill `torch.randn(shape, generator=g)`, so a seed matches a host run element for element.
 */
fun latentNoise(seed: Long, channels: Int, height: Int, width: Int): FloatArray =
  Pcg32(seed).gaussians(channels * height * width)

/**
 * Initial noise for a transformer that works directly in token space.
 *
 * A DiT does not iterate on a `C×H×W` grid; it iterates on the `tokens × packedChannels`
 * sequence, and the reference pipeline draws its noise in exactly that shape. It holds the
 * same number of values as [latentNoise] would for the same image, which is why getting it
 * wrong is invisible: packing noise as though it were a latent image, denoising it, and
 * unpacking it again yields a differently-shuffled but equally valid draw, so the run
 * completes and only the seed stops meaning anything.
 */
fun tokenNoise(seed: Long, tokens: Int, packedChannels: Int): FloatArray =
  Pcg32(seed).gaussians(tokens * packedChannels)

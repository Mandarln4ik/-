package dev.neuroforge.core

import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Aggregated timings for one (model, accelerator) pair.
 *
 * Reports percentiles rather than just a mean because that is what NPU behaviour actually
 * looks like: the first runs pay for graph compilation and buffer setup, and a thermally
 * throttled phone produces a long right tail. A mean folds both into one number that
 * describes neither the steady state nor the worst case.
 */
data class BenchmarkResult(
  val label: String,
  val accelerator: Accel,
  val samples: List<Double>,
  val warmupMillis: Double,
  val failed: String? = null,
) {
  val ok: Boolean get() = failed == null && samples.isNotEmpty()

  val median: Double get() = percentile(50.0)
  val p95: Double get() = percentile(95.0)
  val best: Double get() = samples.minOrNull() ?: Double.NaN
  val worst: Double get() = samples.maxOrNull() ?: Double.NaN

  val mean: Double get() = if (samples.isEmpty()) Double.NaN else samples.sum() / samples.size

  val stdDev: Double
    get() {
      if (samples.size < 2) return 0.0
      val m = mean
      return sqrt(samples.sumOf { (it - m) * (it - m) } / (samples.size - 1))
    }

  /** Inferences per second at the median. */
  val throughput: Double get() = if (median > 0) 1000.0 / median else 0.0

  /**
   * Linear interpolation between order statistics (the "R-7" definition, as used by NumPy
   * and Excel), so a 20-sample run still gives a meaningful p95 instead of snapping to the
   * single worst sample.
   */
  fun percentile(p: Double): Double {
    if (samples.isEmpty()) return Double.NaN
    val sorted = samples.sorted()
    if (sorted.size == 1) return sorted[0]
    val rank = (p / 100.0) * (sorted.size - 1)
    val lo = rank.toInt()
    val hi = (lo + 1).coerceAtMost(sorted.size - 1)
    val frac = rank - lo
    return sorted[lo] + (sorted[hi] - sorted[lo]) * frac
  }

  fun summary(): String = when {
    failed != null -> "$label / $accelerator: FAILED — $failed"
    else -> "$label / $accelerator: median %.3f ms (p95 %.3f, best %.3f), %.1f inf/s, warmup %.0f ms"
      .format(median, p95, best, throughput, warmupMillis)
  }

  /**
   * True when every sample landed on the same whole millisecond.
   *
   * The signature of a clock that cannot see the thing being measured. It is worth calling
   * out rather than reporting the resulting 1.00x speedup as a finding: this app shipped
   * exactly that, and it read as "the NPU is no faster" when it meant "the timer has
   * millisecond resolution and the graph takes about two of them".
   */
  val quantised: Boolean
    get() = samples.size > 1 && samples.all { it == kotlin.math.floor(it) }
}

/** A whole comparison run: the same model across every accelerator that would load it. */
data class BenchmarkReport(val results: List<BenchmarkResult>) {

  /** The fastest accelerator that actually ran. */
  val winner: BenchmarkResult? get() = results.filter { it.ok }.minByOrNull { it.median }

  /**
   * Speedup of the NPU over the CPU baseline, or null when either side did not run.
   *
   * Reported as a ratio of medians — the honest comparison, since it is the steady-state
   * number a user experiences rather than a best-case cherry-pick.
   */
  fun npuSpeedupOverCpu(): Double? {
    val npu = results.firstOrNull { it.accelerator == Accel.NPU && it.ok } ?: return null
    val cpu = results.firstOrNull { it.accelerator == Accel.CPU && it.ok } ?: return null
    if (npu.median <= 0) return null
    return cpu.median / npu.median
  }

  fun table(): String = buildString {
    appendLine("%-6s %10s %10s %10s %9s %8s".format("accel", "median", "p95", "best", "inf/s", "± sd"))
    results.forEach { r ->
      if (r.ok) {
        appendLine(
          "%-6s %8.3fms %8.3fms %8.3fms %9.1f %7.3f"
            .format(r.accelerator.name, r.median, r.p95, r.best, r.throughput, r.stdDev)
        )
      } else {
        appendLine("%-6s %s".format(r.accelerator.name, r.failed ?: "no samples"))
      }
    }
    npuSpeedupOverCpu()?.let { appendLine("NPU speedup over CPU: %.2fx".format(it)) }
    interpretation()?.let { appendLine(it) }
  }

  /**
   * What the numbers mean, when they mean something other than what they look like.
   *
   * A ratio near 1.00 has three quite different causes and the user cannot tell them apart
   * from the ratio alone: the clock could not resolve the difference, the two results
   * overlap inside their own spread, or the NPU genuinely does not help on this graph. The
   * third is a real and useful finding; the first two are not findings at all.
   */
  fun interpretation(): String? {
    val npu = results.firstOrNull { it.accelerator == Accel.NPU && it.ok } ?: return null
    val cpu = results.firstOrNull { it.accelerator == Accel.CPU && it.ok } ?: return null

    if (npu.quantised && cpu.quantised) {
      return "Every sample was a whole millisecond, so this graph is too quick for the " +
        "clock here. Benchmark something heavier - the upscaler - for a number that means " +
        "anything."
    }

    val gap = kotlin.math.abs(cpu.median - npu.median)
    val noise = npu.stdDev + cpu.stdDev
    if (gap < noise) {
      return "The gap (%.3f ms) is smaller than the run-to-run spread (%.3f ms), so these two are tied, not equal."
        .format(gap, noise)
    }

    val speedup = cpu.median / npu.median
    return when {
      speedup >= 1.2 -> "The NPU is %.2fx faster than the CPU here.".format(speedup)
      speedup <= 0.83 -> "The CPU is %.2fx faster than the NPU here - this graph is not one an APU accelerates."
        .format(1.0 / speedup)
      else -> "The NPU and CPU are within 20%% of each other on this graph."
    }
  }
}

/**
 * Discards the warm-up samples from a raw timing run.
 *
 * The first invocation on an NPU includes on-device graph compilation, which on a large
 * model can be seconds — folding it into the steady-state number would be misleading, but
 * throwing it away silently would hide a real first-run cost the user waits through, so it
 * is returned separately as [BenchmarkResult.warmupMillis].
 */
fun summarize(
  label: String,
  accelerator: Accel,
  rawSamples: List<Double>,
  warmupCount: Int,
): BenchmarkResult {
  if (rawSamples.isEmpty()) {
    return BenchmarkResult(label, accelerator, emptyList(), 0.0, failed = "no samples recorded")
  }
  val n = warmupCount.coerceIn(0, rawSamples.size - 1)
  val warmup = if (n == 0) 0.0 else rawSamples.take(n).sum()
  return BenchmarkResult(
    label = label,
    accelerator = accelerator,
    samples = rawSamples.drop(n),
    warmupMillis = warmup,
  )
}

/** Human-readable byte size, for model lists and memory estimates. */
fun formatBytes(bytes: Long): String {
  if (bytes < 1024) return "$bytes B"
  val units = listOf("KiB", "MiB", "GiB", "TiB")
  var v = bytes.toDouble() / 1024
  var i = 0
  while (v >= 1024 && i < units.size - 1) { v /= 1024; i++ }
  return if (v >= 100) "${v.roundToInt()} ${units[i]}" else "%.1f %s".format(v, units[i])
}

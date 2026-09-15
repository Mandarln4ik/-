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
    else -> "$label / $accelerator: median %.2f ms (p95 %.2f, best %.2f), %.1f inf/s, warmup %.0f ms"
      .format(median, p95, best, throughput, warmupMillis)
  }
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
    appendLine("%-10s %10s %10s %10s %10s".format("accel", "median", "p95", "best", "inf/s"))
    results.forEach { r ->
      if (r.ok) {
        appendLine(
          "%-10s %9.2fms %9.2fms %9.2fms %10.1f"
            .format(r.accelerator.name, r.median, r.p95, r.best, r.throughput)
        )
      } else {
        appendLine("%-10s %s".format(r.accelerator.name, r.failed ?: "no samples"))
      }
    }
    npuSpeedupOverCpu()?.let { appendLine("NPU speedup over CPU: %.2fx".format(it)) }
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

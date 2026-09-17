package dev.neuroforge.runtime

import android.content.Context
import android.util.Log
import dev.neuroforge.core.Accel
import dev.neuroforge.core.BenchmarkReport
import dev.neuroforge.core.BenchmarkResult
import dev.neuroforge.core.ModelSpec
import dev.neuroforge.core.Pcg32
import dev.neuroforge.core.summarize
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlin.coroutines.coroutineContext

/**
 * Runs the same graph on every accelerator and reports what each one actually costs.
 *
 * This is the app's evidence layer. "Uses the NPU" is a claim; a median latency on NPU next
 * to a median latency on CPU, measured on this phone with this graph, is a fact — and
 * occasionally an uncomfortable one, because for some graphs the NPU loses. Reporting that
 * honestly is the point: it tells you which stage is worth moving and which is not.
 */
object Benchmark {

  private const val TAG = "Benchmark"

  /**
   * @param iterations timed runs per accelerator, after [warmup].
   * @param warmup untimed runs. The first NPU execution includes on-device graph compilation
   *   and buffer setup; folding that into the steady-state median would be wrong, but it is
   *   a real cost, so it is reported separately rather than discarded.
   */
  suspend fun compare(
    context: Context,
    spec: ModelSpec,
    modelFile: File,
    iterations: Int = 20,
    warmup: Int = 3,
    accelerators: List<Accel> = listOf(Accel.NPU, Accel.GPU, Accel.CPU),
  ): BenchmarkReport = withContext(Dispatchers.Default) {
    val results = mutableListOf<BenchmarkResult>()

    for (accel in accelerators) {
      coroutineContext.ensureActive()
      results += runCatching { measureOne(context, spec, modelFile, accel, iterations, warmup) }
        .getOrElse { e ->
          Log.w(TAG, "${spec.id} on $accel failed", e)
          BenchmarkResult(
            label = spec.displayName,
            accelerator = accel,
            samples = emptyList(),
            warmupMillis = 0.0,
            failed = e.message ?: e.javaClass.simpleName,
          )
        }
    }

    BenchmarkReport(results)
  }

  private suspend fun measureOne(
    context: Context,
    spec: ModelSpec,
    modelFile: File,
    accel: Accel,
    iterations: Int,
    warmup: Int,
  ): BenchmarkResult {
    // A single-element preference plus strict mode: this must measure *this* accelerator,
    // so a silent fallback would make the whole comparison meaningless.
    LiteRtSession.load(context, spec.displayName, modelFile, listOf(accel), strict = true)
      .use { session ->
      if (session.accelerator != accel) {
        return BenchmarkResult(
          spec.displayName, accel, emptyList(), 0.0,
          failed = "fell back to ${session.accelerator}",
        )
      }

      val input = syntheticInput(spec)
      val samples = ArrayList<Double>(iterations + warmup)

      repeat(warmup + iterations) {
        coroutineContext.ensureActive()
        session.writeInputForTiming(0, input)
        samples += session.run()
      }

      return summarize(spec.displayName, accel, samples, warmup)
    }
  }

  /**
   * Input tensor of the right size filled with reproducible noise.
   *
   * Noise rather than zeros on purpose: some backends short-circuit on constant or sparse
   * inputs, and a benchmark that measures the fast path for data no real image produces is
   * worse than no benchmark.
   */
  private fun syntheticInput(spec: ModelSpec): FloatArray {
    val shape = spec.inputShape
      ?: error("${spec.id} has no declared input shape, so it cannot be benchmarked blind")
    val n = shape.fold(1) { a, b -> a * b }
    val rng = Pcg32(0xBEEF)
    return FloatArray(n) { rng.nextFloat() }
  }
}

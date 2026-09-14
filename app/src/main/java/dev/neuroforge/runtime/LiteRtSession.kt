package dev.neuroforge.runtime

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import dev.neuroforge.core.Accel
import java.io.File

/** Maps this app's accelerator vocabulary onto LiteRT's. */
internal fun Accel.toLiteRt(): Accelerator = when (this) {
  Accel.NPU -> Accelerator.NPU
  Accel.GPU -> Accelerator.GPU
  Accel.CPU -> Accelerator.CPU
}

/**
 * Process-wide LiteRT environment.
 *
 * Created once: it is what registers the vendor NPU accelerator providers, and building it
 * per model would re-scan for runtimes on every load.
 */
object LiteRt {

  private const val TAG = "LiteRt"

  @Volatile private var env: Environment? = null

  /** Never throws — a device with no NPU provider still has to run on GPU/CPU. */
  fun environment(context: Context): Environment? {
    env?.let { return it }
    synchronized(this) {
      env?.let { return it }
      val created = runCatching {
        Environment.create(BuiltinNpuAcceleratorProvider(context.applicationContext))
      }.onFailure {
        Log.w(TAG, "No NPU accelerator provider; GPU/CPU only", it)
      }.getOrNull()
      env = created
      return created
    }
  }
}

/** What happened while trying to place a graph on hardware. */
data class LoadAttempt(val accelerator: Accel, val ok: Boolean, val millis: Long, val error: String?)

/**
 * A loaded graph, pinned to the accelerator that actually took it.
 *
 * Owns its input and output [TensorBuffer]s for its whole lifetime rather than allocating
 * per call: on a tiled 4K upscale that is the difference between 25 buffer allocations and
 * 25 × 2 × tile-sized allocations, on the thread the user is waiting on.
 */
class LiteRtSession private constructor(
  val label: String,
  val accelerator: Accel,
  val attempts: List<LoadAttempt>,
  val compileMillis: Long,
  private val model: CompiledModel,
) : AutoCloseable {

  private val inputs: List<TensorBuffer> = model.createInputBuffers()
  private val outputs: List<TensorBuffer> = model.createOutputBuffers()

  val inputCount: Int get() = inputs.size
  val outputCount: Int get() = outputs.size

  /** Writes float data into input tensor [index]. */
  fun writeInput(index: Int, data: FloatArray) {
    inputs[index].writeFloat(data)
  }

  /** Writes integer data into input tensor [index] — token ids, timestep indices. */
  fun writeInput(index: Int, data: IntArray) {
    inputs[index].writeInt(data)
  }

  /** Runs the graph. Returns the elapsed wall-clock in milliseconds. */
  fun run(): Long {
    val t0 = SystemClock.elapsedRealtimeNanos()
    model.run(inputs, outputs)
    return (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000
  }

  /** Reads output tensor [index] as floats. */
  fun readOutput(index: Int): FloatArray = outputs[index].readFloat()

  override fun close() {
    runCatching { inputs.forEach { it.close() } }
    runCatching { outputs.forEach { it.close() } }
    runCatching { model.close() }
  }

  companion object {

    private const val TAG = "LiteRtSession"

    /**
     * Loads [modelFile], trying [preference] in order and stopping at the first accelerator
     * that takes the whole graph.
     *
     * ### Why one accelerator at a time
     *
     * LiteRT accepts a list and will fall back internally, op by op. That is usually the
     * right call — but it also means you can never answer "did this actually run on the
     * NPU?", which is the entire question this app exists to answer. So each target is
     * tried alone and the winner is recorded, and only if every single-target attempt fails
     * does the mixed list get one last try, so an awkward graph still runs rather than
     * failing outright for the sake of a tidy report.
     */
    fun load(
      context: Context,
      label: String,
      modelFile: File,
      preference: List<Accel>,
    ): LiteRtSession {
      require(modelFile.isFile) { "model file missing: ${modelFile.absolutePath}" }
      val env = LiteRt.environment(context)
      val attempts = mutableListOf<LoadAttempt>()

      for (accel in preference) {
        val t0 = SystemClock.elapsedRealtime()
        val result = runCatching {
          CompiledModel.create(
            modelFile.absolutePath,
            CompiledModel.Options(accel.toLiteRt()),
            env,
          )
        }
        val elapsed = SystemClock.elapsedRealtime() - t0
        result.fold(
          onSuccess = { model ->
            attempts += LoadAttempt(accel, ok = true, millis = elapsed, error = null)
            Log.i(TAG, "$label -> $accel in $elapsed ms")
            return LiteRtSession(label, accel, attempts, elapsed, model)
          },
          onFailure = { e ->
            attempts += LoadAttempt(accel, ok = false, millis = elapsed, error = e.message ?: e.javaClass.simpleName)
            Log.w(TAG, "$label rejected by $accel after $elapsed ms: ${e.message}")
          },
        )
      }

      // Last resort: let LiteRT mix accelerators across the graph. Slower to reason about,
      // but a running model beats a clean failure report.
      val t0 = SystemClock.elapsedRealtime()
      val mixed = runCatching {
        CompiledModel.create(
          modelFile.absolutePath,
          CompiledModel.Options(*preference.map { it.toLiteRt() }.toTypedArray()),
          env,
        )
      }
      val elapsed = SystemClock.elapsedRealtime() - t0
      mixed.onSuccess { model ->
        attempts += LoadAttempt(Accel.CPU, ok = true, millis = elapsed, error = "mixed delegation")
        Log.i(TAG, "$label loaded with mixed delegation in $elapsed ms")
        // Reported as CPU: with per-op fallback in play, claiming a specific accelerator
        // would be a guess, and the conservative label is the honest one.
        return LiteRtSession(label, Accel.CPU, attempts, elapsed, model)
      }

      throw IllegalStateException(
        "No accelerator could load '$label'. Attempts: " +
          attempts.joinToString("; ") { "${it.accelerator}=${it.error}" }
      )
    }
  }
}

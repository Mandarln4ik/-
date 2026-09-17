package dev.neuroforge.runtime

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig
import dev.neuroforge.core.Accel
import dev.neuroforge.core.AcceleratorPolicy
import dev.neuroforge.core.explainLlmFailure
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext

/** Where a model load has got to. There is no percentage to report; see [LlmEngine.load]. */
data class LlmLoadProgress(
  val stage: String,
  val elapsedMillis: Long,
  val tried: List<String> = emptyList(),
)

/** How a chat is sampled. Mirrors LiteRT-LM's `SamplerConfig`. */
data class SamplingOptions(
  val temperature: Double = 0.7,
  val topK: Int = 40,
  val topP: Double = 0.9,
)

/**
 * A loaded text model and one conversation on it.
 *
 * Text generation does not go through the `CompiledModel` API the vision graphs use.
 * LiteRT-LM owns the KV cache and the decode loop, which is why a chat cannot be assembled
 * out of the same pieces — and also why it is the one runtime here that exposes an NPU
 * backend for a *generative* model. The diffusion transformer cannot reach the APU; this
 * can.
 *
 * @param accelerator the backend that actually accepted the model, not the one requested.
 * @param contextTokens the window the engine was configured with, for the context gauge.
 * @param loadMillis how long [load] took, including any just-in-time NPU compilation.
 */
class LlmEngine private constructor(
  val accelerator: Accel,
  val contextTokens: Int,
  val loadMillis: Long,
  val attempts: List<String>,
  private val engine: Engine,
) : AutoCloseable {

  private var conversation: Conversation? = null

  /**
   * Starts a fresh conversation, discarding any previous one.
   *
   * The engine holds conversation state, so this is what "new chat" actually means; the
   * message list in the UI is a record of it, not the state itself.
   */
  suspend fun startConversation(
    systemPrompt: String,
    sampling: SamplingOptions = SamplingOptions(),
  ): Unit = withContext(Dispatchers.IO) {
    closeConversation()
    conversation = engine.createConversation(
      ConversationConfig(
        systemInstruction = Contents.of(systemPrompt),
        samplerConfig = SamplerConfig(
          temperature = sampling.temperature,
          topK = sampling.topK,
          topP = sampling.topP,
        ),
      )
    )
  }

  /**
   * Streams the reply as text deltas.
   *
   * Each [Message] carries a list of [Content] parts, and only the text ones mean anything
   * in a chat. An earlier version of this called `toString()` on the message, which is not a
   * subtly worse rendering — it puts the data class's own debug representation into the
   * conversation instead of the model's words.
   */
  fun send(prompt: String): Flow<String> {
    val conv = conversation ?: error("No conversation started")
    return conv.sendMessageAsync(prompt).map { it.text() }
  }

  private fun Message.text(): String =
    contents.contents.joinToString("") { part ->
      when (part) {
        is Content.Text -> part.text
        else -> ""
      }
    }

  private fun closeConversation() {
    runCatching { conversation?.close() }
    conversation = null
  }

  override fun close() {
    closeConversation()
    runCatching { engine.close() }
  }

  companion object {
    private const val TAG = "LlmEngine"

    /**
     * Loads [modelFile], trying the backends [policy] allows in order.
     *
     * [onProgress] reports a stage and an elapsed time, not a percentage: `initialize()` is
     * one blocking native call with no progress channel, and inventing a bar that advances
     * on a timer would be a decoration rather than information. The elapsed figure is worth
     * showing because the first NPU load can include just-in-time graph compilation and
     * takes tens of seconds, which is otherwise indistinguishable from a hang.
     *
     * Reaching the NPU needs the vendor runtime on the loader's search path, which is not
     * set for a normal app process — hence [configureNativeRuntime]. Without it the NPU
     * backend fails to initialise for a reason that looks nothing like "library not found".
     */
    suspend fun load(
      context: Context,
      modelFile: File,
      policy: AcceleratorPolicy = AcceleratorPolicy.AUTO,
      contextTokens: Int = 2048,
      onProgress: (LlmLoadProgress) -> Unit = {},
    ): LlmEngine = withContext(Dispatchers.IO) {
      require(modelFile.isFile) { "model file missing: ${modelFile.absolutePath}" }
      val started = SystemClock.elapsedRealtime()
      fun elapsed() = SystemClock.elapsedRealtime() - started

      val libDir = context.applicationInfo.nativeLibraryDir
      val targets = policy.order
      if (Accel.NPU in targets) configureNativeRuntime(libDir)

      val failures = mutableListOf<String>()
      for (accel in targets) {
        onProgress(LlmLoadProgress("Loading on $accel", elapsed(), failures.toList()))
        val backend = when (accel) {
          Accel.NPU -> Backend.NPU(nativeLibraryDir = libDir)
          Accel.GPU -> Backend.GPU()
          Accel.CPU -> Backend.CPU()
        }
        val attempt = runCatching {
          val created = Engine(
            EngineConfig(
              modelPath = modelFile.absolutePath,
              backend = backend,
              maxNumTokens = contextTokens,
              // Required for just-in-time compilation; without it the NPU backend has
              // nowhere to put the graph it builds and fails on every load.
              cacheDir = context.cacheDir.path,
            )
          )
          created.initialize()
          created
        }
        attempt.fold(
          onSuccess = {
            Log.i(TAG, "${modelFile.name} -> $accel in ${elapsed()} ms")
            onProgress(LlmLoadProgress("Ready on $accel", elapsed(), failures.toList()))
            return@withContext LlmEngine(
              accel, contextTokens, elapsed(), failures.toList(), it,
            )
          },
          onFailure = { e ->
            val explained = explainLlmFailure(accel, e.message)
            failures += explained
            Log.w(TAG, "${modelFile.name} rejected by $accel: $explained", e)
          },
        )
      }
      error(
        "No backend loaded ${modelFile.name} after ${elapsed()} ms.\n" +
          failures.joinToString("\n")
      )
    }

    @Volatile private var nativeRuntimeConfigured = false

    /**
     * Puts the vendor NPU runtime directories on the native loader's search path.
     *
     * These are the locations a MediaTek or Qualcomm runtime is installed to by the
     * vendor. An app process does not search them by default, so the NPU backend cannot
     * dlopen what it needs and reports an initialisation failure instead.
     */
    private fun configureNativeRuntime(nativeLibraryDir: String) {
      if (nativeRuntimeConfigured) return
      runCatching {
        android.system.Os.setenv(
          "LD_LIBRARY_PATH",
          "$nativeLibraryDir:/vendor/lib64:/vendor/lib64/snap:/system/lib64",
          true,
        )
        android.system.Os.setenv(
          "ADSP_LIBRARY_PATH",
          "$nativeLibraryDir;/vendor/lib64/snap;/vendor/dsp;/vendor/lib/rfsa/adsp",
          true,
        )
      }.onFailure { Log.w(TAG, "could not set native library paths: ${it.message}") }
      nativeRuntimeConfigured = true
    }
  }
}

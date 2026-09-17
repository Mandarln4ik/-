package dev.neuroforge.runtime

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import dev.neuroforge.core.Accel
import dev.neuroforge.core.AcceleratorPolicy
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext

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
 */
class LlmEngine private constructor(
  val accelerator: Accel,
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
    temperature: Double = 0.7,
    topK: Int = 40,
    topP: Double = 0.9,
  ): Unit = withContext(Dispatchers.IO) {
    closeConversation()
    conversation = engine.createConversation(
      ConversationConfig(
        systemInstruction = Contents.of(systemPrompt),
        samplerConfig = SamplerConfig(temperature = temperature, topK = topK, topP = topP),
      )
    )
  }

  /** Streams the reply token group by token group. */
  fun send(prompt: String): Flow<String> = callbackFlow {
    val conv = conversation
    if (conv == null) {
      close(IllegalStateException("No conversation started"))
      return@callbackFlow
    }
    conv.sendMessageAsync(prompt, object : MessageCallback {
      override fun onMessage(message: Message) {
        trySend(message.toString())
      }

      override fun onDone() {
        close()
      }

      override fun onError(throwable: Throwable) {
        close(throwable)
      }
    })
    awaitClose { }
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
     * Reaching the NPU needs the vendor runtime on the loader's search path, which is not
     * set for a normal app process — hence [configureNativeRuntime]. Without it the NPU
     * backend fails to initialise for a reason that looks nothing like "library not found".
     */
    suspend fun load(
      context: Context,
      modelFile: File,
      policy: AcceleratorPolicy = AcceleratorPolicy.AUTO,
    ): LlmEngine = withContext(Dispatchers.IO) {
      require(modelFile.isFile) { "model file missing: ${modelFile.absolutePath}" }

      val libDir = context.applicationInfo.nativeLibraryDir
      val targets = policy.order
      if (Accel.NPU in targets) configureNativeRuntime(libDir)

      val failures = mutableListOf<String>()
      for (accel in targets) {
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
              cacheDir = context.cacheDir.path,
            )
          )
          created.initialize()
          created
        }
        attempt.fold(
          onSuccess = {
            Log.i(TAG, "${modelFile.name} -> $accel")
            return@withContext LlmEngine(accel, it)
          },
          onFailure = { e ->
            failures += "$accel=${e.message ?: e.javaClass.simpleName}"
            Log.w(TAG, "${modelFile.name} rejected by $accel", e)
          },
        )
      }
      error("No backend loaded ${modelFile.name}. Tried: ${failures.joinToString("; ")}")
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

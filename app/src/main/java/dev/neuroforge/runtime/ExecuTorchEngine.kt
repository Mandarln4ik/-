package dev.neuroforge.runtime

import android.os.SystemClock
import android.util.Log
import dev.neuroforge.core.Accel
import dev.neuroforge.core.TextBackend
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModule
import org.pytorch.executorch.extension.llm.LlmModuleConfig

/**
 * PyTorch ExecuTorch, running a `.pte` model.
 *
 * ## Why this is CPU
 *
 * ExecuTorch has vendor backends, MediaTek's NeuroPilot among them, but they are compiled
 * into the runtime when it is built. The public `org.pytorch:executorch-android` artifact
 * carries XNNPACK and nothing else — established by unpacking the AAR and looking at what
 * `jni/arm64-v8a` actually references. So this reports [Accel.CPU] rather than trying the
 * NPU and reporting a fallback: there is no NPU path in this binary to fall back from.
 *
 * Reaching the APU through ExecuTorch would mean building the AAR against the NeuroPilot
 * SDK, which is a decision about how this app is distributed rather than something it can
 * do at run time.
 *
 * ## Why the tokenizer is a separate file
 *
 * A `.pte` is the graph alone. ExecuTorch takes the tokenizer as its own path, so a model
 * here is two downloads, and a missing second one fails inside the native loader with a
 * message that does not name it. Hence the explicit check before anything is opened.
 */
class ExecuTorchEngine private constructor(
  override val contextTokens: Int,
  override val loadMillis: Long,
  private val sampling: SamplingOptions,
  private val module: LlmModule,
) : TextEngine {

  override val backend: TextBackend get() = TextBackend.EXECUTORCH

  /** Always CPU. See the class note: the published runtime has no other backend in it. */
  override val accelerator: Accel get() = Accel.CPU

  override val attempts: List<String> get() = emptyList()

  private var systemPrompt: String = ""

  /**
   * ExecuTorch has no conversation object: the module holds a KV cache and `resetContext()`
   * clears it. So "new conversation" is that reset, and the system prompt is remembered
   * here to be prepended to the first turn.
   */
  override suspend fun startConversation(systemPrompt: String, sampling: SamplingOptions) {
    withContext(Dispatchers.IO) {
      this@ExecuTorchEngine.systemPrompt = systemPrompt
      runCatching { module.resetContext() }
        .onFailure { Log.w(TAG, "could not reset context", it) }
    }
  }

  override fun send(prompt: String): Flow<String> = callbackFlow {
    val text = if (systemPrompt.isBlank()) prompt else "$systemPrompt\n\n$prompt"
    // Only the first turn carries the system prompt; after that the module's own cache has
    // it, and repeating it would spend context on a duplicate every turn.
    systemPrompt = ""

    // `LlmGenerationConfig.Builder()` is not callable from Kotlin: these are Kotlin
    // classes and the builder constructors are `internal`, which the compiler enforces
    // across modules even though the bytecode says public. `create()` is the way in.
    val config = LlmGenerationConfig.create()
      .seqLen(contextTokens)
      .temperature(sampling.temperature.toFloat())
      // The prompt is already in the chat above the reply; echoing it back would print it
      // twice in the bubble.
      .echo(false)
      .build()

    val callback = object : LlmCallback {
      override fun onResult(result: String) {
        trySend(result)
      }

      override fun onStats(stats: String) {
        // Free-form JSON from the runtime. Logged rather than parsed: the app measures its
        // own time-to-first-token, and a schema nobody has pinned is not worth depending on.
        Log.i(TAG, "stats: $stats")
      }

      override fun onError(code: Int, message: String) {
        close(IllegalStateException("ExecuTorch error $code: $message"))
      }
    }

    // `generate` blocks until the reply is finished, so it needs its own thread; the flow
    // stays open until it returns.
    val job = CoroutineScope(Dispatchers.IO).launch {
      try {
        module.generate(text, config, callback)
        close()
      } catch (e: Throwable) {
        close(e)
      }
    }

    awaitClose {
      runCatching { module.stop() }
      job.cancel()
    }
  }
    // The runtime emits from its own thread while the collector recomposes on the main one.
    // callbackFlow's default 64-slot channel would start refusing sends part-way through a
    // long reply, and trySend drops silently - a lost token is a corrupted word.
    .buffer(Channel.UNLIMITED)

  override fun close() {
    runCatching { module.stop() }
    runCatching { module.close() }
  }

  companion object {
    private const val TAG = "ExecuTorchEngine"

    suspend fun load(
      modelFile: File,
      tokenizerFile: File?,
      contextTokens: Int,
      sampling: SamplingOptions,
      onProgress: (LlmLoadProgress) -> Unit = {},
    ): ExecuTorchEngine = withContext(Dispatchers.IO) {
      require(modelFile.isFile) { "model file missing: ${modelFile.absolutePath}" }
      val tokenizer = tokenizerFile?.takeIf { it.isFile } ?: error(
        "${modelFile.name} needs its tokenizer alongside it. A .pte is the graph only; " +
          "ExecuTorch loads the tokenizer from a separate file, so both have to be " +
          "downloaded."
      )

      val started = SystemClock.elapsedRealtime()
      onProgress(LlmLoadProgress("Loading on CPU (ExecuTorch)", 0))

      val module = LlmModule(
        LlmModuleConfig.create()
          .modulePath(modelFile.absolutePath)
          .tokenizerPath(tokenizer.absolutePath)
          .temperature(sampling.temperature.toFloat())
          .modelType(LlmModule.MODEL_TYPE_TEXT)
          .build()
      )
      module.load()

      val elapsed = SystemClock.elapsedRealtime() - started
      Log.i(TAG, "${modelFile.name} loaded in $elapsed ms")
      onProgress(LlmLoadProgress("Ready on CPU (ExecuTorch)", elapsed))
      ExecuTorchEngine(contextTokens, elapsed, sampling, module)
    }
  }
}

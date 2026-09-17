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
import com.google.ai.edge.litertlm.tool
import dev.neuroforge.core.Accel
import dev.neuroforge.core.AcceleratorPolicy
import dev.neuroforge.core.TextBackend
import dev.neuroforge.core.explainLlmFailure
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
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
 * @param contextTokens the window the engine was configured with, for the context gauge.
 * @param loadMillis how long [load] took, including any just-in-time NPU compilation.
 */
class LiteRtLmEngine private constructor(
  override val accelerator: Accel,
  override val contextTokens: Int,
  override val loadMillis: Long,
  override val attempts: List<String>,
  override val supportsImages: Boolean,
  private val engine: Engine,
) : TextEngine {

  override val backend: TextBackend get() = TextBackend.LITERT_LM

  private var conversation: Conversation? = null

  /**
   * Starts a fresh conversation, discarding any previous one.
   *
   * The engine holds conversation state, so this is what "new chat" actually means; the
   * message list in the UI is a record of it, not the state itself.
   */
  override suspend fun startConversation(
    systemPrompt: String,
    sampling: SamplingOptions,
    tools: Set<String>,
  ): Unit = withContext(Dispatchers.IO) {
    closeConversation()
    conversation = engine.createConversation(
      ConversationConfig(
        systemInstruction = Contents.of(systemPrompt),
        // Real function calling, not a prompt convention: with automatic tool calling on,
        // the runtime invokes these itself when the model asks and folds the result back
        // into the same turn. A model that was not trained to emit tool calls simply never
        // does, which costs nothing but the description in its context.
        tools = DeviceTools.bind(tools, accelerator).map { tool(it) },
        automaticToolCalling = true,
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
  override fun send(prompt: String, imagePaths: List<String>): Flow<String> {
    val conv = conversation ?: error("No conversation started")
    if (imagePaths.isEmpty()) {
      return conv.sendMessageAsync(prompt).map { it.text() }
    }
    check(supportsImages) {
      "This model has no vision executor, so it cannot be shown an image. Pick a model " +
        "the Models tab marks as able to see, or send the message without the photo."
    }
    // The image goes first: every multimodal template this runtime supports expects the
    // picture before the question about it, and the reference sample orders it that way.
    val contents = Contents.of(
      imagePaths.map { Content.ImageFile(it) } + Content.Text(prompt),
    )
    return conv.sendMessageAsync(contents).map { it.text() }
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
    private const val TAG = "LiteRtLmEngine"

    /**
     * Images the engine will accept in one turn.
     *
     * Each one costs a few hundred tokens of context after the vision tower encodes it, so
     * a handful is already most of a small window. Four is enough to compare things and
     * small enough that the reply still has room.
     */
    private const val MAX_IMAGES_PER_TURN = 4

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
      vision: Boolean = false,
      onProgress: (LlmLoadProgress) -> Unit = {},
    ): LiteRtLmEngine = withContext(Dispatchers.IO) {
      require(modelFile.isFile) { "model file missing: ${modelFile.absolutePath}" }
      val started = SystemClock.elapsedRealtime()
      fun elapsed() = SystemClock.elapsedRealtime() - started

      val libDir = context.applicationInfo.nativeLibraryDir
      val targets = policy.order
      if (Accel.NPU in targets) configureNativeRuntime(libDir)

      val failures = mutableListOf<String>()

      // Vision first, then the same accelerators again without it.
      //
      // A vision executor is a separate thing from the text one and can fail on its own —
      // the tower may want a backend this chip's driver will not give it. Falling back to
      // text-only is much better than failing the load, because the model still answers
      // questions; what must not happen is the app going on to believe it can see. That is
      // why the second pass is a distinct attempt rather than a retry, and why the engine
      // carries whether vision actually came up rather than whether it was asked for.
      val passes = if (vision) listOf(true, false) else listOf(false)

      for (withVision in passes) {
        for (accel in targets) {
          val label = if (withVision) "$accel with vision" else "$accel"
          onProgress(LlmLoadProgress("Loading on $label", elapsed(), failures.toList()))
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
                // Upstream's guidance is to match the vision executor to the text backend
                // and let the cascade handle the combinations that do not start.
                visionBackend = if (withVision) backend else null,
                maxNumTokens = contextTokens,
                maxNumImages = if (withVision) MAX_IMAGES_PER_TURN else null,
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
              Log.i(TAG, "${modelFile.name} -> $label in ${elapsed()} ms")
              if (vision && !withVision) {
                failures += "Loaded without vision: the text model works, but this build " +
                  "of the runtime would not start a vision executor on this device, so " +
                  "photos cannot be sent to it."
              }
              onProgress(LlmLoadProgress("Ready on $label", elapsed(), failures.toList()))
              return@withContext LiteRtLmEngine(
                accel, contextTokens, elapsed(), failures.toList(), withVision, it,
              )
            },
            onFailure = { e ->
              val explained = explainLlmFailure(accel, e.message)
              failures += if (withVision) "$explained (with vision)" else explained
              Log.w(TAG, "${modelFile.name} rejected by $label: $explained", e)
            },
          )
        }
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

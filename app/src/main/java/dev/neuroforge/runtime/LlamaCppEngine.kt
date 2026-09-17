package dev.neuroforge.runtime

import android.os.SystemClock
import android.util.Log
import dev.neuroforge.core.Accel
import dev.neuroforge.core.StopReason
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

/**
 * llama.cpp, running a GGUF file.
 *
 * ## Why this backend exists
 *
 * GGUF is where the open-weights world publishes. LiteRT-LM reads `.litertlm` and ExecuTorch
 * reads `.pte`, and between them that is a handful of models converted by two vendors;
 * essentially every community quantisation of every open model is a `.gguf`. Without this
 * the Models tab can only offer what Google and Meta chose to convert.
 *
 * ## Why this is CPU
 *
 * llama.cpp has Vulkan and OpenCL backends that do run on Android, but each is a build-time
 * option with its own driver caveats, and neither has ever run on this phone's Mali G1-Ultra
 * during development. Shipping one unexercised and letting the app claim GPU would be the
 * same kind of unverified claim the accelerator badge exists to prevent. [Accel.CPU] is what
 * this reports because it is what the binary contains.
 *
 * ## How multi-turn works without re-reading the conversation
 *
 * The context keeps a KV cache, so only the part of the transcript that is not in it yet
 * needs decoding. Each turn formats the whole conversation with the model's own chat
 * template, then feeds the suffix past what was already fed — which is the template's
 * end-of-turn marker, the new user block, and the assistant opener. A tenth turn therefore
 * costs one turn of prefill rather than ten. If a template ever produces something that is
 * not an extension of the previous transcript, [send] notices that the prefix no longer
 * matches, clears the cache and feeds the lot; correctness does not depend on the
 * optimisation holding.
 */
class LlamaCppEngine private constructor(
  override val contextTokens: Int,
  override val loadMillis: Long,
  override val attempts: List<String>,
  private val model: Long,
  private val context: Long,
  private val templated: Boolean,
) : TextEngine {

  override val backend: TextBackend get() = TextBackend.LLAMA_CPP

  /** Always CPU. See the class note: no GPU backend is compiled into this APK. */
  override val accelerator: Accel get() = Accel.CPU

  override var lastStop: StopReason? = null
    private set

  override var lastTokens: Int = 0
    private set

  private var sampling = SamplingOptions()

  /** The conversation so far, in the shape `llama_chat_apply_template` takes. */
  private val roles = mutableListOf<String>()
  private val contents = mutableListOf<String>()

  /** Exactly the text that has been decoded into the KV cache. */
  private var fed = ""

  override suspend fun startConversation(systemPrompt: String, sampling: SamplingOptions) {
    withContext(Dispatchers.IO) {
      this@LlamaCppEngine.sampling = sampling
      LlamaCppNative.clearMemory(context)
      roles.clear()
      contents.clear()
      fed = ""
      if (systemPrompt.isNotBlank()) {
        roles += "system"
        contents += systemPrompt
      }
    }
  }

  override fun send(prompt: String): Flow<String> = callbackFlow {
    roles += "user"
    contents += prompt

    val full = format(addAssistant = true)
    // A template that is not a pure extension of the last one — the system prompt moved,
    // say — means the cache no longer describes this transcript and has to go.
    val reuse = fed.isNotEmpty() && full.startsWith(fed)
    if (!reuse && fed.isNotEmpty()) {
      Log.i(TAG, "chat template is not an extension of the cached prefix; reprocessing")
      LlamaCppNative.clearMemory(context)
    }
    val delta = if (reuse) full.substring(fed.length) else full
    // BOS belongs at the start of the sequence and nowhere else, so it rides only the
    // batch that begins a fresh cache.
    val addSpecial = !reuse

    val reply = StringBuilder()
    var tokens = 0

    val job = CoroutineScope(Dispatchers.IO).launch {
      try {
        val stop = LlamaCppNative.generate(
          model = model,
          context = context,
          prompt = delta,
          addSpecial = addSpecial,
          // Bounded by the window rather than by a guess: the native side reports
          // CONTEXT_FULL when it runs out of KV slots, which is the honest answer.
          maxTokens = contextTokens,
          temperature = sampling.temperature.toFloat(),
          topK = sampling.topK,
          topP = sampling.topP.toFloat(),
          seed = SEED_RANDOM,
        ) { token ->
          tokens++
          reply.append(token)
          // The buffer below is unlimited, so the only way a send fails is the collector
          // having gone — which is exactly when the native loop should stop.
          trySend(token).isSuccess
        }
        lastStop = stopReason(stop)
        lastTokens = tokens
        if (lastStop == StopReason.CONTEXT_FULL || lastStop == StopReason.ERROR) {
          // Both of these can land part-way through prefill, so some unknown prefix of the
          // delta is in the cache and the rest is not. Tracking a partial commit is more
          // machinery than it is worth for a turn that has already failed: drop the cache
          // and let the next turn feed the transcript again from the beginning.
          LlamaCppNative.clearMemory(context)
          fed = ""
        } else {
          // Everything fed plus everything generated is now in the cache. The end-of-turn
          // marker is not — sampling stops at that token without decoding it — so the next
          // turn's suffix starts with it, which is exactly what the template will produce.
          fed = (if (reuse) fed else "") + delta + reply
        }
        roles += "assistant"
        contents += reply.toString()
        close()
      } catch (e: Throwable) {
        // The native side threw, so how much of the prompt reached the cache is unknown.
        // Drop the cache and the turn together rather than leaving the two describing
        // different conversations.
        runCatching { LlamaCppNative.clearMemory(context) }
        fed = ""
        lastStop = StopReason.ERROR
        lastTokens = tokens
        if (reply.isNotEmpty()) {
          roles += "assistant"
          contents += reply.toString()
        } else if (roles.lastOrNull() == "user") {
          // Nothing came back at all, so the turn may as well not have been asked.
          roles.removeAt(roles.lastIndex)
          contents.removeAt(contents.lastIndex)
        }
        close(e)
      }
    }

    awaitClose { job.cancel() }
  }
    // The native decode loop hands tokens over from an IO thread while the collector is
    // recomposing on the main one. callbackFlow's default 64-slot channel would make
    // trySend start failing on a long reply, and a dropped token is a corrupted word.
    .buffer(Channel.UNLIMITED)

  private fun format(addAssistant: Boolean): String {
    if (!templated) return fallbackPrompt(addAssistant)
    return LlamaCppNative.formatChat(
      model, roles.toTypedArray(), contents.toTypedArray(), addAssistant,
    ) ?: fallbackPrompt(addAssistant)
  }

  /**
   * ChatML, used when the file's own template is one llama.cpp cannot apply.
   *
   * It is a guess, and the load panel says so — most current instruction-tuned models use
   * this shape, but a model that does not will answer oddly rather than not at all, which
   * is the failure worth naming out loud.
   */
  private fun fallbackPrompt(addAssistant: Boolean): String = buildString {
    for (i in roles.indices) {
      append("<|im_start|>").append(roles[i]).append('\n')
      append(contents[i]).append("<|im_end|>\n")
    }
    if (addAssistant) append("<|im_start|>assistant\n")
  }

  override fun close() {
    LlamaCppNative.freeContext(context)
    LlamaCppNative.freeModel(model)
  }

  companion object {
    private const val TAG = "LlamaCppEngine"

    /** `LLAMA_DEFAULT_SEED`; llama.cpp reads it as unsigned and seeds from the clock. */
    private const val SEED_RANDOM = -1

    private fun stopReason(code: Int): StopReason = when (code) {
      LlamaCppNative.Stop.COMPLETE -> StopReason.COMPLETE
      LlamaCppNative.Stop.CANCELLED -> StopReason.CANCELLED
      LlamaCppNative.Stop.CONTEXT_FULL -> StopReason.CONTEXT_FULL
      LlamaCppNative.Stop.ERROR -> StopReason.ERROR
      // Hitting the token limit is the window filling up from the caller's point of view:
      // maxTokens is set to the window, so there is nowhere else for it to have stopped.
      else -> StopReason.CONTEXT_FULL
    }

    /**
     * Number of threads to decode on.
     *
     * Half the cores, which on every big.LITTLE phone this runs on is the performance
     * cluster. Using all of them is slower, not faster: the little cores finish their
     * share late and every other thread waits at the end of each layer.
     */
    private fun decodeThreads(): Int =
      (Runtime.getRuntime().availableProcessors() / 2).coerceIn(2, 8)

    suspend fun load(
      modelFile: File,
      contextTokens: Int,
      sampling: SamplingOptions,
      onProgress: (LlmLoadProgress) -> Unit = {},
    ): LlamaCppEngine = withContext(Dispatchers.IO) {
      require(modelFile.isFile) { "model file missing: ${modelFile.absolutePath}" }
      LlamaCppNative.available.getOrElse {
        error(
          "llama.cpp did not load on this device (${it.javaClass.simpleName}: ${it.message}). " +
            "The APK builds libllama_jni.so for arm64-v8a only, so this is expected on an " +
            "emulator and a packaging fault on a phone."
        )
      }

      val started = SystemClock.elapsedRealtime()
      onProgress(LlmLoadProgress("Reading GGUF", 0))

      val model = LlamaCppNative.loadModel(modelFile.absolutePath)
      if (model == 0L) {
        error(
          "${modelFile.name} could not be read as GGUF. Either the download is incomplete " +
            "or it uses a quantisation this llama.cpp build does not know; check the file " +
            "size against the one on Hugging Face."
        )
      }

      val notes = mutableListOf<String>()
      LlamaCppNative.describeModel(model).takeIf { it.isNotBlank() }?.let { notes += it }

      // Running past the trained window does not fail, it degrades — the model simply has
      // not seen those positions. Clamping and saying so beats a chat that turns to noise
      // a thousand tokens in for no visible reason.
      val trained = LlamaCppNative.trainedContextSize(model)
      val asked = contextTokens
      val wanted = if (trained > 0 && asked > trained) {
        notes += "Context capped at $trained; that is what this model was trained for."
        trained
      } else {
        asked
      }

      val threads = decodeThreads()
      notes += "Decoding on $threads threads."
      onProgress(LlmLoadProgress("Allocating a $wanted-token context", 0, notes))
      val context = LlamaCppNative.newContext(model, wanted, threads)
      if (context == 0L) {
        LlamaCppNative.freeModel(model)
        error(
          "A $wanted-token context would not fit in memory for ${modelFile.name}. " +
            "Lower the context window in Settings, or use a smaller quantisation."
        )
      }

      // Probing with an empty conversation is the cheapest way to find out whether this
      // file's template is one llama.cpp can apply, and it is worth knowing at load time
      // rather than discovering it from a strange first reply.
      val templated = LlamaCppNative.formatChat(
        model, arrayOf("user"), arrayOf("hello"), true,
      ) != null
      if (!templated) {
        notes += "This file's chat template is not one llama.cpp recognises; falling back " +
          "to ChatML, which may not match how the model was trained."
      }

      val opened = LlamaCppNative.contextSize(context).takeIf { it > 0 } ?: wanted
      val elapsed = SystemClock.elapsedRealtime() - started
      Log.i(TAG, "${modelFile.name} loaded in $elapsed ms, $opened tokens, $threads threads")
      onProgress(LlmLoadProgress("Ready on CPU (llama.cpp)", elapsed, notes))
      LlamaCppEngine(opened, elapsed, notes, model, context, templated)
    }
  }
}

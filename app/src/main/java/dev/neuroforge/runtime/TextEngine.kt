package dev.neuroforge.runtime

import android.content.Context
import dev.neuroforge.core.Accel
import dev.neuroforge.core.AcceleratorPolicy
import dev.neuroforge.core.StopReason
import dev.neuroforge.core.TextBackend
import dev.neuroforge.core.backendFor
import java.io.File
import kotlinx.coroutines.flow.Flow

/** Where a model load has got to. There is no percentage to report; see [TextEngines.load]. */
data class LlmLoadProgress(
  val stage: String,
  val elapsedMillis: Long,
  val tried: List<String> = emptyList(),
)

/** How a chat is sampled. Every backend takes some subset of this. */
data class SamplingOptions(
  val temperature: Double = 0.7,
  val topK: Int = 40,
  val topP: Double = 0.9,
)

/**
 * A loaded text model, whichever runtime is holding it.
 *
 * The three runtimes differ in what they read and what silicon they reach, and nothing
 * above this line should care which one answered. What callers do need is the two facts
 * that change what they show: which backend it turned out to be, and which accelerator it
 * actually landed on — neither of which is the same as what was asked for.
 */
interface TextEngine : AutoCloseable {
  val backend: TextBackend

  /** The accelerator that accepted the model, not the one requested. */
  val accelerator: Accel

  /** Context window the engine opened. May be smaller than the one asked for. */
  val contextTokens: Int

  /** How long loading took, including any just-in-time compilation. */
  val loadMillis: Long

  /** Per-backend failures on the way to the one that worked, already in plain words. */
  val attempts: List<String>

  /**
   * Why the last [send] ended, when the runtime says so rather than just closing the stream.
   *
   * Only llama.cpp reports one today: it is the only backend here that owns the decode loop,
   * so it is the only one that can tell "the model emitted end-of-turn" from "there was no
   * KV slot left". Null means the caller should fall back to inferring it, which is what an
   * empty reply or a thrown exception already gives it.
   */
  val lastStop: StopReason? get() = null

  /** Tokens in the last reply, counted rather than estimated. 0 when the backend cannot say. */
  val lastTokens: Int get() = 0

  /**
   * Whether this engine can be shown an image.
   *
   * False for everything except a LiteRT-LM engine that opened a model with a vision tower
   * and got a vision executor to initialise. Both halves matter: the runtime has to support
   * images *and* the weights have to contain something that can see. A text-only Gemma 3 1B
   * on LiteRT-LM answers this false, which is what stops the app sending a photo into a
   * model that will either error or, worse, confidently describe an image it never got.
   */
  val supportsImages: Boolean get() = false

  /**
   * Whether this runtime can call tools.
   *
   * Only LiteRT-LM, which takes an OpenAPI description and runs the function itself when
   * the model asks. Whether any given *model* emits tool calls is a separate matter and
   * not something the app can know in advance: one that was not trained for it simply
   * never calls anything, which costs only the descriptions' place in its context.
   */
  val supportsTools: Boolean get() = backend.supportsTools

  /**
   * Starts a fresh conversation, discarding any previous one.
   *
   * @param tools names from `AiTools` the user has enabled. Ignored by a runtime that
   *   cannot call them.
   */
  suspend fun startConversation(
    systemPrompt: String,
    sampling: SamplingOptions,
    tools: Set<String> = emptySet(),
  )

  /**
   * Streams the reply as text deltas.
   *
   * @param imagePaths absolute paths to images to send with the turn. Only meaningful when
   *   [supportsImages]; the engines that cannot see reject a non-empty list rather than
   *   dropping it, because silently ignoring an attachment is how a model ends up answering
   *   about a photo nobody showed it. Text attachments never reach here — they are folded
   *   into [prompt] by the caller, since that works the same for every backend.
   */
  fun send(prompt: String, imagePaths: List<String> = emptyList()): Flow<String>
}

/** Opens whichever runtime reads the file in front of it. */
object TextEngines {

  /**
   * Loads [modelFile] with the backend that reads its format.
   *
   * Routing on the extension rather than on the setting is deliberate: a `.gguf` cannot be
   * opened by LiteRT-LM whatever is selected, and "wrong backend selected" is a worse answer
   * than just using the one that reads the file. The setting decides which models are
   * offered in the first place.
   *
   * @param tokenizerFile required by ExecuTorch, which keeps the tokenizer outside the
   *   model; ignored by the other two, which carry it inside the bundle.
   * @param vision whether the catalogue says this model has a vision tower. Only a hint to
   *   ask the runtime for a vision executor; whether one actually starts is reported back
   *   by [TextEngine.supportsImages].
   */
  suspend fun load(
    context: Context,
    modelFile: File,
    policy: AcceleratorPolicy = AcceleratorPolicy.AUTO,
    contextTokens: Int = 2048,
    sampling: SamplingOptions = SamplingOptions(),
    tokenizerFile: File? = null,
    vision: Boolean = false,
    onProgress: (LlmLoadProgress) -> Unit = {},
  ): TextEngine = when (backendFor(modelFile.name)) {
    TextBackend.LITERT_LM, null ->
      LiteRtLmEngine.load(context, modelFile, policy, contextTokens, vision, onProgress)

    TextBackend.EXECUTORCH ->
      ExecuTorchEngine.load(modelFile, tokenizerFile, contextTokens, sampling, onProgress)

    TextBackend.LLAMA_CPP ->
      LlamaCppEngine.load(modelFile, contextTokens, sampling, onProgress)
  }
}

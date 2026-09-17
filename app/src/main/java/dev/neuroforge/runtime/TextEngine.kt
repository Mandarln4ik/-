package dev.neuroforge.runtime

import android.content.Context
import dev.neuroforge.core.Accel
import dev.neuroforge.core.AcceleratorPolicy
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

  /** Starts a fresh conversation, discarding any previous one. */
  suspend fun startConversation(systemPrompt: String, sampling: SamplingOptions)

  /** Streams the reply as text deltas. */
  fun send(prompt: String): Flow<String>
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
   */
  suspend fun load(
    context: Context,
    modelFile: File,
    policy: AcceleratorPolicy = AcceleratorPolicy.AUTO,
    contextTokens: Int = 2048,
    sampling: SamplingOptions = SamplingOptions(),
    tokenizerFile: File? = null,
    onProgress: (LlmLoadProgress) -> Unit = {},
  ): TextEngine = when (backendFor(modelFile.name)) {
    TextBackend.LITERT_LM, null ->
      LiteRtLmEngine.load(context, modelFile, policy, contextTokens, onProgress)

    TextBackend.EXECUTORCH ->
      ExecuTorchEngine.load(modelFile, tokenizerFile, contextTokens, sampling, onProgress)

    TextBackend.LLAMA_CPP -> error(
      "llama.cpp is not built into this APK yet, so ${modelFile.name} cannot be opened. " +
        "It needs a native build of llama.cpp shipped with the app; until then, use a " +
        ".litertlm or .pte model."
    )
  }
}

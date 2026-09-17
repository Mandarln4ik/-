package dev.neuroforge.core

/**
 * Why a reply stopped.
 *
 * Named rather than inferred from an empty result, because "the model produced nothing" and
 * "the model filled the context" and "you pressed Stop" look identical in a chat bubble and
 * lead to completely different next steps.
 */
enum class StopReason(val label: String) {
  /** The model emitted its end-of-turn marker. */
  COMPLETE("finished"),

  /** Cancelled from the UI. */
  CANCELLED("stopped by you"),

  /** The conversation reached the context window. */
  CONTEXT_FULL("context full"),

  /** The runtime raised an error part-way through. */
  ERROR("error"),

  /** The stream closed without producing a single token. */
  EMPTY("no output"),
}

/**
 * What one reply cost.
 *
 * @param ttftMillis time from sending to the first token arriving. On an NPU this is where
 *   prefill and any just-in-time graph compilation land, so it is often most of a short
 *   reply's wall clock and is worth separating from the decode rate.
 * @param decodeMillis time from the first token to the last.
 * @param tokens how many tokens the reply came to.
 * @param tokensEstimated whether [tokens] is a guess from the text — see [estimateTokens] —
 *   or a count the runtime kept. Only llama.cpp keeps one, because it is the only backend
 *   here that owns its decode loop. The difference shows up as the `≈` in [line]: a figure
 *   that is off by a third and one that is exact should not be printed the same way.
 */
data class ReplyStats(
  val ttftMillis: Long = 0,
  val decodeMillis: Long = 0,
  val tokens: Int = 0,
  val stop: StopReason = StopReason.COMPLETE,
  val detail: String? = null,
  val tokensEstimated: Boolean = true,
) {
  val totalMillis: Long get() = ttftMillis + decodeMillis

  /** Tokens per second during decode, excluding the wait for the first one. */
  val tokensPerSecond: Double
    get() = if (decodeMillis > 0) tokens * 1000.0 / decodeMillis else 0.0

  /** One line for under a chat bubble. */
  fun line(accelerator: Accel?): String = buildString {
    accelerator?.let { append(it.name).append(" · ") }
    append("${ttftMillis}ms to first token")
    if (tokens > 0) {
      append(if (tokensEstimated) " · ≈$tokens tok" else " · $tokens tok")
      if (tokensPerSecond > 0) append(" · %.1f tok/s".format(tokensPerSecond))
    }
    if (stop != StopReason.COMPLETE) append(" · ${stop.label}")
  }
}

/**
 * Rough token count for a piece of text.
 *
 * **An estimate, and labelled as one everywhere it is shown.** LiteRT-LM does not expose a
 * token count for a reply, and the tokenizer lives inside the `.litertlm` bundle where this
 * app cannot reach it, so there is nothing exact to report. Google's own sample apps use
 * `length / 4`; this splits ASCII from everything else because that ratio is a Latin-text
 * figure and is roughly half as good on Cyrillic, which would make the context gauge read
 * about twice as optimistic as reality for a Russian conversation.
 */
fun estimateTokens(text: String): Int {
  if (text.isEmpty()) return 0
  var ascii = 0
  var other = 0
  for (ch in text) if (ch.code < 128) ascii++ else other++
  return maxOf(1, ascii / 4 + other / 2)
}

/**
 * How much of the context window a conversation has spent.
 *
 * @param used estimated tokens already in the window.
 * @param limit the model's `maxNumTokens`.
 */
data class ContextBudget(val used: Int, val limit: Int) {
  val remaining: Int get() = (limit - used).coerceAtLeast(0)
  val fraction: Float get() = if (limit > 0) (used.toFloat() / limit).coerceIn(0f, 1f) else 0f

  /** True once the window is tight enough that the next turn may be truncated. */
  val tight: Boolean get() = fraction >= 0.85f

  fun line(): String = "≈$used / $limit tokens · $remaining left"
}

/** Estimated context used by a whole conversation, including the system prompt. */
fun contextUsed(systemPrompt: String, messages: List<String>): Int =
  estimateTokens(systemPrompt) + messages.sumOf { estimateTokens(it) }

/**
 * Turns a LiteRT-LM engine failure into something a person can act on.
 *
 * The raw messages are native-layer strings that name a symbol rather than a cause. The one
 * that matters most here is `TF_LITE_AUX`: it means the bundle carried no ahead-of-time NPU
 * payload for this chip and just-in-time compilation did not happen either — which is not a
 * broken app and not a broken NPU, it is the wrong file for this phone.
 */
fun explainLlmFailure(backend: Accel, raw: String?): String {
  val message = raw?.takeIf { it.isNotBlank() } ?: "no reason given"
  return when {
    message.contains("TF_LITE_AUX", ignoreCase = true) ->
      "$backend: this bundle has no ahead-of-time NPU payload for this chip, and " +
        "just-in-time compilation did not run. Download a build named for this SoC if the " +
        "repository publishes one."
    message.contains("not readable", ignoreCase = true) ||
      message.contains("FileNotFound", ignoreCase = true) ->
      "$backend: the model file is missing or unreadable — re-download it."
    message.contains("out of memory", ignoreCase = true) ||
      message.contains("OutOfMemory", ignoreCase = true) ->
      "$backend: ran out of memory loading this model. A smaller build, or a shorter " +
        "context in Settings, will fit."
    message.contains("failed to compile", ignoreCase = true) ->
      "$backend: the accelerator would not compile this graph. That is a property of the " +
        "model, not a fault — it will run on another backend."
    else -> "$backend: $message"
  }
}

package dev.neuroforge.core

/**
 * What a conversation produces.
 *
 * Separate kinds rather than one chat that guesses: a text reply, an image and a piece of
 * audio need different models, different controls and different rendering, and a single
 * "smart" chat that infers which one you meant gets it wrong precisely when the prompt is
 * ambiguous.
 */
enum class ChatKind(val label: String, val role: ModelRole) {
  TEXT("Text", ModelRole.TEXT_CHAT),
  IMAGE("Image", ModelRole.DENOISER),
  MUSIC("Music", ModelRole.AUDIO);

  /** Whether this app can currently run this kind at all. */
  val supported: Boolean get() = this != MUSIC

  /** Why a kind is unavailable, for the UI to show instead of a dead button. */
  val unsupportedReason: String?
    get() = if (this == MUSIC) {
      "No music model ships as a LiteRT graph yet. The tab exists so it is obvious this " +
        "is missing rather than hidden; when one is published it drops into the catalogue."
    } else {
      null
    }
}

/** Who said it. */
enum class Speaker { USER, MODEL }

/**
 * One turn.
 *
 * @param imagePath set for a generated image; the text then holds the prompt.
 * @param accelerator which target produced it, so a slow reply can be explained.
 * @param thinking the model's reasoning, kept apart from the answer so it can be collapsed.
 *   Separate rather than inline because it is usually longer than the reply and answers a
 *   different question — "how did it get here" instead of "what did it say".
 * @param stats what the reply cost. Null for a user turn and for replies from before this
 *   was recorded.
 * @param attachments files sent with a user turn. Kept on the message rather than only in
 *   the prompt so the bubble can show what was sent: a folded-in text file is invisible in
 *   the prompt text, and reopening a chat months later should still say which file the
 *   question was about.
 */
data class ChatMessage(
  val speaker: Speaker,
  val text: String,
  val imagePath: String? = null,
  val accelerator: Accel? = null,
  val millis: Long = 0,
  val timestamp: Long = 0,
  val thinking: String? = null,
  val stats: ReplyStats? = null,
  val attachments: List<Attachment> = emptyList(),
)

/**
 * Splits a reply into its reasoning and its answer.
 *
 * Qwen3 and the other reasoning models wrap their working in `<think>…</think>` and expect
 * the host to present it separately. Left inline it is the first and largest thing in the
 * bubble, which buries the actual answer; dropped entirely it takes away the one thing that
 * explains a wrong result.
 *
 * Handles the stream arriving in pieces: an opened but unclosed block is all reasoning so
 * far, which is what makes the thinking panel fill live rather than appearing at the end.
 */
fun splitThinking(raw: String): Pair<String?, String> {
  val open = raw.indexOf(OPEN_THINK)
  if (open < 0) return null to raw

  val bodyStart = open + OPEN_THINK.length
  val close = raw.indexOf(CLOSE_THINK, bodyStart)
  if (close < 0) {
    // Still inside the block: everything after the marker is reasoning, and anything before
    // it is answer text the model emitted first.
    val thinking = raw.substring(bodyStart)
    return thinking.ifBlank { null } to raw.substring(0, open).trim()
  }

  val thinking = raw.substring(bodyStart, close).trim()
  val answer = (raw.substring(0, open) + raw.substring(close + CLOSE_THINK.length)).trim()
  return thinking.ifBlank { null } to answer
}

private const val OPEN_THINK = "<think>"
private const val CLOSE_THINK = "</think>"

/**
 * A conversation, pinned to one model.
 *
 * The model is chosen when the chat is created and does not change: replies in a single
 * thread that silently came from different models would be impossible to interpret, and
 * for a text model the conversation state lives inside the engine anyway.
 */
data class Chat(
  val id: String,
  val kind: ChatKind,
  val modelId: String,
  val title: String,
  val messages: List<ChatMessage> = emptyList(),
  val createdAt: Long = 0,
) {
  /**
   * When this conversation was last touched.
   *
   * Falls back to [createdAt] so an empty chat still sorts sensibly instead of dropping to
   * the bottom of the list the moment it is created — which is exactly when someone is
   * looking for it.
   */
  val lastActivity: Long
    get() = messages.maxOfOrNull { it.timestamp }?.takeIf { it > 0 } ?: createdAt

  /** A title derived from the first thing actually said, for a chat still called "New chat". */
  fun derivedTitle(): String {
    if (title.isNotBlank() && title != DEFAULT_TITLE) return title
    val first = messages.firstOrNull { it.speaker == Speaker.USER }?.text?.trim()
    if (first.isNullOrBlank()) return DEFAULT_TITLE
    return if (first.length <= 40) first else first.take(37).trimEnd() + "…"
  }

  companion object {
    const val DEFAULT_TITLE = "New chat"
  }
}

/**
 * Conversations in the order they should be listed: most recently used first.
 *
 * By last message rather than by creation, because the one being worked on is the one to
 * hand. Ties break on id so the order is stable across recompositions.
 */
fun List<Chat>.mostRecentFirst(): List<Chat> =
  sortedWith(compareByDescending<Chat> { it.lastActivity }.thenBy { it.id })

/** Models that can drive a chat of this [kind]. */
fun modelsFor(kind: ChatKind, catalogue: List<ModelSpec> = ModelCatalog.all): List<ModelSpec> =
  catalogue.filter { it.role == kind.role }

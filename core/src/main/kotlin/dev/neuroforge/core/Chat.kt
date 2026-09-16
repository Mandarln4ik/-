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
 */
data class ChatMessage(
  val speaker: Speaker,
  val text: String,
  val imagePath: String? = null,
  val accelerator: Accel? = null,
  val millis: Long = 0,
  val timestamp: Long = 0,
)

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

/** Models that can drive a chat of this [kind]. */
fun modelsFor(kind: ChatKind, catalogue: List<ModelSpec> = ModelCatalog.all): List<ModelSpec> =
  catalogue.filter { it.role == kind.role }

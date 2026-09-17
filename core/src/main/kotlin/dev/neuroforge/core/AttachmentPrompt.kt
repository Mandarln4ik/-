package dev.neuroforge.core

/**
 * A prompt with its text attachments folded in, and what had to be cut to make it fit.
 *
 * [dropped] is reported rather than swallowed because silently truncating a file is the
 * kind of thing that makes a model look stupid: it answers about the half it was given and
 * there is no way to tell from the reply that it never saw the rest.
 */
data class FoldedPrompt(val text: String, val dropped: List<String> = emptyList()) {
  val wasTruncated: Boolean get() = dropped.isNotEmpty()
}

/**
 * Builds the prompt for a turn that has text files attached.
 *
 * A text file has nowhere to go but into the prompt, so it competes with the conversation
 * for the context window. The budget here is in tokens and deliberately conservative: it
 * leaves room for the reply, because a prompt that fills the window produces an empty
 * answer, which is the least informative failure there is.
 *
 * Files are fenced and named. Models follow an instruction about "the file above" far more
 * reliably when the boundary is explicit, and it also stops a file that happens to contain
 * something prompt-shaped from reading as part of the user's own message.
 *
 * @param read pulls the file's text. Passed in rather than done here so this stays testable
 *   off-device; the caller is what knows about storage.
 * @param tokenBudget how many tokens all the attachments together may occupy.
 */
fun foldTextAttachments(
  prompt: String,
  attachments: List<Attachment>,
  tokenBudget: Int,
  read: (Attachment) -> String,
): FoldedPrompt {
  val texts = attachments.filter { it.kind == AttachmentKind.TEXT }
  if (texts.isEmpty()) return FoldedPrompt(prompt)

  // Divided evenly rather than first-come: attaching a large log and a small config should
  // not mean the config never arrives.
  val perFile = (tokenBudget / texts.size).coerceAtLeast(MIN_TOKENS_PER_FILE)
  val dropped = mutableListOf<String>()

  val blocks = texts.map { attachment ->
    val raw = runCatching { read(attachment) }.getOrElse { error ->
      dropped += attachment.fileName
      return@map "--- ${attachment.fileName} (could not be read: ${error.message}) ---"
    }
    val budgetChars = perFile * CHARS_PER_TOKEN
    val body = if (raw.length <= budgetChars) {
      raw
    } else {
      dropped += attachment.fileName
      raw.take(budgetChars).substringBeforeLast('\n', raw.take(budgetChars)) +
        "\n… truncated, ${raw.length - budgetChars} more characters"
    }
    "--- ${attachment.fileName} ---\n$body\n--- end of ${attachment.fileName} ---"
  }

  val header = if (texts.size == 1) "Attached file:" else "Attached files:"
  val text = buildString {
    append(header).append('\n')
    append(blocks.joinToString("\n\n"))
    if (prompt.isNotBlank()) append("\n\n").append(prompt)
  }
  return FoldedPrompt(text, dropped)
}

/**
 * Tokens the attachments may use, given the window and what the conversation already holds.
 *
 * Half the free space, never more than a fixed ceiling. Half because the model still has to
 * answer, and the ceiling because a very large window does not make it a good idea to paste
 * a megabyte in: prefill is linear and the phone is not a server.
 */
fun attachmentTokenBudget(contextTokens: Int, usedTokens: Int): Int {
  val free = (contextTokens - usedTokens).coerceAtLeast(0)
  return (free / 2).coerceAtMost(MAX_ATTACHMENT_TOKENS)
}

/** Roughly what [estimateTokens] assumes in reverse, for ASCII source and prose. */
private const val CHARS_PER_TOKEN = 4

private const val MIN_TOKENS_PER_FILE = 64

private const val MAX_ATTACHMENT_TOKENS = 4096

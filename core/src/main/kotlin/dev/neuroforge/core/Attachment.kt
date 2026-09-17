package dev.neuroforge.core

/**
 * What kind of thing was attached, and therefore how it reaches the model.
 *
 * The split is not cosmetic. An image has to go to the runtime as an image, through a
 * model that was trained to see one; a text file has nowhere to go but into the prompt,
 * where it costs context. Those are different failure modes — "this model cannot see" and
 * "this file does not fit" — so they are different kinds here rather than one blob the
 * engine has to sniff.
 */
enum class AttachmentKind { IMAGE, TEXT }

/**
 * A file the user attached to a message.
 *
 * [localPath] rather than the picker's `content://` URI on purpose: that URI is a
 * short-lived grant to another app's document, and a chat is stored and reopened days
 * later. The file is copied into the app's own storage when it is attached, so reopening
 * a conversation does not depend on the other app still being installed, the SD card still
 * being mounted, or a permission that expired when the process died.
 */
data class Attachment(
  val id: String,
  val fileName: String,
  val kind: AttachmentKind,
  val localPath: String,
  val sizeBytes: Long,
  val mimeType: String,
)

/**
 * Which kind [mimeType]/[fileName] is, or null when the app cannot do anything useful with it.
 *
 * MIME first because it is what the picker actually reports; the extension is the fallback
 * for the providers that return `application/octet-stream` for everything. Deciding this in
 * one place is what lets the composer reject a file at the moment it is picked instead of
 * at the moment it is sent.
 */
fun attachmentKindFor(mimeType: String?, fileName: String): AttachmentKind? {
  val mime = mimeType?.lowercase().orEmpty()
  val name = fileName.lowercase()

  if (mime.startsWith("image/")) return AttachmentKind.IMAGE
  if (IMAGE_EXTENSIONS.any { name.endsWith(it) }) return AttachmentKind.IMAGE

  // A lot of perfectly readable formats are not text/*: JSON is application/json, and most
  // source files come back as octet-stream. Extension decides those.
  if (mime.startsWith("text/")) return AttachmentKind.TEXT
  if (TEXT_EXTENSIONS.any { name.endsWith(it) }) return AttachmentKind.TEXT
  if (mime in TEXT_MIME_TYPES) return AttachmentKind.TEXT

  return null
}

/** Said to the user when a file is refused, naming what would work. */
fun unsupportedAttachmentReason(fileName: String): String =
  "$fileName is not something this app can pass to a model. Images go to a vision model " +
    "as images; text and source files are read into the prompt. A PDF, archive or " +
    "spreadsheet would need converting first."

private val IMAGE_EXTENSIONS =
  listOf(".png", ".jpg", ".jpeg", ".webp", ".bmp", ".heic", ".heif", ".gif")

private val TEXT_EXTENSIONS = listOf(
  ".txt", ".md", ".markdown", ".log", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".yaml",
  ".yml", ".toml", ".ini", ".cfg", ".conf", ".properties", ".sql", ".sh", ".bash", ".zsh",
  ".bat", ".ps1", ".py", ".kt", ".kts", ".java", ".c", ".h", ".cc", ".cpp", ".hpp", ".rs",
  ".go", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".html", ".htm", ".css", ".scss", ".swift",
  ".rb", ".php", ".pl", ".lua", ".r", ".m", ".gradle", ".cmake", ".dockerfile", ".env",
  ".diff", ".patch",
)

private val TEXT_MIME_TYPES = setOf(
  "application/json", "application/xml", "application/x-yaml", "application/yaml",
  "application/javascript", "application/x-sh", "application/sql", "application/toml",
)

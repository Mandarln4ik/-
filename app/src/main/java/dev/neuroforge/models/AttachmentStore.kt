package dev.neuroforge.models

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import dev.neuroforge.core.Attachment
import dev.neuroforge.core.AttachmentKind
import dev.neuroforge.core.attachmentKindFor
import dev.neuroforge.core.unsupportedAttachmentReason
import java.io.File
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Attached files, copied into the app's own storage.
 *
 * The picker hands back a `content://` URI, which is a temporary grant to read a document
 * another app owns. That is fine for the next few seconds and useless afterwards: the grant
 * dies with the process, the other app can be uninstalled, and a chat is something people
 * reopen days later. So the bytes are copied once, at the moment of attaching, and
 * everything downstream deals in ordinary file paths.
 *
 * It also means the runtime gets a real path. LiteRT-LM's `Content.ImageFile` takes an
 * absolute path and hands it to native code, which cannot resolve a content URI at all.
 */
class AttachmentStore(private val context: Context) {

  private val root = File(context.filesDir, "attachments")

  /**
   * Copies [uri] in and describes it.
   *
   * @throws IllegalArgumentException when the file is not something a model can be given —
   *   thrown here, at the moment of picking, so the refusal names the file while the user
   *   is still looking at the picker rather than after they have typed a question about it.
   */
  suspend fun store(chatId: String, uri: Uri): Attachment = withContext(Dispatchers.IO) {
    val resolver = context.contentResolver
    val displayName = queryDisplayName(uri) ?: uri.lastPathSegment ?: "attachment"
    val mimeType = resolver.getType(uri).orEmpty()

    val kind = attachmentKindFor(mimeType, displayName)
      ?: throw IllegalArgumentException(unsupportedAttachmentReason(displayName))

    val id = UUID.randomUUID().toString()
    val dir = File(root, chatId).apply { mkdirs() }
    // The id prefixes the name so two photos both called IMG_0001.jpg do not collide, and
    // the real name survives for the chip and for the prompt.
    val target = File(dir, "$id-${displayName.take(MAX_NAME_LENGTH).replace('/', '_')}")

    val copied = resolver.openInputStream(uri)?.use { input ->
      target.outputStream().use { output -> input.copyTo(output) }
    } ?: throw IllegalStateException(
      "$displayName could not be opened. The app that provided it may have revoked access."
    )

    require(copied <= MAX_BYTES) {
      target.delete()
      "$displayName is ${copied / 1024 / 1024} MB. Attachments are capped at " +
        "${MAX_BYTES / 1024 / 1024} MB: an image is resized before it reaches the model " +
        "anyway, and a text file that large cannot fit in any context window here."
    }

    Attachment(
      id = id,
      fileName = displayName,
      kind = kind,
      localPath = target.absolutePath,
      sizeBytes = copied,
      mimeType = mimeType,
    )
  }

  /** Reads a text attachment. Bounded, because the caller's budget is in tokens not bytes. */
  suspend fun readText(attachment: Attachment): String = withContext(Dispatchers.IO) {
    require(attachment.kind == AttachmentKind.TEXT) { "not a text attachment" }
    val file = File(attachment.localPath)
    require(file.isFile) { "${attachment.fileName} is no longer on disk" }
    file.reader().use { reader ->
      val buffer = CharArray(MAX_TEXT_CHARS)
      val read = reader.read(buffer)
      if (read <= 0) "" else String(buffer, 0, read)
    }
  }

  /** Drops the files for a chat that has been deleted. */
  suspend fun forget(chatId: String): Unit = withContext(Dispatchers.IO) {
    runCatching { File(root, chatId).deleteRecursively() }
      .onFailure { Log.w(TAG, "could not remove attachments for $chatId", it) }
  }

  /** Drops one attachment's file, for a message never sent. */
  suspend fun discard(attachment: Attachment): Unit = withContext(Dispatchers.IO) {
    runCatching { File(attachment.localPath).delete() }
  }

  private fun queryDisplayName(uri: Uri): String? = runCatching {
    context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
      ?.use { cursor ->
        val column = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
        if (column >= 0 && cursor.moveToFirst()) cursor.getString(column) else null
      }
  }.getOrNull()

  private companion object {
    const val TAG = "AttachmentStore"

    /** 32 MB. Well past any useful phone photo, well short of anything that would hurt. */
    const val MAX_BYTES = 32L * 1024 * 1024

    /**
     * Characters read from a text attachment.
     *
     * Deliberately larger than any prompt budget: the trimming that matters happens in
     * `foldTextAttachments`, where it knows the context window. This only stops a
     * pathological file from being held in memory whole.
     */
    const val MAX_TEXT_CHARS = 1_000_000

    const val MAX_NAME_LENGTH = 80
  }
}

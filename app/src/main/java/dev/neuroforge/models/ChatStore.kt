package dev.neuroforge.models

import android.content.Context
import android.util.Log
import dev.neuroforge.core.Accel
import dev.neuroforge.core.Chat
import dev.neuroforge.core.ChatKind
import dev.neuroforge.core.ChatMessage
import dev.neuroforge.core.Speaker
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * Conversations on disk.
 *
 * Plain JSON in one file rather than a database: a few dozen chats with a few dozen turns
 * each is kilobytes, and a schema migration on a phone is a worse problem to have than a
 * full rewrite on every save. An unreadable file is dropped rather than propagated — a
 * corrupt chat history should not stop the app opening.
 */
class ChatStore(context: Context) {

  private val file = File(context.filesDir, "chats.json")

  suspend fun load(): List<Chat> = withContext(Dispatchers.IO) {
    if (!file.isFile) return@withContext emptyList()
    runCatching { parse(file.readText()) }
      .onFailure { Log.w(TAG, "chat history unreadable, starting empty", it) }
      .getOrDefault(emptyList())
  }

  suspend fun save(chats: List<Chat>): Unit = withContext(Dispatchers.IO) {
    runCatching {
      val tmp = File(file.parentFile, "${file.name}.tmp")
      tmp.writeText(serialize(chats))
      if (!tmp.renameTo(file)) {
        tmp.copyTo(file, overwrite = true)
        tmp.delete()
      }
    }.onFailure { Log.w(TAG, "could not save chats", it) }
  }

  private fun serialize(chats: List<Chat>): String {
    val array = JSONArray()
    chats.forEach { chat ->
      array.put(
        JSONObject().apply {
          put("id", chat.id)
          put("kind", chat.kind.name)
          put("modelId", chat.modelId)
          put("title", chat.title)
          put("createdAt", chat.createdAt)
          put(
            "messages",
            JSONArray().apply {
              chat.messages.forEach { m ->
                put(
                  JSONObject().apply {
                    put("speaker", m.speaker.name)
                    put("text", m.text)
                    m.imagePath?.let { put("imagePath", it) }
                    m.accelerator?.let { put("accelerator", it.name) }
                    put("millis", m.millis)
                    put("timestamp", m.timestamp)
                  }
                )
              }
            },
          )
        }
      )
    }
    return array.toString()
  }

  private fun parse(text: String): List<Chat> {
    val array = JSONArray(text)
    return (0 until array.length()).mapNotNull { i ->
      val obj = array.optJSONObject(i) ?: return@mapNotNull null
      val kind = enumOrNull<ChatKind>(obj.optString("kind")) ?: return@mapNotNull null
      val messages = obj.optJSONArray("messages")?.let { msgs ->
        (0 until msgs.length()).mapNotNull { j ->
          val m = msgs.optJSONObject(j) ?: return@mapNotNull null
          ChatMessage(
            speaker = enumOrNull<Speaker>(m.optString("speaker")) ?: Speaker.MODEL,
            text = m.optString("text"),
            imagePath = m.optString("imagePath").takeIf { it.isNotBlank() },
            accelerator = enumOrNull<Accel>(m.optString("accelerator")),
            millis = m.optLong("millis"),
            timestamp = m.optLong("timestamp"),
          )
        }
      }.orEmpty()

      Chat(
        id = obj.optString("id").ifBlank { return@mapNotNull null },
        kind = kind,
        modelId = obj.optString("modelId"),
        title = obj.optString("title", Chat.DEFAULT_TITLE),
        messages = messages,
        createdAt = obj.optLong("createdAt"),
      )
    }
  }

  /** A stored name can outlive its constant across an update; drop the entry rather than crash. */
  private inline fun <reified T : Enum<T>> enumOrNull(name: String): T? =
    enumValues<T>().firstOrNull { it.name == name }

  private companion object {
    const val TAG = "ChatStore"
  }
}

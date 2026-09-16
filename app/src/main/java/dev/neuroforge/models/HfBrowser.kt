package dev.neuroforge.models

import android.util.Log
import dev.neuroforge.core.formatBytes
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray

/** One file in a Hugging Face repository. */
data class RepoFile(val path: String, val sizeBytes: Long) {
  val isModel: Boolean get() = path.endsWith(".tflite", true) || path.endsWith(".litertlm", true)
  val display: String get() = if (sizeBytes > 0) "$path — ${formatBytes(sizeBytes)}" else path
}

/**
 * Lists what a Hugging Face repository actually contains.
 *
 * The reason this is a user-facing feature rather than an internal detail: every model
 * download that failed in this app failed because a file name in the catalogue was a
 * guess. Letting someone point at a repository and see its real contents removes the
 * guesser from the loop entirely — which is a better fix than guessing more carefully.
 */
class HfBrowser {

  private val http = OkHttpClient.Builder()
    .connectTimeout(20, TimeUnit.SECONDS)
    .readTimeout(30, TimeUnit.SECONDS)
    .followRedirects(true)
    .build()

  /**
   * Files in [repoId], model files first.
   *
   * @throws IOException with the server's own wording, so a private or misspelled
   *   repository is distinguishable from a network problem.
   */
  suspend fun list(repoId: String): List<RepoFile> = withContext(Dispatchers.IO) {
    val clean = repoId.trim().trim('/')
    require(clean.count { it == '/' } == 1 && clean.length > 2) {
      "A repository id looks like 'owner/name' — got '$repoId'"
    }

    val url = "https://huggingface.co/api/models/$clean/tree/main?recursive=true"
    http.newCall(Request.Builder().url(url).build()).execute().use { response ->
      if (response.code == 404) {
        throw IOException("No repository '$clean' — check the spelling, or it may be private.")
      }
      if (!response.isSuccessful) {
        throw IOException("HTTP ${response.code} listing '$clean'")
      }
      val body = response.body?.string().orEmpty()
      val array = runCatching { JSONArray(body) }.getOrElse {
        throw IOException("Unexpected reply from Hugging Face for '$clean'")
      }

      val files = (0 until array.length()).mapNotNull { i ->
        val item = array.optJSONObject(i) ?: return@mapNotNull null
        if (item.optString("type") != "file") return@mapNotNull null
        val path = item.optString("path").takeIf { it.isNotBlank() } ?: return@mapNotNull null
        RepoFile(path, item.optLong("size", 0L))
      }
      Log.i(TAG, "$clean: ${files.size} files, ${files.count { it.isModel }} runnable")
      // Runnable graphs first, then everything else alphabetically: the point of the
      // screen is to find a model, not to browse a directory.
      files.sortedWith(compareByDescending<RepoFile> { it.isModel }.thenBy { it.path })
    }
  }

  private companion object {
    const val TAG = "HfBrowser"
  }
}

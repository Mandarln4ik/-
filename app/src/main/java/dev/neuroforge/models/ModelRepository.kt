package dev.neuroforge.models

import android.content.Context
import android.util.Log
import dev.neuroforge.core.ModelFile
import dev.neuroforge.core.ModelSource
import dev.neuroforge.core.chooseModelFile
import dev.neuroforge.core.ModelSpec
import dev.neuroforge.core.Provenance
import dev.neuroforge.core.Tar
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.security.DigestInputStream
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import java.util.zip.GZIPInputStream
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import kotlin.coroutines.coroutineContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray

/** Where a model stands on this device. */
sealed interface ModelState {
  /** Present and usable. */
  data class Ready(val file: File, val bytes: Long) : ModelState

  /** Not downloaded yet. */
  data object Absent : ModelState

  /** On disk but wrong — truncated, or the digest did not match. */
  data class Corrupt(val reason: String) : ModelState

  /** Published upstream in a format that needs converting first; see docs/MODELS.md. */
  data class NeedsConversion(val recipe: String) : ModelState
}

/** Progress of one file's download. */
data class DownloadProgress(
  val fileName: String,
  val downloaded: Long,
  val total: Long,
  val bytesPerSecond: Long,
  val stage: String = "downloading",
) {
  val fraction: Float get() = if (total > 0) (downloaded.toFloat() / total).coerceIn(0f, 1f) else 0f
  val etaSeconds: Long
    get() = if (bytesPerSecond > 0) (total - downloaded) / bytesPerSecond else -1
}

/**
 * Fetches and keeps track of model weights.
 *
 * Two things drive the design. The files are large — the diffusion transformer alone is
 * over 2 GiB — so a partially written file is never given its final name, and nothing is
 * handed to the runtime until its length and, where pinned, its digest check out.
 *
 * And some models are published only as a `.tar.gz` release asset. Unpacking those here,
 * on the phone, is deliberate: the alternative is telling someone to fetch and unpack it
 * on a desktop first, which is the step this app is meant to remove.
 */
class ModelRepository(private val context: Context) {

  private val root: File = File(context.filesDir, "models").apply { mkdirs() }

  private val http: OkHttpClient by lazy {
    OkHttpClient.Builder()
      .connectTimeout(30, TimeUnit.SECONDS)
      // No read timeout: a slow link on a 2 GiB body is not a stalled connection, and
      // cutting it at 30 s would make the large models undownloadable on exactly the
      // connections that most need patience.
      .readTimeout(0, TimeUnit.SECONDS)
      .retryOnConnectionFailure(true)
      .followRedirects(true)
      .build()
  }

  fun fileFor(spec: ModelSpec, file: ModelFile): File = File(root, "${spec.id}/${file.fileName}")

  private fun partFor(spec: ModelSpec, file: ModelFile): File =
    File(root, "${spec.id}/${file.fileName}.part")

  /** Total bytes currently occupied by downloaded models. */
  fun diskUsage(): Long = root.walkTopDown().filter { it.isFile }.sumOf { it.length() }

  /** State of every file in [spec]; the model is usable only when all are [ModelState.Ready]. */
  fun stateOf(spec: ModelSpec): Map<String, ModelState> =
    spec.files.associate { it.fileName to stateOfFile(spec, it) }

  /** Convenience: is every file of [spec] present? */
  fun isReady(spec: ModelSpec): Boolean = stateOf(spec).values.all { it is ModelState.Ready }

  fun stateOfFile(spec: ModelSpec, file: ModelFile): ModelState {
    val target = fileFor(spec, file)
    if (target.isFile) {
      // For an archive the stored size describes the download, not the extracted member,
      // so only a direct download can be length-checked here.
      if (!file.isArchive && file.sizeBytes > 0 && target.length() != file.sizeBytes) {
        return ModelState.Corrupt(
          "expected ${file.sizeBytes} bytes, found ${target.length()} — delete and re-download"
        )
      }
      return ModelState.Ready(target, target.length())
    }
    if (file.provenance == Provenance.NEEDS_CONVERSION) {
      return ModelState.NeedsConversion("docs/MODELS.md — ${file.originLabel()}")
    }
    return ModelState.Absent
  }

  /**
   * Downloads one file and, if it is an archive, unpacks the member the catalogue names.
   *
   * The body is written to a `.part` sibling and only renamed once every check passes, so
   * an interrupted download can never be mistaken for a complete model — a failure mode
   * that otherwise surfaces as an unreadable native crash much later.
   *
   * Downloads restart rather than resume. Range resumption was removed when archives
   * arrived: a resumed archive cannot be hashed as one stream, and a checksum that is
   * sometimes skipped is worse than one that always runs.
   */
  fun download(spec: ModelSpec, file: ModelFile): Flow<DownloadProgress> = flow {
    val target = fileFor(spec, file)
    target.parentFile?.mkdirs()
    val part = partFor(spec, file)
    part.delete()

    val digest = MessageDigest.getInstance("SHA-256")
    val url = resolveUrl(file)
    val request = Request.Builder().url(url).build()

    http.newCall(request).execute().use { response ->
      if (!response.isSuccessful) {
        throw IOException(
          "HTTP ${response.code} fetching ${file.fileName} from ${file.originLabel()}"
        )
      }
      val body = response.body ?: throw IOException("empty body for ${file.fileName}")
      val total = body.contentLength().takeIf { it > 0 } ?: file.sizeBytes

      var written = 0L
      var lastEmit = System.nanoTime()
      var lastBytes = 0L
      var rate = 0L

      DigestInputStream(body.byteStream(), digest).use { source ->
        part.outputStream().buffered(1 shl 16).use { out ->
          val buffer = ByteArray(1 shl 16)
          while (true) {
            coroutineContext.ensureActive()
            val n = source.read(buffer)
            if (n <= 0) break
            out.write(buffer, 0, n)
            written += n

            val now = System.nanoTime()
            val dt = now - lastEmit
            if (dt > 200_000_000L) { // ~5 updates/s is plenty for a progress bar
              rate = ((written - lastBytes) * 1_000_000_000L) / dt
              lastEmit = now
              lastBytes = written
              emit(DownloadProgress(file.fileName, written, total, rate))
            }
          }
        }
      }
      emit(DownloadProgress(file.fileName, written, total, rate))
    }

    if (file.sizeBytes > 0 && part.length() != file.sizeBytes) {
      part.delete()
      throw IOException(
        "${file.fileName}: downloaded ${part.length()} bytes, catalogue expects ${file.sizeBytes}"
      )
    }

    file.sha256?.let { expected ->
      val actual = digest.digest().joinToString("") { "%02x".format(it) }
      if (!actual.equals(expected, ignoreCase = true)) {
        part.delete()
        throw IOException("${file.fileName}: checksum mismatch (got $actual)")
      }
      Log.i(TAG, "${file.fileName}: checksum verified")
    }

    val member = file.archiveMember
    if (member != null) {
      emit(DownloadProgress(file.fileName, part.length(), part.length(), 0, stage = "unpacking"))
      extractFromTarGz(part, member, target)
      part.delete()
    } else if (!part.renameTo(target)) {
      throw IOException("could not finalise ${file.fileName}")
    }

    Log.i(TAG, "ready: ${file.fileName} (${target.length()} bytes)")
  }.flowOn(Dispatchers.IO)

  /**
   * Works out the URL to fetch, listing the repository when the expected name is not there.
   *
   * A catalogued file name is a claim about someone else's repository, and repositories get
   * re-quantised and renamed. Rather than let that surface as a 404 on the device — which is
   * exactly how this failed — an HTTP 404 triggers one listing call, and the file is chosen
   * from what actually exists.
   */
  private fun resolveUrl(file: ModelFile): String {
    val declared = file.downloadUrl()
    val source = file.source
    if (source !is ModelSource.HuggingFace) return declared
    if (headOk(declared)) return declared

    Log.w(TAG, "${source.path} is not in ${source.repoId}; listing the repository")
    val available = listHuggingFaceFiles(source.repoId)
    if (available.isEmpty()) {
      throw IOException(
        "'${source.path}' was not found in ${source.repoId} and the repository could not " +
          "be listed. Check the model page, or side-load the file."
      )
    }
    val chosen = chooseModelFile(available, source.path, source.hints)
      ?: throw IOException(
        "'${source.path}' was not found in ${source.repoId}. It contains: " +
          available.joinToString(", ").take(400)
      )
    Log.i(TAG, "resolved ${source.path} -> $chosen")
    return "${ModelFile.HF_ENDPOINT}/${source.repoId}/resolve/main/$chosen?download=true"
  }

  /** True when the URL exists; a failed probe is treated as "present" so a flaky network
   *  does not trigger a pointless listing and a misleading error. */
  private fun headOk(url: String): Boolean = runCatching {
    http.newCall(Request.Builder().url(url).head().build()).execute()
      .use { it.isSuccessful || it.code != 404 }
  }.getOrDefault(true)

  /** File paths at the root of a Hugging Face model repository. */
  private fun listHuggingFaceFiles(repoId: String): List<String> = runCatching {
    val url = "${ModelFile.HF_ENDPOINT}/api/models/$repoId/tree/main"
    http.newCall(Request.Builder().url(url).build()).execute().use { response ->
      if (!response.isSuccessful) return emptyList()
      val body = response.body?.string().orEmpty()
      val array = JSONArray(body)
      (0 until array.length()).mapNotNull { i ->
        array.optJSONObject(i)?.takeIf { it.optString("type") == "file" }?.optString("path")
      }.filter { it.isNotBlank() }
    }
  }.onFailure { Log.w(TAG, "could not list $repoId", it) }.getOrDefault(emptyList())

  /**
   * Pulls one member out of a gzipped tar into [target].
   *
   * Skips AppleDouble sidecars explicitly. An archive built on macOS carries a `._name`
   * entry beside every real one, with the same extension — matching on the extension alone
   * extracts a few hundred bytes of resource fork, and the failure then appears inside the
   * model loader rather than here.
   */
  private fun extractFromTarGz(archive: File, member: String, target: File) {
    GZIPInputStream(archive.inputStream().buffered(1 shl 16)).use { gz ->
      val entry = Tar.seek(gz) { candidate ->
        candidate.isFile &&
          !Tar.isAppleDoubleSidecar(candidate.name) &&
          candidate.name.substringAfterLast('/') == member
      } ?: throw IOException("'$member' not found inside ${archive.name}")

      val tmp = File(target.parentFile, "${target.name}.extract")
      tmp.outputStream().buffered(1 shl 16).use { out ->
        copyExactly(gz, out, entry.size)
      }
      if (!tmp.renameTo(target)) {
        tmp.delete()
        throw IOException("could not write extracted $member")
      }
      Log.i(TAG, "extracted $member (${entry.size} bytes)")
    }
  }

  private fun copyExactly(source: InputStream, sink: java.io.OutputStream, bytes: Long) {
    val buffer = ByteArray(1 shl 16)
    var remaining = bytes
    while (remaining > 0) {
      val n = source.read(buffer, 0, minOf(remaining, buffer.size.toLong()).toInt())
      if (n < 0) throw IOException("archive ended after ${bytes - remaining} of $bytes bytes")
      sink.write(buffer, 0, n)
      remaining -= n
    }
  }

  /** Removes every file of [spec], including a stale partial download. */
  fun delete(spec: ModelSpec) {
    File(root, spec.id).deleteRecursively()
  }

  /** Adopts a file the user side-loaded (adb push, file picker) as [spec]'s weights. */
  suspend fun importFile(spec: ModelSpec, file: ModelFile, source: File): Unit =
    withContext(Dispatchers.IO) {
      val target = fileFor(spec, file)
      target.parentFile?.mkdirs()
      source.copyTo(target, overwrite = true)
    }

  private companion object {
    const val TAG = "ModelRepository"
  }
}

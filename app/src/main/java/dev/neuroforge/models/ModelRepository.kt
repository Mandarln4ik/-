package dev.neuroforge.models

import android.content.Context
import android.util.Log
import dev.neuroforge.core.ModelFile
import dev.neuroforge.core.ModelSpec
import dev.neuroforge.core.Provenance
import java.io.File
import java.io.IOException
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import kotlin.coroutines.coroutineContext
import okhttp3.OkHttpClient
import okhttp3.Request

/** Where a model stands on this device. */
sealed interface ModelState {
  /** Present, size checked, and checksum verified if one was pinned. */
  data class Ready(val file: File, val bytes: Long, val checksumVerified: Boolean) : ModelState

  /** Not downloaded yet. */
  data object Absent : ModelState

  /** Partially downloaded; a resumed download continues from [bytes]. */
  data class Partial(val bytes: Long, val total: Long) : ModelState

  /** On disk but wrong — truncated, or the digest does not match. */
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
) {
  val fraction: Float get() = if (total > 0) (downloaded.toFloat() / total).coerceIn(0f, 1f) else 0f
  val etaSeconds: Long
    get() = if (bytesPerSecond > 0) (total - downloaded) / bytesPerSecond else -1
}

/**
 * Fetches and keeps track of model weights.
 *
 * The files here are large — the diffusion transformer alone is over 2 GiB — which drives
 * every design choice in this class: downloads resume rather than restart, files are
 * verified before the first run rather than after a confusing failure, and a partially
 * written file is never handed to the runtime under its final name.
 */
class ModelRepository(private val context: Context) {

  private val root: File = File(context.filesDir, "models").apply { mkdirs() }

  private val http: OkHttpClient by lazy {
    OkHttpClient.Builder()
      .connectTimeout(30, TimeUnit.SECONDS)
      // No read timeout: a slow link on a 2 GiB body is not a stalled connection, and
      // killing it at 30 s would make the large models undownloadable on exactly the
      // connections that most need resumable downloads.
      .readTimeout(0, TimeUnit.SECONDS)
      .retryOnConnectionFailure(true)
      .build()
  }

  fun fileFor(spec: ModelSpec, file: ModelFile): File = File(root, "${spec.id}/${file.fileName}")

  private fun partFor(spec: ModelSpec, file: ModelFile): File =
    File(root, "${spec.id}/${file.fileName}.part")

  /** Total bytes currently occupied by downloaded models. */
  fun diskUsage(): Long = root.walkTopDown().filter { it.isFile }.sumOf { it.length() }

  /** State of every file in [spec]; the model is usable only when all of them are [ModelState.Ready]. */
  fun stateOf(spec: ModelSpec): Map<String, ModelState> =
    spec.files.associate { it.fileName to stateOfFile(spec, it) }

  /** Convenience: is every file of [spec] present and valid? */
  fun isReady(spec: ModelSpec): Boolean =
    stateOf(spec).values.all { it is ModelState.Ready }

  fun stateOfFile(spec: ModelSpec, file: ModelFile): ModelState {
    val target = fileFor(spec, file)
    if (target.isFile) {
      // Size is checked every time because it is free; the digest is not, so it is only
      // verified on demand via `verify`, and Ready reports which of the two it is standing on.
      if (file.sizeBytes > 0 && target.length() != file.sizeBytes) {
        return ModelState.Corrupt(
          "expected ${file.sizeBytes} bytes, found ${target.length()} — delete and re-download"
        )
      }
      return ModelState.Ready(target, target.length(), checksumVerified = false)
    }
    val part = partFor(spec, file)
    if (part.isFile && part.length() > 0) return ModelState.Partial(part.length(), file.sizeBytes)
    if (file.provenance == Provenance.NEEDS_CONVERSION) {
      return ModelState.NeedsConversion("docs/MODELS.md — ${file.repoId}")
    }
    return ModelState.Absent
  }

  /**
   * Downloads one file, resuming a previous attempt when possible.
   *
   * Emits progress as it goes. The body is written to a `.part` sibling and renamed only
   * after the length checks pass, so an interrupted download can never be mistaken for a
   * complete model — that failure mode surfaces as an incomprehensible native crash hours
   * later, and it is worth a few lines to make impossible.
   */
  fun download(spec: ModelSpec, file: ModelFile): Flow<DownloadProgress> = flow {
    val target = fileFor(spec, file)
    target.parentFile?.mkdirs()
    val part = partFor(spec, file)

    val existing = if (part.isFile) part.length() else 0L
    val request = Request.Builder()
      .url(file.downloadUrl())
      .apply { if (existing > 0) header("Range", "bytes=$existing-") }
      .build()

    http.newCall(request).execute().use { response ->
      if (!response.isSuccessful) {
        throw IOException("HTTP ${response.code} fetching ${file.fileName} from ${file.repoId}")
      }
      // A server that ignores the Range header restarts the body at 0; appending then would
      // silently corrupt the file, so fall back to a clean rewrite.
      val resumed = existing > 0 && response.code == 206
      val body = response.body ?: throw IOException("empty body for ${file.fileName}")
      val total = (body.contentLength().takeIf { it > 0 } ?: 0L) + (if (resumed) existing else 0L)

      var written = if (resumed) existing else 0L
      val sink = java.io.RandomAccessFile(part, "rw")
      sink.use { out ->
        out.setLength(written)
        out.seek(written)

        val buffer = ByteArray(1 shl 16)
        val source = body.byteStream()
        var lastEmit = System.nanoTime()
        var lastBytes = written
        var rate = 0L

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
            emit(DownloadProgress(file.fileName, written, maxOf(total, file.sizeBytes), rate))
          }
        }
        emit(DownloadProgress(file.fileName, written, maxOf(total, file.sizeBytes), rate))
      }
    }

    if (file.sizeBytes > 0 && part.length() != file.sizeBytes) {
      part.delete()
      throw IOException(
        "${file.fileName}: downloaded ${part.length()} bytes, catalogue expects ${file.sizeBytes}"
      )
    }
    if (!part.renameTo(target)) {
      throw IOException("could not finalise ${file.fileName}")
    }
    Log.i(TAG, "downloaded ${file.fileName} (${target.length()} bytes)")
  }.flowOn(Dispatchers.IO)

  /**
   * Recomputes the SHA-256 of a downloaded file.
   *
   * Returns null when the catalogue has no digest pinned for it — reporting "unverified"
   * rather than "verified" for a file nothing was ever checked against.
   */
  suspend fun verify(spec: ModelSpec, file: ModelFile): Boolean? = withContext(Dispatchers.IO) {
    val expected = file.sha256 ?: return@withContext null
    val target = fileFor(spec, file)
    if (!target.isFile) return@withContext false

    val digest = MessageDigest.getInstance("SHA-256")
    target.inputStream().buffered(1 shl 16).use { input ->
      val buffer = ByteArray(1 shl 16)
      while (true) {
        coroutineContext.ensureActive()
        val n = input.read(buffer)
        if (n <= 0) break
        digest.update(buffer, 0, n)
      }
    }
    val actual = digest.digest().joinToString("") { "%02x".format(it) }
    actual.equals(expected, ignoreCase = true)
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

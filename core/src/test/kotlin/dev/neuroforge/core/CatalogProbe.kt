package dev.neuroforge.core

import java.net.HttpURLConnection
import java.net.URL

/**
 * Checks that every catalogued file can actually be fetched — the way the app fetches it.
 *
 * This exists because of a failure that happened three times: a file name in the catalogue
 * was a plausible guess, could not be verified from the authoring environment (Hugging Face
 * is unreachable there), and 404'd on someone's phone at the moment they pressed Download.
 *
 * The important detail is *what* it checks. An earlier version of this check probed the
 * declared URL with a HEAD and failed on anything but 200 — which is stricter than the app,
 * because the app falls back to listing the repository and picking the file with
 * [chooseModelFile]. A check stricter than production fails on entries that would have
 * worked, so this runs the same resolution the device runs and reports which file it landed
 * on. Pinning that name back into the catalogue then saves the device one round trip; it is
 * not what makes the download work.
 *
 * Lives in the test source set so the HTTP code never ships inside the app.
 */
fun main() {
  var failures = 0
  var resolvedByListing = 0

  ModelCatalog.all.forEach { spec ->
    println("── ${spec.id}")
    spec.files.forEach { file ->
      when (val outcome = probe(file)) {
        is Outcome.Declared -> println("   ok        ${file.fileName}  ${size(outcome.bytes)}")
        is Outcome.Resolved -> {
          resolvedByListing++
          println("   resolved  ${file.fileName} -> ${outcome.actual}  ${size(outcome.bytes)}")
          println("             pin this into the catalogue to save the device a round trip")
        }
        is Outcome.Failed -> {
          failures++
          println("   FAIL      ${file.fileName}: ${outcome.reason}")
          outcome.listing.take(60).forEach { println("               $it") }
          if (outcome.listing.size > 60) println("               … ${outcome.listing.size - 60} more")
        }
      }
    }
  }

  // The reference host loop's own constants, printed so the pipeline's tensor contract is
  // something read on every commit rather than remembered from one.
  dump("Bonsai pipeline_meta.json", "$HF/litert-community/Bonsai-Image-ternary-4B/resolve/main/pipeline_meta.json", 1200)
  dump("Bonsai chat template (tokenizer_config.json)", "$HF/litert-community/Bonsai-Image-ternary-4B/resolve/main/tokenizer/tokenizer_config.json", 4000)

  println()
  println("$failures unresolvable, $resolvedByListing resolved by listing")
  if (failures > 0) kotlin.system.exitProcess(1)
}

private const val HF = "https://huggingface.co"

private sealed interface Outcome {
  data class Declared(val bytes: Long) : Outcome
  data class Resolved(val actual: String, val bytes: Long) : Outcome
  data class Failed(val reason: String, val listing: List<String>) : Outcome
}

private fun probe(file: ModelFile): Outcome {
  val declared = file.downloadUrl()
  head(declared)?.let { return Outcome.Declared(it) }

  val source = file.source as? ModelSource.HuggingFace
    ?: return Outcome.Failed("direct download did not answer 2xx: $declared", emptyList())

  val listing = listRepo(source.repoId)
  if (listing.isEmpty()) {
    return Outcome.Failed("not at the declared path and ${source.repoId} could not be listed", emptyList())
  }
  val chosen =
    chooseModelFile(listing, source.path, source.hints, source.extension, source.requires)
    ?: return Outcome.Failed(
      "no ${source.extension} file in ${source.repoId} matches '${source.path}'", listing,
    )
  val url = "$HF/${source.repoId}/resolve/main/$chosen?download=true"
  val bytes = head(url)
    ?: return Outcome.Failed("listing offered '$chosen' but it did not answer 2xx", listing)
  return Outcome.Resolved(chosen, bytes)
}

/**
 * The file's length in bytes, or null if it does not answer 2xx.
 *
 * The length is reported because the catalogue refuses to carry a size nobody measured, and
 * this is the measurement. Hugging Face answers a HEAD on a resolve URL with the real
 * object length in `x-linked-size`; `Content-Length` after the redirect is the same number.
 */
private fun head(url: String): Long? = runCatching {
  val c = URL(url).openConnection() as HttpURLConnection
  c.instanceFollowRedirects = true
  c.requestMethod = "HEAD"
  c.connectTimeout = 30_000
  c.readTimeout = 30_000
  try {
    if (c.responseCode !in 200..299) return@runCatching null
    c.getHeaderField("x-linked-size")?.toLongOrNull() ?: c.contentLengthLong.coerceAtLeast(0L)
  } finally {
    c.disconnect()
  }
}.getOrNull()

private fun size(bytes: Long): String = if (bytes > 0) "${formatBytes(bytes)} ($bytes)" else ""

/** Every file in the repository, including subdirectories — the tokenizer lives in one. */
private fun listRepo(repoId: String): List<String> = runCatching {
  val body = URL("$HF/api/models/$repoId/tree/main?recursive=true").readText()
  // Deliberately not a JSON parser: `core` has no dependencies and this is a diagnostic.
  Regex("\"path\"\\s*:\\s*\"([^\"]+)\"").findAll(body).map { it.groupValues[1] }.toList()
}.getOrDefault(emptyList())

private fun dump(title: String, url: String, limit: Int) {
  println()
  println("=== $title ===")
  println(runCatching { URL(url).readText().take(limit) }.getOrElse { "(unavailable: ${it.message})" })
}

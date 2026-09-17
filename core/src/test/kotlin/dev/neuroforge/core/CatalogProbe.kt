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
 * The important detail is *what* it checks. An earlier version probed the declared URL with
 * a HEAD and failed on anything but 200 — which is stricter than the app, because the app
 * falls back to listing the repository and picking the file with [chooseModelFile]. A check
 * stricter than production fails on entries that would have worked, so this runs the same
 * resolution the device runs and reports which file it landed on. Pinning that name back
 * into the catalogue then saves the device one round trip; it is not what makes the
 * download work.
 *
 * A gated repository is a pass, not a failure, provided the file is really there: the
 * remedy is a person accepting a licence in a browser, which CI has no token for and no
 * retry can achieve. What matters is that the catalogue is pointing at something real.
 *
 * Lives in the test source set so the HTTP code never ships inside the app.
 */
fun main() {
  var failures = 0
  var resolvedByListing = 0
  var gated = 0

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
        is Outcome.Gated -> {
          gated++
          println("   gated     ${file.fileName}: present in the repository, HTTP ${outcome.code}")
          println("             expected — the catalogue marks this repository as gated")
        }
        is Outcome.Failed -> {
          failures++
          println("   FAIL      ${file.fileName}: ${outcome.reason}")
          outcome.listing.take(60).forEach { println("               $it") }
          if (outcome.listing.size > 60) println("               … ${outcome.listing.size - 60} more")
        }
      }
      // What else the repository offers for this entry. Printed because a repository that
      // publishes one bundle per accelerator has a build compiled for a specific NPU, and
      // that is only discoverable by looking.
      candidates(file).takeIf { it.size > 1 }?.let { all ->
        println("             candidates: ${all.joinToString(", ")}")
      }
    }
  }

  // The reference host loop's own constants, printed so the pipeline's tensor contract is
  // something read on every commit rather than remembered from one.
  dump("Bonsai pipeline_meta.json (head)", "$HF/$BONSAI/resolve/main/pipeline_meta.json", 400)
  chatTemplate()

  println()
  println("$failures unresolvable, $resolvedByListing resolved by listing, $gated gated")
  if (failures > 0) kotlin.system.exitProcess(1)
}

private const val HF = "https://huggingface.co"
private const val BONSAI = "litert-community/Bonsai-Image-ternary-4B"

private sealed interface Outcome {
  data class Declared(val bytes: Long) : Outcome
  data class Resolved(val actual: String, val bytes: Long) : Outcome
  data class Gated(val code: Int) : Outcome
  data class Failed(val reason: String, val listing: List<String>) : Outcome
}

private fun probe(file: ModelFile): Outcome {
  val declared = file.downloadUrl()
  val first = head(declared)
  if (first.ok) return Outcome.Declared(first.bytes)

  val source = file.source as? ModelSource.HuggingFace
    ?: return Outcome.Failed("direct download answered HTTP ${first.code}: $declared", emptyList())

  val listing = listRepo(source.repoId)
  if (listing.isEmpty()) {
    return Outcome.Failed(
      "HTTP ${first.code} at the declared path and ${source.repoId} could not be listed",
      emptyList(),
    )
  }
  val chosen =
    chooseModelFile(listing, source.path, source.hints, source.extension, source.requires)
      ?: return Outcome.Failed(
        "no ${source.extension} file in ${source.repoId} matches '${source.path}'", listing,
      )

  val second = head("$HF/${source.repoId}/resolve/main/$chosen?download=true")
  return when {
    second.ok && chosen == source.path -> Outcome.Declared(second.bytes)
    second.ok -> Outcome.Resolved(chosen, second.bytes)
    // The file is in the listing but will not serve. For a repository the catalogue already
    // marks as gated that is the documented behaviour, not a broken entry.
    source.gated && (second.code == 401 || second.code == 403) -> Outcome.Gated(second.code)
    else -> Outcome.Failed("'$chosen' is listed but answered HTTP ${second.code}", listing)
  }
}

/** Everything in the repository this entry could resolve to, for the log. */
private fun candidates(file: ModelFile): List<String> {
  val source = file.source as? ModelSource.HuggingFace ?: return emptyList()
  return listRepo(source.repoId)
    .filter { it.endsWith(source.extension, ignoreCase = true) }
    .filter { name -> source.requires.all { name.contains(it, ignoreCase = true) } }
}

private data class Head(val ok: Boolean, val code: Int, val bytes: Long)

/**
 * HEADs [url], reporting the status and the object's length.
 *
 * The length is reported because the catalogue refuses to carry a size nobody measured, and
 * this is the measurement. Hugging Face answers a HEAD on a resolve URL with the real
 * object length in `x-linked-size`; `Content-Length` after the redirect is the same number.
 */
private fun head(url: String): Head = runCatching {
  val c = URL(url).openConnection() as HttpURLConnection
  c.instanceFollowRedirects = true
  c.requestMethod = "HEAD"
  c.connectTimeout = 30_000
  c.readTimeout = 30_000
  try {
    val code = c.responseCode
    val bytes = c.getHeaderField("x-linked-size")?.toLongOrNull()
      ?: c.contentLengthLong.coerceAtLeast(0L)
    Head(code in 200..299, code, bytes)
  } finally {
    c.disconnect()
  }
}.getOrElse { Head(false, -1, 0L) }

/** Every file in the repository, including subdirectories — the tokenizer lives in one. */
private val listings = HashMap<String, List<String>>()

private fun listRepo(repoId: String): List<String> = listings.getOrPut(repoId) {
  runCatching {
    val body = URL("$HF/api/models/$repoId/tree/main?recursive=true").readText()
    // Deliberately not a JSON parser: `core` has no dependencies and this is a diagnostic.
    Regex("\"path\"\\s*:\\s*\"([^\"]+)\"").findAll(body).map { it.groupValues[1] }.toList()
  }.getOrDefault(emptyList())
}

private fun size(bytes: Long): String = if (bytes > 0) "${formatBytes(bytes)} ($bytes)" else ""

private fun dump(title: String, url: String, limit: Int) {
  println()
  println("=== $title ===")
  println(runCatching { URL(url).readText().take(limit) }.getOrElse { "(unavailable: ${it.message})" })
}

/**
 * Finds and prints the repository's chat template, wherever it is kept.
 *
 * Two places, because transformers moved: older checkpoints embed it as a `chat_template`
 * string inside `tokenizer_config.json`, newer ones save it as a separate
 * `chat_template.jinja`. The Bonsai release has no `chat_template` key at all, so the
 * template this app renders as a constant was still unverified after the first run of this
 * check — which is exactly the situation this whole file exists to end.
 *
 * The template is Jinja and is not rendered on a phone, so the app carries the rendered
 * form as a constant. This is what keeps that constant honest: a divergence shows up here
 * rather than as an image that quietly answers a slightly different prompt.
 */
private fun chatTemplate() {
  println()
  println("=== Bonsai tokenizer/ ===")
  val tokenizerFiles = listRepo(BONSAI).filter { it.startsWith("tokenizer") }
  tokenizerFiles.forEach { println("  $it") }

  val fromJinja = tokenizerFiles.firstOrNull { it.endsWith("chat_template.jinja") }
    ?.let { fetch("$HF/$BONSAI/resolve/main/$it") }
  val fromConfig = fetch("$HF/$BONSAI/resolve/main/tokenizer/tokenizer_config.json")
    ?.let { body ->
      Regex("\"chat_template\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"")
        .find(body)?.groupValues?.get(1)?.replace("\\n", "\n")?.replace("\\\"", "\"")
    }

  println()
  println("=== Bonsai chat template ===")
  val template = fromJinja ?: fromConfig
  if (template == null) {
    println("(not found in tokenizer_config.json or a chat_template.jinja)")
  } else {
    println("source: ${if (fromJinja != null) "chat_template.jinja" else "tokenizer_config.json"}")
    // The head shows how a user turn is framed; the `add_generation_prompt` block is what
    // decides the tail this app appends, and it is at the very end of a long template — so
    // printing the first N characters shows everything except the part being verified.
    println("--- how a user turn is framed ---")
    println(template.lineSequence().filter { it.contains("im_start") }.take(4).joinToString("\n"))
    println("--- add_generation_prompt ---")
    val at = template.indexOf("add_generation_prompt")
    println(if (at >= 0) template.substring(at).take(900) else "(not in the template)")
  }

  println()
  println("this app renders: ${Bonsai.chatPrompt("PROMPT").replace("\n", "\\n")}")
  if (template != null) {
    listOf("<|im_start|>", "<|im_end|>", "<think>", "enable_thinking").forEach {
      println("  template mentions $it: ${template.contains(it)}")
    }
  }
}

private fun fetch(url: String): String? = runCatching { URL(url).readText() }.getOrNull()

package dev.neuroforge.pipeline

import android.content.Context
import android.util.Log
import dev.neuroforge.core.BpeTokenizer
import dev.neuroforge.core.ModelSpec
import dev.neuroforge.models.ModelRepository
import java.io.File
import org.json.JSONObject

/** A prompt as the text encoder takes it: fixed-length ids, plus which of them are real. */
data class TokenizedPrompt(val ids: IntArray, val mask: IntArray) {
  val length: Int get() = ids.size

  /** How many positions hold prompt rather than padding — useful for a truncation warning. */
  val realTokens: Int get() = mask.count { it != 0 }

  override fun equals(other: Any?): Boolean =
    this === other || (other is TokenizedPrompt && ids.contentEquals(other.ids) && mask.contentEquals(other.mask))

  override fun hashCode(): Int = 31 * ids.contentHashCode() + mask.contentHashCode()
}

/**
 * Loads a HuggingFace `tokenizer.json` and turns prompts into the id sequence a text encoder
 * graph expects.
 *
 * The tokenizer travels with the checkpoint for a reason: ids are meaningless without the
 * exact vocabulary that produced them, and a mismatched pair does not fail — it produces an
 * image that answers a slightly different prompt. So this refuses to guess: no tokenizer
 * file next to the weights is an error, not a fallback to something approximate.
 */
class PromptTokenizer(
  private val bpe: BpeTokenizer,
  private val padId: Int,
) {

  /**
   * Encodes [text] to exactly [maxTokens] ids plus the matching attention mask.
   *
   * Text encoder graphs are converted at a fixed sequence length, so the result is padded
   * on the right — the side these tokenizers pad on by default — or truncated to fit. The
   * mask is what tells the encoder which of those positions to read; without it the model
   * attends to a few hundred padding tokens as though they were prompt.
   */
  fun encode(text: String, maxTokens: Int): TokenizedPrompt {
    val body = bpe.encode(text)
    if (body.size > maxTokens) {
      Log.w(TAG, "prompt truncated from ${body.size} to $maxTokens tokens")
    }
    val kept = minOf(body.size, maxTokens)
    val ids = IntArray(maxTokens) { if (it < kept) body[it] else padId }
    val mask = IntArray(maxTokens) { if (it < kept) 1 else 0 }
    return TokenizedPrompt(ids, mask)
  }

  companion object {
    private const val TAG = "PromptTokenizer"

    /** Cached per model directory: parsing a 10 MB vocabulary per generation is wasteful. */
    private val cache = HashMap<String, PromptTokenizer>()

    fun forModel(
      context: Context,
      repository: ModelRepository,
      spec: ModelSpec,
    ): PromptTokenizer {
      val dir = repository.fileFor(spec, spec.files.first()).parentFile
        ?: error("model directory missing for ${spec.id}")
      return forDirectory(dir)
    }

    @Synchronized
    fun forDirectory(dir: File): PromptTokenizer {
      cache[dir.absolutePath]?.let { return it }
      val file = File(dir, "tokenizer.json")
      require(file.isFile) {
        "tokenizer.json not found in ${dir.absolutePath}. It ships with the checkpoint — " +
          "download it alongside the weights; ids from a different vocabulary silently " +
          "encode a different prompt."
      }
      val config = File(dir, "tokenizer_config.json").takeIf { it.isFile }?.readText()
      return parse(file.readText(), config).also { cache[dir.absolutePath] = it }
    }

    /**
     * Reads the `model` section of a `tokenizer.json`, and the padding token out of
     * `tokenizer_config.json` when that is available.
     *
     * Uses `org.json` rather than a generated schema on purpose: the file's shape varies
     * between checkpoints (added-token tables, normalizer chains, optional fields), and only
     * the vocabulary, the merges and a handful of special ids are needed here.
     *
     * @param configJson contents of `tokenizer_config.json`. Optional, but it is the file
     *   that *states* the padding token; without it the pad id is picked from a list of
     *   conventional names, which is a guess that happens to be right for most checkpoints.
     */
    fun parse(json: String, configJson: String? = null): PromptTokenizer {
      val root = JSONObject(json)
      val model = root.optJSONObject("model")
        ?: error("tokenizer.json has no 'model' section")

      val vocabJson = model.optJSONObject("vocab")
        ?: error("tokenizer.json has no vocabulary")
      val vocab = HashMap<String, Int>(vocabJson.length())
      vocabJson.keys().forEach { key -> vocab[key] = vocabJson.getInt(key) }

      val mergesJson = model.optJSONArray("merges")
      val merges = buildList {
        if (mergesJson != null) {
          for (i in 0 until mergesJson.length()) {
            // Newer exports store merges as ["a", "b"] pairs; older ones as "a b" strings.
            when (val entry = mergesJson.get(i)) {
              is String -> add(entry)
              is org.json.JSONArray ->
                if (entry.length() == 2) add("${entry.getString(0)} ${entry.getString(1)}")
              else -> Unit
            }
          }
        }
      }

      // Added tokens are ids handed to the model, not text to segment. Passing them to the
      // tokenizer is what keeps a chat marker like <|im_start|> from being byte-pair-encoded
      // into ordinary punctuation, which would turn the conversation frame into content.
      val specials = HashMap<String, Int>()
      root.optJSONArray("added_tokens")?.let { added ->
        for (i in 0 until added.length()) {
          val t = added.getJSONObject(i)
          specials[t.getString("content")] = t.getInt("id")
        }
      }

      fun lookup(vararg names: String): Int? =
        names.firstNotNullOfOrNull { specials[it] ?: vocab[it] }

      // The checkpoint's own answer first; the conventional names only as a fallback.
      val declaredPad = configJson
        ?.let { runCatching { JSONObject(it).optString("pad_token") }.getOrNull() }
        ?.takeIf { it.isNotBlank() }
      val pad = declaredPad?.let { lookup(it) }
        ?: lookup("<|endoftext|>", "<pad>", "<|im_end|>", "</s>")
        ?: vocab.values.minOrNull()
        ?: error("tokenizer.json has neither a pad token nor a usable vocabulary")

      return PromptTokenizer(
        bpe = BpeTokenizer.from(vocab, merges, specialTokens = specials),
        padId = pad,
      )
    }
  }
}

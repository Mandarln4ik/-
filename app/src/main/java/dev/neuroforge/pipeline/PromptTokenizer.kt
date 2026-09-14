package dev.neuroforge.pipeline

import android.content.Context
import android.util.Log
import dev.neuroforge.core.BpeTokenizer
import dev.neuroforge.core.ModelSpec
import dev.neuroforge.models.ModelRepository
import java.io.File
import org.json.JSONObject

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
  private val bosId: Int?,
  private val eosId: Int?,
  private val padId: Int,
) {

  /**
   * Encodes [text] to exactly [maxTokens] ids.
   *
   * Text encoder graphs are converted at a fixed sequence length, so the result is padded
   * or truncated to fit. Truncation keeps room for the end-of-sequence marker rather than
   * cutting it off, since a missing EOS changes how the encoder reads the whole sequence.
   */
  fun encodePadded(text: String, maxTokens: Int): IntArray {
    val body = bpe.encode(text).toMutableList()
    val reserved = (if (bosId != null) 1 else 0) + (if (eosId != null) 1 else 0)
    val room = (maxTokens - reserved).coerceAtLeast(0)
    if (body.size > room) {
      Log.w(TAG, "prompt truncated from ${body.size} to $room tokens")
      while (body.size > room) body.removeAt(body.size - 1)
    }

    val ids = ArrayList<Int>(maxTokens)
    bosId?.let { ids.add(it) }
    ids.addAll(body)
    eosId?.let { ids.add(it) }
    while (ids.size < maxTokens) ids.add(padId)
    return ids.toIntArray()
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
      return parse(file.readText()).also { cache[dir.absolutePath] = it }
    }

    /**
     * Reads the `model` section of a `tokenizer.json`.
     *
     * Uses `org.json` rather than a generated schema on purpose: the file's shape varies
     * between checkpoints (added-token tables, normalizer chains, optional fields), and only
     * the vocabulary, the merges and a handful of special ids are needed here.
     */
    fun parse(json: String): PromptTokenizer {
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

      val specials = HashMap<String, Int>()
      root.optJSONArray("added_tokens")?.let { added ->
        for (i in 0 until added.length()) {
          val t = added.getJSONObject(i)
          specials[t.getString("content")] = t.getInt("id")
        }
      }

      fun lookup(vararg names: String): Int? =
        names.firstNotNullOfOrNull { specials[it] ?: vocab[it] }

      val pad = lookup("<|endoftext|>", "<pad>", "<|im_end|>", "</s>")
        ?: vocab.values.minOrNull()
        ?: error("tokenizer.json has neither a pad token nor a usable vocabulary")

      return PromptTokenizer(
        bpe = BpeTokenizer.from(vocab, merges),
        bosId = lookup("<|startoftext|>", "<s>", "<|im_start|>"),
        eosId = lookup("<|endoftext|>", "</s>", "<|im_end|>"),
        padId = pad,
      )
    }
  }
}

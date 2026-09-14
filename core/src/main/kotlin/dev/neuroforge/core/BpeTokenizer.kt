package dev.neuroforge.core

/**
 * Byte-level byte-pair-encoding tokenizer — the GPT-2 scheme that CLIP, Qwen and most
 * text encoders in diffusion pipelines use.
 *
 * Implemented on device rather than shipped as a precomputed table because the prompt is
 * arbitrary user text: there is no lookup table for "whatever they typed". The scheme has
 * three parts, and all three have to match the reference exactly or the text encoder
 * receives ids that mean something else entirely — which shows up as an image that ignores
 * half the prompt rather than as an error:
 *
 *  1. **Byte-to-unicode**: every one of the 256 byte values maps to a printable code point,
 *     so any input, including emoji and Cyrillic, becomes a string the merge table covers
 *     and nothing is ever out-of-vocabulary.
 *  2. **Pre-tokenization**: split on a fixed pattern so merges never cross word boundaries.
 *  3. **Merges**: repeatedly join the adjacent pair with the lowest rank.
 *
 * @param vocab token string to id.
 * @param merges ordered merge rules; position in the list *is* the rank, so earlier means
 *   higher priority.
 */
class BpeTokenizer(
  private val vocab: Map<String, Int>,
  merges: List<Pair<String, String>>,
  private val unkId: Int? = null,
) {

  private val ranks: Map<Pair<String, String>, Int> =
    merges.withIndex().associate { (i, pair) -> pair to i }

  /** Cache of word → merged symbols; prompts repeat words far more than they repeat prompts. */
  private val cache = HashMap<String, List<String>>()

  val vocabSize: Int get() = vocab.size

  /** Encodes [text] to token ids. */
  fun encode(text: String): IntArray {
    val ids = ArrayList<Int>()
    for (word in preTokenize(text)) {
      val mapped = word.toByteArray(Charsets.UTF_8)
        .joinToString("") { BYTE_TO_UNICODE[it.toInt() and 0xFF].toString() }
      for (symbol in bpe(mapped)) {
        val id = vocab[symbol] ?: unkId
        if (id != null) ids.add(id) else {
          // A symbol absent from the vocabulary means the merge table and the vocabulary
          // came from different checkpoints. Failing loudly beats emitting silent nonsense.
          error("token '$symbol' is not in the vocabulary — mismatched tokenizer files?")
        }
      }
    }
    return ids.toIntArray()
  }

  /** Applies the merge rules to one pre-tokenized word. */
  private fun bpe(word: String): List<String> {
    cache[word]?.let { return it }
    if (word.length <= 1) return listOf(word).also { cache[word] = it }

    var symbols = word.map { it.toString() }
    while (symbols.size > 1) {
      // Lowest rank among adjacent pairs wins; no candidate means no merge is possible.
      var bestRank = Int.MAX_VALUE
      var bestIndex = -1
      for (i in 0 until symbols.size - 1) {
        val rank = ranks[symbols[i] to symbols[i + 1]] ?: continue
        if (rank < bestRank) { bestRank = rank; bestIndex = i }
      }
      if (bestIndex < 0) break

      val merged = ArrayList<String>(symbols.size - 1)
      var i = 0
      while (i < symbols.size) {
        if (i == bestIndex) {
          merged.add(symbols[i] + symbols[i + 1])
          i += 2
        } else {
          merged.add(symbols[i])
          i++
        }
      }
      symbols = merged
    }
    cache[word] = symbols
    return symbols
  }

  private fun preTokenize(text: String): List<String> =
    PRE_TOKENIZE.findAll(text).map { it.value }.filter { it.isNotEmpty() }.toList()

  companion object {

    /**
     * The GPT-2 pre-tokenization pattern.
     *
     * Keeps a leading space attached to the word that follows it — which is why "cat" and
     * " cat" are different tokens in these vocabularies, and why dropping the space would
     * shift every id in a sentence.
     */
    private val PRE_TOKENIZE = Regex(
      """'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    /**
     * The reversible byte→code-point table from the GPT-2 reference implementation.
     *
     * Printable ASCII and Latin-1 ranges map to themselves; every other byte is lifted into
     * an unused block starting at U+0100, so the mapping stays one-to-one and no byte
     * sequence is unrepresentable.
     */
    val BYTE_TO_UNICODE: CharArray = buildByteToUnicode()

    private fun buildByteToUnicode(): CharArray {
      val direct = buildList {
        addAll('!'.code..'~'.code)
        addAll('¡'.code..'¬'.code)
        addAll('®'.code..'ÿ'.code)
      }.toMutableList()
      val mapped = direct.toMutableList()
      var next = 0
      for (b in 0 until 256) {
        if (b !in direct) {
          direct.add(b)
          mapped.add(256 + next)
          next++
        }
      }
      val table = CharArray(256)
      for (i in direct.indices) table[direct[i]] = mapped[i].toChar()
      return table
    }

    /**
     * Builds a tokenizer from the `vocab` and `merges` sections of a HuggingFace
     * `tokenizer.json`, already parsed into plain collections.
     *
     * @param mergeLines each entry is `"left right"`, as stored in the file.
     */
    fun from(vocab: Map<String, Int>, mergeLines: List<String>, unkId: Int? = null): BpeTokenizer {
      val merges = mergeLines.mapNotNull { line ->
        val parts = line.trim().split(' ')
        if (parts.size == 2) parts[0] to parts[1] else null
      }
      return BpeTokenizer(vocab, merges, unkId)
    }
  }
}

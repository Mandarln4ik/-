package dev.neuroforge.core

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BpeTokenizerTest {

  @Test
  fun `byte to unicode table is a bijection over all 256 bytes`() {
    // If two bytes mapped to the same code point the encoding would stop being reversible
    // and some inputs would silently tokenize as others.
    val table = BpeTokenizer.BYTE_TO_UNICODE
    assertEquals(256, table.size)
    assertEquals(256, table.toSet().size, "byte->unicode table is not injective")
  }

  @Test
  fun `printable ascii maps to itself`() {
    val table = BpeTokenizer.BYTE_TO_UNICODE
    for (c in '!'..'~') assertEquals(c, table[c.code])
  }

  @Test
  fun `control bytes are lifted out of the way`() {
    // Byte 0 is not printable, so it must be remapped into the U+0100 block rather than
    // emitted as a NUL that no vocabulary contains.
    assertTrue(BpeTokenizer.BYTE_TO_UNICODE[0].code >= 256)
    assertTrue(BpeTokenizer.BYTE_TO_UNICODE[' '.code].code >= 256)
  }

  @Test
  fun `added tokens are emitted whole instead of being merged`() {
    // A chat marker is an id the model was handed, not text to segment. Without literal
    // matching the vocabulary here would BPE "<|im_start|>" into ordinary punctuation and
    // the encoder would read the conversation frame as content.
    val vocab = mapOf("a" to 0, "b" to 1, "<" to 2, "|" to 3, ">" to 4, "im" to 5)
    val tok = BpeTokenizer(
      vocab,
      merges = emptyList(),
      specialTokens = mapOf("<|im_start|>" to 99),
    )
    assertContentEquals(intArrayOf(0, 99, 1), tok.encode("a<|im_start|>b"))
  }

  @Test
  fun `a longer added token wins over one that is its prefix`() {
    val tok = BpeTokenizer(
      mapOf("a" to 0),
      merges = emptyList(),
      specialTokens = mapOf("<|im" to 7, "<|im_start|>" to 8),
    )
    assertContentEquals(intArrayOf(8), tok.encode("<|im_start|>"))
    assertContentEquals(intArrayOf(7), tok.encode("<|im"))
  }

  @Test
  fun `text between added tokens is still byte-pair encoded as one run`() {
    // Splitting on the marker must not also split the surrounding words, or merges would
    // stop at an arbitrary boundary and the ids would drift from the reference.
    val vocab = mapOf("a" to 0, "b" to 1, "ab" to 2)
    val plain = BpeTokenizer(vocab, listOf("a" to "b"))
    val withSpecials = BpeTokenizer(vocab, listOf("a" to "b"), specialTokens = mapOf("<s>" to 9))
    assertContentEquals(plain.encode("ab"), withSpecials.encode("ab"))
    assertContentEquals(intArrayOf(9, 2), withSpecials.encode("<s>ab"))
  }

  @Test
  fun `merges are applied by rank not left to right`() {
    // "ab" ranks after "bc", so a greedy left-to-right pass would wrongly produce [ab, c].
    val vocab = mapOf("a" to 0, "b" to 1, "c" to 2, "ab" to 3, "bc" to 4, "abc" to 5)
    val tok = BpeTokenizer(vocab, listOf("b" to "c", "a" to "bc"))
    assertContentEquals(intArrayOf(5), tok.encode("abc"))
  }

  @Test
  fun `a leading space is part of the following token`() {
    val vocab = mapOf(
      "cat" to 1,
      BpeTokenizer.BYTE_TO_UNICODE[' '.code] + "cat" to 2,
    )
    val merges = listOf(
      BpeTokenizer.BYTE_TO_UNICODE[' '.code].toString() to "c",
      BpeTokenizer.BYTE_TO_UNICODE[' '.code] + "c" to "a",
      BpeTokenizer.BYTE_TO_UNICODE[' '.code] + "ca" to "t",
      "c" to "a",
      "ca" to "t",
    )
    val tok = BpeTokenizer(vocab, merges)
    // The two must not collide: " cat" and "cat" are different tokens in these vocabularies.
    assertContentEquals(intArrayOf(1), tok.encode("cat"))
    assertContentEquals(intArrayOf(2), tok.encode(" cat"))
  }

  @Test
  fun `non-latin text is representable`() {
    // Cyrillic goes through the byte mapping as multi-byte UTF-8; with no merges each byte
    // stays its own symbol, and none of them may be missing from the byte table.
    val bytes = "мир".toByteArray(Charsets.UTF_8)
    val vocab = bytes.withIndex().associate { (i, b) ->
      BpeTokenizer.BYTE_TO_UNICODE[b.toInt() and 0xFF].toString() to i
    }
    val tok = BpeTokenizer(vocab, emptyList())
    assertEquals(bytes.size, tok.encode("мир").size)
  }

  @Test
  fun `unknown symbols fail loudly rather than silently`() {
    val tok = BpeTokenizer(mapOf("a" to 0), emptyList())
    val e = runCatching { tok.encode("z") }
    assertTrue(e.isFailure, "a symbol outside the vocabulary must not be dropped in silence")
  }

  @Test
  fun `from parses merge lines`() {
    val tok = BpeTokenizer.from(
      vocab = mapOf("a" to 0, "b" to 1, "ab" to 2),
      mergeLines = listOf("a b", "", "  "),
    )
    assertContentEquals(intArrayOf(2), tok.encode("ab"))
  }

  @Test
  fun `empty input yields no tokens`() {
    assertContentEquals(intArrayOf(), BpeTokenizer(mapOf("a" to 0), emptyList()).encode(""))
  }
}

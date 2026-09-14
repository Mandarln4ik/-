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

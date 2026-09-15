package dev.neuroforge.core

import java.io.EOFException
import java.io.InputStream

/**
 * One entry's header in a tar stream.
 *
 * @param name path as stored, which for archives built on macOS includes AppleDouble
 *   sidecar files that must be skipped rather than mistaken for the payload.
 */
data class TarEntry(val name: String, val size: Long, val isFile: Boolean)

/**
 * Minimal tar reader.
 *
 * Present because some models are published only as a `.tar.gz` release asset, and the
 * alternative is telling the user to unpack it on a desktop — the exact "you need a PC"
 * step this app exists to remove. Tar is simple enough to read directly: fixed 512-byte
 * headers, each followed by a payload padded to a 512-byte boundary.
 *
 * Deliberately partial. No symlinks, no GNU long names, no sparse files; those cannot
 * appear in a single-model release asset, and quietly mishandling them would be worse
 * than not claiming support.
 */
object Tar {

  const val BLOCK = 512

  private const val TYPE_REGULAR = '0'.code.toByte()
  private const val TYPE_IMPLICIT_REGULAR: Byte = 0

  /**
   * Reads headers until [predicate] matches, then returns that entry.
   *
   * Returns null if the archive ends first. On a match the stream is left positioned at the
   * start of that entry's payload, so the caller reads exactly [TarEntry.size] bytes.
   */
  fun seek(input: InputStream, predicate: (TarEntry) -> Boolean): TarEntry? {
    val header = ByteArray(BLOCK)
    while (true) {
      if (!readFully(input, header)) return null
      // A zero-filled block marks the end of the archive.
      if (header.all { it == TYPE_IMPLICIT_REGULAR }) return null

      val entry = parseHeader(header)
      if (predicate(entry)) return entry
      skip(input, paddedSize(entry.size), entry.name)
    }
  }

  /** Size of an entry's payload including padding to the next block boundary. */
  fun paddedSize(size: Long): Long = ((size + BLOCK - 1) / BLOCK) * BLOCK

  /**
   * Parses a ustar header block.
   *
   * Name is at offset 0 (100 bytes), size at 124 (12 bytes, octal), type flag at 156.
   */
  fun parseHeader(header: ByteArray): TarEntry {
    require(header.size >= BLOCK) { "tar header must be $BLOCK bytes, was ${header.size}" }
    val name = cString(header, 0, 100)
    val size = octal(header, 124, 12)
    val type = header[156]
    return TarEntry(
      name = name,
      size = size,
      // '0' and an implicit zero byte both mean a regular file; anything else (directory,
      // link, device) carries no payload we would want.
      isFile = type == TYPE_REGULAR || type == TYPE_IMPLICIT_REGULAR,
    )
  }

  /**
   * True for the AppleDouble metadata files macOS's tar writes alongside real entries.
   *
   * They carry the same base name as the file they describe, so a naive "ends with
   * .tflite" match picks up a few hundred bytes of resource fork instead of the model —
   * which then fails deep inside the LiteRT loader rather than here.
   */
  fun isAppleDoubleSidecar(name: String): Boolean =
    name.substringAfterLast('/').startsWith("._")

  private fun skip(input: InputStream, bytes: Long, entryName: String) {
    var remaining = bytes
    val scratch = ByteArray(8192)
    while (remaining > 0) {
      val skipped = input.skip(remaining)
      if (skipped > 0) {
        remaining -= skipped
        continue
      }
      // skip() is allowed to return 0 at any time; read instead so this cannot spin.
      val chunk = minOf(remaining, scratch.size.toLong()).toInt()
      val n = input.read(scratch, 0, chunk)
      if (n < 0) throw EOFException("tar stream ended inside '$entryName'")
      remaining -= n
    }
  }

  private fun cString(bytes: ByteArray, offset: Int, length: Int): String {
    var end = offset
    val limit = offset + length
    while (end < limit && bytes[end].toInt() != 0) end++
    return String(bytes, offset, end - offset, Charsets.UTF_8).trim()
  }

  private fun octal(bytes: ByteArray, offset: Int, length: Int): Long {
    var value = 0L
    for (i in offset until offset + length) {
      val c = bytes[i].toInt()
      if (c == 0 || c == ' '.code) continue
      require(c in '0'.code..'7'.code) {
        "tar size field is not octal at byte $i (saw '${c.toChar()}')"
      }
      value = value * 8 + (c - '0'.code)
    }
    return value
  }

  /** Fills [into] completely; false when the stream ends before a single byte was read. */
  private fun readFully(input: InputStream, into: ByteArray): Boolean {
    var read = 0
    while (read < into.size) {
      val n = input.read(into, read, into.size - read)
      if (n < 0) {
        if (read == 0) return false
        throw EOFException("tar header truncated: got $read of ${into.size} bytes")
      }
      read += n
    }
    return true
  }
}

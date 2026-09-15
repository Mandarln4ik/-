package dev.neuroforge.core

import java.io.ByteArrayInputStream
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class TarTest {

  @Test
  fun `finds the requested member and leaves the stream on its payload`() {
    val archive = tar(
      "._model.tflite" to ByteArray(10) { 1 },
      "model.tflite" to ByteArray(600) { (it % 251).toByte() },
    )
    val input = ByteArrayInputStream(archive)

    val entry = Tar.seek(input) { it.isFile && it.name.endsWith(".tflite") && !Tar.isAppleDoubleSidecar(it.name) }

    assertEquals("model.tflite", entry?.name)
    assertEquals(600L, entry?.size)
    val payload = ByteArray(600)
    input.read(payload)
    assertContentEquals(ByteArray(600) { (it % 251).toByte() }, payload)
  }

  @Test
  fun `apple double sidecars are recognised`() {
    // This is the real trap: macOS tar writes a `._name` sibling with the same extension,
    // and picking it up hands the model loader a few hundred bytes of resource fork.
    assertTrue(Tar.isAppleDoubleSidecar("._esrgan_int8.tflite"))
    assertTrue(Tar.isAppleDoubleSidecar("./._esrgan_int8.tflite"))
    assertTrue(!Tar.isAppleDoubleSidecar("esrgan_int8.tflite"))
    assertTrue(!Tar.isAppleDoubleSidecar("dir/esrgan_int8.tflite"))
  }

  @Test
  fun `skips over entries that do not match, including large ones`() {
    val archive = tar(
      "a.bin" to ByteArray(2000) { 7 },
      "b.bin" to ByteArray(3) { 8 },
      "wanted.tflite" to byteArrayOf(42, 43),
    )
    val entry = Tar.seek(ByteArrayInputStream(archive)) { it.name == "wanted.tflite" }
    assertEquals(2L, entry?.size)
  }

  @Test
  fun `returns null when nothing matches`() {
    val archive = tar("a.bin" to ByteArray(5))
    assertNull(Tar.seek(ByteArrayInputStream(archive)) { it.name == "missing" })
  }

  @Test
  fun `returns null on an empty archive`() {
    assertNull(Tar.seek(ByteArrayInputStream(ByteArray(Tar.BLOCK * 2)) ) { true })
  }

  @Test
  fun `payload padding rounds up to the block size`() {
    assertEquals(0L, Tar.paddedSize(0))
    assertEquals(512L, Tar.paddedSize(1))
    assertEquals(512L, Tar.paddedSize(512))
    assertEquals(1024L, Tar.paddedSize(513))
  }

  @Test
  fun `directory entries are not reported as files`() {
    val header = header("some/dir/", 0, type = '5'.code.toByte())
    assertTrue(!Tar.parseHeader(header).isFile)
  }

  @Test
  fun `a non-octal size field is rejected`() {
    val header = header("x", 0)
    header[124] = '9'.code.toByte() // 9 is not an octal digit
    assertTrue(runCatching { Tar.parseHeader(header) }.isFailure)
  }

  // --- helpers ---------------------------------------------------------------------

  private fun tar(vararg entries: Pair<String, ByteArray>): ByteArray {
    val out = java.io.ByteArrayOutputStream()
    entries.forEach { (name, payload) ->
      out.write(header(name, payload.size.toLong()))
      out.write(payload)
      val pad = (Tar.paddedSize(payload.size.toLong()) - payload.size).toInt()
      out.write(ByteArray(pad))
    }
    out.write(ByteArray(Tar.BLOCK * 2)) // end-of-archive marker
    return out.toByteArray()
  }

  private fun header(name: String, size: Long, type: Byte = '0'.code.toByte()): ByteArray {
    val h = ByteArray(Tar.BLOCK)
    name.toByteArray().copyInto(h, 0)
    // Size is stored as 11 octal digits followed by a space, as ustar specifies.
    val octal = size.toString(8).padStart(11, '0')
    octal.toByteArray().copyInto(h, 124)
    h[135] = ' '.code.toByte()
    h[156] = type
    return h
  }
}

package dev.neuroforge.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class AttachmentTest {

  private fun attachment(name: String, kind: AttachmentKind = AttachmentKind.TEXT) =
    Attachment(
      id = name, fileName = name, kind = kind,
      localPath = "/data/$name", sizeBytes = 0, mimeType = "",
    )

  @Test
  fun `images are recognised by mime type and by extension`() {
    // The picker usually reports a real MIME type, but plenty of providers return
    // octet-stream for everything, which is why the extension is a fallback rather than a
    // nicety.
    assertEquals(AttachmentKind.IMAGE, attachmentKindFor("image/png", "shot.png"))
    assertEquals(AttachmentKind.IMAGE, attachmentKindFor("image/jpeg", "no-extension"))
    assertEquals(AttachmentKind.IMAGE, attachmentKindFor("application/octet-stream", "photo.HEIC"))
    assertEquals(AttachmentKind.IMAGE, attachmentKindFor(null, "diagram.webp"))
  }

  @Test
  fun `source files count as text even when their mime type does not say so`() {
    // application/octet-stream is what a file manager hands back for most source files, and
    // refusing those would make the feature useless for the thing it is most wanted for.
    assertEquals(AttachmentKind.TEXT, attachmentKindFor("text/plain", "notes.txt"))
    assertEquals(AttachmentKind.TEXT, attachmentKindFor("application/json", "data.json"))
    assertEquals(AttachmentKind.TEXT, attachmentKindFor("application/octet-stream", "Main.kt"))
    assertEquals(AttachmentKind.TEXT, attachmentKindFor(null, "build.gradle"))
    assertEquals(AttachmentKind.TEXT, attachmentKindFor(null, "server.log"))
  }

  @Test
  fun `what the app cannot use is refused at pick time, with a reason`() {
    assertNull(attachmentKindFor("application/pdf", "paper.pdf"))
    assertNull(attachmentKindFor("application/zip", "bundle.zip"))
    assertNull(attachmentKindFor("video/mp4", "clip.mp4"))
    assertTrue(unsupportedAttachmentReason("paper.pdf").contains("paper.pdf"))
  }

  @Test
  fun `a text file is fenced and named in the prompt`() {
    val folded = foldTextAttachments(
      "what does this do?", listOf(attachment("Main.kt")), tokenBudget = 1000,
    ) { "fun main() = Unit" }

    assertTrue(folded.text.contains("--- Main.kt ---"), folded.text)
    assertTrue(folded.text.contains("fun main() = Unit"), folded.text)
    assertTrue(folded.text.contains("--- end of Main.kt ---"), folded.text)
    // The question has to survive, and come after the file.
    assertTrue(folded.text.endsWith("what does this do?"), folded.text)
    assertTrue(!folded.wasTruncated)
  }

  @Test
  fun `a file too big for the budget is cut, and the cut is reported`() {
    // Silently truncating is the failure worth avoiding: the model answers about the half it
    // was given and nothing in the reply says the rest never arrived.
    val huge = "x".repeat(100_000)
    val folded = foldTextAttachments("summarise", listOf(attachment("big.log")), 100) { huge }

    assertTrue(folded.wasTruncated)
    assertEquals(listOf("big.log"), folded.dropped)
    assertTrue(folded.text.contains("truncated"), folded.text)
    assertTrue(folded.text.length < 10_000, "budget was not applied: ${folded.text.length}")
  }

  @Test
  fun `the budget is split between files so a small one still arrives whole`() {
    // First-come budgeting would let one large log eat everything and leave the small config
    // the question was actually about entirely out.
    val files = listOf(attachment("huge.log"), attachment("app.conf"))
    val folded = foldTextAttachments("why does it fail?", files, tokenBudget = 200) { file ->
      if (file.fileName == "huge.log") "x".repeat(100_000) else "port = 8080"
    }

    assertTrue(folded.text.contains("port = 8080"), "the small file was dropped")
    assertEquals(listOf("huge.log"), folded.dropped)
  }

  @Test
  fun `a file that cannot be read says so instead of vanishing`() {
    val folded = foldTextAttachments("read it", listOf(attachment("gone.txt")), 1000) {
      error("permission denied")
    }
    assertTrue(folded.text.contains("could not be read"), folded.text)
    assertEquals(listOf("gone.txt"), folded.dropped)
  }

  @Test
  fun `images are left alone by the text folding`() {
    // An image goes to the runtime as an image. Reading its bytes into the prompt would be
    // both useless and enormous.
    val folded = foldTextAttachments(
      "what is this?", listOf(attachment("cat.png", AttachmentKind.IMAGE)), 1000,
    ) { error("should not read an image") }
    assertEquals("what is this?", folded.text)
  }

  @Test
  fun `the budget leaves room for the model to answer`() {
    // A prompt that fills the window produces an empty reply, which is the least
    // informative failure there is.
    assertTrue(attachmentTokenBudget(contextTokens = 2048, usedTokens = 0) <= 1024)
    assertEquals(0, attachmentTokenBudget(contextTokens = 2048, usedTokens = 2048))
    assertEquals(0, attachmentTokenBudget(contextTokens = 2048, usedTokens = 9000))
    // And a huge window is not licence to paste a megabyte: prefill is linear.
    assertTrue(attachmentTokenBudget(contextTokens = 1_000_000, usedTokens = 0) <= 4096)
  }
}

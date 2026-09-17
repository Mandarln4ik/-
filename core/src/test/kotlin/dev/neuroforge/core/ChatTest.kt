package dev.neuroforge.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ChatTest {

  @Test
  fun `a chat titles itself from the first thing said`() {
    val chat = Chat("c1", ChatKind.TEXT, "m", Chat.DEFAULT_TITLE).copy(
      messages = listOf(
        ChatMessage(Speaker.USER, "explain flow matching"),
        ChatMessage(Speaker.MODEL, "It is a way to..."),
      )
    )
    assertEquals("explain flow matching", chat.derivedTitle())
  }

  @Test
  fun `a long opening line is truncated rather than wrapped across the list`() {
    val long = "a".repeat(120)
    val title = Chat("c", ChatKind.TEXT, "m", Chat.DEFAULT_TITLE)
      .copy(messages = listOf(ChatMessage(Speaker.USER, long)))
      .derivedTitle()
    assertTrue(title.length <= 40, "title was ${title.length} chars")
    assertTrue(title.endsWith("…"))
  }

  @Test
  fun `an explicit title is kept`() {
    val chat = Chat("c", ChatKind.IMAGE, "m", "Fox studies")
      .copy(messages = listOf(ChatMessage(Speaker.USER, "a red fox")))
    assertEquals("Fox studies", chat.derivedTitle())
  }

  @Test
  fun `an empty chat keeps the default title`() {
    assertEquals(Chat.DEFAULT_TITLE, Chat("c", ChatKind.TEXT, "m", Chat.DEFAULT_TITLE).derivedTitle())
  }

  @Test
  fun `a chat opened by the model alone still has no title of its own`() {
    val chat = Chat("c", ChatKind.TEXT, "m", Chat.DEFAULT_TITLE)
      .copy(messages = listOf(ChatMessage(Speaker.MODEL, "Hello")))
    assertEquals(Chat.DEFAULT_TITLE, chat.derivedTitle())
  }

  @Test
  fun `each kind selects the models built for it`() {
    assertTrue(modelsFor(ChatKind.TEXT).all { it.role == ModelRole.TEXT_CHAT })
    assertTrue(modelsFor(ChatKind.TEXT).isNotEmpty(), "no text model in the catalogue")
    assertTrue(modelsFor(ChatKind.IMAGE).all { it.role == ModelRole.DENOISER })
  }

  @Test
  fun `music is declared unsupported with a reason rather than offered and broken`() {
    // Showing a tab that cannot work is worse than saying why it cannot: the second is
    // information, the first is a dead end discovered after a download.
    assertTrue(!ChatKind.MUSIC.supported)
    assertTrue(!ChatKind.MUSIC.unsupportedReason.isNullOrBlank())
    assertTrue(modelsFor(ChatKind.MUSIC).isEmpty())
  }

  @Test
  fun `supported kinds have no excuse attached`() {
    assertTrue(ChatKind.TEXT.supported && ChatKind.TEXT.unsupportedReason == null)
    assertTrue(ChatKind.IMAGE.supported && ChatKind.IMAGE.unsupportedReason == null)
  }
}

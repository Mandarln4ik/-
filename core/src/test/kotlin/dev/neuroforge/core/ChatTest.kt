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

  @Test
  fun `thinking is separated from the answer`() {
    val (thinking, answer) = splitThinking("<think>weigh the options</think>The answer is 4.")
    assertEquals("weigh the options", thinking)
    assertEquals("The answer is 4.", answer)
  }

  @Test
  fun `a reply with no thinking block is left alone`() {
    val (thinking, answer) = splitThinking("Just an answer.")
    assertEquals(null, thinking)
    assertEquals("Just an answer.", answer)
  }

  @Test
  fun `an unclosed block streams as reasoning so the panel fills live`() {
    // Mid-stream state: the model has opened <think> and is still inside it. Waiting for
    // the close tag would leave the panel empty for most of the reply.
    val (thinking, answer) = splitThinking("<think>still working on")
    assertEquals("still working on", thinking)
    assertEquals("", answer)
  }

  @Test
  fun `an empty thinking block is treated as none`() {
    // Qwen3 with enable_thinking=false emits exactly this, and an empty grey panel is worse
    // than no panel.
    assertEquals(null, splitThinking("<think>\n\n</think>\n\nHello").first)
    assertEquals("Hello", splitThinking("<think>\n\n</think>\n\nHello").second)
  }

  @Test
  fun `chats list most recently used first`() {
    val old = Chat("a", ChatKind.TEXT, "m", "old", createdAt = 100)
    val recent = Chat(
      "b", ChatKind.TEXT, "m", "recent", createdAt = 50,
      messages = listOf(ChatMessage(Speaker.USER, "hi", timestamp = 900)),
    )
    assertEquals(listOf("b", "a"), listOf(old, recent).mostRecentFirst().map { it.id })
  }

  @Test
  fun `a chat with no messages sorts by when it was created`() {
    // Otherwise a brand new chat falls to the bottom at the exact moment it is opened.
    val fresh = Chat("new", ChatKind.TEXT, "m", "n", createdAt = 500)
    val stale = Chat(
      "old", ChatKind.TEXT, "m", "o", createdAt = 1,
      messages = listOf(ChatMessage(Speaker.USER, "x", timestamp = 200)),
    )
    assertEquals(500L, fresh.lastActivity)
    assertEquals(listOf("new", "old"), listOf(stale, fresh).mostRecentFirst().map { it.id })
  }

  @Test
  fun `token estimates account for non-latin text`() {
    // Google's samples use length/4, which is a Latin-text ratio. On Cyrillic that reads
    // about twice as optimistic as reality, and the context gauge is the thing it misleads.
    val latin = estimateTokens("aaaaaaaa")
    val cyrillic = estimateTokens("абвгдежз")
    assertEquals(2, latin)
    assertEquals(4, cyrillic)
    assertEquals(0, estimateTokens(""))
  }

  @Test
  fun `the context budget reports what is left`() {
    val budget = ContextBudget(used = 1800, limit = 2048)
    assertEquals(248, budget.remaining)
    assertTrue(budget.tight)
    assertTrue(!ContextBudget(100, 2048).tight)
    assertEquals(0, ContextBudget(used = 5000, limit = 2048).remaining)
  }

  @Test
  fun `a stalled NPU load is explained rather than echoed`() {
    val explained = explainLlmFailure(Accel.NPU, "Internal: TF_LITE_AUX payload absent")
    assertTrue(explained.contains("no ahead-of-time NPU payload"), explained)
    assertTrue(explained.contains("named for this SoC"), explained)
  }

  @Test
  fun `an unrecognised failure is passed through with its backend`() {
    assertEquals("GPU: something odd", explainLlmFailure(Accel.GPU, "something odd"))
    assertTrue(explainLlmFailure(Accel.CPU, null).contains("no reason given"))
  }

  @Test
  fun `the reply line names the accelerator and the cost`() {
    val stats = ReplyStats(ttftMillis = 420, decodeMillis = 2000, tokens = 60)
    val line = stats.line(Accel.NPU)
    assertTrue(line.contains("NPU"), line)
    assertTrue(line.contains("420ms to first token"), line)
    assertTrue(line.contains("≈60 tok"), line)
    assertTrue(line.contains("30.0 tok/s"), line)
  }

  @Test
  fun `an interrupted reply says so`() {
    val line = ReplyStats(ttftMillis = 100, stop = StopReason.CANCELLED).line(Accel.CPU)
    assertTrue(line.contains("stopped by you"), line)
  }
}

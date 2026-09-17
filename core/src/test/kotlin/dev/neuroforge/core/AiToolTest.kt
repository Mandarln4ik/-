package dev.neuroforge.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class AiToolTest {

  @Test
  fun `a tool with no parameters describes itself without a parameters key`() {
    // LiteRT-LM's spec makes `parameters` optional, and an empty object is not the same as
    // an absent one to every model that reads the description.
    val json = AiTools.CLOCK.describeJson()
    assertTrue(json.startsWith("""{"name":"current_time","description":"""), json)
    assertTrue(!json.contains("parameters"), json)
    assertTrue(json.endsWith("}"), json)
  }

  @Test
  fun `a tool with parameters produces the shape the runtime parses`() {
    val json = AiTools.CALCULATOR.describeJson()
    assertTrue(json.contains(""""parameters":{"type":"object","properties":{"""), json)
    assertTrue(json.contains(""""expression":{"type":"string","description":"""), json)
    assertTrue(json.contains(""""required":["expression"]"""), json)
  }

  @Test
  fun `quotes and newlines in a description do not produce broken json`() {
    // The description is prose written by whoever adds the tool. An apostrophe is fine; a
    // quotation mark or a newline would end the JSON string early and the runtime would
    // reject the whole tool with an error naming the tool rather than the quoting.
    val tool = AiTool(
      name = "quoter",
      description = "He said \"hello\"\nthen left\\.",
      parameters = listOf(
        AiToolParameter("text", AiToolParameterType.STRING, "a \"quoted\" value")
      ),
    )
    val json = tool.describeJson()
    assertTrue(json.contains("""\"hello\""""), json)
    assertTrue(json.contains("""\n"""), json)
    assertTrue(json.contains("""\\"""), json)
    // Balanced quotes is the property that matters: an odd count means the string escaped.
    assertEquals(0, json.count { it == '"' } % 2, json)
  }

  @Test
  fun `tool names are unique and lookup works`() {
    val names = AiTools.all.map { it.name }
    assertEquals(names.size, names.toSet().size, "two tools share a name")
    AiTools.all.forEach { assertEquals(it, AiTools.byName(it.name)) }
    assertEquals(null, AiTools.byName("nothing_like_this"))
  }

  @Test
  fun `arithmetic follows precedence rather than left to right`() {
    assertEquals(14.0, evaluateArithmetic("2 + 3 * 4"))
    assertEquals(20.0, evaluateArithmetic("(2 + 3) * 4"))
    assertEquals(-6.0, evaluateArithmetic("-2 * 3"))
    assertEquals(2.5, evaluateArithmetic("5 / 2"))
    assertEquals(1.0, evaluateArithmetic("7 % 3"))
    assertEquals(57.5, evaluateArithmetic("(17 * 3.5) - 2"))
  }

  @Test
  fun `bad input fails with a message rather than a wrong number`() {
    // The string comes from the model, so it will sometimes be nonsense. Answering "0" to
    // a malformed expression would be worse than saying it could not be evaluated.
    assertFailsWith<IllegalArgumentException> { evaluateArithmetic("2 +") }
    assertFailsWith<IllegalArgumentException> { evaluateArithmetic("(2 + 3") }
    assertFailsWith<IllegalArgumentException> { evaluateArithmetic("2 / 0") }
    assertFailsWith<IllegalArgumentException> { evaluateArithmetic("rm -rf /") }
    assertFailsWith<IllegalArgumentException> { evaluateArithmetic("") }
  }

  @Test
  fun `whole results lose the decimal point`() {
    // A model handed "4.0" repeats it into a sentence where a person expects "4".
    assertEquals("4", formatArithmetic(evaluateArithmetic("2 + 2")))
    assertEquals("2.5", formatArithmetic(evaluateArithmetic("5 / 2")))
    assertEquals("undefined", formatArithmetic(Double.NaN))
  }
}

package dev.neuroforge.core

/**
 * A tool the model can call, described the way LiteRT-LM wants it.
 *
 * Real function calling, not a prompt trick. LiteRT-LM takes an OpenAPI-shaped description
 * and, with automatic tool calling on, runs the function itself when the model asks and
 * feeds the result back into the turn. That only works where the runtime supports it —
 * which here means LiteRT-LM and nothing else — and only with a model trained to emit tool
 * calls at all.
 *
 * The description lives in `core` so the schema can be unit-tested without a device. The
 * *execution* does not: a tool that reads the clock or the SoC needs Android, so the app
 * supplies the body and this supplies the contract.
 */
data class AiTool(
  val name: String,
  val description: String,
  val parameters: List<AiToolParameter> = emptyList(),
  /** Off by default; a tool the user has not enabled is never offered to the model. */
  val enabledByDefault: Boolean = false,
) {
  /**
   * The JSON LiteRT-LM's `OpenApiTool.getToolDescriptionJsonString()` must return.
   *
   * Hand-built rather than serialized from a library: `core` has no dependencies, the
   * shape is three keys deep, and the format is pinned by someone else's parser — which
   * makes it exactly the kind of thing worth having a test read character by character.
   */
  fun describeJson(): String {
    val properties = parameters.joinToString(",") { p ->
      """"${p.name}":{"type":"${p.type.jsonType}","description":${quote(p.description)}}"""
    }
    val required = parameters.filter { it.required }.joinToString(",") { quote(it.name) }
    return buildString {
      append("""{"name":"$name","description":""").append(quote(description))
      if (parameters.isNotEmpty()) {
        append(""","parameters":{"type":"object","properties":{""").append(properties)
        append("""},"required":[""").append(required).append("]}")
      }
      append("}")
    }
  }
}

/** One argument of a tool. */
data class AiToolParameter(
  val name: String,
  val type: AiToolParameterType,
  val description: String,
  val required: Boolean = true,
)

enum class AiToolParameterType(val jsonType: String) {
  STRING("string"),
  NUMBER("number"),
  BOOLEAN("boolean"),
}

/**
 * Escapes a string into a JSON string literal, including the quotes.
 *
 * Needed because a tool description is prose written by whoever adds the tool, and an
 * apostrophe or a newline in it would otherwise produce JSON that the runtime's parser
 * rejects with an error naming the tool rather than the quoting.
 */
private fun quote(value: String): String = buildString {
  append('"')
  value.forEach { c ->
    when (c) {
      '"' -> append("\\\"")
      '\\' -> append("\\\\")
      '\n' -> append("\\n")
      '\r' -> append("\\r")
      '\t' -> append("\\t")
      else -> if (c < ' ') append("\\u%04x".format(c.code)) else append(c)
    }
  }
  append('"')
}

/**
 * The tools this app knows how to run.
 *
 * Small and local on purpose. A phone assistant that can read its own clock, say what
 * silicon it is running on and do arithmetic covers the three things a small model is
 * reliably bad at on its own; anything that reaches the network would undo the point of
 * running the model on the device in the first place.
 */
object AiTools {

  val CLOCK = AiTool(
    name = "current_time",
    description = "The current date and time on this device, with its time zone. Use it " +
      "for anything that depends on today's date; the model's own idea of the date is " +
      "whenever it was trained.",
    enabledByDefault = true,
  )

  val DEVICE = AiTool(
    name = "device_info",
    description = "This phone's model, chipset and which accelerator the app is currently " +
      "using for inference.",
    enabledByDefault = true,
  )

  val CALCULATOR = AiTool(
    name = "calculate",
    description = "Evaluates an arithmetic expression and returns the number. Supports " +
      "+ - * / %, parentheses and decimals. Use it instead of doing the arithmetic in " +
      "your head.",
    parameters = listOf(
      AiToolParameter(
        name = "expression",
        type = AiToolParameterType.STRING,
        description = "The expression to evaluate, for example (17 * 3.5) - 2",
      )
    ),
    enabledByDefault = true,
  )

  val all: List<AiTool> = listOf(CLOCK, DEVICE, CALCULATOR)

  fun byName(name: String): AiTool? = all.firstOrNull { it.name == name }
}

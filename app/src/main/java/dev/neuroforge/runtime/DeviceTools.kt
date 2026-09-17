package dev.neuroforge.runtime

import android.os.Build
import com.google.ai.edge.litertlm.OpenApiTool
import dev.neuroforge.core.Accel
import dev.neuroforge.core.AiTool
import dev.neuroforge.core.AiTools
import dev.neuroforge.core.evaluateArithmetic
import dev.neuroforge.core.formatArithmetic
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import org.json.JSONObject

/**
 * The bodies behind [AiTools], as LiteRT-LM wants them.
 *
 * `core` describes each tool so the schema can be tested off-device; this runs them, which
 * needs Android for two of the three. With `automaticToolCalling` on, the runtime calls
 * these itself when the model asks and feeds the result back into the same turn — the app
 * never sees the round trip, it just gets a better answer.
 *
 * Everything here is local and cheap. Nothing reaches the network: a model that runs on
 * the device precisely so nothing leaves it should not gain a tool that phones out.
 */
object DeviceTools {

  /** Wraps an [AiTool] and a body into the interface `tool(...)` takes. */
  private class Bound(
    private val spec: AiTool,
    private val body: (JSONObject) -> Any,
  ) : OpenApiTool {
    override fun getToolDescriptionJsonString(): String = spec.describeJson()

    override fun execute(paramsJsonString: String): String {
      // A tool that throws takes the whole turn down with it, and the model then has no
      // idea why. Returning the error as the tool's result lets it say what happened, or
      // try different arguments.
      val result = runCatching {
        val params = runCatching { JSONObject(paramsJsonString) }.getOrElse { JSONObject() }
        body(params)
      }.getOrElse { e -> mapOf("error" to (e.message ?: e.javaClass.simpleName)) }
      return JSONObject(result as? Map<*, *> ?: mapOf("result" to result)).toString()
    }
  }

  /**
   * Builds the enabled tools.
   *
   * @param accelerator what the loaded engine actually landed on, so `device_info` reports
   *   the truth rather than the policy that was requested.
   */
  fun bind(enabled: Set<String>, accelerator: Accel?): List<OpenApiTool> =
    AiTools.all.filter { it.name in enabled }.map { spec ->
      when (spec.name) {
        AiTools.CLOCK.name -> Bound(spec) { clock() }
        AiTools.DEVICE.name -> Bound(spec) { device(accelerator) }
        AiTools.CALCULATOR.name -> Bound(spec) { params -> calculate(params) }
        else -> error("no implementation for tool ${spec.name}")
      }
    }

  private fun clock(): Map<String, Any> {
    val now = Date()
    val zone = TimeZone.getDefault()
    return mapOf(
      // ISO 8601 because every model has seen far more of it than of any locale format.
      "iso8601" to SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssXXX", Locale.US).format(now),
      "readable" to SimpleDateFormat("EEEE d MMMM yyyy, HH:mm", Locale.getDefault()).format(now),
      "timezone" to zone.id,
    )
  }

  private fun device(accelerator: Accel?): Map<String, Any> = mapOf(
    "device" to "${Build.MANUFACTURER} ${Build.MODEL}",
    "chipset" to "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}",
    "android" to Build.VERSION.RELEASE,
    // The accelerator that accepted the model, not the one the settings asked for.
    "inference_on" to (accelerator?.name ?: "not loaded"),
  )

  private fun calculate(params: JSONObject): Map<String, Any> {
    val expression = params.optString("expression").takeIf { it.isNotBlank() }
      ?: return mapOf("error" to "no expression was given")
    return mapOf(
      "expression" to expression,
      "result" to formatArithmetic(evaluateArithmetic(expression)),
    )
  }
}

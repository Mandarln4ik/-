package dev.neuroforge.runtime

import android.content.Context
import android.os.Build
import android.util.Log
import dev.neuroforge.core.Accel
import java.io.File

/**
 * What the app could establish about this device's inference hardware.
 *
 * @param npuAvailable whether LiteRT's own compatibility check accepts this SoC. This is the
 *   authoritative *static* answer; it does not promise a given graph will compile.
 * @param npuRuntimePresent whether vendor runtime shared objects were found in the process.
 *   Without them the NPU path cannot load however compatible the silicon is.
 * @param verifiedAccelerators accelerators that actually loaded and ran a graph. Empty until
 *   a probe run happens — this is the only claim here that is proof rather than inference.
 */
data class DeviceReport(
  val socManufacturer: String,
  val socModel: String,
  val boardPlatform: String,
  val deviceModel: String,
  val androidRelease: String,
  val abis: List<String>,
  val npuVendor: NpuVendor,
  val npuAvailable: Boolean,
  val npuRuntimePresent: Boolean,
  val apuNodes: List<String>,
  val verifiedAccelerators: List<Accel> = emptyList(),
  val notes: List<String> = emptyList(),
) {
  /** One-line verdict for the top of the Device screen. */
  fun verdict(): String = when {
    Accel.NPU in verifiedAccelerators -> "NPU active — graphs are compiling and running on the APU"
    npuAvailable && !npuRuntimePresent ->
      "NPU supported by this SoC, but the vendor runtime libraries are missing"
    npuAvailable -> "NPU supported — not yet proven on a real graph"
    npuVendor == NpuVendor.MEDIATEK ->
      "MediaTek SoC, but LiteRT does not list this model as supported yet"
    else -> "No NPU path available — running on GPU/CPU"
  }
}

enum class NpuVendor { MEDIATEK, QUALCOMM, GOOGLE_TENSOR, SAMSUNG, UNKNOWN }

/**
 * Works out what inference hardware is really here.
 *
 * Deliberately separates three different kinds of claim, because conflating them is how
 * "NPU accelerated" ends up meaning nothing:
 *
 *  1. **What the SoC is** — from [Build], cheap and reliable.
 *  2. **What LiteRT says it supports** — [com.google.ai.edge.litert.NpuCompatibilityChecker],
 *     a static allow-list of SoC models, which lags new silicon.
 *  3. **What actually ran** — filled in by [AcceleratorSelector] after a graph really
 *     compiled and executed. Only this one is evidence.
 *
 * The third matters most on a very new chip: the allow-list can say "no" on hardware that
 * works, and a driver can say "yes" and then fail to compile the graph you care about.
 */
object AcceleratorProbe {

  private const val TAG = "AcceleratorProbe"

  /**
   * MediaTek APU device nodes.
   *
   * Advisory only: app processes usually cannot stat these under SELinux, so absence proves
   * nothing. Presence is a strong positive signal, which is why it is still worth reporting.
   */
  private val APU_NODES = listOf("/dev/apusys", "/dev/mdla", "/dev/vpu", "/dev/apu")

  /** Vendor runtime shared objects LiteRT dlopen's to reach an APU. */
  private val MEDIATEK_RUNTIME_LIBS = listOf(
    "libneuron_adapter.so",
    "libneuron_adapter_mgvi.so",
    "libneuronusdk_adapter.mtk.so",
  )

  fun probe(context: Context): DeviceReport {
    val socManufacturer = Build.SOC_MANUFACTURER ?: "unknown"
    val socModel = Build.SOC_MODEL ?: "unknown"
    val boardPlatform = systemProperty("ro.board.platform") ?: "unknown"

    val vendor = classifyVendor(socManufacturer, socModel, boardPlatform)
    val notes = mutableListOf<String>()

    val npuAvailable = try {
      checkLiteRtSupport(vendor)
    } catch (t: Throwable) {
      // A missing class here means the LiteRT NPU surface is not on the classpath at all.
      notes += "LiteRT NPU compatibility check unavailable: ${t.javaClass.simpleName}"
      Log.w(TAG, "NpuCompatibilityChecker unavailable", t)
      false
    }

    val runtimePresent = mediatekRuntimePresent(context)
    if (vendor == NpuVendor.MEDIATEK && !runtimePresent) {
      notes += "No NeuroPilot runtime .so found — see docs/NPU_RUNTIME.md"
    }

    val nodes = APU_NODES.filter { runCatching { File(it).exists() }.getOrDefault(false) }
    if (vendor == NpuVendor.MEDIATEK && nodes.isEmpty()) {
      notes += "APU device nodes not visible to this process (normal under SELinux; not a failure)"
    }

    return DeviceReport(
      socManufacturer = socManufacturer,
      socModel = socModel,
      boardPlatform = boardPlatform,
      deviceModel = "${Build.MANUFACTURER} ${Build.MODEL}",
      androidRelease = "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})",
      abis = Build.SUPPORTED_ABIS.toList(),
      npuVendor = vendor,
      npuAvailable = npuAvailable,
      npuRuntimePresent = runtimePresent,
      apuNodes = nodes,
      notes = notes,
    )
  }

  private fun checkLiteRtSupport(vendor: NpuVendor): Boolean = when (vendor) {
    NpuVendor.MEDIATEK -> com.google.ai.edge.litert.NpuCompatibilityChecker.Mediatek.isDeviceSupported()
    NpuVendor.QUALCOMM -> com.google.ai.edge.litert.NpuCompatibilityChecker.Qualcomm.isDeviceSupported()
    NpuVendor.GOOGLE_TENSOR -> com.google.ai.edge.litert.NpuCompatibilityChecker.GoogleTensor.isDeviceSupported()
    else -> false
  }

  private fun classifyVendor(manufacturer: String, model: String, platform: String): NpuVendor {
    val haystack = "$manufacturer $model $platform".lowercase()
    return when {
      "mediatek" in haystack || haystack.contains(Regex("\\bmt6\\d{3}")) -> NpuVendor.MEDIATEK
      "qualcomm" in haystack || haystack.startsWith("sm") || "kona" in haystack -> NpuVendor.QUALCOMM
      "google" in haystack || "tensor" in haystack || "gs10" in haystack -> NpuVendor.GOOGLE_TENSOR
      "samsung" in haystack || "exynos" in haystack -> NpuVendor.SAMSUNG
      else -> NpuVendor.UNKNOWN
    }
  }

  /**
   * Looks for a NeuroPilot runtime in the places it can legitimately live: bundled into the
   * APK's native library directory, or installed on the system partition by the vendor.
   */
  private fun mediatekRuntimePresent(context: Context): Boolean {
    val dirs = buildList {
      add(File(context.applicationInfo.nativeLibraryDir))
      add(File("/vendor/lib64"))
      add(File("/system/lib64"))
      add(File("/odm/lib64"))
    }
    return dirs.any { dir ->
      MEDIATEK_RUNTIME_LIBS.any { lib ->
        runCatching { File(dir, lib).exists() }.getOrDefault(false)
      }
    }
  }

  /** Reads a build system property; these are not part of the public SDK, hence reflection. */
  private fun systemProperty(key: String): String? = runCatching {
    val cls = Class.forName("android.os.SystemProperties")
    val get = cls.getMethod("get", String::class.java)
    (get.invoke(null, key) as? String)?.takeIf { it.isNotBlank() }
  }.getOrNull()
}

package dev.neuroforge.core

/**
 * A runtime that can generate text on this phone.
 *
 * Three of them, and the differences that matter are not performance — they are which model
 * files each one reads and which silicon each one can actually reach. Those two facts decide
 * everything else, so they are properties here rather than something to discover by trying.
 *
 * The honest summary, from inspecting the shipped artifacts rather than from documentation:
 *
 * | backend    | file        | reaches the APU |
 * |------------|-------------|-----------------|
 * | LiteRT-LM  | `.litertlm` | **yes**         |
 * | ExecuTorch | `.pte`      | no              |
 * | llama.cpp  | `.gguf`     | no              |
 *
 * Only the first has an NPU path that a publicly buildable app can use. The other two are
 * here because they read formats LiteRT cannot, which is a different kind of useful: far
 * more models are published as GGUF than as anything else.
 */
enum class TextBackend(
  val label: String,
  /** File extension this runtime loads. Also what the catalogue filters on. */
  val extension: String,
  /** Accelerators this runtime can place a *generative* model on, best first. */
  val accelerators: List<Accel>,
) {
  /**
   * Google's LiteRT-LM.
   *
   * The only one of the three with `Backend.NPU` for a generative model, and the reason
   * this app started here. `litert-community` publishes bundles compiled per SoC —
   * `..._mt6991.litertlm` and friends — so on a matching phone the graph is already built
   * for that APU rather than compiled at load.
   */
  LITERT_LM("LiteRT-LM", ".litertlm", listOf(Accel.NPU, Accel.GPU, Accel.CPU)),

  /**
   * PyTorch ExecuTorch.
   *
   * **CPU only, and that is a property of the published artifact, not a limitation of the
   * framework.** ExecuTorch does have vendor backends, including one for MediaTek's
   * NeuroPilot, but they are compiled in when the runtime is built. The public
   * `org.pytorch:executorch-android` AAR carries XNNPACK and nothing else — checked by
   * unpacking it and looking at the symbols in `jni/arm64-v8a`. Reaching the APU through
   * ExecuTorch would mean building the AAR against the NeuroPilot SDK, which is not
   * something an app can do for you.
   *
   * Worth having anyway: `.pte` is what PyTorch exports to, so models land here first.
   */
  EXECUTORCH("ExecuTorch", ".pte", listOf(Accel.CPU)),

  /**
   * llama.cpp.
   *
   * The reason to want it is GGUF: more open models are published in that format than in
   * every other on-device format combined, and neither of the other two runtimes reads it.
   *
   * CPU here. llama.cpp does have GPU backends on Android — OpenCL and Vulkan — but they
   * are build-time options with their own compatibility caveats, and claiming a GPU path
   * that has not been run on this device would be the same mistake as claiming an NPU one.
   */
  LLAMA_CPP("llama.cpp", ".gguf", listOf(Accel.CPU));

  /** True when this runtime can put a generative model on the NPU. */
  val reachesNpu: Boolean get() = Accel.NPU in accelerators

  /** One line for the settings list, stating the trade rather than selling the option. */
  val summary: String
    get() = when (this) {
      LITERT_LM ->
        "Reads .litertlm. The only backend here that reaches the APU, and on this chip it " +
          "picks a bundle compiled for it."
      EXECUTORCH ->
        "Reads .pte, what PyTorch exports. CPU only: the public runtime is built with " +
          "XNNPACK and no vendor backend."
      LLAMA_CPP ->
        "Reads .gguf, the format most open models ship in. CPU."
    }

  companion object {
    /** Restores a stored choice, falling back to the one that can reach the NPU. */
    fun fromStoredName(name: String?): TextBackend =
      entries.firstOrNull { it.name == name } ?: LITERT_LM
  }
}

/**
 * Which backend should load [fileName].
 *
 * Decided by extension rather than by the setting, because a `.gguf` cannot be opened by
 * LiteRT-LM whatever the user picked — and failing with "wrong backend selected" is a worse
 * answer than simply using the one that reads the file. The setting decides which models
 * are *offered*; this decides what happens once one is chosen.
 */
fun backendFor(fileName: String): TextBackend? =
  TextBackend.entries.firstOrNull { fileName.endsWith(it.extension, ignoreCase = true) }

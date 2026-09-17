package dev.neuroforge.runtime

/**
 * The `llama_jni` shared object, one method per JNI entry point in `src/main/cpp/llama_jni.cpp`.
 *
 * Declared as a Kotlin `object` rather than a class of statics on purpose: an `external fun`
 * on an object compiles to an instance method, which is the `(JNIEnv*, jobject)` shape the
 * C++ side is written against. Adding `@JvmStatic` here would change the signature and the
 * calls would fail to bind at run time rather than at compile time.
 *
 * Every function that takes a pointer accepts 0 and does nothing with it, so a failed load
 * cannot turn into a native crash on the way out.
 */
internal object LlamaCppNative {

  /** Mirrors the `STOP_*` constants in llama_jni.cpp. */
  object Stop {
    const val COMPLETE = 0
    const val LIMIT = 1
    const val CANCELLED = 2
    const val CONTEXT_FULL = 3
    const val ERROR = 4
  }

  /** Receives each token's text; returning false stops the native decode loop. */
  fun interface TokenSink {
    fun onToken(token: String): Boolean
  }

  /**
   * Whether libllama_jni.so is present and loadable.
   *
   * Held as a [Result] rather than thrown from an initialiser because there is one entirely
   * ordinary way for this to fail: the APK ships arm64-v8a only, so on an x86_64 emulator
   * the library is simply not there. That has to be a message, not a crash on first launch.
   */
  val available: Result<Unit> by lazy {
    runCatching {
      System.loadLibrary("llama_jni")
      init()
    }
  }

  external fun init()

  /** @return a model pointer, or 0 if the file could not be read as GGUF. */
  external fun loadModel(path: String): Long

  external fun newContext(model: Long, contextTokens: Int, threads: Int): Long

  external fun contextSize(context: Long): Int

  /** The context length the model was trained at; 0 if unknown. */
  external fun trainedContextSize(model: Long): Int

  /** Architecture, parameter count and quantisation, in llama.cpp's own words. */
  external fun describeModel(model: Long): String

  /** Drops the KV cache. */
  external fun clearMemory(context: Long)

  /**
   * Applies the model's own chat template.
   *
   * @return the formatted transcript, or null if llama.cpp does not recognise the
   *   template in the file's metadata.
   */
  external fun formatChat(
    model: Long,
    roles: Array<String>,
    contents: Array<String>,
    addAssistant: Boolean,
  ): String?

  /** @return one of the [Stop] codes. */
  external fun generate(
    model: Long,
    context: Long,
    prompt: String,
    addSpecial: Boolean,
    maxTokens: Int,
    temperature: Float,
    topK: Int,
    topP: Float,
    seed: Int,
    sink: TokenSink,
  ): Int

  external fun freeContext(context: Long)

  external fun freeModel(model: Long)
}

# LiteRT reaches its native layer through JNI, so its classes cannot be renamed or stripped.
-keep class com.google.ai.edge.litert.** { *; }
-keep class org.tensorflow.lite.** { *; }
-dontwarn com.google.ai.edge.litert.**

# Vendor NPU accelerator providers are looked up reflectively at runtime.
-keepclassmembers class * implements com.google.ai.edge.litert.** { *; }

-keepattributes *Annotation*, Signature, InnerClasses

# The llama.cpp bridge. R8's default rules keep `native` methods and the classes that
# declare them, so LlamaCppNative itself survives — but TokenSink.onToken is an ordinary
# interface method that the C++ side looks up by name with GetMethodID, and nothing in the
# defaults protects a name that only native code knows about. Renaming it would break
# every GGUF chat in release builds and in release builds only, which is the worst place
# for a bug of this kind to live.
-keep class dev.neuroforge.runtime.LlamaCppNative { *; }
-keep interface dev.neuroforge.runtime.LlamaCppNative$TokenSink { *; }
-keepclassmembers class * implements dev.neuroforge.runtime.LlamaCppNative$TokenSink {
    boolean onToken(java.lang.String);
}

# LiteRT-LM reaches liblitertlm_jni.so the same way LiteRT does. ExecuTorch ships its own
# consumer rules in the AAR and needs nothing here; this one does not, as far as its
# artifact shows.
-keep class com.google.ai.edge.litertlm.** { *; }
-dontwarn com.google.ai.edge.litertlm.**

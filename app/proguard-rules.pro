# LiteRT reaches its native layer through JNI, so its classes cannot be renamed or stripped.
-keep class com.google.ai.edge.litert.** { *; }
-keep class org.tensorflow.lite.** { *; }
-dontwarn com.google.ai.edge.litert.**

# Vendor NPU accelerator providers are looked up reflectively at runtime.
-keepclassmembers class * implements com.google.ai.edge.litert.** { *; }

-keepattributes *Annotation*, Signature, InnerClasses

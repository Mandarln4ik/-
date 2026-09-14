plugins {
  kotlin("jvm") version "2.3.0"
}

group = "dev.neuroforge"
version = "1.0"

// Java 11 bytecode: this module is consumed by the Android app, whose compileOptions target
// the same level. Deliberately no `jvmToolchain` — that would make the build download a JDK.
java {
  sourceCompatibility = JavaVersion.VERSION_11
  targetCompatibility = JavaVersion.VERSION_11
}

kotlin {
  compilerOptions {
    jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11)
  }
}

dependencies {
  testImplementation(kotlin("test"))
}

tasks.test {
  useJUnitPlatform()
  testLogging { events("failed", "skipped") }
}

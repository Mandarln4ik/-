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

// Prints every catalogued download as `id<TAB>file<TAB>url<TAB>repo`, for CI to probe.
tasks.register<JavaExec>("dumpCatalog") {
  group = "verification"
  mainClass.set("dev.neuroforge.core.CatalogDumpKt")
  classpath = sourceSets["main"].runtimeClasspath
}

tasks.test {
  useJUnitPlatform()
  testLogging { events("failed", "skipped") }
}

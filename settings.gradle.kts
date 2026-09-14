pluginManagement {
  repositories {
    google {
      content {
        includeGroupByRegex("com\\.android.*")
        includeGroupByRegex("com\\.google.*")
        includeGroupByRegex("androidx.*")
      }
    }
    mavenCentral()
    gradlePluginPortal()
  }
}

dependencyResolutionManagement {
  repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
  repositories {
    google()
    mavenCentral()
  }
}

rootProject.name = "NeuroForge"

// `core` is a standalone, Android-free Kotlin/JVM build: the scheduler math, the
// tiling geometry, the tokenizer and the model catalogue live there so they can be
// unit-tested on a plain JVM without an Android SDK (`cd core && ./gradlew test`).
// Gradle substitutes `dev.neuroforge:core` in :app with this included build.
includeBuild("core")

include(":app")

plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.kotlin.android)
  alias(libs.plugins.kotlin.compose)
  alias(libs.plugins.kotlin.serialization)
}

android {
  namespace = "dev.neuroforge"
  compileSdk = 36

  defaultConfig {
    applicationId = "dev.neuroforge"
    // 31 is where Build.SOC_MODEL arrives, and it is the floor the LiteRT NPU samples use.
    minSdk = 31
    targetSdk = 36
    versionCode = 7
    versionName = "1.4.0"

    // The NPU accelerators ship arm64 only; shipping other ABIs would only grow the APK
    // with builds that can never reach the hardware this app exists for.
    ndk { abiFilters.add("arm64-v8a") }

    // llama.cpp is compiled from source into libllama_jni.so. See src/main/cpp/CMakeLists.txt
    // for what each flag buys; -PllamaArmArch overrides the instruction-set baseline for a
    // build that only has to run on one known phone.
    externalNativeBuild {
      cmake {
        arguments += "-DCMAKE_BUILD_TYPE=Release"
        arguments += "-DGGML_CPU_ARM_ARCH=" +
          (project.findProperty("llamaArmArch") as String? ?: "armv8.2-a+dotprod+fp16")
      }
    }
  }

  // Release signing is opt-in via environment variables, so CI can sign when the repository
  // has the secrets and everyone else can still build. A missing keystore leaves the release
  // APK unsigned rather than failing the build or silently substituting the debug key.
  val keystorePath: String? = System.getenv("KEYSTORE_PATH")?.takeIf { it.isNotBlank() }
  signingConfigs {
    if (keystorePath != null) {
      create("release") {
        storeFile = file(keystorePath)
        storePassword = System.getenv("KEYSTORE_PASSWORD")
        keyAlias = System.getenv("KEY_ALIAS")
        keyPassword = System.getenv("KEY_PASSWORD")
      }
    }
  }

  buildTypes {
    release {
      isMinifyEnabled = true
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
      signingConfig = signingConfigs.findByName("release")
    }
    debug {
      isMinifyEnabled = false
    }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
  }

  externalNativeBuild {
    cmake {
      path = file("src/main/cpp/CMakeLists.txt")
      version = "3.22.1"
    }
  }

  // Pinned rather than left to AGP's default so CI installs exactly what the build wants;
  // an unpinned NDK is a build that breaks when the runner image rolls.
  ndkVersion = "27.2.12479018"

  buildFeatures { compose = true }

  packaging {
    resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" }
    // Vendor NPU runtimes are dlopen'd by the LiteRT native layer, which needs them
    // extracted on disk rather than mapped out of the APK.
    jniLibs { useLegacyPackaging = true }
  }

  lint {
    // The build must not fail on style; a red lint on a sample is noise, a red compile is not.
    abortOnError = false
  }
}

kotlin {
  compilerOptions {
    jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11)
  }
}

dependencies {
  // Android-free maths and planning, unit-tested on the JVM. Substituted from the
  // `core` included build by settings.gradle.kts.
  implementation("dev.neuroforge:core:1.0")

  implementation(libs.litert)
  // Text generation runs on a different runtime: it owns the KV cache and the decode
  // loop, and is the only one here with an NPU backend for a generative model.
  implementation(libs.litertlm)
  // A second text runtime, for the models that are published as .pte rather than
  // .litertlm. CPU only; see TextBackend for why that is the artifact and not the choice.
  implementation(libs.executorch)

  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.activity.compose)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.compose)
  implementation(platform(libs.androidx.compose.bom))
  implementation(libs.androidx.ui)
  implementation(libs.androidx.ui.graphics)
  implementation(libs.androidx.ui.tooling.preview)
  implementation(libs.androidx.material3)
  implementation(libs.androidx.material.icons.extended)

  implementation(libs.kotlinx.coroutines.android)
  implementation(libs.kotlinx.serialization.json)
  implementation(libs.okhttp)

  debugImplementation(libs.androidx.ui.tooling)

  testImplementation(libs.junit)
  androidTestImplementation(libs.androidx.junit)
  androidTestImplementation(libs.androidx.espresso.core)
}

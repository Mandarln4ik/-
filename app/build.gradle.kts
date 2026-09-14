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
    versionCode = 1
    versionName = "1.0"

    // The NPU accelerators ship arm64 only; shipping other ABIs would only grow the APK
    // with builds that can never reach the hardware this app exists for.
    ndk { abiFilters.add("arm64-v8a") }
  }

  buildTypes {
    release {
      isMinifyEnabled = true
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
    debug {
      isMinifyEnabled = false
    }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
  }

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
  implementation(libs.androidx.datastore.preferences)

  implementation(libs.kotlinx.coroutines.android)
  implementation(libs.kotlinx.serialization.json)
  implementation(libs.okhttp)

  debugImplementation(libs.androidx.ui.tooling)

  testImplementation(libs.junit)
  androidTestImplementation(libs.androidx.junit)
  androidTestImplementation(libs.androidx.espresso.core)
}

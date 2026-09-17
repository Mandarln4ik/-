# Builds libllama_jni.so for the host, so the JNI layer can be run against a real model.
#
# The point is the class of bug that only appears at run time: a C++ exception crossing the
# JNI boundary, or a GGML_ASSERT firing. Both abort the process. On a phone that is a silent
# disappearance with no Java stack trace, and the emulator in CI cannot help -- it is x86_64
# and the APK ships arm64 only, so the library is not even there. A host build is the only
# place this code can actually be executed by CI.
#
# It reuses src/main/cpp/CMakeLists.txt rather than copying it, so the flags under test are
# the flags that ship.

cmake_minimum_required(VERSION 3.22.1)
project("llama_jni_host" CXX)

find_package(JNI REQUIRED)

add_subdirectory(../../main/cpp llama_jni_build)
target_include_directories(llama_jni PRIVATE ${JNI_INCLUDE_DIRS})

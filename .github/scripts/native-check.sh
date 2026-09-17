#!/usr/bin/env bash
# Runs app/src/test/native against a host build of the JNI layer.
#
# Executes the same llama_jni.cpp the APK ships, compiled for this machine instead of for
# arm64, and drives it from the same Kotlin binding the app uses. See NativeCheck.java for
# what is actually asserted and why none of it is provable at compile time.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
work="${root}/build/native-check"
mkdir -p "${work}"

echo "--- configuring host build"
cmake -S "${root}/app/src/test/native" -B "${work}/cmake" -DCMAKE_BUILD_TYPE=Release

echo "--- building libllama_jni for the host"
cmake --build "${work}/cmake" --target llama_jni -j "$(nproc)"

lib="$(find "${work}/cmake" -name 'libllama_jni.so' -o -name 'libllama_jni.dylib' | head -1)"
test -n "${lib}" || { echo "no library was produced"; exit 1; }

echo "--- writing a tiny model"
# gguf-py comes from the llama.cpp source CMake just fetched, so it always matches the
# version the library was built from.
gguf_py="$(find "${work}/cmake" -type d -path '*llama-src/gguf-py' | head -1)"
test -n "${gguf_py}" || { echo "could not find gguf-py in the fetched source"; exit 1; }
python3 -c 'import numpy' 2>/dev/null ||
  python3 -m pip install --quiet --disable-pip-version-check numpy
python3 "${root}/app/src/test/native/tiny_gguf.py" "${work}/tiny.gguf" "${gguf_py}"

echo "--- compiling the driver"
# The Kotlin binding is taken from the app's own build output rather than recompiled, so the
# test binds to exactly the class the APK contains.
classes="${root}/app/build/tmp/kotlin-classes/debug"
test -f "${classes}/dev/neuroforge/runtime/LlamaCppNative.class" || {
  echo "run :app:compileDebugKotlin first - ${classes} has no binding to test"
  exit 1
}
# The version-numbered jar specifically: 'kotlin-stdlib-*' also matches
# kotlin-stdlib-common, which does not carry the classes the driver needs.
stdlib="$(find ~/.gradle/caches -regex '.*/kotlin-stdlib-[0-9][^/]*\.jar' | head -1)"
test -n "${stdlib}" || { echo "no kotlin-stdlib jar in the Gradle cache"; exit 1; }
javac -nowarn -cp "${classes}:${stdlib}" -d "${work}" "${root}/app/src/test/native/NativeCheck.java"

echo "--- running"
java -cp "${work}:${classes}:${stdlib}" NativeCheck "${lib}" "${work}/tiny.gguf"

#!/usr/bin/env bash
# Checks the minified release APK still contains what the native layer looks up by name.
#
# None of this is visible in a debug build: R8 only runs on release. The llama.cpp bridge
# is reached from C++ by string — the class name is baked into libllama_jni.so as
# Java_dev_neuroforge_runtime_LlamaCppNative_*, and the callback is found with
# GetMethodID(cls, "onToken", ...). If R8 renames either, every GGUF chat fails with an
# UnsatisfiedLinkError or a null method id in the build people actually install.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"

apk="$(find "${root}/app/build/outputs/apk/release" -name '*.apk' | head -1)"
test -n "${apk}" || { echo "no release APK found"; exit 1; }
echo "checking $(basename "${apk}")"

SDK="${ANDROID_HOME:-$ANDROID_SDK_ROOT}"
dexdump="$(find "$SDK/build-tools" -name dexdump -type f | sort | tail -1)"
test -n "${dexdump}" || { echo "no dexdump in the build tools"; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
unzip -q -o "${apk}" 'classes*.dex' 'lib/*/libllama_jni.so' -d "${work}"

# The native library has to be in there at all.
test -n "$(find "${work}/lib" -name 'libllama_jni.so' 2>/dev/null)" || {
  echo "FAIL: libllama_jni.so is not packaged in the release APK"
  exit 1
}
echo "ok: libllama_jni.so is packaged"

symbols="$("${dexdump}" -d "${work}"/classes*.dex 2>/dev/null || true)"

fail=0
check() {
  if printf '%s' "${symbols}" | grep -q -- "$1"; then
    echo "ok: $2"
  else
    echo "FAIL: $2 — R8 renamed or removed it; see app/proguard-rules.pro"
    fail=1
  fi
}

# The JNI entry points resolve by fully-qualified class name.
check "Ldev/neuroforge/runtime/LlamaCppNative;" "LlamaCppNative kept its name"
# GetMethodID looks this one up as a plain string.
check "onToken" "TokenSink.onToken kept its name"

exit "${fail}"

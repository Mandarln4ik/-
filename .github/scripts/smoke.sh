#!/usr/bin/env bash
# Installs the debug APK on a running emulator and checks the app is still alive after it
# has had time to finish starting.
#
# A file rather than an inline `script:` block because android-emulator-runner executes the
# script one line at a time, each through its own `sh -c`. Under that, `set -o pipefail` is
# an invalid option and a multi-line `if ... fi` is a syntax error at the first line - both
# of which failed the step *before the APK was installed* and looked exactly like the launch
# crash this check exists to catch. One line in the workflow, one real shell here.
set -u

APK=app/build/outputs/apk/debug/app-debug.apk

adb install -r "$APK"
adb logcat -c
adb shell am start -W -n dev.neuroforge/.MainActivity

# Long enough for the first frame, the view model's init and the device probe to run. The
# v1.1.0 crash was in a property initialiser, so it happened during this window rather than
# at the `am start` above, which reports success either way.
sleep 12

# Two independent signals, because either alone lies: a crash dialog can be dismissed
# leaving the process alive, and a process can die silently.
if adb logcat -d | grep -E "FATAL EXCEPTION|AndroidRuntime.*dev\.neuroforge"; then
  echo "::error::The app crashed on launch"
  adb logcat -d | grep -A 40 "FATAL EXCEPTION" | head -60
  exit 1
fi

if [ -z "$(adb shell pidof dev.neuroforge)" ]; then
  echo "::error::The app is not running 12s after launch"
  adb logcat -d | tail -80
  exit 1
fi

echo "app is alive"

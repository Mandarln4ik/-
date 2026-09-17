#!/data/data/com.termux/files/usr/bin/bash
#
# One-time setup for converting models to LiteRT ON THE PHONE.
#
# Run this inside Termux. It builds a Debian aarch64 userland alongside Android and
# installs the conversion toolchain into it.
#
# Why this works, when "run the converter inside the app" does not: an Android app is a
# sandboxed JVM process with no Python and no way to host one. Termux is a different
# thing - a full Linux userland running on the same ARM64 cores - and every piece of the
# toolchain publishes an aarch64 Linux build:
#
#   torch                manylinux_2_28_aarch64
#   tensorflow           manylinux_2_27_aarch64
#   ai-edge-litert       manylinux_2_27_aarch64
#   ai-edge-torch        pure Python
#   ai-edge-quantizer    pure Python
#
# So there is no emulation here and no x86 anywhere: the phone's own cores run the
# converter natively.
#
# Honest status: the wheels above were verified to exist and resolve for aarch64. A full
# conversion has NOT been run on a phone by the author, who has no ARM64 device. The two
# things most likely to bite are memory and time - see docs/ON_DEVICE_CONVERSION.md.

set -euo pipefail

DISTRO=debian
VENV=/root/neuroforge-venv

say() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }

say "Checking this is Termux"
if [ ! -d /data/data/com.termux ]; then
  echo "This script expects Termux. Install it from F-Droid or GitHub —" >&2
  echo "the Google Play build is abandoned and cannot install packages." >&2
  exit 1
fi

say "Installing proot-distro"
pkg update -y
pkg install -y proot-distro

say "Installing a Debian aarch64 userland (skipped if present)"
proot-distro install "$DISTRO" || echo "already installed, continuing"

say "Installing the conversion toolchain inside Debian"
proot-distro login "$DISTRO" -- bash -eu <<SETUP
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y python3 python3-pip python3-venv build-essential

  python3 -m venv "$VENV"
  . "$VENV/bin/activate"
  pip install --upgrade pip wheel

  # CPU-only torch: the phone's GPU is not usable by torch here, and the CUDA builds
  # have no aarch64 variant anyway.
  pip install --index-url https://download.pytorch.org/whl/cpu torch
  pip install ai-edge-torch ai-edge-quantizer ai-edge-litert huggingface_hub

  python3 - <<'CHECK'
import platform, torch, ai_edge_torch
print("machine:", platform.machine())
print("torch:", torch.__version__)
print("ai_edge_torch:", ai_edge_torch.__version__)
CHECK
SETUP

say "Done. Convert a model with:"
echo "  proot-distro login $DISTRO -- bash -c '. $VENV/bin/activate && python3 /root/convert.py --help'"
echo
echo "Copy tools/android/convert.py into the Debian root first:"
echo "  proot-distro login $DISTRO -- bash -c 'cat > /root/convert.py' < convert.py"

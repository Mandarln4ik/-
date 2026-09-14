# Vendor NPU runtime libraries go here

This directory is deliberately empty in the repository.

LiteRT talks to a MediaTek APU through NeuroPilot runtime shared objects
(`libneuron_adapter.so` and friends) that MediaTek and Google distribute separately from
the Maven artifact — they are not redistributable here. Drop the `arm64-v8a` set for
**LiteRT 2.1.6** into this directory and rebuild.

The app is built to work without them: `AcceleratorProbe` reports the NPU as unavailable,
every stage falls back to GPU or CPU, and the Device screen says so in as many words rather
than silently running slower than you expect.

See `docs/NPU_RUNTIME.md` for where to obtain them.

#!/usr/bin/env python3
"""Ahead-of-time compile a LiteRT model for a phone's NPU.

This is the step that turns a `.tflite` into a graph the APU runs *without* compiling it
first. Without it the vendor runtime either compiles the graph just-in-time on the device —
which is the tens of seconds a first NPU load spends — or gives up and the model quietly
lands on the CPU.

    python3 compile_for_npu.py model.tflite --soc MT6991 --out ./npu

## What this needs, and where it can run

    pip install ai-edge-litert ai-edge-litert-sdk-mediatek

**x86_64 Linux only.** Verified by inspecting both wheels: the compiler plugin
`vendors/mediatek/compiler/libLiteRtCompilerPlugin_MediaTek.so` is present in
`ai_edge_litert-*-manylinux_2_27_x86_64.whl` and absent from the `aarch64` one. So this
cannot run in Termux on the phone, however much of the rest of the toolchain does — the
piece that does the compiling is not built for ARM. A desktop, a CI runner or a Colab
notebook, then; the output is a file you copy to the phone.

`ai-edge-litert-sdk-mediatek` supplies `libneuron_adapter.so` from MediaTek's NeuroPilot
SDK. Without it the plugin loads, recognises the SoC, and then fails with "Failed to load
NeuronAdapter shared library" — which reads like a broken install rather than a missing
package, so it is worth naming.

## Not every model compiles

Compilation is per-operator: the plugin partitions the graph and takes what it can. A model
whose operators it rejects comes back with zero partitions, and that is a real answer about
that model rather than an error to work around. Measured here on a MT6991 target:

  * `selfie_multiclass_256x256.tflite` — compiles, 2.5 s.
  * `esrgan_int8.tflite` — rejected, 0 of 910 operators. Its PReLU layers have an output
    type the MediaTek compiler considers incompatible with their inputs. That is why this
    app's upscaler runs through the NPU's just-in-time path instead.

So the script prints the partition result rather than only succeeding or failing, because
"which operators did it take" is the question worth answering.
"""

import argparse
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", help="path to the .tflite to compile")
    parser.add_argument(
        "--soc",
        default="MT6991",
        help="SoC model, e.g. MT6991 (Dimensity 9400), MT6989, MT6993, or SM8750 for "
        "Qualcomm. Read it off the phone: the app's Device tab shows Build.SOC_MODEL.",
    )
    parser.add_argument("--out", default="./npu", help="output directory")
    args = parser.parse_args()

    model = pathlib.Path(args.model)
    if not model.is_file():
        print(f"no such model: {model}", file=sys.stderr)
        return 2

    try:
        from ai_edge_litert.aot import aot_compile as aot
        from ai_edge_litert.aot.vendors.mediatek import target as mtk
        from ai_edge_litert.aot.vendors.qualcomm import target as qnn
    except ImportError as e:
        print(f"install the toolchain first: pip install ai-edge-litert  ({e})", file=sys.stderr)
        return 2

    soc = args.soc.upper()
    if soc in mtk.SocModel.__members__:
        target = mtk.Target(mtk.SocModel[soc])
        _warn_if_sdk_missing()
    elif soc in qnn.SocModel.__members__:
        target = qnn.Target(qnn.SocModel[soc])
    else:
        mtk_socs = ", ".join(sorted(mtk.SocModel.__members__))
        print(f"unsupported SoC {soc!r}. MediaTek targets: {mtk_socs}", file=sys.stderr)
        return 2

    print(f"compiling {model.name} for {soc}...")
    result = aot.aot_compile(str(model), keep_going=True, target=[target])

    for backend, error in result.failed_backends:
        print(f"  {backend.id()}: {error}", file=sys.stderr)

    if not result.models_with_backend:
        print(
            "\nNothing compiled. The error file named above lists the operators the plugin\n"
            "refused; a model it cannot partition is a fact about that model, and it will\n"
            "still run through the just-in-time path or on CPU.",
            file=sys.stderr,
        )
        return 1

    out = pathlib.Path(args.out)
    result.export(output_dir=str(out), model_name=model.stem)
    for produced in sorted(out.glob(f"{model.stem}*")):
        print(f"  wrote {produced}  ({produced.stat().st_size / 1e6:.1f} MB)")
    print("\nCopy it to the phone and point the app at it from the Models tab.")
    return 0


def _warn_if_sdk_missing() -> None:
    try:
        import ai_edge_litert_sdk_mediatek  # noqa: F401
    except ImportError:
        print(
            "warning: ai-edge-litert-sdk-mediatek is not installed. The plugin will load,\n"
            "         recognise the SoC, and then fail on the NeuronAdapter library.\n"
            "         pip install ai-edge-litert-sdk-mediatek",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Convert a PyTorch model to a LiteRT graph NeuroForge can run — on the phone.

Runs inside the Debian aarch64 userland that `setup.sh` builds under Termux, so the
phone's own ARM cores do the work and no desktop is involved.

Two things this deliberately does not hide:

*Fixed input shapes are mandatory.* An NPU compiles one shape and replays it; a dynamic
axis forces a recompile per call, or a refusal. `--shape` is therefore required and is
the single most common reason a converted model then fails to load on the APU.

*Quantisation needs calibration data that looks like the real input.* Full-integer
quantisation records activation ranges by running the model over samples. Feed it noise
and the ranges are wrong in a way that shows up as banding and seams, not as an error.
`--calib` should point at images resembling what the model will actually see.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_shape(text: str) -> tuple[int, ...]:
    try:
        dims = tuple(int(p) for p in text.replace("x", ",").split(",") if p)
    except ValueError as exc:  # pragma: no cover - argument parsing
        raise SystemExit(f"--shape must be integers, got {text!r}") from exc
    if not dims:
        raise SystemExit("--shape must not be empty")
    if any(d <= 0 for d in dims):
        raise SystemExit(
            f"--shape must be fully static, got {dims}. A dynamic axis cannot be "
            "compiled once and replayed, which is the whole point of targeting an NPU."
        )
    return dims


def report_memory() -> None:
    """Prints available RAM, because this is where phone conversions actually die."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            info = {
                line.split(":")[0]: int(line.split()[1])
                for line in fh
                if line.split(":")[0] in ("MemTotal", "MemAvailable")
            }
        total = info.get("MemTotal", 0) / 1024 / 1024
        avail = info.get("MemAvailable", 0) / 1024 / 1024
        log(f"memory: {avail:.1f} GiB available of {total:.1f} GiB")
        if avail < 6:
            log(
                "WARNING: under 6 GiB free. Conversion holds the model and its "
                "intermediate representation at once; close other apps first."
            )
    except OSError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="Hugging Face repo id to fetch the checkpoint from")
    ap.add_argument("--checkpoint", help="Local .pt/.pth/.safetensors checkpoint instead of --repo")
    ap.add_argument("--entry", required=True,
                    help="import path to a callable returning an nn.Module in eval mode, "
                         "e.g. 'my_pkg.models:build_upscaler'")
    ap.add_argument("--shape", required=True,
                    help="static input shape, e.g. 1,3,256,256 — no dynamic axes")
    ap.add_argument("--out", required=True, help="output .tflite path")
    ap.add_argument("--quantize", choices=("none", "dynamic_int8", "full_int8"), default="none")
    ap.add_argument("--calib", help="directory of representative inputs, required for full_int8")
    args = ap.parse_args()

    if args.quantize == "full_int8" and not args.calib:
        raise SystemExit(
            "full_int8 needs --calib. Calibration records activation ranges by running "
            "the model over real samples; without them the ranges are guesses and the "
            "output degrades silently rather than failing."
        )

    # Validated before torch is imported: importing the toolchain on a phone takes a
    # noticeable while, and learning about a typo in --shape after that wait is the kind
    # of small cruelty that makes a tool unpleasant to use.
    shape = parse_shape(args.shape)

    if args.calib and not Path(args.calib).is_dir():
        raise SystemExit(f"--calib is not a directory: {args.calib}")

    report_memory()

    import torch  # imported late so --help works without the toolchain
    import ai_edge_torch

    log(f"torch {torch.__version__} on {torch.__config__.show().splitlines()[0][:60]}")

    if args.repo:
        from huggingface_hub import snapshot_download
        log(f"fetching {args.repo}")
        local = snapshot_download(args.repo)
        log(f"fetched to {local}")
        os.environ.setdefault("NEUROFORGE_CHECKPOINT", local)
    elif args.checkpoint:
        os.environ.setdefault("NEUROFORGE_CHECKPOINT", args.checkpoint)

    module_name, _, factory_name = args.entry.partition(":")
    if not factory_name:
        raise SystemExit("--entry must be 'module:callable'")
    log(f"importing {module_name}.{factory_name}")
    sys.path.insert(0, str(Path.cwd()))
    module = __import__(module_name, fromlist=[factory_name])
    model = getattr(module, factory_name)()
    model.eval()

    log(f"tracing with a static input of {shape}")
    sample = (torch.randn(*shape),)

    started = time.time()
    edge = ai_edge_torch.convert(model, sample)
    log(f"converted in {time.time() - started:.0f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.quantize == "none":
        edge.export(str(out))
    else:
        interim = out.with_suffix(".fp32.tflite")
        edge.export(str(interim))
        log(f"quantising ({args.quantize})")
        quantize(interim, out, args.quantize, args.calib)
        interim.unlink(missing_ok=True)

    size_mb = out.stat().st_size / 1024 / 1024
    log(f"wrote {out} ({size_mb:.1f} MiB)")
    log("Copy it into NeuroForge's model directory, or import it from the Models tab:")
    log("  /sdcard/Android/data/dev.neuroforge/files/models/<model-id>/")
    return 0


def quantize(src: Path, dst: Path, mode: str, calib: str | None) -> None:
    """Applies the requested recipe with ai-edge-quantizer."""
    from ai_edge_quantizer import quantizer

    qt = quantizer.Quantizer(str(src))
    recipe = "dynamic_wi8_afp32" if mode == "dynamic_int8" else "static_wi8_ai8"
    qt.load_quantization_recipe(recipe)

    if mode == "full_int8":
        import numpy as np
        from PIL import Image

        samples = sorted(Path(calib).glob("*"))
        if not samples:
            raise SystemExit(f"no calibration samples found in {calib}")
        log(f"calibrating on {len(samples)} samples")
        data = []
        for path in samples:
            arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            data.append({"input": arr[None, ...]})
        qt.calibrate(data)

    result = qt.quantize()
    result.export_model(str(dst))


if __name__ == "__main__":
    raise SystemExit(main())

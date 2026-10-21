"""Edge-deployment benchmark: *what does it actually cost to run each detector, and what do
the tricks (ONNX, INT8, CoreML, threads, input size) really buy?*

Every configuration is measured in its own subprocess (`emot.export.edge_worker`) on the
same 150 MOT17 frames (6 sequences x 25, 16:9 only) and scored with COCO-style AP against
ground truth -- so each row has latency, memory **and** accuracy, not a proxy.

Experiment groups (all resumable, rows keyed by id):
  main        6 backbones x {torch cpu, torch mps, onnx fp32, int8 dynamic, int8 static,
              int8 static + fp32 head, onnx coreml}
  threads     onnx fp32 / int8 static at 1, 2, 4 and all threads (constrained-CPU proxy)
  resolution  onnx fp32 at long side 320 ... 1280 for the three yolo11 sizes
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import onnxruntime
import torch

from emot.detection.yolo import BACKBONES
from emot.export.onnx_export import (
    export_onnx,
    quantize_onnx_dynamic,
    quantize_onnx_static,
    rect_shape_16x9,
)

logger = logging.getLogger(__name__)
EVAL_SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
FRAMES_PER_SEQ = 25
CALIB_FRAMES_PER_SEQ = 11


def sample_frames(data_root: Path, per_seq: int, offset: int) -> list[tuple[str, int]]:
    frames = []
    for seq in EVAL_SEQUENCES:
        n = len(list((data_root / seq / "img1").glob("*.jpg")))
        step = (n - 2 * offset) // per_seq
        frames += [(seq, offset + i * step) for i in range(per_seq)]
    return frames


def disjoint_from(
    frames: list[tuple[str, int]], reference: list[tuple[str, int]], margin: int = 3
) -> list[tuple[str, int]]:
    """Shifts frames forward until none lies within `margin` frames of a reference frame, so
    calibration never sees the evaluation frames *or their near-duplicates*."""
    out = []
    for seq, frame in frames:
        taken = [f for s, f in reference if s == seq]
        while any(abs(frame - f) <= margin for f in taken):
            frame += margin + 1
        out.append((seq, frame))
    return out


def machine_info() -> dict:
    return {
        "cpu": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        or platform.processor(),
        "cores": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "ort_providers": onnxruntime.get_available_providers(),
    }


def run_worker(spec: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(spec, fh)
        spec_path = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "emot.export.edge_worker", spec_path],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        os.unlink(spec_path)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--export-dir", type=Path, default=Path("exports"))
    parser.add_argument("--output", type=Path, default=Path("results/edge_benchmark.json"))
    parser.add_argument("--groups", nargs="+", default=["main", "threads", "resolution"])
    parser.add_argument("--backbones", nargs="+", default=[b.name for b in BACKBONES])
    args = parser.parse_args()

    weights = {b.name: b.weights for b in BACKBONES}
    eval_frames = sample_frames(args.data_root, FRAMES_PER_SEQ, offset=7)
    calib_frames = disjoint_from(
        sample_frames(args.data_root, CALIB_FRAMES_PER_SEQ, offset=19), eval_frames
    )
    calib_paths = [args.data_root / s / "img1" / f"{f:06d}.jpg" for s, f in calib_frames]
    assert not set(eval_frames) & set(calib_frames), "calibration must not overlap evaluation"

    doc = json.loads(args.output.read_text()) if args.output.exists() else {"rows": []}
    doc["machine"] = machine_info()
    doc["protocol"] = {
        "eval_frames": len(eval_frames),
        "eval_sequences": EVAL_SEQUENCES,
        "calibration_frames": len(calib_frames),
        "warmup_frames": 5,
        "conf_floor": 0.1,
        "input": "16:9 rectangular, stride-32 padded",
    }
    done = {r["id"] for r in doc["rows"]}

    def measure(row_id: str, group: str, backbone: str, label: dict, spec: dict) -> None:
        if row_id in done:
            return
        logger.info("measuring %s", row_id)
        spec = {**spec, "data_root": str(args.data_root), "frames": eval_frames}
        result = run_worker(spec)
        doc["rows"].append({"id": row_id, "group": group, "backbone": backbone, **label, **result})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, indent=2))
        logger.info(
            "  %.1f ms  AP50=%.3f  RSS=%.0f MB",
            result["mean_ms"],
            result["ap50"],
            result["peak_rss_mb"],
        )

    def onnx_assets(backbone: str, long_side: int, static: bool = True):
        shape = rect_shape_16x9(long_side)
        fp32 = export_onnx(weights[backbone], args.export_dir / backbone, shape)
        out = {"fp32": fp32}
        if static:
            out["int8dyn"] = quantize_onnx_dynamic(fp32)
            out["int8st"] = quantize_onnx_static(fp32, calib_paths, shape)
            out["int8sh"] = quantize_onnx_static(fp32, calib_paths, shape, keep_head_fp32=True)
        return out

    if "main" in args.groups:
        for bb in args.backbones:
            assets = onnx_assets(bb, 640)
            measure(
                f"main/{bb}/torch_cpu",
                "main",
                bb,
                {
                    "variant": "torch_cpu",
                    "size_mb": Path(f"weights/{weights[bb]}").stat().st_size / 2**20,
                },
                {"kind": "torch", "weights": weights[bb], "device": "cpu"},
            )
            measure(
                f"main/{bb}/torch_mps",
                "main",
                bb,
                {
                    "variant": "torch_mps",
                    "size_mb": Path(f"weights/{weights[bb]}").stat().st_size / 2**20,
                },
                {"kind": "torch", "weights": weights[bb], "device": "mps"},
            )
            for key, name in [
                ("fp32", "onnx_fp32"),
                ("int8dyn", "onnx_int8_dynamic"),
                ("int8st", "onnx_int8_static"),
                ("int8sh", "onnx_int8_static_fp32head"),
            ]:
                measure(
                    f"main/{bb}/{name}",
                    "main",
                    bb,
                    {"variant": name, "size_mb": assets[key].stat().st_size / 2**20},
                    {"kind": "onnx", "model": str(assets[key])},
                )
            measure(
                f"main/{bb}/onnx_coreml",
                "main",
                bb,
                {"variant": "onnx_coreml", "size_mb": assets["fp32"].stat().st_size / 2**20},
                {"kind": "onnx", "model": str(assets["fp32"]), "provider": "coreml"},
            )

    if "threads" in args.groups:
        for bb in [b for b in ("yolo11n", "yolo11s") if b in args.backbones]:
            assets = onnx_assets(bb, 640)
            for key, name in [("fp32", "onnx_fp32"), ("int8st", "onnx_int8_static")]:
                for threads in (1, 2, 4, 0):
                    measure(
                        f"threads/{bb}/{name}/{threads}",
                        "threads",
                        bb,
                        {
                            "variant": name,
                            "threads": threads or os.cpu_count(),
                            "size_mb": assets[key].stat().st_size / 2**20,
                        },
                        {"kind": "onnx", "model": str(assets[key]), "threads": threads},
                    )

    if "resolution" in args.groups:
        for bb in [b for b in ("yolo11n", "yolo11s", "yolo11m") if b in args.backbones]:
            for long_side in (320, 480, 640, 960, 1280):
                assets = onnx_assets(bb, long_side, static=False)
                measure(
                    f"resolution/{bb}/{long_side}",
                    "resolution",
                    bb,
                    {
                        "variant": "onnx_fp32",
                        "long_side": long_side,
                        "size_mb": assets["fp32"].stat().st_size / 2**20,
                    },
                    {"kind": "onnx", "model": str(assets["fp32"])},
                )


if __name__ == "__main__":
    main()

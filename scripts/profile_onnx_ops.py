"""Per-operator time of ONNX Runtime on CPU for FP32 vs INT8 (dynamic / static) -- the evidence
behind *why* quantisation does or doesn't speed a convolutional detector up.

Writes `profile` into results/edge_benchmark.json: for each variant, total ms per frame by
operator type (after ORT's graph optimisation, so fused/QLinear operators show by their real name).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from emot.detection.onnx_detector import letterbox

BACKBONE = "yolo11s"
SHAPE = (384, 640)
N_FRAMES = 20
VARIANTS = {
    "fp32": "fp32",
    "int8_dynamic": "int8dyn",
    "int8_static": "int8st",
}


def profile(model: Path, frames: list[np.ndarray], threads: int) -> list[dict]:
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for frame in frames:
        canvas, _ = letterbox(frame, SHAPE)
        x = np.ascontiguousarray(
            canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
        )
        sess.run(None, {name: x})
    path = sess.end_profiling()
    events = json.loads(Path(path).read_text())
    Path(path).unlink()
    totals: dict[str, float] = defaultdict(float)
    for e in events:
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
            totals[e["args"]["op_name"]] += e["dur"] / 1000.0  # us -> ms
    grand = sum(totals.values())
    rows = [{"op": k, "ms": v / len(frames), "share": v / grand} for k, v in totals.items()]
    return sorted(rows, key=lambda r: -r["ms"])[:8]


def main() -> None:
    root = Path("data/mot17/MOT17-04/img1")
    frames = [cv2.imread(str(root / f"{i:06d}.jpg")) for i in range(100, 100 + 10 * N_FRAMES, 10)]
    out = {}
    for threads in (1, 0):
        for label, suffix in VARIANTS.items():
            model = Path(f"exports/{BACKBONE}/{BACKBONE}_{SHAPE[0]}x{SHAPE[1]}.{suffix}.onnx")
            key = f"{label}/{threads or 'all'}"
            out[key] = profile(model, frames, threads)
            total = sum(r["ms"] for r in out[key])
            print(
                key,
                f"{total:.1f} ms in top-8 ops:",
                ", ".join(f"{r['op']} {r['ms']:.1f}" for r in out[key][:4]),
            )
    path = Path("results/edge_benchmark.json")
    doc = json.loads(path.read_text())
    doc["profile"] = {"backbone": BACKBONE, "frames": N_FRAMES, "variants": out}
    path.write_text(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()

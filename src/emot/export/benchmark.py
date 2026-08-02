"""Latency/size/detection-quality benchmark across PyTorch FP32, ONNX
FP32, and ONNX INT8 (dynamically quantized) for a single YOLO backbone.

Ultralytics' `YOLO` class dispatches to the right backend purely from the
weights file extension (`.pt` -> PyTorch, `.onnx` -> onnxruntime), so the
same `PersonDetector` wrapper used for the main tracking pipeline is
reused here unchanged rather than writing a second, ONNX-specific
inference path -- one real code path per concern, not two that could
silently drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emot.detection.yolo import Detection, PersonDetector
from emot.tracking.iou import iou_matrix


@dataclass(frozen=True)
class ExportBenchmarkResult:
    variant: str  # "pytorch_fp32" | "onnx_fp32" | "onnx_int8"
    file_size_mb: float
    mean_latency_ms: float
    p95_latency_ms: float
    mean_detections_per_frame: float
    mean_iou_vs_reference: float | None  # None for the reference variant itself


def _benchmark_variant(
    weights_path: Path, images: list[np.ndarray]
) -> tuple[list[list[Detection]], list[float]]:
    detector = PersonDetector(str(weights_path))
    latencies_ms = []
    all_detections = []
    for image in images:
        detections, elapsed = detector.detect(image)
        latencies_ms.append(elapsed * 1000)
        all_detections.append(detections)
    return all_detections, latencies_ms


def _mean_iou_vs_reference(
    reference: list[list[Detection]], candidate: list[list[Detection]]
) -> float:
    """Mean best-match IoU between each reference-frame box set and the
    candidate's, averaged over frames -- a cheap proxy for "how much did
    quantization change what the model actually detects," not a
    substitute for a full mAP evaluation (documented in
    docs/limitations.md as a deliberate scope trade-off: mAP needs
    dataset-wide ground truth matching machinery this benchmark doesn't
    otherwise need)."""
    frame_ious = []
    for ref_dets, cand_dets in zip(reference, candidate, strict=True):
        if not ref_dets and not cand_dets:
            frame_ious.append(1.0)
            continue
        if not ref_dets or not cand_dets:
            frame_ious.append(0.0)
            continue
        ref_boxes = np.array([d.xyxy for d in ref_dets])
        cand_boxes = np.array([d.xyxy for d in cand_dets])
        ious = iou_matrix(ref_boxes, cand_boxes)
        frame_ious.append(float(ious.max(axis=1).mean()))
    return float(np.mean(frame_ious)) if frame_ious else 0.0


def benchmark_variant(
    variant_name: str,
    weights_path: Path,
    images: list[np.ndarray],
    reference_detections: list[list[Detection]] | None = None,
) -> tuple[ExportBenchmarkResult, list[list[Detection]]]:
    detections, latencies_ms = _benchmark_variant(weights_path, images)

    # First run includes backend warmup (session/graph build) -- excluded
    # from the latency stats so cold-start cost doesn't dominate a
    # benchmark meant to characterize steady-state edge inference.
    steady_state = latencies_ms[1:] if len(latencies_ms) > 1 else latencies_ms

    mean_iou = (
        _mean_iou_vs_reference(reference_detections, detections)
        if reference_detections is not None
        else None
    )

    result = ExportBenchmarkResult(
        variant=variant_name,
        file_size_mb=weights_path.stat().st_size / (1024 * 1024),
        mean_latency_ms=float(np.mean(steady_state)),
        p95_latency_ms=float(np.percentile(steady_state, 95)),
        mean_detections_per_frame=float(np.mean([len(d) for d in detections])),
        mean_iou_vs_reference=mean_iou,
    )
    return result, detections

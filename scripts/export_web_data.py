"""Exports everything the static web app needs into web/public/data/ (committed, so the site
builds in CI without MOT17, weights or caches):

  manifest.json        sequences, trackers, per-clip metrics
  clips/<seq>.mp4      10 s H.264 clip (+ .jpg poster)
  tracks/<seq>.json    per-tracker tracks, raw detections, ground truth, identity switches
  results.json         the full benchmark (tracking grid, detection AP, edge benchmark)

Clips are the 10-second window of each sequence where ByteTrack makes the most identity
switches -- deliberately the *hard* part, and labelled as such in the UI.

MOT17 (c) MOTChallenge, CC BY-NC-SA 3.0 -- clips and ground-truth overlays carry that licence.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from emot.benchmark.aggregate import (
    SEQUENCES,
    load_detection_results,
    load_edge_benchmark,
    load_tracking_results,
    sequence_table,
)
from emot.benchmark.caches import CachedDetections, detection_path
from emot.benchmark.replay import VARIANTS, load_tuned, replay, score_with_events, sequence_frames
from emot.data.mot17 import load_ground_truth, load_seqinfo
from emot.viz.palette import TRACKER_COLORS

OUT = Path("web/public/data")
BACKBONE = "yolo11s"
CLIP_SECONDS = 10
EXPORT_TRACKERS = ["bytetrack", "botsort_cmc", "botsort", "deepsort", "deepsort_osnet_cascade"]
BLURBS = {
    "bytetrack": "Motion-only. Kalman + IoU, with a second pass that rescues low-confidence boxes.",
    "botsort_cmc": "ByteTrack plus camera-motion compensation (optical flow + RANSAC).",
    "botsort": "Camera-motion compensation plus OSNet appearance fused into the association cost.",
    "deepsort": "Kalman + gated cosine distance on a generic ImageNet ResNet18 embedding.",
    "deepsort_osnet_cascade": (
        "DeepSORT with a real ReID net (OSNet) and the paper's matching cascade."
    ),
}


def pick_window(switch_frames: list[int], n_frames: int, clip_frames: int) -> int:
    """Start frame (1-based) of the window with the most identity switches."""
    counts = np.zeros(n_frames + 2)
    for f in switch_frames:
        counts[f] += 1
    csum = np.concatenate([[0], np.cumsum(counts)])
    best, best_start = -1.0, 1
    for start in range(1, n_frames - clip_frames + 2, max(1, clip_frames // 20)):
        total = csum[start + clip_frames] - csum[start]
        if total > best:
            best, best_start = total, start
    return best_start


def encode_clip(seq_dir: Path, fps: float, start: int, k: int, n_out: int, out_w: int, out: Path):
    out_fps = fps / k
    vf = f"select='not(mod(n,{k}))',setpts=N/({out_fps}*TB),scale={out_w}:-2:flags=lanczos"
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        str(start),
        "-i",
        str(seq_dir / "img1" / "%06d.jpg"),
        "-vf",
        vf,
        "-frames:v",
        str(n_out),
        "-r",
        str(out_fps),
        "-c:v",
        "libx264",
        "-crf",
        "29",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(out),
            "-frames:v",
            "1",
            "-q:v",
            "4",
            str(out.with_suffix(".jpg")),
        ],
        check=True,
    )


def boxes_json(rows, scale):
    return [[int(r[0]), *[int(round(v * scale)) for v in r[1:5]]] for r in rows]


def main() -> None:
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    (OUT / "tracks").mkdir(parents=True, exist_ok=True)
    tuned = load_tuned()
    seq_meta = {r["sequence"]: r for r in sequence_table().to_dict("records")}
    manifest_seqs = []

    for seq in SEQUENCES:
        seq_dir = Path("data/mot17") / seq
        info = load_seqinfo(seq_dir)
        gt = load_ground_truth(seq_dir)
        frames = sequence_frames(info)
        cached = CachedDetections(detection_path(seq, BACKBONE))

        full = {}
        for name in EXPORT_TRACKERS:
            tracks = replay(VARIANTS[name], seq, BACKBONE, frames, tuned, cached=cached)
            full[name] = (tracks, *score_with_events(gt, tracks, frames))
        switch_frames = [e["frame"] for e in full["bytetrack"][2]]

        clip_len = int(round(info.frame_rate * CLIP_SECONDS))
        start = pick_window(switch_frames, info.seq_length, clip_len)
        k = max(1, round(info.frame_rate / 15))
        n_out = clip_len // k
        window = range(start, start + n_out * k)
        scale = (960 if info.im_width >= 1280 else info.im_width) / info.im_width
        out_w = int(round(info.im_width * scale))
        out_h = int(round(info.im_height * scale / 2) * 2)

        encode_clip(seq_dir, info.frame_rate, start, k, n_out, out_w, OUT / "clips" / f"{seq}.mp4")

        out_frames = [start + k * i for i in range(n_out)]
        payload: dict = {"trackers": {}, "switches": {}, "metrics": {}}
        for name in EXPORT_TRACKERS:
            tracks, _, events = full[name]
            payload["trackers"][name] = [
                boxes_json([(tid, *xyxy) for tid, xyxy, _ in tracks.get(f, [])], scale)
                for f in out_frames
            ]
            clip_metrics = score_with_events(gt, tracks, window)[0]
            payload["metrics"][name] = {
                "hota": clip_metrics.hota,
                "idf1": clip_metrics.idf1,
                "mota": clip_metrics.mota,
                "assa": clip_metrics.assa,
                "deta": clip_metrics.deta,
                "num_switches": clip_metrics.num_switches,
            }
            sw = []
            for e in events:
                if start <= e["frame"] < start + n_out * k:
                    f = e["frame"]
                    g = gt[(gt["frame"] == f) & (gt["id"] == e["gt_id"])]
                    box = (
                        [
                            int(round(v * scale))
                            for v in (
                                g.iloc[0]["bb_left"],
                                g.iloc[0]["bb_top"],
                                g.iloc[0]["bb_right"],
                                g.iloc[0]["bb_bottom"],
                            )
                        ]
                        if len(g)
                        else None
                    )
                    sw.append(
                        {
                            "f": (f - start) // k,
                            "gt": e["gt_id"],
                            "from": e["from_track"],
                            "to": e["to_track"],
                            "box": box,
                        }
                    )
            payload["switches"][name] = sw
        payload["detections"] = [
            [
                [int(round(v * scale)) for v in cached.boxes[i]] + [round(float(cached.conf[i]), 2)]
                for i in cached.rows(f)
                if cached.conf[i] >= 0.3
            ]
            for f in out_frames
        ]
        gt_vis = gt[gt["visibility"] >= 0.25]
        payload["gt"] = [
            boxes_json(
                gt_vis[gt_vis["frame"] == f][
                    ["id", "bb_left", "bb_top", "bb_right", "bb_bottom"]
                ].to_numpy(),
                scale,
            )
            for f in out_frames
        ]
        (OUT / "tracks" / f"{seq}.json").write_text(json.dumps(payload, separators=(",", ":")))

        meta = seq_meta[seq]
        manifest_seqs.append(
            {
                "id": seq,
                "camera": meta["camera"],
                "title": f"{seq.replace('MOT17-', 'Scene ')} · {meta['camera']} camera",
                "video": f"clips/{seq}.mp4",
                "poster": f"clips/{seq}.jpg",
                "tracks": f"tracks/{seq}.json",
                "width": out_w,
                "height": out_h,
                "fps": info.frame_rate / k,
                "frames": n_out,
                "sourceFps": info.frame_rate,
                "startFrame": start,
                "meanPeople": meta["mean_people_per_frame"],
                "gtTracks": meta["gt_tracks"],
                "cameraMotionPx": meta["camera_motion_px"],
                "sourceSize": [info.im_width, info.im_height],
            }
        )
        print(
            seq,
            "clip start",
            start,
            "switches in clip (ByteTrack):",
            payload["metrics"]["bytetrack"]["num_switches"],
        )

    manifest = {
        "backbone": BACKBONE,
        "clipSeconds": CLIP_SECONDS,
        "sequences": manifest_seqs,
        "trackers": [
            {
                "id": t,
                "label": VARIANTS[t].label,
                "color": TRACKER_COLORS[t],
                "blurb": BLURBS[t],
                "cmc": VARIANTS[t].use_cmc,
                "reid": VARIANTS[t].use_reid or VARIANTS[t].family == "deepsort",
                "embedder": VARIANTS[t].embedder,
            }
            for t in EXPORT_TRACKERS
        ],
        "license": "MOT17 (c) MOTChallenge, CC BY-NC-SA 3.0",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))

    trk = load_tracking_results()
    det = load_detection_results()
    sweep_trk = load_tracking_results(imgsz=None)
    sweep_trk = sweep_trk[sweep_trk["tracker"].isin(["bytetrack", "botsort_cmc"])]
    sweep_det = load_detection_results(imgsz=None)
    sweep = (
        sweep_trk.groupby(["backbone", "imgsz", "tracker"])[
            ["hota", "idf1", "mota", "assa", "num_switches"]
        ]
        .mean()
        .reset_index()
        .merge(
            sweep_det.groupby(["backbone", "imgsz"])[["ap50", "ap"]].mean().reset_index(),
            on=["backbone", "imgsz"],
        )
    )
    edge = load_edge_benchmark()
    tuned_doc = json.loads(Path("results/tuned_tracker_configs.json").read_text())
    results = {
        "tracking": trk.drop(columns=["label"]).round(4).to_dict("records"),
        "resolutionSweep": sweep.round(4).to_dict("records"),
        "detection": det.round(4).to_dict("records"),
        "sequences": list(seq_meta.values()),
        "edge": edge,
        "variants": {
            k: {
                "label": v.label,
                "family": v.family,
                "cmc": v.use_cmc,
                "reid": v.use_reid,
                "embedder": v.embedder,
                "cascade": v.cascade,
                "color": TRACKER_COLORS.get(k, "#999999"),
            }
            for k, v in VARIANTS.items()
        },
        "tuning": {
            k: tuned_doc[k]
            for k in ("tuned_on_sequence", "tuned_on_backbone", "objective", "n_trials")
        },
    }
    (OUT / "results.json").write_text(json.dumps(results, separators=(",", ":")))

    (OUT / "LICENSE-DATA.txt").write_text(
        "Video clips, ground-truth overlays and detections in this folder are derived from\n"
        "MOT17 (Milan et al., arXiv:1603.00831), (c) MOTChallenge, licensed CC BY-NC-SA 3.0.\n"
        "Non-commercial use with attribution; derivatives must carry the same licence.\n"
    )
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"web data written to {OUT} ({total:.1f} MB)")
    shutil.which("ffmpeg") or print("warning: ffmpeg missing")


if __name__ == "__main__":
    main()

"""Writes results/sequences.json: per-sequence facts derived from the data itself
(resolution, fps, length, GT density, measured camera motion)."""

from __future__ import annotations

import json
from pathlib import Path

from emot.benchmark.aggregate import MOVING_CAMERA_PX, SEQUENCES, camera_motion_px
from emot.data.mot17 import load_ground_truth, load_seqinfo

rows = []
for seq in SEQUENCES:
    d = Path("data/mot17") / seq
    info = load_seqinfo(d)
    gt = load_ground_truth(d)
    motion = camera_motion_px(seq)
    rows.append(
        {
            "sequence": seq,
            "width": info.im_width,
            "height": info.im_height,
            "fps": info.frame_rate,
            "frames": info.seq_length,
            "gt_tracks": int(gt["id"].nunique()),
            "gt_boxes": int(len(gt)),
            "mean_people_per_frame": round(len(gt) / info.seq_length, 1),
            "camera_motion_px": None if motion is None else round(motion, 2),
            "camera": None
            if motion is None
            else ("moving" if motion > MOVING_CAMERA_PX else "static"),
        }
    )
Path("results/sequences.json").write_text(json.dumps(rows, indent=2))
for r in rows:
    print(r)

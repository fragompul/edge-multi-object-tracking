"""Renders the README / docs GIFs from the exported web clips (same data the web app plays):

hero_tracking.gif         BoT-SORT with trails on a busy static-camera clip
compare_camera_motion.gif ByteTrack vs BoT-SORT (CMC) on a hand-held clip, with a live
                          identity-switch counter and red rings where identities break
compare_reid.gif          DeepSORT (generic ResNet18) vs DeepSORT (OSNet + cascade)
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from emot.viz.draw import FONT, TrackView, TrailBuffer, _alpha_rect, draw_hud, draw_tracks
from emot.viz.palette import MUTED, NAVY, RED, TRACKER_COLORS, hex_to_bgr

DATA = Path("web/public/data")
OUT = Path("docs/assets")


def load(seq: str):
    tracks = json.loads((DATA / "tracks" / f"{seq}.json").read_text())
    cap = cv2.VideoCapture(str(DATA / "clips" / f"{seq}.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames, tracks


def render_panel(
    frame,
    rows,
    trails: TrailBuffer,
    title: str,
    color: str,
    lines: list[str],
    switch_events,
    idx: int,
):
    img = frame.copy()
    views = [TrackView(r[0], np.array(r[1:5], float)) for r in rows]
    trails.push(views)
    draw_tracks(img, views, trails.trails(), thickness=2)
    for e in switch_events:
        age = idx - e["f"]
        if 0 <= age <= 12 and e["box"]:
            x1, y1, x2, y2 = e["box"]
            c = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            axes = (int((x2 - x1) / 2 + 8), int((y2 - y1) / 2 + 8))
            cv2.ellipse(img, c, axes, 0, 0, 360, hex_to_bgr(RED), 3, cv2.LINE_AA)
            cv2.putText(
                img,
                f"ID {e['from']}>{e['to']}",
                (x1, max(18, y1 - 8)),
                FONT,
                0.6,
                hex_to_bgr(RED),
                2,
                cv2.LINE_AA,
            )
    draw_hud(img, title, lines, accent=color)
    return img


def encode_gif(frames: list[np.ndarray], fps: float, width: int, out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            h, w = f.shape[:2]
            cv2.imwrite(
                f"{tmp}/{i:04d}.png",
                cv2.resize(f, (width, int(h * width / w)), interpolation=cv2.INTER_AREA),
            )
        palette = f"{tmp}/p.png"
        src = f"{tmp}/%04d.png"
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                src,
                "-vf",
                "palettegen=max_colors=96:stats_mode=diff",
                palette,
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                src,
                "-i",
                palette,
                "-lavfi",
                "paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                "-loop",
                "0",
                str(out),
            ],
            check=True,
        )
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")


def single(seq: str, tracker: str, out: Path, start: int, n: int, width: int) -> None:
    frames, t = load(seq)
    fps = json.loads((DATA / "manifest.json").read_text())["sequences"][0]["fps"]
    trails = TrailBuffer(30)
    label = {"botsort": "BoT-SORT (CMC + ReID)"}[tracker]
    outs = []
    for i in range(start, start + n):
        rows = t["trackers"][tracker][i]
        outs.append(
            render_panel(
                frames[i],
                rows,
                trails,
                f"EMOT | {label}",
                TRACKER_COLORS[tracker],
                [f"{len(rows)} people tracked"],
                [],
                i,
            )
        )
    encode_gif(outs, 10, width, out)
    _ = fps


def compare(
    seq: str, a: str, b: str, labels: tuple[str, str], out: Path, start: int, n: int, width: int
) -> None:
    frames, t = load(seq)
    ta, tb = TrailBuffer(25), TrailBuffer(25)
    outs = []
    for i in range(start, start + n):
        panels = []
        for name, label, trails in ((a, labels[0], ta), (b, labels[1], tb)):
            sw = t["switches"][name]
            so_far = sum(1 for e in sw if start <= e["f"] <= i)
            panels.append(
                render_panel(
                    frames[i],
                    t["trackers"][name][i],
                    trails,
                    label,
                    TRACKER_COLORS[name],
                    [f"identity switches: {so_far}"],
                    sw,
                    i,
                )
            )
        gap = np.full((panels[0].shape[0], 6, 3), hex_to_bgr(NAVY), np.uint8)
        outs.append(np.hstack([panels[0], gap, panels[1]]))
    encode_gif(outs, 10, width, out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    single("MOT17-04", "botsort", OUT / "hero_tracking.gif", 0, 70, 760)
    compare(
        "MOT17-10",
        "bytetrack",
        "botsort_cmc",
        ("ByteTrack", "BoT-SORT + camera-motion compensation"),
        OUT / "compare_camera_motion.gif",
        14,
        56,
        820,
    )
    compare(
        "MOT17-02",
        "deepsort",
        "deepsort_osnet_cascade",
        ("DeepSORT (ImageNet ResNet18)", "DeepSORT (OSNet ReID + cascade)"),
        OUT / "compare_reid.gif",
        10,
        60,
        900,
    )
    _ = (MUTED, _alpha_rect)


if __name__ == "__main__":
    main()

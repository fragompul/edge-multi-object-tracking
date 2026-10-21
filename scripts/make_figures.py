"""Regenerates every figure in docs/figures/ from results/ and the caches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from emot.benchmark.aggregate import (
    SEQUENCES,
    load_detection_results,
    load_edge_benchmark,
    load_tracking_results,
    sequence_table,
)
from emot.benchmark.caches import cmc_path
from emot.viz import figures as F

OUT = Path("docs/figures")


def reid_stats(cache: Path = Path("results/reid_separation.json")) -> dict:
    """Cosine similarities between GT crops of the same / different people, for both
    embedders (cached: needs MOT17 frames and the OSNet weights)."""
    if cache.exists():
        return json.loads(cache.read_text())
    import cv2

    from emot.benchmark.precompute import crop_rgb
    from emot.data.mot17 import frame_path, load_ground_truth
    from emot.reid import build_embedder

    rng = np.random.default_rng(0)
    out: dict = {}
    embedders = {n: build_embedder(n) for n in ("resnet18", "osnet_x0_25")}
    acc = {n: {"same": [], "different": []} for n in embedders}
    for seq in ("MOT17-02", "MOT17-04", "MOT17-09", "MOT17-11"):
        d = Path("data/mot17") / seq
        gt = load_ground_truth(d, min_visibility=0.6)
        for f1, f2 in [(60, 120), (200, 280), (330, 400)]:
            a, b = gt[gt.frame == f1], gt[gt.frame == f2]
            ids = sorted(set(a.id) & set(b.id))
            if len(ids) < 4:
                continue
            crops = {}
            for f, df in ((f1, a), (f2, b)):
                img = cv2.imread(str(frame_path(d, f)))
                crops[f] = [
                    crop_rgb(
                        img,
                        df[df.id == i]
                        .iloc[0][["bb_left", "bb_top", "bb_right", "bb_bottom"]]
                        .to_numpy(float),
                    )
                    for i in ids
                ]
            for name, emb in embedders.items():
                ea, eb = emb.embed_crops(crops[f1]), emb.embed_crops(crops[f2])
                s = ea @ eb.T
                acc[name]["same"] += np.diag(s).tolist()
                acc[name]["different"] += s[~np.eye(len(ids), dtype=bool)].tolist()
    for name, d in acc.items():
        same, diff = np.array(d["same"]), np.array(d["different"])
        dprime = (same.mean() - diff.mean()) / np.sqrt(0.5 * (same.var() + diff.var()))
        out[name] = {
            "same": same.round(4).tolist(),
            "different": rng.choice(diff, min(len(diff), 1500), replace=False).round(4).tolist(),
            "dprime": float(dprime),
            "n_same": int(len(same)),
            "mean_same": float(same.mean()),
            "mean_diff": float(diff.mean()),
        }
    cache.write_text(json.dumps(out))
    return out


def main() -> None:
    trk = load_tracking_results()
    det = load_detection_results()
    seq = sequence_table()
    edge_doc = load_edge_benchmark()
    edge = pd.DataFrame(edge_doc["rows"])
    sweep_trk = load_tracking_results(imgsz=None)
    sweep_det = load_detection_results(imgsz=None)
    sweep = pd.DataFrame()
    if not sweep_trk.empty and (sweep_trk["imgsz"] != 640).any():
        sweep = (
            sweep_trk.groupby(["backbone", "imgsz", "tracker"])[["hota", "mota", "idf1"]]
            .mean()
            .reset_index()
        )

    F.leaderboard(trk, OUT / "leaderboard.png")
    F.heatmap(trk, OUT / "heatmap_hota.png", "hota", seq)
    F.ablation(trk, OUT / "ablation_cmc_reid.png", seq)
    F.id_switches(trk, OUT / "id_switches.png")
    if not edge.empty:
        F.backbone_scatter(det, trk, edge, OUT / "backbone_pareto.png")
        F.edge_latency(edge, OUT / "edge_latency.png")
        F.edge_resolution(edge, sweep, OUT / "edge_resolution.png")
        F.edge_threads(edge, OUT / "edge_threads.png")
        if "profile" in edge_doc:
            F.op_profile(edge_doc["profile"], OUT / "edge_op_profile.png")
    cmc = {s: np.load(cmc_path(s)) for s in SEQUENCES if cmc_path(s).exists()}
    if cmc:
        F.camera_motion(cmc, OUT / "camera_motion.png")
    F.reid_separation(reid_stats(), OUT / "reid_separation.png")
    _ = sweep_det
    print("figures written to", OUT)


if __name__ == "__main__":
    main()

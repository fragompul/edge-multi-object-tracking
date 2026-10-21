"""Brand-styled matplotlib figures for the docs and README.

Every function takes tidy DataFrames (see `emot.benchmark.aggregate`) and writes one PNG,
so figures are regenerated from `results/` -- never hand-edited.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from emot.viz import palette as P  # noqa: E402

_FONTS = ["Avenir Next", "Helvetica Neue", "Inter", "Arial", "DejaVu Sans"]
_available = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in _FONTS if f in _available), "DejaVu Sans")

TRACKER_ORDER = [
    "bytetrack",
    "botsort_cmc",
    "botsort",
    "deepsort",
    "deepsort_osnet_cascade",
]
LABELS = {
    "bytetrack": "ByteTrack",
    "botsort_cmc": "BoT-SORT\n(CMC)",
    "botsort": "BoT-SORT\n(CMC + ReID)",
    "botsort_reid": "BoT-SORT\n(ReID only)",
    "deepsort": "DeepSORT\n(ResNet18)",
    "deepsort_osnet": "DeepSORT\n(OSNet)",
    "deepsort_osnet_cascade": "DeepSORT\n(OSNet + cascade)",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": P.NAVY,
            "axes.facecolor": P.NAVY_2,
            "savefig.facecolor": P.NAVY,
            "axes.edgecolor": "#2B3563",
            "axes.labelcolor": P.INK,
            "text.color": P.INK,
            "xtick.color": P.MUTED,
            "ytick.color": P.MUTED,
            "grid.color": "#222B55",
            "axes.grid": True,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "font.family": FONT,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlecolor": P.INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.labelcolor": P.INK,
        }
    )


def _save(fig: plt.Figure, path: Path, caption: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if caption:
        fig.text(0.01, 0.005, caption, color=P.MUTED, fontsize=8, ha="left", va="bottom")
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def leaderboard(df: pd.DataFrame, path: Path) -> Path:
    """Mean HOTA / IDF1 / MOTA per tracker variant."""
    style()
    order = [t for t in TRACKER_ORDER if t in set(df["tracker"])]
    means = df.groupby("tracker")[["hota", "idf1", "mota", "assa", "deta"]].mean().loc[order]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(order))
    w = 0.26
    for i, (col, name, color) in enumerate(
        [("hota", "HOTA", P.CYAN), ("idf1", "IDF1", P.MAGENTA), ("mota", "MOTA", P.AMBER)]
    ):
        bars = ax.bar(x + (i - 1) * w, means[col], w, label=name, color=color, zorder=3)
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.005,
                f"{b.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=P.INK,
            )
    ax.set_xticks(x, [LABELS[t] for t in order])
    ax.set_ylim(0, max(means[["hota", "idf1", "mota"]].max()) * 1.18)
    ax.set_title("Tracker leaderboard -- mean over 7 MOT17 sequences x 6 YOLO backbones")
    ax.legend(ncol=3, loc="upper left")
    return _save(fig, path)


def heatmap(
    df: pd.DataFrame, path: Path, metric: str = "hota", seq_meta: pd.DataFrame | None = None
) -> Path:
    """Tracker x sequence matrix of a metric, mean over backbones."""
    style()
    order = [t for t in TRACKER_ORDER if t in set(df["tracker"])]
    pivot = df.pivot_table(index="tracker", columns="sequence", values=metric, aggfunc="mean").loc[
        order
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    im = ax.imshow(pivot.values, cmap="magma", aspect="auto")
    ax.grid(False)
    cams = {}
    if seq_meta is not None and not seq_meta.empty:
        cams = dict(zip(seq_meta["sequence"], seq_meta["camera"], strict=False))
    ax.set_xticks(
        range(pivot.shape[1]),
        [f"{s.replace('MOT17-', '')}\n{cams.get(s, '')}" for s in pivot.columns],
    )
    ax.set_yticks(range(len(order)), [LABELS[t].replace("\n", " ") for t in order])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if v < np.nanmedian(pivot.values) else "black",
            )
    ax.set_title(f"{metric.upper()} by tracker and sequence (mean over backbones)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    return _save(fig, path)


def ablation(df: pd.DataFrame, path: Path, seq_meta: pd.DataFrame) -> Path:
    """What camera-motion compensation and ReID each buy, on static vs moving-camera clips."""
    style()
    cams = dict(zip(seq_meta["sequence"], seq_meta["camera"], strict=False))
    d = df.assign(camera=df["sequence"].map(cams))
    steps = ["bytetrack", "botsort_cmc", "botsort_reid", "botsort"]
    steps = [s for s in steps if s in set(d["tracker"])]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, cam in zip(axes, ["static", "moving"], strict=True):
        sub = d[d["camera"] == cam].groupby("tracker")[["hota", "assa", "idf1"]].mean().loc[steps]
        x = np.arange(len(steps))
        for k, (col, color) in enumerate(
            [("hota", P.CYAN), ("assa", P.VIOLET), ("idf1", P.MAGENTA)]
        ):
            bars = ax.bar(
                x + (k - 1) * 0.26, sub[col], 0.26, color=color, label=col.upper(), zorder=3
            )
            for b in bars:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.004,
                    f"{b.get_height():.3f}",
                    ha="center",
                    fontsize=7.5,
                )
        n = int((pd.Series(cams) == cam).sum())
        ax.set_title(f"{cam.capitalize()} camera ({n} sequences)")
        ax.set_xticks(x, [LABELS[s] for s in steps], fontsize=9)
    axes[0].legend(ncol=3, loc="upper left")
    axes[0].set_ylim(0, 0.75)
    fig.suptitle(
        "Ablation: what camera-motion compensation and appearance each contribute",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    return _save(fig, path)


def id_switches(df: pd.DataFrame, path: Path) -> Path:
    style()
    order = [t for t in TRACKER_ORDER if t in set(df["tracker"])]
    sub = df.pivot_table(
        index="sequence", columns="tracker", values="num_switches", aggfunc="mean"
    )[order]
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    x = np.arange(len(sub))
    w = 0.8 / len(order)
    for k, t in enumerate(order):
        ax.bar(
            x + (k - (len(order) - 1) / 2) * w,
            sub[t],
            w,
            label=LABELS[t].replace("\n", " "),
            color=P.TRACKER_COLORS[t],
            zorder=3,
        )
    ax.set_xticks(x, [s.replace("MOT17-", "") for s in sub.index])
    ax.set_ylabel("identity switches (mean over backbones)")
    ax.set_title("Identity switches by sequence -- lower is better")
    ax.legend(ncol=3, fontsize=8.5, loc="upper left")
    return _save(fig, path)


def backbone_scatter(det: pd.DataFrame, trk: pd.DataFrame, edge: pd.DataFrame, path: Path) -> Path:
    """Detector quality (AP50) and downstream HOTA vs CPU latency, one point per backbone."""
    style()
    lat = edge[(edge["group"] == "main") & (edge["variant"] == "onnx_fp32")].set_index("backbone")[
        "mean_ms"
    ]
    ap = det.groupby("backbone")["ap50"].mean()
    hota = trk[trk["tracker"] == "bytetrack"].groupby("backbone")["hota"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax, series, label in [(axes[0], ap, "detection AP50"), (axes[1], hota, "ByteTrack HOTA")]:
        for bb in series.index:
            color = P.CYAN if bb.startswith("yolo11") else P.MAGENTA
            ax.scatter(
                lat[bb], series[bb], s=140, color=color, zorder=3, edgecolor=P.NAVY, linewidth=2
            )
            ax.annotate(
                bb, (lat[bb], series[bb]), xytext=(8, 6), textcoords="offset points", fontsize=9
            )
        ax.set_xlabel("CPU latency per frame, ms (ONNX FP32)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs latency")
    axes[0].scatter([], [], color=P.CYAN, label="YOLO11")
    axes[0].scatter([], [], color=P.MAGENTA, label="YOLOv8")
    axes[0].legend(loc="lower right")
    return _save(fig, path)


# ---------------------------------------------------------------- edge benchmark figures
RUNTIMES = [
    ("torch_cpu", "PyTorch CPU", "#8B93B0"),
    ("torch_mps", "PyTorch Apple GPU", P.AMBER),
    ("onnx_fp32", "ONNX CPU FP32", P.CYAN),
    ("onnx_int8_dynamic", "ONNX INT8 dynamic", P.VIOLET),
    ("onnx_int8_static", "ONNX INT8 static", P.MAGENTA),
    ("onnx_coreml", "ONNX CoreML", P.LIME),
]


def edge_latency(edge: pd.DataFrame, path: Path) -> Path:
    style()
    main = edge[edge["group"] == "main"]
    backbones = list(main["backbone"].unique())
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(backbones))
    w = 0.8 / len(RUNTIMES)
    for k, (v, label, color) in enumerate(RUNTIMES):
        vals = [
            main[(main["backbone"] == b) & (main["variant"] == v)]["mean_ms"].mean()
            for b in backbones
        ]
        ax.bar(x + (k - (len(RUNTIMES) - 1) / 2) * w, vals, w, label=label, color=color, zorder=3)
    ax.set_xticks(x, backbones)
    ax.set_ylabel("ms per frame (lower is better)")
    ax.set_title("Detector latency by runtime -- 150 MOT17 frames, 384x640 input")
    ax.legend(ncol=3, loc="upper left", fontsize=9)
    return _save(fig, path)


def edge_resolution(edge: pd.DataFrame, sweep: pd.DataFrame, path: Path) -> Path:
    """Tracking quality (HOTA) and detection AP50 against CPU latency across input sizes."""
    style()
    res = edge[edge["group"] == "resolution"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"yolo11n": P.CYAN, "yolo11s": P.VIOLET, "yolo11m": P.MAGENTA}
    for bb, color in colors.items():
        r = res[res["backbone"] == bb].sort_values("long_side")
        axes[0].plot(r["mean_ms"], r["ap50"], "-o", color=color, label=bb, lw=2, ms=7, zorder=3)
        for _, row in r.iterrows():
            axes[0].annotate(
                int(row["long_side"]),
                (row["mean_ms"], row["ap50"]),
                xytext=(6, -12),
                textcoords="offset points",
                fontsize=8,
                color=P.MUTED,
            )
        if not sweep.empty:
            s = sweep[(sweep["backbone"] == bb) & (sweep["tracker"] == "botsort_cmc")].sort_values(
                "imgsz"
            )
            lat = r.set_index("long_side")["mean_ms"]
            xs = [lat[i] for i in s["imgsz"] if i in lat.index]
            ys = [h for i, h in zip(s["imgsz"], s["hota"], strict=True) if i in lat.index]
            axes[1].plot(xs, ys, "-o", color=color, label=bb, lw=2, ms=7, zorder=3)
    for ax, ylab, title in [
        (axes[0], "detection AP50", "Detector accuracy vs latency"),
        (axes[1], "HOTA (BoT-SORT + CMC)", "Tracking quality vs latency"),
    ]:
        ax.set_xscale("log")
        ax.set_xlabel("CPU latency per frame, ms (log; labels = input long side px)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend()
    fig.suptitle(
        "Input resolution is a bigger lever than model size", fontsize=14, fontweight="bold", y=1.02
    )
    return _save(fig, path)


def edge_threads(edge: pd.DataFrame, path: Path) -> Path:
    style()
    t = edge[edge["group"] == "threads"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for (bb, v), color, ls in [
        (("yolo11n", "onnx_fp32"), P.CYAN, "-"),
        (("yolo11n", "onnx_int8_static"), P.MAGENTA, "-"),
        (("yolo11s", "onnx_fp32"), P.CYAN, "--"),
        (("yolo11s", "onnx_int8_static"), P.MAGENTA, "--"),
    ]:
        r = t[(t["backbone"] == bb) & (t["variant"] == v)].sort_values("threads")
        ax.plot(
            r["threads"],
            r["mean_ms"],
            ls,
            marker="o",
            color=color,
            lw=2,
            label=f"{bb} {'FP32' if v == 'onnx_fp32' else 'INT8'}",
        )
    ax.set_xlabel("intra-op threads")
    ax.set_ylabel("ms per frame")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 10], ["1", "2", "4", "10"])
    ax.set_title("FP32 does not scale with threads; INT8 does")
    ax.legend()
    return _save(fig, path)


def op_profile(profile: dict, path: Path) -> Path:
    """Where the time goes per operator, FP32 vs INT8 (single thread and all threads)."""
    style()
    variants = profile["variants"]
    keys = [
        k
        for k in [
            "fp32/1",
            "int8_dynamic/1",
            "int8_static/1",
            "fp32/all",
            "int8_dynamic/all",
            "int8_static/all",
        ]
        if k in variants
    ]
    ops = sorted({r["op"] for k in keys for r in variants[k]})
    palette = [
        P.CYAN,
        P.MAGENTA,
        P.AMBER,
        P.VIOLET,
        P.LIME,
        "#FB923C",
        "#34D399",
        "#F87171",
        "#60A5FA",
        "#E879F9",
    ]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    left = np.zeros(len(keys))
    labels = [
        k.replace("/1", "\n1 thread").replace("/all", "\nall threads").replace("_", " ")
        for k in keys
    ]
    for i, op in enumerate(ops):
        vals = np.array([next((r["ms"] for r in variants[k] if r["op"] == op), 0.0) for k in keys])
        if vals.sum() < 0.8:
            continue
        ax.barh(
            range(len(keys)), vals, left=left, color=palette[i % len(palette)], label=op, zorder=3
        )
        left += vals
    ax.set_yticks(range(len(keys)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("ms per frame (yolo11s, top operators)")
    ax.set_title("Where the time goes: convolution kernels, FP32 vs INT8")
    ax.legend(ncol=4, fontsize=8, loc="lower right")
    return _save(fig, path)


def camera_motion(cmc: dict[str, np.ndarray], path: Path) -> Path:
    """Cumulative estimated camera path per sequence: tripod vs hand-held, from optical flow."""
    style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for name, warps in cmc.items():
        path_xy = np.cumsum(warps[:, :, 2], axis=0)
        moving = np.linalg.norm(warps[1:, :, 2], axis=1).mean() > 1.0
        color = P.MAGENTA if moving else P.CYAN
        axes[0].plot(path_xy[:, 0], path_xy[:, 1], color=color, lw=1.6, alpha=0.9)
        axes[0].annotate(name.replace("MOT17-", ""), path_xy[-1], fontsize=9, color=color)
        axes[1].bar(
            name.replace("MOT17-", ""),
            np.linalg.norm(warps[1:, :, 2], axis=1).mean(),
            color=color,
            zorder=3,
        )
    axes[0].set_title("Cumulative camera translation (px)")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].set_title("Mean global motion per frame (px)")
    axes[1].axhline(1.0, color=P.AMBER, ls="--", lw=1)
    axes[1].text(
        0.02,
        1.2,
        "moving-camera threshold",
        color=P.AMBER,
        fontsize=8,
        transform=axes[1].get_yaxis_transform(),
    )
    return _save(fig, path)


def reid_separation(stats: dict, path: Path) -> Path:
    """Same-identity vs different-identity cosine similarity, generic vs ReID embeddings."""
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, key, title in [
        (axes[0], "resnet18", "ResNet18 (ImageNet features)"),
        (axes[1], "osnet_x0_25", "OSNet x0.25 (MSMT17 ReID)"),
    ]:
        d = stats[key]
        bins = np.linspace(-0.2, 1.0, 40)
        ax.hist(
            d["different"],
            bins=bins,
            color=P.MAGENTA,
            alpha=0.75,
            label="different people",
            density=True,
            zorder=3,
        )
        ax.hist(
            d["same"],
            bins=bins,
            color=P.CYAN,
            alpha=0.75,
            label="same person",
            density=True,
            zorder=3,
        )
        ax.set_title(f"{title}\nseparation (d') = {d['dprime']:.2f}")
        ax.set_xlabel("cosine similarity")
        ax.legend()
    return _save(fig, path)

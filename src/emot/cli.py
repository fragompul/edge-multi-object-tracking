"""`emot` command line: run the tracking pipeline on any video source, compare
trackers side by side, inspect benchmark results, export models, and query
tracks in natural language."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from emot import __version__

app = typer.Typer(
    name="emot",
    help="EMOT -- edge multi-object tracking: detect, track, benchmark, deploy.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

TRACKERS = ["bytetrack", "botsort", "botsort_cmc", "deepsort", "deepsort_osnet_cascade"]


def _auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _build_detector(
    model: str, onnx: Path | None, provider: str, conf: float, device: str, imgsz: int, threads: int
):
    if onnx is not None:
        from emot.detection.onnx_detector import OnnxPersonDetector

        return OnnxPersonDetector(
            onnx, confidence_threshold=conf, provider=provider, num_threads=threads
        )
    from emot.detection.yolo import PersonDetector

    return PersonDetector(model, confidence_threshold=conf, device=device, imgsz=imgsz)


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"emot {__version__}")


@app.command()
def track(
    source: Annotated[
        str, typer.Argument(help="Video file, webcam index, rtsp:// URL, or image folder")
    ],
    tracker: Annotated[str, typer.Option(help=f"One of {TRACKERS}")] = "botsort",
    model: Annotated[
        str, typer.Option(help="Ultralytics checkpoint (auto-downloaded)")
    ] = "yolo11n.pt",
    onnx: Annotated[Path | None, typer.Option(help="Use an exported ONNX detector instead")] = None,
    provider: Annotated[
        str, typer.Option(help="ONNX Runtime provider: cpu | coreml | cuda")
    ] = "cpu",
    imgsz: Annotated[int, typer.Option(help="PyTorch detector input size")] = 640,
    conf: Annotated[float, typer.Option(help="Detector confidence floor")] = 0.1,
    device: Annotated[
        str, typer.Option(help="PyTorch/ReID device: auto | cpu | mps | cuda")
    ] = "auto",
    threads: Annotated[int, typer.Option(help="ONNX intra-op threads (0 = all cores)")] = 0,
    save: Annotated[Path | None, typer.Option(help="Write the annotated video here (.mp4)")] = None,
    jsonl: Annotated[Path | None, typer.Option(help="Write per-frame tracks as JSON lines")] = None,
    show: Annotated[bool, typer.Option(help="Open a live preview window (q to quit)")] = False,
    max_frames: Annotated[int | None, typer.Option(help="Stop after N frames")] = None,
    trail: Annotated[int, typer.Option(help="Trail length in frames (0 disables)")] = 40,
) -> None:
    """Detect and track people in a video, with live FPS / latency stats."""
    import cv2

    from emot.pipeline.sources import FrameSource
    from emot.pipeline.stream import TrackingPipeline
    from emot.tracking.online import OnlineTracker

    dev = _auto_device() if device == "auto" else device
    detector = _build_detector(model, onnx, provider, conf, dev, imgsz, threads)
    online = OnlineTracker(tracker, device=dev)
    label = f"EMOT | {tracker}" + (f" | {onnx.stem}" if onnx else f" | {Path(model).stem}")
    pipeline = TrackingPipeline(detector, online, label=label, trail_length=trail or 1)

    def on_frame(idx, frame, boxes, stats):  # noqa: ANN001
        if show:
            cv2.imshow("EMOT", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    try:
        result = pipeline.run(
            FrameSource(source), on_frame if show else None, save, jsonl, max_frames
        )
    except KeyboardInterrupt:
        console.print("[yellow]stopped by user[/yellow]")
        return
    finally:
        if show:
            cv2.destroyAllWindows()
    s = result.stats
    console.print(
        f"[bold cyan]{result.frames} frames[/] in {result.wall_seconds:.1f}s  "
        f"[bold]{result.avg_fps:.1f} FPS[/]  (detect {s.detect_ms:.1f} ms | "
        f"track {s.track_ms:.1f} ms per frame, dropped {result.dropped})"
    )
    if save:
        console.print(f"video  -> {save}")
    if jsonl:
        console.print(f"tracks -> {jsonl}")


@app.command()
def results(
    results_dir: Annotated[Path, typer.Option()] = Path("results"),
    metric: Annotated[str, typer.Option(help="mota | idf1 | hota | num_switches ...")] = "hota",
) -> None:
    """Summarise the committed benchmark: mean metric per tracker variant."""
    import pandas as pd

    rows = []
    for f in results_dir.glob("MOT17-*/*/*.json"):
        if f.stem == "detection":
            continue
        rows.append(json.loads(f.read_text()))
    if not rows:
        console.print("[red]no results found[/red]")
        raise typer.Exit(1)
    df = pd.DataFrame(rows)
    table = Table(
        title=f"Mean {metric.upper()} over {df.sequence.nunique()} sequences x "
        f"{df.backbone.nunique()} backbones"
    )
    for col in ("tracker", "MOTA", "IDF1", "HOTA", "AssA", "ID switches"):
        table.add_column(col, justify="left" if col == "tracker" else "right")
    grouped = df.groupby("tracker")[["mota", "idf1", "hota", "assa", "num_switches"]].mean()
    for name, r in grouped.sort_values(metric, ascending=metric == "num_switches").iterrows():
        table.add_row(
            str(name),
            f"{r.mota:.3f}",
            f"{r.idf1:.3f}",
            f"{r.hota:.3f}",
            f"{r.assa:.3f}",
            f"{r.num_switches:.0f}",
        )
    console.print(table)


@app.command()
def ask(
    tracks: Annotated[Path, typer.Argument(help="Tracks JSONL written by `emot track --jsonl`")],
    question: Annotated[
        str, typer.Argument(help="e.g. 'How many people crossed the middle line?'")
    ],
    fps: Annotated[float, typer.Option(help="Video frame rate")] = 30.0,
    width: Annotated[int, typer.Option(help="Video width in px")] = 1920,
    height: Annotated[int, typer.Option(help="Video height in px")] = 1080,
    offline: Annotated[
        bool, typer.Option(help="Use the rule-based planner (no API key needed)")
    ] = False,
    model: Annotated[str, typer.Option(help="Claude model for the agent")] = "claude-opus-5-5",
    trace: Annotated[bool, typer.Option(help="Show every tool call and result")] = False,
) -> None:
    """Ask a question about tracked video in plain English."""
    from emot.agent import TrackStore, ask_claude, ask_offline

    store = TrackStore.from_jsonl(tracks, fps, width, height)
    if offline:
        result = ask_offline(store, question)
    else:
        import os

        if not os.environ.get("ANTHROPIC_API_KEY"):
            console.print(
                "[yellow]ANTHROPIC_API_KEY is not set -- falling back to the offline planner.[/yellow]"
            )
            result = ask_offline(store, question)
        else:
            result = ask_claude(store, question, model=model)
    console.print(f"[bold cyan]{result.text}[/]")
    if trace:
        for call in result.tool_calls:
            console.print(
                f"  [dim]tool[/] {call.name}({json.dumps(call.arguments)}) -> {json.dumps(call.result)}"
            )
    console.print(
        f"[dim]({'offline planner' if result.offline else result.model}, {result.turns} turn(s))[/dim]"
    )


if __name__ == "__main__":
    app()

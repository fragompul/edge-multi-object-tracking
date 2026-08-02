"""Downloads a 4-sequence subset of MOT17 from the public Hugging Face
mirror `Lekim89/MOT17` and lays it out in the canonical MOTChallenge
directory convention:

    data/mot17/<sequence>/img1/000001.jpg ...
    data/mot17/<sequence>/gt/gt.txt
    data/mot17/<sequence>/seqinfo.ini

Only the `-FRCNN` detector-variant folder is fetched per sequence: MOT17
ships each sequence three times (`-DPM`/`-FRCNN`/`-SDP`) with identical
images and only the public detection file differing, and this project
runs its own YOLO detectors rather than using the bundled ones.

The official host (motchallenge.net) is unreachable from this
environment (verified: connection refused, not a transient failure), so
this project sources the same data from the community HF mirror instead.

Original dataset: Milan et al., "MOT16: A Benchmark for Multi-Object
Tracking", arXiv:1603.00831 (MOT17 extends MOT16 with refined ground
truth).
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

HF_BASE = "https://huggingface.co/datasets/Lekim89/MOT17/resolve/main"
DEFAULT_SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-11", "MOT17-13"]
FRAME_COUNTS = {
    "MOT17-02": 600,
    "MOT17-04": 1050,
    "MOT17-05": 837,
    "MOT17-09": 525,
    "MOT17-10": 654,
    "MOT17-11": 900,
    "MOT17-13": 750,
}

logger = logging.getLogger(__name__)


def _fetch_bytes(url: str, timeout: int = 30) -> bytes:
    req = Request(url, headers={"User-Agent": "emot-dataset-downloader"})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed, trusted HF host
        return resp.read()


def _download_one(remote_path: str, dest: Path, max_retries: int = 6) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_BASE}/{remote_path}"
    for attempt in range(max_retries):
        try:
            data = _fetch_bytes(url)
            dest.write_bytes(data)
            return True
        except URLError as exc:
            # HF's CDN rate-limits (HTTP 429) hard under concurrent load;
            # retrying immediately just draws more 429s. Exponential
            # backoff with jitter (not a fixed delay) so a whole worker
            # pool doesn't retry in lockstep and re-trigger the limit.
            delay = min(30.0, (2**attempt) * 0.5) + random.uniform(0, 0.5)
            logger.warning(
                "attempt %d failed for %s: %s (retrying in %.1fs)", attempt + 1, url, exc, delay
            )
            time.sleep(delay)
    return False


def _sequence_jobs(sequence: str, output_dir: Path) -> list[tuple[str, Path]]:
    remote_seq = f"{sequence}-FRCNN"
    local_dir = output_dir / sequence
    jobs: list[tuple[str, Path]] = [
        (f"train/{remote_seq}/gt/gt.txt", local_dir / "gt" / "gt.txt"),
        (f"train/{remote_seq}/seqinfo.ini", local_dir / "seqinfo.ini"),
    ]
    for frame_idx in range(1, FRAME_COUNTS[sequence] + 1):
        filename = f"{frame_idx:06d}.jpg"
        jobs.append(
            (f"train/{remote_seq}/img1/{filename}", local_dir / "img1" / filename)
        )
    return jobs


def download_mot17_subset(
    output_dir: Path,
    sequences: list[str] = DEFAULT_SEQUENCES,
    max_workers: int = 12,
) -> dict[str, int]:
    jobs: list[tuple[str, Path]] = []
    for sequence in sequences:
        jobs.extend(_sequence_jobs(sequence, output_dir))

    logger.info("Downloading %d files across %s with %d workers...", len(jobs), sequences, max_workers)
    ok, failed = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, remote, dest): dest for remote, dest in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            if future.result():
                ok += 1
            else:
                failed += 1
                logger.error("permanently failed: %s", futures[future])
            if i % 200 == 0 or i == len(jobs):
                logger.info("progress: %d/%d (ok=%d, failed=%d)", i, len(jobs), ok, failed)

    counts = {"downloaded": ok, "failed": failed, "total": len(jobs)}
    logger.info("Done: %s", counts)
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/mot17", type=Path)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    counts = download_mot17_subset(args.output, args.sequences, args.workers)
    if counts["failed"] > 0:
        raise SystemExit(f"{counts['failed']} file(s) failed to download after retries")


if __name__ == "__main__":
    main()

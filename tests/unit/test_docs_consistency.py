"""The README quotes headline numbers; fail loudly if they drift from the committed results."""

from pathlib import Path

import pytest

from emot.benchmark.aggregate import load_tracking_results

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def means():
    df = load_tracking_results(ROOT / "results")
    if df.empty:
        pytest.skip("results/ not present")
    return df.groupby("tracker")[["hota", "num_switches"]].mean()


def test_readme_headline_table_matches_results(means):
    readme = (ROOT / "README.md").read_text()
    for tracker, label in [
        ("botsort", "BoT-SORT (CMC + ReID)"),
        ("botsort_cmc", "BoT-SORT (CMC)"),
        ("deepsort_osnet_cascade", "DeepSORT (OSNet + cascade)"),
        ("bytetrack", "ByteTrack"),
        ("deepsort", "DeepSORT (ResNet18)"),
    ]:
        row = next(
            line for line in readme.splitlines()
            if line.startswith(f"| {label}") or f"| **{label}**" in line
        )
        assert f"{means.loc[tracker, 'hota']:.3f}" in row, label
        assert f"{means.loc[tracker, 'num_switches']:.0f}" in row, label

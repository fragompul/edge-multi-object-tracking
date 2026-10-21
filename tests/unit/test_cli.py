import json

from typer.testing import CliRunner

from emot import __version__
from emot.cli import app

runner = CliRunner()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0 and __version__ in result.output


def test_ask_offline_answers_from_a_tracks_file(tmp_path):
    rows = [
        {"frame": f, "tracks": [{"id": 1, "xyxy": [10 + 8 * f, 100, 30 + 8 * f, 200], "conf": 0.9}]}
        for f in range(1, 61)
    ]
    path = tmp_path / "tracks.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    result = runner.invoke(
        app,
        [
            "ask",
            str(path),
            "How many people are there?",
            "--offline",
            "--fps",
            "30",
            "--width",
            "640",
            "--height",
            "360",
        ],
    )
    assert result.exit_code == 0 and "1 distinct" in result.output


def test_results_command_reports_missing_data(tmp_path):
    result = runner.invoke(app, ["results", "--results-dir", str(tmp_path)])
    assert result.exit_code == 1

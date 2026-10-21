import json
from types import SimpleNamespace

import numpy as np
import pytest

from emot.agent import TOOL_SCHEMAS, TrackStore, ask_claude, ask_offline


def _store() -> TrackStore:
    """10 s @ 10 fps, 1000x500 frame. Track 1 walks left->right across the middle;
    track 2 stands in the left half; track 3 is only visible for the first 2 seconds."""
    frames = {}
    for f in range(1, 101):
        tracks = {
            1: np.array([10 + 9.5 * f, 200, 30 + 9.5 * f, 300]),
            2: np.array([100, 200, 120, 300]),
        }
        if f <= 20:
            tracks[3] = np.array([700, 200, 720, 300])
        frames[f] = tracks
    return TrackStore(frames, fps=10.0, width=1000, height=500)


def test_count_people_respects_window_and_region():
    s = _store()
    assert s.count_people()["unique_people"] == 3
    assert s.count_people(start_s=3, end_s=10)["unique_people"] == 2  # track 3 left at 2 s
    left = s.count_people(region="left")
    assert left["unique_people"] == 2  # tracks 1 (early) and 2
    assert s.count_people(region="right")["unique_people"] == 2  # tracks 1 (late) and 3


def test_line_crossings_counts_the_walker_once():
    r = _store().line_crossings(0.5, 0, 0.5, 1)
    assert r["total"] == 1 and r["first_events"][0]["track_id"] == 1


def test_dwell_and_summary_tools():
    s = _store()
    dwell = s.dwell_time("left", top_k=1)
    assert dwell["longest"][0]["track_id"] == 2 and dwell["longest"][0]["seconds"] == pytest.approx(
        10
    )
    assert s.scene_summary()["peak_people"] == 3


def test_tools_report_bad_arguments_instead_of_raising():
    s = _store()
    assert "error" in s.call("count_people", {"region": "narnia"})
    assert "error" in s.call("nope", {})
    assert "error" in s.call("track_info", {"track_id": 99})


def test_offline_planner_routes_common_questions():
    s = _store()
    assert "3 distinct" in ask_offline(s, "How many people are there?").text
    crossing = ask_offline(s, "How many people crossed the vertical line?")
    assert crossing.tool_calls[0].name == "line_crossings" and "1 crossings" in crossing.text
    assert (
        ask_offline(s, "How long did people stay in the left half?").tool_calls[0].name
        == "dwell_time"
    )
    assert ask_offline(s, "tell me about track 2").tool_calls[0].name == "track_info"
    assert ask_offline(s, "what is the meaning of life").tool_calls == []
    assert (
        ask_offline(s, "how many people in the last 5 seconds").tool_calls[0].arguments["start_s"]
        == 5.0
    )


def test_tool_schemas_are_valid_json_schema_objects():
    for tool in TOOL_SCHEMAS:
        assert tool["input_schema"]["type"] == "object"
        assert set(tool["input_schema"].get("required", [])) <= set(
            tool["input_schema"]["properties"]
        )
    json.dumps(TOOL_SCHEMAS)


class _FakeClaude:
    """Scripted stand-in for `anthropic.Anthropic()`: one tool call, then an answer."""

    def __init__(self):
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        if len(self.requests) == 1:
            block = SimpleNamespace(
                type="tool_use", id="toolu_1", name="count_people", input={"region": "left"}
            )
            return SimpleNamespace(stop_reason="tool_use", content=[block], usage=usage)
        text = SimpleNamespace(type="text", text="Two people were in the left half.")
        return SimpleNamespace(stop_reason="end_turn", content=[text], usage=usage)


def test_claude_loop_executes_tools_and_returns_trace():
    client = _FakeClaude()
    answer = ask_claude(_store(), "who is on the left?", client=client)
    assert answer.text == "Two people were in the left half."
    assert answer.turns == 2 and answer.tool_calls[0].result["unique_people"] == 2
    second = client.requests[1]["messages"]
    assert second[-1]["role"] == "user"
    result = second[-1]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_1"
    assert json.loads(result["content"])["unique_people"] == 2
    assert client.requests[0]["model"] == "claude-opus-5-5"
    assert "tool_choice" not in client.requests[0] and "thinking" not in client.requests[0]


def test_claude_loop_surfaces_refusals():
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **_: SimpleNamespace(stop_reason="refusal", content=[], usage=None)
        )
    )
    answer = ask_claude(_store(), "x", client=client)
    assert answer.refused

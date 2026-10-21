"""Offline question router: no LLM, no network.

A small intent grammar (counts, crossings, dwell time, crowding, busiest
moments, single tracks, summary) maps a question onto the *same* tools the
Claude agent uses and fills a sentence template from the result. It exists so
the CLI, the CI tests and the web demo answer common questions without an API
key -- and as an honest baseline for what the LLM adds (free-form phrasing,
multi-step questions, unseen intents).
"""

from __future__ import annotations

import re

from emot.agent.agent import AgentAnswer, ToolCall
from emot.agent.tools import REGION_PRESETS, TrackStore

_REGIONS = "|".join(sorted(set(REGION_PRESETS) - {"full"}))
_NUM = r"(\d+(?:\.\d+)?)"


def _region(q: str) -> str | None:
    m = re.search(
        rf"\b(?:in|on|at|inside|within)?\s*the\s+({_REGIONS})(?:\s+(?:half|side|part|area|region))?",
        q,
    )
    return m.group(1) if m else None


def _window(q: str) -> tuple[float | None, float | None]:
    m = re.search(rf"between\s+{_NUM}\s*(?:s|sec|seconds)?\s*(?:and|to|-)\s*{_NUM}", q)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(rf"first\s+{_NUM}\s*(?:s|sec|seconds)", q)
    if m:
        return 0.0, float(m.group(1))
    return None, None  # "last N seconds" needs the duration: resolved by the caller


def ask_offline(store: TrackStore, question: str) -> AgentAnswer:
    q = question.lower().strip().rstrip("?")
    calls: list[ToolCall] = []

    def call(name: str, **kwargs):
        result = store.call(name, kwargs)
        calls.append(ToolCall(name, kwargs, result))
        return result

    start, end = _window(q)
    last = re.search(rf"last\s+{_NUM}\s*(?:s|sec|seconds)", q)
    if last:
        start, end = max(0.0, store.duration_s - float(last.group(1))), None
    region = _region(q)

    if m := re.search(r"\b(?:track|id|person)\s*#?(\d+)\b", q):
        r = call("track_info", track_id=int(m.group(1)))
        text = (
            r["error"]
            if "error" in r
            else f"Track {r['track_id']} is visible from {r['first_seen_s']} s to {r['last_seen_s']} s "
            f"({r['frames_visible']} frames), moving at a median "
            f"{r.get('median_speed_bh_per_s', 0)} body-heights/s."
        )
    elif re.search(r"\b(cross|crossed|crossing|enter|entered|exit|exited|pass|passed)\b", q):
        horizontal = "horizontal" in q
        coords = (
            dict(x1=0, y1=0.5, x2=1, y2=0.5) if horizontal else dict(x1=0.5, y1=0, x2=0.5, y2=1)
        )
        r = call("line_crossings", **coords, start_s=start, end_s=end)
        which = "horizontal middle" if horizontal else "vertical middle"
        text = (
            f"{r['total']} crossings of the {which} line: {r['a_to_b']} from side A to B and "
            f"{r['b_to_a']} from B to A."
        )
    elif re.search(r"\b(how long|dwell|stay|stayed|linger|spent)\b", q):
        r = call("dwell_time", region=region or "center")
        text = (
            f"{r['people_who_entered']} people entered the {region or 'center'} region; mean stay "
            f"{r['mean_dwell_s']} s, longest {r['max_dwell_s']} s."
        )
    elif re.search(r"\b(crowd|crowded|packed)\b", q):
        n = re.search(r"(\d+)\s*(?:or more|\+)?\s*(?:people|persons|pedestrians)", q)
        r = call("crowding", threshold=int(n.group(1)) if n else 10, region=region)
        text = f"{r['count']} crowding episode(s)" + (
            ": "
            + ", ".join(
                f"{e['start_s']}-{e['end_s']} s (peak {e['peak']})" for e in r["episodes"][:3]
            )
            if r["count"]
            else "."
        )
    elif re.search(r"\b(busiest|peak|most people|rush|busy)\b", q):
        r = call("busiest_moments", top_k=3, region=region)
        top = r["moments"][0] if r["moments"] else None
        text = (
            f"The busiest second is {top['second']} s with ~{top['mean_people']} people "
            f"(peak {top['peak']})."
            if top
            else "No people were tracked."
        )
    elif re.search(r"\b(how many|count|number of)\b", q):
        r = call("count_people", start_s=start, end_s=end, region=region)
        text = (
            f"{r['unique_people']} distinct people (track IDs); at most {r['max_simultaneous']} "
            f"visible at once, {r['mean_simultaneous']} on average."
        )
    elif re.search(r"\b(summary|summarize|overview|describe|what happens)\b", q):
        r = call("scene_summary")
        text = (
            f"{r['duration_s']} s of video with {r['unique_tracks']} distinct tracks; "
            f"{r['mean_people_per_frame']} people per frame on average, peaking at "
            f"{r['peak_people']} at {r['peak_time_s']} s."
        )
    else:
        text = (
            "I can answer counts, line crossings, dwell time, crowding, busiest moments, "
            "single tracks and a summary offline; use the Claude agent for open-ended questions."
        )
    return AgentAnswer(text=text, tool_calls=calls, turns=1, model="offline-planner", offline=True)

"""Claude tool-use agent over a `TrackStore` (manual agentic loop).

The model is given the analytics tools and the video's metadata, calls
whichever tools it needs (several per turn, in parallel when independent),
and writes the answer from their results. Every tool call and result is kept
in the returned trace so an answer can be audited number by number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from emot.agent.tools import TOOL_SCHEMAS, TrackStore

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """\
You answer questions about a video in which people were detected and tracked. You can \
only see the video through the provided tools, which return exact, deterministic \
statistics computed from the tracking output.

- Never estimate or compute counts yourself: call a tool and report what it returned.
- Coordinates are normalised to 0-1 with the origin at the top-left; (0.5, 0) to (0.5, 1) \
is a vertical line down the middle of the frame.
- Track IDs are anonymous, session-local labels, not identities. A person who leaves and \
re-enters may receive a new ID, so "unique people" is an upper bound; say so when it matters.
- For line crossings, side A / side B are defined by the line's orientation; describe the \
direction in plain words (e.g. "left to right") only when you can derive it from the \
coordinates you passed, otherwise report A->B / B->A.
- If a question cannot be answered from the available tools, say what is missing instead \
of guessing. Keep answers short and lead with the number.
"""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class AgentAnswer:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    turns: int = 0
    model: str = ""
    offline: bool = False
    refused: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


def _context_line(store: TrackStore) -> str:
    return (
        f"Video: {store.width}x{store.height} px, {store.fps:g} fps, "
        f"{store.duration_s:.1f} s, {len(store.trajectories)} distinct tracks."
    )


def ask_claude(
    store: TrackStore,
    question: str,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    effort: str = "low",
    max_turns: int = 8,
) -> AgentAnswer:
    """Runs the tool-use loop until the model answers (or `max_turns` is hit)."""
    if client is None:
        import anthropic  # optional dependency: pip install "emot[agent]"

        client = anthropic.Anthropic()

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"{_context_line(store)}\n\nQuestion: {question}"}
    ]
    answer = AgentAnswer(text="", model=model)

    for turn in range(1, max_turns + 1):
        response = client.messages.create(
            model=model,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            output_config={"effort": effort},
            messages=messages,
        )
        answer.turns = turn
        usage = getattr(response, "usage", None)
        answer.input_tokens += getattr(usage, "input_tokens", 0) or 0
        answer.output_tokens += getattr(usage, "output_tokens", 0) or 0

        if response.stop_reason == "refusal":
            answer.refused = True
            answer.text = "The model declined to answer this request."
            return answer

        if response.stop_reason != "tool_use":
            answer.text = "".join(b.text for b in response.content if b.type == "text").strip()
            return answer

        # Pass the assistant turn back unchanged (thinking blocks included), then answer
        # *all* tool calls of the turn in one user message.
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = store.call(block.name, dict(block.input))
            answer.tool_calls.append(ToolCall(block.name, dict(block.input), result))
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                    "is_error": "error" in result,
                }
            )
        messages.append({"role": "user", "content": results})

    answer.text = "Stopped after reaching the maximum number of tool-use turns."
    return answer

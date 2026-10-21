"""Natural-language questions over tracking output: a typed tool layer that
does the counting, and a Claude tool-use agent (or an offline planner) that
decides which tools to call."""

from emot.agent.agent import AgentAnswer, ask_claude
from emot.agent.planner import ask_offline
from emot.agent.tools import TOOL_SCHEMAS, TrackStore

__all__ = ["TOOL_SCHEMAS", "AgentAnswer", "TrackStore", "ask_claude", "ask_offline"]

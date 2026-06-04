"""Core domain types. Pure data, no framework/IO dependencies.

Everything inward of the adapters layer speaks in these types. FastAPI DTOs,
LangGraph state, and provider SDK objects are translated to/from these at the
boundary so the domain never sees a vendor type.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    # Present on assistant messages that requested tools.
    tool_calls: tuple["ToolCall", ...] = ()
    # Present on tool-role messages: which call this result answers.
    tool_call_id: str | None = None
    name: str | None = None  # tool name, for tool-role messages


@dataclass(frozen=True)
class ToolCall:
    """A model's request to invoke a tool. `id` ties a result back to a call."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ToolSpec:
    """JSON-schema description of a tool, handed to the model so it can call it."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass
class LLMResponse:
    """A single model generation. Either has text, tool calls, or both."""
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    # Raw usage for the cost tracker; provider-shaped, opaque to the domain.
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None

    @property
    def wants_tools(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def token_usage(self) -> tuple[int, int]:
        """Return (input_tokens, output_tokens) normalized across provider formats."""
        input_t = self.usage.get("input_tokens", self.usage.get("prompt_tokens", 0))
        output_t = self.usage.get("output_tokens", self.usage.get("completion_tokens", 0))
        return input_t, output_t


@dataclass
class AgentState:
    """Mutable working state for one agent run. The ReAct loop owns this."""
    messages: list[Message] = field(default_factory=list)
    # Hard ceiling on think->act iterations. Critical with small models that
    # can loop forever; the loop treats hitting this as a failure, not success.
    max_iterations: int = 8
    iteration: int = 0
    scratch: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    output: str
    state: AgentState
    stopped_reason: str  # "final_answer" | "max_iterations" | "error"


@dataclass(frozen=True)
class AgentEvent:
    """A single streaming event emitted during an agent run, serialized to SSE."""
    type: str  # "token" | "tool_call" | "tool_result" | "interrupt" | "final" | "error" | "thinking" (handle_rejection only)
    text: str = ""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    is_error: bool = False
    stopped_reason: str = ""

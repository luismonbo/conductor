"""What may leave the server for a browser (SL-3).

Everything a browser receives about a run passes through here: the SSE frames
(to_browser), a thread's transcript (transcript_for_browser) and the text of a
failed run (error_event). Deny by default: an exception never becomes browser
text, tool arguments and results reach a browser only where the deployment
allows them (show_tool_details), and an event type this module does not know
is dropped. #8 adds its sub-agent rule to to_browser.
See docs/superpowers/specs/2026-10-13-egress-hardening-design.md.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from harness.core.llm.errors import ContentFiltered
from harness.core.types import AgentEvent, Message, Role

CONTENT_FILTERED_TEXT = (
    "I can't respond to that message. Your next message will start a new conversation."
)
UNAVAILABLE_TEXT = "The assistant is unavailable right now. Please try again later."

# Event types a browser receives unchanged. Tool events are handled below;
# any other type is dropped.
_PASS_THROUGH = frozenset({"token", "final", "error", "thinking", "interrupt"})


def error_event(exc: BaseException) -> AgentEvent:
    """A failed run as a browser sees it: a fixed sentence, never the exception."""
    if isinstance(exc, ContentFiltered):
        return AgentEvent(
            type="error", text=CONTENT_FILTERED_TEXT, stopped_reason="content_filter"
        )
    return AgentEvent(type="error", text=UNAVAILABLE_TEXT, stopped_reason="error")


def to_browser(event: AgentEvent, *, show_tool_details: bool) -> AgentEvent | None:
    """The event as a browser may see it, or None when it must not see it at all.

    Without tool details a browser still gets each tool event's name, call id
    and error flag (enough to show "Searching documents…"); the fields are
    blanked, not removed, so the wire schema the UIs parse stays the same.
    """
    if event.type in _PASS_THROUGH:
        return event
    if event.type == "tool_call":
        return event if show_tool_details else dataclasses.replace(event, args={})
    if event.type == "tool_result":
        return event if show_tool_details else dataclasses.replace(event, text="")
    return None


def transcript_for_browser(
    messages: list[Message], *, show_tool_details: bool
) -> list[dict[str, Any]]:
    """A thread's messages as GET /threads/{id} returns them: no system prompt,
    and the same tool-detail rule as the stream."""
    return [
        {
            "role": m.role.value,
            "content": "" if m.role == Role.TOOL and not show_tool_details else m.content,
            "tool_calls": [
                {
                    "name": tc.name,
                    "args": tc.arguments if show_tool_details else {},
                    "call_id": tc.id,
                }
                for tc in (m.tool_calls or ())
            ],
            "tool_call_id": m.tool_call_id or "",
            "name": m.name or "",
        }
        for m in messages
        if m.role != Role.SYSTEM
    ]

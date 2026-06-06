"""Server-Sent Events (SSE) frame formatting and parsing.

This implements the wire format used by the EventSource / SSE protocol
(https://html.spec.whatwg.org/multipage/server-sent-events.html) as adopted
by OpenAI-style streaming chat completions:

* A *frame* is one or more ``field: value`` lines followed by a blank line.
* The ``data`` field may span multiple lines; each physical line is emitted as
  its own ``data:`` line and the parser rejoins them with ``"\n"``.
* The optional ``event`` field names the event type (defaults to ``"message"``).
* A leading space after the colon is part of the format and is stripped on
  parse, so ``"data: hi"`` and ``"data:hi"`` both decode to ``"hi"``.

Part of LLM Streaming by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Sentinel payload that signals the end of an OpenAI-style token stream.
DONE = "[DONE]"


class SSEEvent(BaseModel):
    """A single parsed Server-Sent Event.

    Attributes:
        event: The event type. ``"message"`` is the SSE default when the source
            omits an explicit ``event:`` field.
        data: The decoded payload, with multi-line ``data:`` fields rejoined by
            newlines.
    """

    event: str = Field(default="message")
    data: str = Field(default="")


def format_sse(data: str, event: str | None = None) -> str:
    """Format ``data`` (and optional ``event``) into one SSE frame.

    Multi-line ``data`` is split on newlines and each line is emitted as its own
    ``data:`` field, per the SSE specification. The frame is terminated by a
    blank line (``"\\n\\n"``).

    Args:
        data: The payload to send. May contain newlines.
        event: Optional event type. When given, an ``event:`` line is prepended.

    Returns:
        A complete SSE frame string ending in a blank line, for example
        ``"event: token\\ndata: hello\\n\\n"``.
    """
    lines: list[str] = []
    if event is not None:
        lines.append(f"event: {event}")
    # An empty payload still needs one (empty) data line so the frame is valid.
    for line in data.split("\n"):
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


def parse_sse(text: str) -> list[SSEEvent]:
    """Parse a stream of SSE frames back into :class:`SSEEvent` objects.

    Frames are separated by blank lines. Within a frame, repeated ``data:``
    lines are concatenated with newlines. Lines beginning with ``":"`` are
    treated as comments and ignored. Trailing partial frames without a blank
    line terminator are still parsed.

    Args:
        text: The raw SSE byte stream decoded to ``str``.

    Returns:
        The parsed events in order of appearance.
    """
    events: list[SSEEvent] = []
    event_type: str | None = None
    data_lines: list[str] = []
    have_fields = False

    def flush() -> None:
        nonlocal event_type, data_lines, have_fields
        if have_fields:
            events.append(
                SSEEvent(
                    event=event_type if event_type is not None else "message",
                    data="\n".join(data_lines),
                )
            )
        event_type = None
        data_lines = []
        have_fields = False

    # Normalize line endings so CRLF streams parse identically to LF streams.
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw_line == "":
            flush()
            continue
        if raw_line.startswith(":"):
            # Comment line — ignored, but it does keep the frame "open".
            continue
        field, _, value = raw_line.partition(":")
        # A single leading space after the colon is stripped per the SSE spec.
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_type = value
            have_fields = True
        elif field == "data":
            data_lines.append(value)
            have_fields = True
        # Other fields (id, retry, ...) are accepted but not surfaced here.
        else:
            have_fields = True
    flush()
    return events

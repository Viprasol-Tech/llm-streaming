from __future__ import annotations

from llm_streaming.sse import DONE, SSEEvent, format_sse, parse_sse


def test_format_sse_basic_frame() -> None:
    assert format_sse("hello", event="token") == "event: token\ndata: hello\n\n"


def test_format_sse_without_event() -> None:
    assert format_sse("hello") == "data: hello\n\n"


def test_format_sse_multiline_data_splits_each_line() -> None:
    frame = format_sse("line one\nline two", event="msg")
    assert frame == "event: msg\ndata: line one\ndata: line two\n\n"


def test_format_parse_round_trip() -> None:
    frame = format_sse("hello world", event="token")
    events = parse_sse(frame)
    assert events == [SSEEvent(event="token", data="hello world")]


def test_parse_multiline_data_rejoined_with_newline() -> None:
    frame = format_sse("alpha\nbeta\ngamma", event="msg")
    (event,) = parse_sse(frame)
    assert event.event == "msg"
    assert event.data == "alpha\nbeta\ngamma"


def test_parse_multiple_frames() -> None:
    stream = format_sse("a", event="token") + format_sse("b", event="token")
    events = parse_sse(stream)
    assert [e.data for e in events] == ["a", "b"]


def test_parse_defaults_event_to_message() -> None:
    (event,) = parse_sse("data: hi\n\n")
    assert event.event == "message"
    assert event.data == "hi"


def test_parse_strips_single_leading_space_only() -> None:
    # "data:  x" -> one leading space stripped, leaving " x".
    (event,) = parse_sse("data:  x\n\n")
    assert event.data == " x"


def test_parse_ignores_comment_lines() -> None:
    (event,) = parse_sse(": this is a comment\ndata: payload\n\n")
    assert event.data == "payload"


def test_parse_handles_crlf_line_endings() -> None:
    (event,) = parse_sse("event: token\r\ndata: hi\r\n\r\n")
    assert event == SSEEvent(event="token", data="hi")


def test_parse_trailing_frame_without_blank_line() -> None:
    (event,) = parse_sse("data: tail")
    assert event.data == "tail"


def test_done_sentinel_round_trips() -> None:
    frame = format_sse(DONE, event="done")
    (event,) = parse_sse(frame)
    assert event.event == "done"
    assert event.data == DONE

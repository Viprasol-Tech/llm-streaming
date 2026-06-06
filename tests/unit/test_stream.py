from __future__ import annotations

from llm_streaming.providers import FakeLLM, echo_llm
from llm_streaming.sse import DONE, parse_sse
from llm_streaming.stream import collect, stream_completion, stream_tokens, tokenize


def test_tokenize_is_lossless() -> None:
    text = "hello  world from Viprasol"
    assert "".join(tokenize(text)) == text


def test_tokenize_token_count() -> None:
    # Three space-delimited tokens (each trailing space stays attached).
    assert tokenize("a b c") == ["a ", "b ", "c"]


def test_stream_tokens_emits_done_sentinel_last() -> None:
    frames = list(stream_tokens(["a ", "b ", "c"]))
    # 3 token frames + 1 sentinel frame.
    assert len(frames) == 4
    events = parse_sse("".join(frames))
    assert events[-1].event == "done"
    assert events[-1].data == DONE


def test_stream_tokens_done_sentinel_present_in_stream() -> None:
    stream = "".join(stream_tokens(["x"]))
    assert DONE in stream


def test_collect_reassembles_original_text() -> None:
    text = "Hello from Viprasol Tech! Streaming works."
    frames = stream_tokens(tokenize(text))
    assert collect(frames) == text


def test_collect_accepts_concatenated_stream_string() -> None:
    text = "one two three"
    stream = "".join(stream_tokens(tokenize(text)))
    assert collect(stream) == text


def test_collect_drops_done_sentinel() -> None:
    frames = list(stream_tokens(tokenize("abc")))
    assert DONE not in collect(frames)


def test_stream_completion_round_trip_with_fake_llm() -> None:
    response = "This is a deterministic fake completion."
    provider = FakeLLM(responses={"prompt": response})
    frames = list(stream_completion("prompt", provider))
    assert collect(frames) == response


def test_stream_completion_with_echo_provider() -> None:
    prompt = "echo me please"
    assert collect(stream_completion(prompt, echo_llm)) == prompt


def test_fake_llm_uses_default_when_prompt_unknown() -> None:
    provider = FakeLLM(default="fallback text")
    assert provider("anything") == "fallback text"

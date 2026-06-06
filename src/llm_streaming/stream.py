"""Token streaming helpers built on the SSE wire format.

* :func:`tokenize` splits text into streamable tokens (whitespace-preserving).
* :func:`stream_tokens` yields one SSE frame per token, then a final
  ``[DONE]`` sentinel frame — mirroring OpenAI-style streaming.
* :func:`stream_completion` runs an :data:`LLMProvider` and streams its output.
* :func:`collect` reassembles the original text from parsed frames.

Part of LLM Streaming by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from llm_streaming.providers import LLMProvider
from llm_streaming.sse import DONE, SSEEvent, format_sse, parse_sse


def tokenize(text: str) -> list[str]:
    """Split ``text`` into tokens that preserve all whitespace.

    Each token carries its trailing whitespace so that concatenating every
    token exactly reconstructs the input. This guarantees a lossless
    round-trip through :func:`stream_tokens` and :func:`collect`.

    Args:
        text: The text to tokenize.

    Returns:
        A list of tokens whose concatenation equals ``text``.
    """
    tokens: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        if char == " ":
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def stream_tokens(tokens: Iterable[str], event: str = "token") -> Iterator[str]:
    """Yield one SSE frame per token, ending with a ``[DONE]`` sentinel frame.

    Args:
        tokens: The tokens to stream, in order.
        event: The SSE event type used for token frames. The terminal sentinel
            is emitted with the ``done`` event type and a ``[DONE]`` payload.

    Yields:
        SSE frame strings, one per token, followed by the sentinel frame.
    """
    for token in tokens:
        yield format_sse(token, event=event)
    yield format_sse(DONE, event="done")


def stream_completion(
    prompt: str,
    provider: LLMProvider,
    event: str = "token",
) -> Iterator[str]:
    """Run ``provider`` on ``prompt`` and stream its completion as SSE frames.

    Args:
        prompt: The prompt to send to the provider.
        provider: Any ``Callable[[str], str]`` (see :data:`LLMProvider`).
        event: The SSE event type for token frames.

    Yields:
        SSE frame strings for each token, then a ``[DONE]`` sentinel frame.
    """
    completion = provider(prompt)
    yield from stream_tokens(tokenize(completion), event=event)


def collect(frames: Iterable[str] | str) -> str:
    """Reassemble the full completion text from streamed SSE frames.

    Accepts either an iterable of frame strings (as produced by
    :func:`stream_tokens`) or a single concatenated SSE stream. The terminal
    ``[DONE]`` sentinel is dropped.

    Args:
        frames: An iterable of SSE frame strings, or one concatenated stream.

    Returns:
        The reassembled text, identical to the streamed source.
    """
    raw = frames if isinstance(frames, str) else "".join(frames)
    events: list[SSEEvent] = parse_sse(raw)
    return "".join(e.data for e in events if not (e.event == "done" and e.data == DONE))

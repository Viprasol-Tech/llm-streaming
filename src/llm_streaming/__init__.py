"""LLM token streaming utilities — SSE formatting and parsing.

Part of LLM Streaming by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from llm_streaming.providers import FakeLLM, LLMProvider, echo_llm
from llm_streaming.sse import DONE, SSEEvent, format_sse, parse_sse
from llm_streaming.stream import collect, stream_tokens

__version__ = "0.1.0"

__all__ = [
    "DONE",
    "FakeLLM",
    "LLMProvider",
    "SSEEvent",
    "__version__",
    "collect",
    "echo_llm",
    "format_sse",
    "parse_sse",
    "stream_tokens",
]

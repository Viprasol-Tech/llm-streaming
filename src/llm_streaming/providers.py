"""Pluggable LLM providers.

The library never calls a real model directly. Instead it accepts an
:data:`LLMProvider` — any ``Callable[[str], str]`` — so callers can plug in
OpenAI, a local model, or anything else. A deterministic :class:`FakeLLM` and an
:func:`echo_llm` are shipped so the demo and tests run fully offline with no API
keys and no network.

Part of LLM Streaming by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Callable

#: A provider maps a prompt string to a completion string. Swap in a real model
#: (e.g. a thin wrapper around the OpenAI SDK) wherever an ``LLMProvider`` is
#: expected.
LLMProvider = Callable[[str], str]


def echo_llm(prompt: str) -> str:
    """A trivial offline provider that echoes its prompt back.

    Args:
        prompt: The input prompt.

    Returns:
        The prompt unchanged.
    """
    return prompt


class FakeLLM:
    """A deterministic, offline LLM provider for demos and tests.

    Given a fixed bank of canned responses keyed by prompt, it returns the
    matching response, falling back to a deterministic default. This makes
    streaming behaviour fully reproducible without any network access.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "This is a fake streamed response from Viprasol Tech.",
    ) -> None:
        """Initialise the fake provider.

        Args:
            responses: Optional mapping of exact prompt to canned response.
            default: Response used when a prompt is not in ``responses``.
        """
        self._responses = dict(responses) if responses else {}
        self._default = default

    def __call__(self, prompt: str) -> str:
        """Return the canned response for ``prompt`` (or the default).

        Args:
            prompt: The input prompt.

        Returns:
            The canned or default completion.
        """
        return self._responses.get(prompt, self._default)

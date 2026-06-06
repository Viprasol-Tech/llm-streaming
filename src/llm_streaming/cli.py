"""Command-line interface for LLM Streaming.

``llm-streaming demo`` streams a sentence token-by-token through the built-in
offline fake provider, printing each SSE frame and then the reassembled text —
no API keys, no network.

Part of LLM Streaming by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import typer
from rich.console import Console

from llm_streaming import __version__
from llm_streaming.providers import FakeLLM
from llm_streaming.stream import collect, stream_completion

app = typer.Typer(add_completion=False, help="LLM Streaming — SSE utilities by Viprasol Tech.")
console = Console()

_DEFAULT_PROMPT = "Stream this sentence token by token over SSE."


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"llm-streaming [bold cyan]{__version__}[/] - by Viprasol Tech")


@app.command()
def demo(
    prompt: str = typer.Option(_DEFAULT_PROMPT, help="Prompt to stream."),
    response: str = typer.Option(
        "Hello from Viprasol Tech! This response is streamed token by token.",
        help="Canned completion the offline fake provider returns.",
    ),
) -> None:
    """Stream a completion token-by-token and print each SSE frame."""
    provider = FakeLLM(responses={prompt: response})
    frames = list(stream_completion(prompt, provider))

    console.print(f"Prompt:   [bold]{prompt}[/]")
    console.print(f"Frames:   {len(frames)} (including the [DONE] sentinel)\n")
    console.print("[dim]--- SSE wire output ---[/]")
    for frame in frames:
        # Show the raw frame with visible line breaks for clarity.
        console.print(frame.replace("\n", "\\n"))

    reassembled = collect(frames)
    console.print("\n[dim]--- reassembled via collect() ---[/]")
    console.print(f"[bold green]{reassembled}[/]")
    if reassembled == response:
        console.print("\n[bold]OK[/] round-trip exact: collect() == original completion")


if __name__ == "__main__":
    app()

import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from .core import run_agent
from .model import create_model
from .tools import Tool, tool

console = Console()

app = typer.Typer(help="Minimal sub-agent CLI")


@tool(
    name="read_file",
    description="Read contents of a file",
    inputs={"path": {"type": "string", "description": "Path to file to read"}},
)
def read_file(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


@tool(
    name="list_directory",
    description="List files in a directory",
    inputs={"path": {"type": "string", "description": "Directory path to list"}},
)
def list_directory(path: str = ".") -> str:
    try:
        import os

        entries = os.listdir(path)
        return "\n".join(entries) if entries else "Empty directory"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="web_fetch",
    description="Fetch content from a URL",
    inputs={"url": {"type": "string", "description": "URL to fetch"}},
)
def web_fetch(url: str) -> str:
    try:
        import requests

        resp = requests.get(url, timeout=10)
        return resp.text[:5000]
    except Exception as e:
        return f"Error: {e}"


DEFAULT_TOOLS = [read_file, list_directory, web_fetch]


@app.command()
def main(
    goal: str = typer.Argument(..., help="Natural language task"),
    mode: str = typer.Option("plan", help="Agent mode: single_step or plan"),
    max_steps: int = typer.Option(5, help="Max steps for plan mode"),
    model_type: str = typer.Option("openai", help="Model type: openai, litellm"),
    model_id: str = typer.Option("gpt-4o-mini", help="Model ID"),
    tool: list[str] = typer.Option(
        ["read_file", "list_directory", "web_fetch"],
        "--tool",
        help="Enable specific tools",
    ),
    api_base: Optional[str] = typer.Option(None, help="API base URL"),
):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY not set[/red]")
        raise typer.Exit(1)

    enabled_tools = []
    available = {t.name: t for t in DEFAULT_TOOLS}
    for t in tool:
        if t in available:
            enabled_tools.append(available[t])
        else:
            console.print(f"[yellow]Warning: Unknown tool '{t}'[/yellow]")

    if not enabled_tools:
        console.print("[yellow]No tools enabled. Using default tools.[/yellow]")
        enabled_tools = DEFAULT_TOOLS

    console.print(Panel(f"[bold]Task:[/bold] {goal}\n[bold]Mode:[/bold] {mode}"))

    try:
        model_kwargs = {}
        if api_base:
            model_kwargs["api_base"] = api_base

        result = run_agent(
            goal=goal,
            mode=mode,
            tools=enabled_tools,
            model_type=model_type,
            model_id=model_id,
            max_steps=max_steps,
            **model_kwargs,
        )

        if result.error:
            console.print(f"[red]Error:[/red] {result.error}")

        console.print(Panel(result.output, title="Result"))

        if result.steps:
            console.print(f"\n[dim]Completed in {len(result.steps)} step(s)[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

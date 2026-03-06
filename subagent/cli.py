import os
import re
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .core import AgentManager, AgentMode, BUILT_IN_AGENTS, Permission, run_agent
from .model import create_model
from .tools import Tool, tool

console = Console()

app = typer.Typer(help="Minimal sub-agent CLI")

SAFE_COMMANDS = {
    "ls",
    "dir",
    "pwd",
    "cat",
    "head",
    "tail",
    "grep",
    "find",
    "echo",
    "date",
    "whoami",
    "git status",
    "git log",
    "git diff",
    "git branch",
    "git checkout",
    "git clone",
    "npm install",
    "npm run",
    "yarn",
    "pip install",
    "pip list",
    "python",
    "python3",
    "curl",
    "wget",
    "mkdir",
    "touch",
    "rm",
    "cp",
    "mv",
    "chmod",
    "tar",
    "zip",
    "unzip",
}

DENIED_COMMANDS = {
    "rm -rf",
    "mkfs",
    "dd",
    ":(){:|:&};:",
    "chown",
    "useradd",
    "passwd",
    "sudo",
    "su ",
    "bash -i",
    "/dev/",
    "proc/",
    "sys/",
    "init",
    "systemctl",
    "service",
}


def is_command_safe(cmd: str) -> bool:
    cmd_lower = cmd.lower().strip()
    for denied in DENIED_COMMANDS:
        if denied in cmd_lower:
            return False
    return True


@tool(
    name="read_file",
    description="Read contents of a file",
    inputs={"path": {"type": "string", "description": "Path to file to read"}},
)
def read_file(path: str) -> str:
    try:
        with open(path, "r") as f:
            content = f.read()
            if len(content) > 10000:
                return content[:10000] + "\n\n... (truncated)"
            return content
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
        return "\n".join(sorted(entries)) if entries else "Empty directory"
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


@tool(
    name="bash",
    description="Execute safe shell commands. Only allows read-only and safe commands.",
    inputs={"command": {"type": "string", "description": "Command to execute"}},
)
def bash(command: str) -> str:
    if not is_command_safe(command):
        return f"Error: Command '{command}' is not allowed for safety reasons"
    try:
        import subprocess

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd(),
        )
        output = result.stdout or result.stderr
        if len(output) > 5000:
            return output[:5000] + "\n... (truncated)"
        return output if output else "Command executed successfully (no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="glob",
    description="Find files matching a pattern",
    inputs={
        "pattern": {"type": "string", "description": "Glob pattern (e.g., **/*.py)"}
    },
)
def glob(pattern: str) -> str:
    try:
        import glob as glob_module

        matches = glob_module.glob(pattern, recursive=True)
        return "\n".join(matches[:100]) if matches else "No matches found"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="grep",
    description="Search for patterns in files",
    inputs={
        "pattern": {"type": "string", "description": "Pattern to search for"},
        "path": {"type": "string", "description": "Path to search in (default: .)"},
    },
)
def grep(pattern: str, path: str = ".") -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["grep", "-r", "--line-number", pattern, path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout or result.stderr
        if len(output) > 5000:
            return output[:5000] + "\n... (truncated)"
        return output if output else "No matches found"
    except Exception as e:
        return f"Error: {e}"


DEFAULT_TOOLS = [read_file, list_directory, web_fetch, bash, glob, grep]

ALL_TOOLS = {t.name: t for t in DEFAULT_TOOLS}


@app.command()
def main(
    goal: str = typer.Argument(..., help="Natural language task"),
    mode: str = typer.Option("plan", help="Agent mode: single_step or plan"),
    max_steps: int = typer.Option(5, help="Max steps for plan mode"),
    model_type: str = typer.Option(
        "openai", help="Model type: openai, anthropic, litellm"
    ),
    model_id: str = typer.Option("gpt-4o-mini", help="Model ID"),
    tool: list[str] = typer.Option(
        ["read_file", "list_directory", "web_fetch", "bash"],
        "--tool",
        help="Enable specific tools",
    ),
    agent: str = typer.Option(
        None,
        "--agent",
        help=f"Use built-in agent: {', '.join(BUILT_IN_AGENTS.keys())}",
    ),
    stream: bool = typer.Option(False, "--stream", help="Stream agent output"),
    api_base: Optional[str] = typer.Option(None, help="API base URL"),
    temperature: float = typer.Option(0.7, help="Temperature for model"),
    permissions: Optional[str] = typer.Option(None, help="JSON permissions config"),
    save_state: Optional[str] = typer.Option(
        None, "--save-state", help="Save agent state to file"
    ),
    load_state: Optional[str] = typer.Option(
        None, "--load-state", help="Load agent state from file"
    ),
):
    api_key = os.environ.get("OPENAI_API_KEY")
    if model_type == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        console.print(
            f"[red]Error: API key not set. Set {'OPENAI_API_KEY' if model_type != 'anthropic' else 'ANTHROPIC_API_KEY'}[/red]"
        )
        raise typer.Exit(1)

    enabled_tools = []
    for t in tool:
        if t in ALL_TOOLS:
            enabled_tools.append(ALL_TOOLS[t])
        else:
            console.print(f"[yellow]Warning: Unknown tool '{t}'[/yellow]")

    if not enabled_tools:
        console.print("[yellow]No tools enabled. Using default tools.[/yellow]")
        enabled_tools = DEFAULT_TOOLS

    agent_permissions = {"*": "allow"}
    if agent:
        if agent in BUILT_IN_AGENTS:
            agent_config = BUILT_IN_AGENTS[agent]
            agent_permissions = agent_config["permissions"]
            console.print(f"[dim]Using built-in agent: {agent}[/dim]")
        else:
            console.print(
                f"[yellow]Warning: Unknown agent '{agent}'. Using default permissions.[/yellow]"
            )

    if permissions:
        try:
            import json

            agent_permissions = json.loads(permissions)
        except json.JSONDecodeError:
            console.print(
                "[yellow]Warning: Invalid permissions JSON, using defaults.[/yellow]"
            )

    console.print(
        Panel(
            f"[bold]Task:[/bold] {goal}\n[bold]Mode:[/bold] {mode}\n[bold]Agent:[/bold] {agent or 'default'}"
        )
    )

    try:
        model_kwargs = {"temperature": temperature}
        if api_base:
            model_kwargs["api_base"] = api_base

        result = run_agent(
            goal=goal,
            mode=mode,
            tools=enabled_tools,
            model_type=model_type,
            model_id=model_id,
            max_steps=max_steps,
            agent_name=agent or "default",
            permissions=agent_permissions,
            stream=stream,
            **model_kwargs,
        )

        if stream:
            console.print("[bold]Streaming output:[/bold]")
            for step in result:
                if step.thought:
                    console.print(f"[dim]Thought:[/dim] {step.thought[:200]}")
                if step.action:
                    console.print(f"[cyan]Action:[/cyan] {step.action}")
                if step.observation:
                    console.print(f"[green]Result:[/green] {step.observation[:500]}")
                console.print()
        else:
            if result.error:
                console.print(f"[red]Error:[/red] {result.error}")

            console.print(Panel(result.output, title="Result"))

            if result.steps:
                console.print(f"\n[dim]Completed in {len(result.steps)} step(s)[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("list-agents")
def list_agents():
    """List all available built-in agents."""
    console.print("[bold]Available Agents:[/bold]\n")
    for name, config in BUILT_IN_AGENTS.items():
        console.print(f"  [cyan]{name}[/cyan]")
        console.print(f"    {config['description'][:60]}...")
        console.print(f"    Model: {config['model_id']}")
        console.print()


@app.command("list-tools")
def list_tools():
    """List all available tools."""
    console.print("[bold]Available Tools:[/bold]\n")
    for name, tool in ALL_TOOLS.items():
        console.print(f"  [cyan]{name}[/cyan]")
        console.print(f"    {tool.description}")
        console.print()


if __name__ == "__main__":
    app()

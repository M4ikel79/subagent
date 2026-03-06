# Subagent

A minimal CLI agent designed for use as a sub-agent in larger orchestrations, or as a standalone CLI tool.

## Design Inspired By

This project intentionally borrows patterns from:

- **single-file-agents**: Tiny self-contained CLI agents with single-purpose focus
- **babyagi-2o**: Minimal ReAct planning loop with tool registration
- **smolagents**: Clean Tool abstraction and model-agnostic design
- **OpenCode**: Agent definitions, permissions system (allow/deny/ask), commands
- **Claude Code**: Built-in subagent types (explore, general, code-reviewer, debugger)

## Features

- **Multiple LLM Backends**: OpenAI, Anthropic, LiteLLM
- **Built-in Agents**: `explore`, `general`, `code-reviewer`, `debugger`
- **Permission System**: Allow/deny/ask per tool (like OpenCode)
- **Safe Shell Execution**: Restricted bash tool
- **Streaming Support**: Real-time output
- **Tool Validation**: Input validation before execution
- **Agent Persistence**: Save/restore agent state
- **Token Usage Tracking**: Track input/output tokens

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## CLI Usage

```bash
# Set your API key
export OPENAI_API_KEY=sk-...
# or for Anthropic
export ANTHROPIC_API_KEY=sk-...

# Basic usage with plan mode (default)
subagent "What files are in the current directory?"

# Single step mode (one LLM call)
subagent "Hello, how are you?" --mode single_step

# Custom max steps
subagent "Read all Python files and summarize them" --max-steps 10

# Use Anthropic model
subagent "Your task here" --model-type anthropic --model-id claude-3-5-sonnet-20241022

# Use built-in agent
subagent "Find all Python files in src/" --agent explore

# Stream output
subagent "Your task here" --stream

# Use specific tools
subagent "Read README.md" --tool read_file --tool bash

# List available agents
subagent list-agents

# List available tools
subagent list-tools

# Custom permissions (JSON)
subagent "Task" --permissions '{"read_file": "allow", "bash": "deny"}'
```

## Programmatic Usage

```python
from subagent import run_agent, Tool, tool, AgentManager, Permission, BUILT_IN_AGENTS

# Using the @tool decorator with validation
@tool(name="my_tool", description="Does something useful")
def my_tool(arg: str) -> str:
    return f"Processed: {arg}"

# Run with custom tools
result = run_agent(
    goal="Use my_tool on 'hello'",
    tools=[my_tool],
    mode="plan",
    max_steps=5,
)

print(result.output)
print(f"Steps: {len(result.steps)}")

# Using built-in agents with permissions
result = run_agent(
    goal="Explore the codebase for API endpoints",
    agent_name="explore",  # Uses explore agent's permissions
    model_type="openai",
)

# Using AgentManager for more control
manager = AgentManager()
manager.register_agent(AgentConfig(
    name="my_agent",
    permissions={"read_file": "allow", "bash": "deny"},
    max_steps=10,
))
agent = manager.create_agent("my_agent", model)
result = agent.run("Task")

# Streaming output
for step in run_agent("Task", stream=True):
    print(f"Thought: {step.thought}")
    if step.action:
        print(f"Action: {step.action}")
    if step.observation:
        print(f"Result: {step.observation}")
```

## Available Tools

- `read_file`: Read file contents (with truncation for large files)
- `list_directory`: List directory contents
- `web_fetch`: Fetch URL content
- `bash`: Safe shell execution (restricted commands)
- `glob`: Find files matching pattern
- `grep`: Search patterns in files

## Built-in Agents

| Agent | Description | Permissions |
|-------|-------------|-------------|
| `explore` | Fast codebase exploration | read_file, list_directory, glob, grep, webfetch |
| `general` | General-purpose multi-step tasks | All allowed |
| `code-reviewer` | Code quality analysis | read_file, glob, grep |
| `debugger` | Bug investigation | read_file, bash, grep |

## Architecture

```
subagent/
├── __init__.py      # Exports public API
├── cli.py           # CLI entrypoint
├── core.py          # Agent loop, AgentManager, Permission, AgentState
├── model.py         # LLM abstraction (OpenAI, Anthropic, LiteLLM)
└── tools.py         # Tool definitions with validation
```

## API

### `run_agent(goal, mode, tools, model, max_steps, model_type, model_id, **kwargs)`

Run the agent with a natural language goal.

- **goal**: Natural language task
- **mode**: "single_step" or "plan" (default: "plan")
- **tools**: List of Tool objects
- **model**: LanguageModel instance (optional)
- **max_steps**: Max iterations (default: 5)
- **model_type**: "openai", "anthropic", or "litellm" (default: "openai")
- **model_id**: Model identifier (default: "gpt-4o-mini")
- **agent_name**: Use built-in agent config
- **permissions**: dict of tool permissions
- **stream**: Stream output in real-time

Returns `AgentResult` with:
- `output`: Final answer string
- `steps`: List of AgentStep objects
- `error`: Error message if any
- `usage`: TokenUsage (input/output tokens)
- `agent_name`: Name of agent used

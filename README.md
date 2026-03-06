# Subagent

A minimal CLI agent using Ollama, designed for use as a sub-agent in larger orchestrations, or as a standalone CLI tool.

## Features

- **Ollama Cloud**: Use models from ollama.com (glm-4, llama, etc.)
- **Built-in Agents**: `explore`, `general`, `code-reviewer`, `debugger`
- **Permission System**: Allow/deny per tool
- **Safe Shell Execution**: Restricted bash tool
- **Streaming Support**: Real-time output
- **Tool Validation**: Input validation before execution

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## CLI Usage

```bash
# Set your Ollama API key
export OLLAMA_API_KEY=your_api_key

# Basic usage with plan mode (default)
subagent "What files are in the current directory?"

# Single step mode
subagent "Hello, how are you?" --mode single_step

# Custom model
subagent "Your task" --model-id glm-4

# Custom max steps
subagent "Read files and summarize" --max-steps 10

# Use built-in agent
subagent "Find Python files" --agent explore

# Stream output
subagent "Your task" --stream

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
from subagent import run_agent, Tool, tool

# Using the @tool decorator
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

# Using built-in agents
result = run_agent(
    goal="Explore the codebase",
    agent_name="explore",
)

# Streaming output
for step in run_agent("Task", stream=True):
    print(f"Thought: {step.thought}")
    if step.action:
        print(f"Action: {step.action}")
```

## Available Tools

- `read_file`: Read file contents
- `list_directory`: List directory contents
- `web_fetch`: Fetch URL content
- `bash`: Safe shell execution (restricted)
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
├── core.py          # Agent loop, AgentManager, Permission
├── model.py         # Ollama model
└── tools.py         # Tool definitions
```

## API

### `run_agent(goal, mode, tools, model, max_steps, model_id, **kwargs)`

- **goal**: Natural language task
- **mode**: "single_step" or "plan" (default: "plan")
- **tools**: List of Tool objects
- **model**: LanguageModel instance (optional)
- **max_steps**: Max iterations (default: 5)
- **model_id**: Ollama model ID (default: "glm-4.6")
- **agent_name**: Use built-in agent config
- **permissions**: dict of tool permissions
- **stream**: Stream output in real-time

Returns `AgentResult` with:
- `output`: Final answer string
- `steps`: List of AgentStep objects
- `error`: Error message if any
- `agent_name`: Name of agent used

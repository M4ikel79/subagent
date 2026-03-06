# Subagent

A minimal CLI agent designed for use as a sub-agent in larger orchestrations, or as a standalone CLI tool.

## Design Inspired By

This project intentionally borrows patterns from three codebases:

- **single-file-agents**: Tiny self-contained CLI agents with single-purpose focus
- **babyagi-2o**: Minimal ReAct planning loop with tool registration
- **smolagents**: Clean Tool abstraction and model-agnostic design

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

# Basic usage with plan mode (default)
subagent "What files are in the current directory?"

# Single step mode (one LLM call)
subagent "Hello, how are you?" --mode single_step

# Custom max steps
subagent "Read all Python files and summarize them" --max-steps 10

# Use a different model
subagent "Your task here" --model-id gpt-4o

# Enable specific tools
subagent "Read the file README.md" --tool read_file
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
    goal="Use my_tool to process 'hello'",
    tools=[my_tool],
    mode="plan",
    max_steps=5,
)

print(result.output)
print(f"Steps: {len(result.steps)}")
```

## Available Tools

- `read_file`: Read file contents
- `list_directory`: List directory contents  
- `web_fetch`: Fetch URL content

## Architecture

```
subagent/
├── __init__.py      # Exports public API
├── cli.py           # CLI entrypoint
├── core.py          # Agent loop and types
├── model.py         # LLM abstraction
└── tools.py         # Tool definitions
```

## API

### `run_agent(goal, mode, tools, model, max_steps, model_type, model_id, **kwargs)`

Run the agent with a natural language goal.

- **goal**: Natural language task
- **mode**: "single_step" or "plan" (default: "plan")
- **tools**: List of Tool objects
- **model**: LanguageModel instance (optional, auto-created if not provided)
- **max_steps**: Max iterations in plan mode (default: 5)
- **model_type**: "openai" or "litellm" (default: "openai")
- **model_id**: Model identifier (default: "gpt-4o-mini")

Returns `AgentResult` with:
- `output`: Final answer string
- `steps**: List of AgentStep objects
- **error**: Error message if any

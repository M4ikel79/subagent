from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    inputs: dict[str, dict[str, str]] = field(default_factory=dict)
    output_type: str = "string"
    func: Callable | None = None

    def __call__(self, **kwargs) -> Any:
        if self.func is None:
            raise NotImplementedError(f"Tool {self.name} has no func defined")
        return self.func(**kwargs)

    def to_json(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        k: {"type": v.get("type", "string"), "description": v.get("description", "")}
                        for k, v in self.inputs.items()
                    },
                    "required": list(self.inputs.keys()),
                },
            },
        }


def tool(name: str, description: str, inputs: dict[str, dict[str, str]] | None = None, output_type: str = "string"):
    def decorator(func: Callable) -> Tool:
        import inspect

        sig = inspect.signature(func)
        param_inputs = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            param_inputs[param_name] = {
                "type": "string",
                "description": param.annotation.__doc__ if hasattr(param.annotation, "__doc__") else "",
            }

        inputs_final = inputs or param_inputs
        t = Tool(
            name=name,
            description=description,
            inputs=inputs_final,
            output_type=output_type,
            func=func,
        )
        return t

    return decorator


class ToolCollection:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def list(self) -> list[Tool]:
        return list(self.tools.values())

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

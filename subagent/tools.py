from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    inputs: dict[str, dict[str, str]] = field(default_factory=dict)
    output_type: str = "string"
    func: Callable | None = None
    output_schema: dict | None = None

    def __call__(self, **kwargs) -> Any:
        if self.func is None:
            raise NotImplementedError(f"Tool {self.name} has no func defined")
        return self.func(**kwargs)

    def validate_input(self, **kwargs) -> tuple[bool, str]:
        for key, schema in self.inputs.items():
            if key not in kwargs:
                if schema.get("required", True):
                    return False, f"Missing required parameter: {key}"
        return True, ""

    def to_json(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        k: {
                            "type": v.get("type", "string"),
                            "description": v.get("description", ""),
                        }
                        for k, v in self.inputs.items()
                    },
                    "required": [
                        k for k, v in self.inputs.items() if v.get("required", True)
                    ],
                },
            },
        }


def tool(
    name: str | None = None,
    description: str = "",
    inputs: dict[str, dict[str, str]] | None = None,
    output_type: str = "string",
    output_schema: dict | None = None,
):
    def decorator(func: Callable) -> Tool:
        import inspect

        sig = inspect.signature(func)
        tool_name = name or func.__name__
        param_inputs = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            param_inputs[param_name] = {
                "type": "string",
                "description": param.annotation.__doc__
                if hasattr(param.annotation, "__doc__")
                else "",
                "required": param.default == inspect.Parameter.empty,
            }

        inputs_final = inputs or param_inputs
        t = Tool(
            name=tool_name,
            description=description,
            inputs=inputs_final,
            output_type=output_type,
            func=func,
            output_schema=output_schema,
        )
        return t

    return decorator


class ToolCollection:
    def __init__(self, tools: list[Tool] | None = None):
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def list(self) -> list[Tool]:
        return list(self.tools.values())

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def unregister(self, name: str):
        if name in self.tools:
            del self.tools[name]

    def filter_by_permission(self, permissions: dict[str, str]) -> "ToolCollection":
        allowed = ToolCollection()
        for tool in self.tools.values():
            action = permissions.get(tool.name, permissions.get("*", "allow"))
            if action == "allow":
                allowed.register(tool)
        return allowed

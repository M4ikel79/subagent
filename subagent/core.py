import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator, Callable

from .model import ChatMessage, LanguageModel, MessageRole, TokenUsage
from .tools import Tool, ToolCollection


class AgentMode(Enum):
    SINGLE_STEP = "single_step"
    PLAN = "plan"


class PermissionAction(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class AgentStep:
    step_number: int
    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None
    is_final: bool = False
    error: str | None = None


@dataclass
class AgentResult:
    output: str
    steps: list[AgentStep] = field(default_factory=list)
    error: str | None = None
    usage: TokenUsage | None = None
    agent_name: str = "default"

    def to_dict(self) -> dict:
        return {
            "output": self.output,
            "agent_name": self.agent_name,
            "steps": [
                {
                    "step_number": s.step_number,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                    "is_final": s.is_final,
                    "error": s.error,
                }
                for s in self.steps
            ],
            "error": self.error,
            "usage": {
                "input": self.usage.input_tokens,
                "output": self.usage.output_tokens,
            }
            if self.usage
            else None,
        }


DEFAULT_SYSTEM_PROMPT = """You are an AI assistant that can use tools to help complete tasks.

You have access to the following tools:
{tools}

When you need to use a tool, respond with a JSON object in the following format:
{{
    "thought": "Your reasoning about what to do",
    "action": "tool_name",
    "action_input": {{"param1": "value1", "param2": "value2"}}
}}

When you have the final answer, respond with:
{{
    "thought": "Your final reasoning",
    "action": "final_answer",
    "action_input": {{"answer": "Your final answer here"}}
}}

IMPORTANT: Always respond with valid JSON only."""


BUILT_IN_AGENTS = {
    "explore": {
        "name": "explore",
        "description": "Fast agent specialized for exploring codebases. Use for finding files by patterns, searching code for keywords, or answering questions about the codebase.",
        "permissions": {
            "read_file": "allow",
            "list_directory": "allow",
            "glob": "allow",
            "grep": "allow",
            "webfetch": "allow",
            "*": "deny",
        },
        "model_id": "gpt-4o-mini",
    },
    "general": {
        "name": "general",
        "description": "General-purpose agent for researching complex questions and executing multi-step tasks.",
        "permissions": {"*": "allow"},
        "model_id": "gpt-4o",
    },
    "code-reviewer": {
        "name": "code-reviewer",
        "description": "Agent specialized for code quality analysis and review.",
        "permissions": {
            "read_file": "allow",
            "glob": "allow",
            "grep": "allow",
            "*": "deny",
        },
        "model_id": "gpt-4o",
    },
    "debugger": {
        "name": "debugger",
        "description": "Agent specialized for root cause analysis and bug investigation.",
        "permissions": {
            "read_file": "allow",
            "bash": "allow",
            "grep": "allow",
            "*": "deny",
        },
        "model_id": "gpt-4o",
    },
}


@dataclass
class AgentConfig:
    name: str
    description: str = ""
    permissions: dict[str, str] = field(default_factory=lambda: {"*": "allow"})
    model_id: str = "gpt-4o-mini"
    model_type: str = "openai"
    max_steps: int = 5
    temperature: float = 0.7
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tools: list[str] = field(default_factory=list)


class Permission:
    @staticmethod
    def check(tool_name: str, permissions: dict[str, str]) -> PermissionAction:
        if tool_name in permissions:
            action = permissions[tool_name]
        elif "*" in permissions:
            action = permissions["*"]
        else:
            action = "allow"
        return PermissionAction(action)

    @staticmethod
    def filter_tools(
        tools: ToolCollection, permissions: dict[str, str]
    ) -> ToolCollection:
        allowed = ToolCollection()
        for tool in tools.list():
            action = Permission.check(tool.name, permissions)
            if action == PermissionAction.ALLOW:
                allowed.register(tool)
        return allowed


@dataclass
class Agent:
    model: LanguageModel
    tools: ToolCollection
    mode: AgentMode = AgentMode.PLAN
    max_steps: int = 5
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    permissions: dict[str, str] = field(default_factory=lambda: {"*": "allow"})
    agent_name: str = "default"
    temperature: float = 0.7
    on_step: Callable[[AgentStep], None] | None = None

    def __post_init__(self):
        self.tools = Permission.filter_tools(self.tools, self.permissions)

    def _format_tools(self) -> str:
        lines = []
        for tool in self.tools.list():
            params = ", ".join(
                f"{k}: {v.get('description', '')}" for k, v in tool.inputs.items()
            )
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines) if lines else "No tools available."

    def _build_messages(self, task: str, history: list[AgentStep]) -> list[ChatMessage]:
        system = self.system_prompt.format(tools=self._format_tools())
        messages = [ChatMessage(role=MessageRole.SYSTEM, content=system)]

        for step in history:
            if step.action and step.action != "final_answer":
                tool_call = {
                    "function": {
                        "name": step.action,
                        "arguments": json.dumps(step.action_input or {}),
                    },
                    "id": f"call_{step.step_number}",
                    "type": "function",
                }
                messages.append(
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=step.thought,
                        tool_calls=[tool_call],
                    )
                )
            if step.observation:
                messages.append(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content=step.observation,
                        tool_call_id=f"call_{step.step_number}",
                    )
                )

        messages.append(ChatMessage(role=MessageRole.USER, content=task))
        return messages

    def _parse_response(
        self, content: str, tool_collection: ToolCollection
    ) -> tuple[str, str, dict | None]:
        try:
            data = json.loads(content.strip())
            thought = data.get("thought", "")
            action = data.get("action", "")
            action_input = data.get("action_input", {})
            return thought, action, action_input
        except json.JSONDecodeError:
            return content, "", {}

    def _execute_tool(
        self, tool_name: str, args: dict, tool_collection: ToolCollection
    ) -> str:
        if tool_name == "final_answer":
            return args.get("answer", "")

        action = Permission.check(tool_name, self.permissions)
        if action == PermissionAction.DENY:
            return f"Error: Tool '{tool_name}' is not allowed by permissions"
        if action == PermissionAction.ASK:
            return f"Error: Tool '{tool_name}' requires approval"

        tool = tool_collection.get(tool_name)
        if tool is None:
            return f"Error: Unknown tool '{tool_name}'"

        valid, error = tool.validate_input(**args)
        if not valid:
            return f"Error: {error}"

        try:
            result = tool(**args)
            return str(result) if result is not None else "Done"
        except Exception as e:
            return f"Error: {str(e)}"

    def run(self, task: str) -> AgentResult:
        history: list[AgentStep] = []
        step_num = 0

        while step_num < self.max_steps:
            step_num += 1
            messages = self._build_messages(task, history)

            try:
                response = self.model.complete(messages)
            except Exception as e:
                return AgentResult(
                    output="", steps=history, error=str(e), agent_name=self.agent_name
                )

            content = response.content

            if response.tool_calls:
                for tc in response.tool_calls:
                    func_name = tc["function"]["name"]
                    try:
                        func_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        func_args = {}

                    observation = self._execute_tool(func_name, func_args, self.tools)
                    step = AgentStep(
                        step_number=step_num,
                        thought=content,
                        action=func_name,
                        action_input=func_args,
                        observation=observation,
                        is_final=(func_name == "final_answer"),
                    )
                    history.append(step)
                    if self.on_step:
                        self.on_step(step)

                    if func_name == "final_answer":
                        return AgentResult(
                            output=observation,
                            steps=history,
                            agent_name=self.agent_name,
                        )
            else:
                thought, action, action_input = self._parse_response(
                    content, self.tools
                )
                if action == "final_answer":
                    step = AgentStep(
                        step_number=step_num,
                        thought=thought,
                        action=action,
                        action_input=action_input,
                        observation=action_input.get("answer", ""),
                        is_final=True,
                    )
                    history.append(step)
                    if self.on_step:
                        self.on_step(step)
                    return AgentResult(
                        output=action_input.get("answer", ""),
                        steps=history,
                        agent_name=self.agent_name,
                    )

                if action:
                    observation = self._execute_tool(action, action_input, self.tools)
                    step = AgentStep(
                        step_number=step_num,
                        thought=thought,
                        action=action,
                        action_input=action_input,
                        observation=observation,
                    )
                    history.append(step)
                    if self.on_step:
                        self.on_step(step)
                else:
                    step = AgentStep(
                        step_number=step_num, thought=content, is_final=True
                    )
                    history.append(step)
                    if self.on_step:
                        self.on_step(step)
                    return AgentResult(
                        output=content, steps=history, agent_name=self.agent_name
                    )

            if self.mode == AgentMode.SINGLE_STEP:
                return AgentResult(
                    output=history[-1].observation or history[-1].thought,
                    steps=history,
                    agent_name=self.agent_name,
                )

        return AgentResult(
            output=history[-1].observation if history else "Max steps reached",
            steps=history,
            error="Max steps reached" if not history[-1].is_final else None,
            agent_name=self.agent_name,
        )

    def run_stream(self, task: str) -> Generator[AgentStep, None, None]:
        history: list[AgentStep] = []
        step_num = 0

        while step_num < self.max_steps:
            step_num += 1
            messages = self._build_messages(task, history)

            try:
                response = self.model.complete(messages)
            except Exception as e:
                yield AgentStep(step_number=step_num, thought="", error=str(e))
                return

            content = response.content

            if response.tool_calls:
                for tc in response.tool_calls:
                    func_name = tc["function"]["name"]
                    try:
                        func_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        func_args = {}

                    observation = self._execute_tool(func_name, func_args, self.tools)
                    step = AgentStep(
                        step_number=step_num,
                        thought=content,
                        action=func_name,
                        action_input=func_args,
                        observation=observation,
                        is_final=(func_name == "final_answer"),
                    )
                    history.append(step)
                    yield step

                    if func_name == "final_answer":
                        return
            else:
                thought, action, action_input = self._parse_response(
                    content, self.tools
                )
                if action == "final_answer":
                    step = AgentStep(
                        step_number=step_num,
                        thought=thought,
                        action=action,
                        action_input=action_input,
                        observation=action_input.get("answer", ""),
                        is_final=True,
                    )
                    history.append(step)
                    yield step
                    return

                if action:
                    observation = self._execute_tool(action, action_input, self.tools)
                    step = AgentStep(
                        step_number=step_num,
                        thought=thought,
                        action=action,
                        action_input=action_input,
                        observation=observation,
                    )
                    history.append(step)
                    yield step
                else:
                    step = AgentStep(
                        step_number=step_num, thought=content, is_final=True
                    )
                    history.append(step)
                    yield step
                    return

            if self.mode == AgentMode.SINGLE_STEP:
                return


class AgentManager:
    def __init__(self, tools: ToolCollection | None = None):
        self.tools = tools or ToolCollection()
        self.agents: dict[str, AgentConfig] = {}
        self._load_builtin_agents()

    def _load_builtin_agents(self):
        for name, config in BUILT_IN_AGENTS.items():
            self.agents[name] = AgentConfig(**config)

    def register_agent(self, config: AgentConfig):
        self.agents[config.name] = config

    def get_agent(self, name: str) -> AgentConfig | None:
        return self.agents.get(name)

    def list_agents(self) -> list[AgentConfig]:
        return list(self.agents.values())

    def create_agent(self, name: str, model: LanguageModel, **kwargs) -> Agent:
        config = self.get_agent(name)
        if not config:
            config = AgentConfig(name=name)

        filtered_tools = Permission.filter_tools(self.tools, config.permissions)

        return Agent(
            model=model,
            tools=filtered_tools,
            mode=kwargs.get("mode", AgentMode.PLAN),
            max_steps=kwargs.get("max_steps", config.max_steps),
            system_prompt=kwargs.get("system_prompt", config.system_prompt),
            permissions=config.permissions,
            agent_name=name,
            temperature=kwargs.get("temperature", config.temperature),
            on_step=kwargs.get("on_step"),
        )


class AgentState:
    def __init__(self, path: str | None = None):
        self.path = path or ".subagent/state.json"
        self.data: dict = {}

    def save(self, result: AgentResult):
        self.data = result.to_dict()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def load(self) -> AgentResult | None:
        if not os.path.exists(self.path):
            return None
        with open(self.path, "r") as f:
            self.data = json.load(f)
        return AgentResult(
            output=self.data.get("output", ""),
            steps=[AgentStep(**s) for s in self.data.get("steps", [])],
            error=self.data.get("error"),
            agent_name=self.data.get("agent_name", "default"),
        )


def run_agent(
    goal: str,
    mode: str = "plan",
    tools: list[Tool] | None = None,
    model: LanguageModel | None = None,
    max_steps: int = 5,
    model_id: str = "glm-4.6",
    agent_name: str = "default",
    permissions: dict[str, str] | None = None,
    stream: bool = False,
    temperature: float = 0.7,
    api_key: str | None = None,
) -> AgentResult | Generator[AgentStep, None, None]:
    from .model import OllamaModel

    agent_mode = AgentMode(mode) if mode == "single_step" else AgentMode.PLAN

    if model is None:
        model = OllamaModel(model_id=model_id, temperature=temperature, api_key=api_key)

    tool_collection = ToolCollection(tools or [])

    agent = Agent(
        model=model,
        tools=tool_collection,
        mode=agent_mode,
        max_steps=max_steps,
        permissions=permissions or {"*": "allow"},
        agent_name=agent_name,
    )

    if stream:
        return agent.run_stream(goal)
    return agent.run(goal)

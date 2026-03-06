import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .model import ChatMessage, LanguageModel, MessageRole
from .tools import Tool, ToolCollection


class AgentMode(Enum):
    SINGLE_STEP = "single_step"
    PLAN = "plan"


@dataclass
class AgentStep:
    step_number: int
    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None
    is_final: bool = False


@dataclass
class AgentResult:
    output: str
    steps: list[AgentStep] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "output": self.output,
            "steps": [
                {
                    "step_number": s.step_number,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                    "is_final": s.is_final,
                }
                for s in self.steps
            ],
            "error": self.error,
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


@dataclass
class Agent:
    model: LanguageModel
    tools: ToolCollection
    mode: AgentMode = AgentMode.PLAN
    max_steps: int = 5
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def __post_init__(self):
        if not self.tools.tools:
            raise ValueError("Agent must have at least one tool")

    def _format_tools(self) -> str:
        lines = []
        for tool in self.tools.list():
            params = ", ".join(f"{k}: {v.get('description', '')}" for k, v in tool.inputs.items())
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines) if lines else "No tools available."

    def _build_messages(self, task: str, history: list[AgentStep]) -> list[ChatMessage]:
        system = self.system_prompt.format(tools=self._format_tools())
        messages = [ChatMessage(role=MessageRole.SYSTEM, content=system)]

        for step in history:
            if step.action and step.action != "final_answer":
                tool_call = {
                    "function": {"name": step.action, "arguments": json.dumps(step.action_input or {})},
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

    def _parse_response(self, content: str, tool_collection: ToolCollection) -> tuple[str, str, dict | None]:
        try:
            data = json.loads(content.strip())
            thought = data.get("thought", "")
            action = data.get("action", "")
            action_input = data.get("action_input", {})
            return thought, action, action_input
        except json.JSONDecodeError:
            return content, "", {}

    def _execute_tool(self, tool_name: str, args: dict, tool_collection: ToolCollection) -> str:
        if tool_name == "final_answer":
            return args.get("answer", "")

        tool = tool_collection.get(tool_name)
        if tool is None:
            return f"Error: Unknown tool '{tool_name}'"

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
                return AgentResult(output="", steps=history, error=str(e))

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

                    if func_name == "final_answer":
                        return AgentResult(output=observation, steps=history)
            else:
                thought, action, action_input = self._parse_response(content, self.tools)
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
                    return AgentResult(output=action_input.get("answer", ""), steps=history)

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
                else:
                    step = AgentStep(step_number=step_num, thought=content, is_final=True)
                    history.append(step)
                    return AgentResult(output=content, steps=history)

            if self.mode == AgentMode.SINGLE_STEP:
                return AgentResult(
                    output=history[-1].observation or history[-1].thought, steps=history
                )

        return AgentResult(
            output=history[-1].observation if history else "Max steps reached",
            steps=history,
            error="Max steps reached" if not history[-1].is_final else None,
        )


def run_agent(
    goal: str,
    mode: str = "plan",
    tools: list[Tool] | None = None,
    model: LanguageModel | None = None,
    max_steps: int = 5,
    model_type: str = "openai",
    model_id: str = "gpt-4o-mini",
    **model_kwargs,
) -> AgentResult:
    from .model import create_model

    agent_mode = AgentMode(mode) if mode == "single_step" else AgentMode.PLAN

    if model is None:
        model = create_model(model_type, model_id, **model_kwargs)

    tool_collection = ToolCollection(tools or [])

    agent = Agent(
        model=model,
        tools=tool_collection,
        mode=agent_mode,
        max_steps=max_steps,
    )

    return agent.run(goal)

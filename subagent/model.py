import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol


class MessageRole:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ChatMessage:
    role: str
    content: str
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class LanguageModel(Protocol):
    def complete(self, messages: list[ChatMessage]) -> ChatMessage:
        ...


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class OpenAICompatibleModel:
    model_id: str
    api_base: str = "https://api.openai.com/v1"
    api_key: str | None = None
    _system_prompt: str = "You are a helpful assistant."

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.api_key is None:
            raise ValueError("API key not provided and OPENAI_API_KEY not set")

    def complete(self, messages: list[ChatMessage]) -> ChatMessage:
        import requests

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_id,
            "messages": [self._msg_to_dict(m) for m in messages],
            "temperature": 0.7,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        msg_data = data["choices"][0]["message"]
        return ChatMessage(
            role=msg_data["role"],
            content=msg_data.get("content", ""),
            tool_calls=msg_data.get("tool_calls"),
        )

    def _msg_to_dict(self, msg: ChatMessage) -> dict:
        result = {"role": msg.role, "content": msg.content}
        if msg.tool_calls:
            result["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id
        return result


@dataclass
class LiteLLMModel:
    model_id: str
    api_key: str | None = None

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")

    def complete(self, messages: list[ChatMessage]) -> ChatMessage:
        try:
            from litellm import completion
        except ImportError:
            raise ImportError("litellm is required for LiteLLMModel. Install with: pip install litellm")

        msgs = [{"role": m.role, "content": m.content} for m in messages]
        response = completion(model=self.model_id, messages=msgs)
        msg_data = response.choices[0].message
        return ChatMessage(
            role=msg_data.role,
            content=msg_data.content or "",
            tool_calls=getattr(msg_data, "tool_calls", None),
        )


def create_model(model_type: str, model_id: str, **kwargs) -> LanguageModel:
    if model_type == "openai":
        return OpenAICompatibleModel(model_id=model_id, **kwargs)
    elif model_type == "litellm":
        return LiteLLMModel(model_id=model_id, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

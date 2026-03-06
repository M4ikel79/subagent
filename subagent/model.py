import os
import json
from dataclasses import dataclass, field
from typing import Any, Generator, Protocol


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


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class LanguageModel(Protocol):
    def complete(self, messages: list[ChatMessage]) -> ChatMessage: ...

    def complete_stream(
        self, messages: list[ChatMessage]
    ) -> Generator[ChatMessage, None, None]: ...


@dataclass
class OpenAICompatibleModel:
    model_id: str
    api_base: str = "https://api.openai.com/v1"
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None

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
            "temperature": self.temperature,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        msg_data = data["choices"][0]["message"]
        return ChatMessage(
            role=msg_data["role"],
            content=msg_data.get("content", ""),
            tool_calls=msg_data.get("tool_calls"),
        )

    def complete_stream(
        self, messages: list[ChatMessage]
    ) -> Generator[ChatMessage, None, None]:
        import requests

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_id,
            "messages": [self._msg_to_dict(m) for m in messages],
            "temperature": self.temperature,
            "stream": True,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        response = requests.post(
            url, headers=headers, json=payload, stream=True, timeout=60
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    yield ChatMessage(
                        role=delta.get("role", "assistant"),
                        content=delta.get("content", ""),
                        tool_calls=delta.get("tool_calls"),
                    )
                except json.JSONDecodeError:
                    continue

    def _msg_to_dict(self, msg: ChatMessage) -> dict:
        result = {"role": msg.role, "content": msg.content}
        if msg.tool_calls:
            result["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id
        return result


@dataclass
class AnthropicModel:
    model_id: str
    api_key: str | None = None
    api_base: str = "https://api.anthropic.com/v1"
    temperature: float = 0.7
    max_tokens: int = 1024

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if self.api_key is None:
            raise ValueError("API key not provided and ANTHROPIC_API_KEY not set")

    def complete(self, messages: list[ChatMessage]) -> ChatMessage:
        import requests

        url = f"{self.api_base}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        system_msg = ""
        filtered_msgs = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_msg = msg.content
            else:
                filtered_msgs.append({"role": msg.role, "content": msg.content})

        payload = {
            "model": self.model_id,
            "messages": filtered_msgs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if system_msg:
            payload["system"] = system_msg

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=data.get("content", [{}])[0].get("text", ""),
        )

    def complete_stream(
        self, messages: list[ChatMessage]
    ) -> Generator[ChatMessage, None, None]:
        import requests

        url = f"{self.api_base}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        system_msg = ""
        filtered_msgs = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_msg = msg.content
            else:
                filtered_msgs.append({"role": msg.role, "content": msg.content})

        payload = {
            "model": self.model_id,
            "messages": filtered_msgs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if system_msg:
            payload["system"] = system_msg

        response = requests.post(
            url, headers=headers, json=payload, stream=True, timeout=60
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk.get("type") == "content_block_delta":
                        delta = chunk.get("delta", {})
                        yield ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=delta.get("text", ""),
                        )
                except json.JSONDecodeError:
                    continue


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
            raise ImportError(
                "litellm is required for LiteLLMModel. Install with: pip install litellm"
            )

        msgs = [{"role": m.role, "content": m.content} for m in messages]
        response = completion(model=self.model_id, messages=msgs)
        msg_data = response.choices[0].message
        return ChatMessage(
            role=msg_data.role,
            content=msg_data.content or "",
            tool_calls=getattr(msg_data, "tool_calls", None),
        )

    def complete_stream(
        self, messages: list[ChatMessage]
    ) -> Generator[ChatMessage, None, None]:
        try:
            from litellm import completion
        except ImportError:
            raise ImportError(
                "litellm is required for LiteLLMModel. Install with: pip install litellm"
            )

        msgs = [{"role": m.role, "content": m.content} for m in messages]
        response = completion(model=self.model_id, messages=msgs, stream=True)
        for chunk in response:
            msg_data = chunk.choices[0].delta
            yield ChatMessage(
                role=msg_data.role or "assistant",
                content=msg_data.content or "",
            )


@dataclass
class NvidiaModel:
    model_id: str
    api_key: str | None = None
    api_base: str = "https://integrate.api.nvidia.com/v1"
    temperature: float = 0.7
    max_tokens: int | None = None

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("NVIDIA_API_KEY")
        if self.api_key is None:
            raise ValueError("API key not provided and NVIDIA_API_KEY not set")

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
            "temperature": self.temperature,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        msg_data = data["choices"][0]["message"]
        return ChatMessage(
            role=msg_data["role"],
            content=msg_data.get("content", ""),
            tool_calls=msg_data.get("tool_calls"),
        )

    def complete_stream(
        self, messages: list[ChatMessage]
    ) -> Generator[ChatMessage, None, None]:
        import requests

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_id,
            "messages": [self._msg_to_dict(m) for m in messages],
            "temperature": self.temperature,
            "stream": True,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        response = requests.post(
            url, headers=headers, json=payload, stream=True, timeout=120
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    yield ChatMessage(
                        role=delta.get("role", "assistant"),
                        content=delta.get("content", ""),
                        tool_calls=delta.get("tool_calls"),
                    )
                except json.JSONDecodeError:
                    continue

    def _msg_to_dict(self, msg: ChatMessage) -> dict:
        result = {"role": msg.role, "content": msg.content}
        if msg.tool_calls:
            result["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id
        return result


def create_model(model_type: str, model_id: str, **kwargs) -> LanguageModel:
    if model_type == "openai":
        return OpenAICompatibleModel(model_id=model_id, **kwargs)
    elif model_type == "anthropic":
        return AnthropicModel(model_id=model_id, **kwargs)
    elif model_type == "litellm":
        return LiteLLMModel(model_id=model_id, **kwargs)
    elif model_type == "nvidia":
        return NvidiaModel(model_id=model_id, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

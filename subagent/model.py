import os
import json
from dataclasses import dataclass
from typing import Generator


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


class LanguageModel:
    def complete(self, messages: list[ChatMessage]) -> ChatMessage: ...
    def complete_stream(
        self, messages: list[ChatMessage]
    ) -> Generator[ChatMessage, None, None]: ...


@dataclass
class OllamaModel:
    model_id: str
    api_key: str | None = None
    api_base: str = "https://ollama.com"
    temperature: float = 0.7
    max_tokens: int | None = None

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("OLLAMA_API_KEY")
        if self.api_key:
            self.api_base = "https://ollama.com"

    def complete(self, messages: list[ChatMessage]) -> ChatMessage:
        import requests

        url = f"{self.api_base}/api/chat"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_id,
            "messages": [self._msg_to_dict(m) for m in messages],
            "temperature": self.temperature,
        }
        if self.max_tokens:
            payload["num_predict"] = self.max_tokens

        response = requests.post(url, json=payload, headers=headers, timeout=180)
        response.raise_for_status()

        content = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                content += chunk.get("message", {}).get("content", "")
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                continue

        return ChatMessage(role=MessageRole.ASSISTANT, content=content)

    def complete_stream(
        self, messages: list[ChatMessage]
    ) -> Generator[ChatMessage, None, None]:
        import requests

        url = f"{self.api_base}/api/chat"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_id,
            "messages": [self._msg_to_dict(m) for m in messages],
            "temperature": self.temperature,
            "stream": True,
        }
        if self.max_tokens:
            payload["num_predict"] = self.max_tokens

        response = requests.post(
            url, json=payload, headers=headers, stream=True, timeout=180
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                if chunk.get("done"):
                    break
                delta = chunk.get("message", {})
                yield ChatMessage(
                    role=MessageRole.ASSISTANT, content=delta.get("content", "")
                )
            except json.JSONDecodeError:
                continue

    def _msg_to_dict(self, msg: ChatMessage) -> dict:
        return {"role": msg.role, "content": msg.content}


def create_model(model_type: str, model_id: str, **kwargs) -> LanguageModel:
    if model_type == "ollama":
        return OllamaModel(model_id=model_id, **kwargs)
    else:
        raise ValueError(f"Only 'ollama' model type is supported. Got: {model_type}")

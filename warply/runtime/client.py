from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _Message:
    content: str


@dataclass(frozen=True)
class _Choice:
    message: _Message


@dataclass(frozen=True)
class _CompletionResponse:
    choices: list[_Choice]


class _ChatCompletions:
    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> _CompletionResponse:
        prompt = messages[-1]["content"] if messages else ""
        content = f"[warply:mock:{model}] {prompt}"
        return _CompletionResponse(choices=[_Choice(message=_Message(content=content))])


class _Chat:
    def __init__(self) -> None:
        self.completions = _ChatCompletions()


class MockOpenAIClient:
    """Small OpenAI-compatible surface for local mock lifecycle tests."""

    def __init__(self, *, base_url: str) -> None:
        self.base_url = base_url
        self.chat = _Chat()

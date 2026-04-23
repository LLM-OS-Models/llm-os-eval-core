from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

class BaseRunner(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        raise NotImplementedError

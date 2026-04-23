from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

TaskType = Literal[
    "md_retrieval",
    "tool_call",
    "terminal",
    "text2sql",
    "coding_agent",
    "docai_ocr",
    "deep_research",
]

class EvalSample(BaseModel):
    sample_id: str
    task_type: TaskType
    difficulty: Literal["easy", "medium", "hard"]
    user_query: str
    artifacts: dict[str, Any] = Field(default_factory=dict)
    gold: dict[str, Any] = Field(default_factory=dict)
    grader: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

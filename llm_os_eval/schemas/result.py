from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field

class EvalResult(BaseModel):
    run_id: str
    sample_id: str
    task_type: str
    model_name: str
    checkpoint_name: str
    prompt_version: str
    raw_output: str | None = None
    parsed_output: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    command_trace: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    final_success: bool = False
    failure_stage: str | None = None
    metric_values: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

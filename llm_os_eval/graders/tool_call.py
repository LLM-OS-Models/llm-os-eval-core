from __future__ import annotations
import json
from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult

class ToolCallEvaluator(BaseEvaluator):
    task_type = "tool_call"

    def build_prompt(self, sample: EvalSample) -> tuple[str, str]:
        tools = sample.artifacts.get("tools", [])
        system_prompt = "너는 tool calling 평가용 어시스턴트다. 항상 JSON 형식의 tool call을 생성하라."
        user_prompt = json.dumps(
            {
                "query": sample.user_query,
                "tools": tools,
                "output_format": {
                    "tool_calls": [{"name": "string", "arguments": {"key": "value"}}]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        return system_prompt, user_prompt

    def grade(self, sample: EvalSample, result: EvalResult) -> EvalResult:
        gold_calls = sample.gold.get("tool_calls", [])
        text = result.raw_output or ""
        try:
            parsed = json.loads(text)
            predicted = parsed.get("tool_calls", [])
            result.parsed_output = parsed
        except Exception:
            predicted = []
            result.failure_stage = "parse"

        selection_ok = 0.0
        if gold_calls and predicted:
            selection_ok = 1.0 if predicted[0].get("name") == gold_calls[0].get("name") else 0.0
        result.metric_values["tool_selection_accuracy"] = selection_ok
        result.final_success = selection_ok > 0
        if not result.final_success and result.failure_stage is None:
            result.failure_stage = "tool_selection"
        return result

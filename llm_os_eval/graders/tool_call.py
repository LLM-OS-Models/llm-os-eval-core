from __future__ import annotations
import json
import re

import jsonschema

from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult


def _extract_json(text: str) -> dict | list | None:
    candidates = [text]
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        candidates.insert(0, fence_match.group(1))
    brace_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if brace_match:
        candidates.insert(0, brace_match.group(1))
    for c in candidates:
        try:
            return json.loads(c.strip())
        except Exception:
            continue
    return None


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
        gold_tools_schema = sample.gold.get("tools_schema", None)
        text = result.raw_output or ""

        parsed = _extract_json(text)
        if parsed is None:
            result.failure_stage = "parse"
            result.metric_values["tool_selection_accuracy"] = 0.0
            result.metric_values["argument_validity"] = 0.0
            result.metric_values["schema_validity"] = 0.0
            result.metric_values["task_success"] = 0.0
            result.final_success = False
            return result

        if isinstance(parsed, dict):
            predicted = parsed.get("tool_calls", [])
        elif isinstance(parsed, list):
            predicted = parsed
        else:
            predicted = []
        result.parsed_output = parsed if isinstance(parsed, dict) else {"tool_calls": predicted}

        if not gold_calls or not predicted:
            result.failure_stage = "tool_selection"
            result.metric_values["tool_selection_accuracy"] = 0.0
            result.metric_values["argument_validity"] = 0.0
            result.metric_values["schema_validity"] = 0.0
            result.metric_values["task_success"] = 0.0
            result.final_success = False
            return result

        selection_ok = 1.0 if predicted[0].get("name") == gold_calls[0].get("name") else 0.0
        result.metric_values["tool_selection_accuracy"] = selection_ok

        gold_args = gold_calls[0].get("arguments", {})
        pred_args = predicted[0].get("arguments", {})
        if gold_args:
            matching = sum(1 for k, v in gold_args.items() if str(pred_args.get(k, "")) == str(v))
            arg_score = matching / len(gold_args)
        else:
            arg_score = 1.0 if not pred_args else 0.5
        result.metric_values["argument_validity"] = arg_score

        if gold_tools_schema:
            try:
                schema = gold_tools_schema if isinstance(gold_tools_schema, dict) else {"type": "object", "properties": gold_tools_schema}
                jsonschema.validate(instance=pred_args, schema=schema)
                result.metric_values["schema_validity"] = 1.0
            except jsonschema.ValidationError:
                result.metric_values["schema_validity"] = 0.0
        else:
            result.metric_values["schema_validity"] = 1.0

        task_ok = 1.0 if selection_ok > 0 and arg_score >= 0.5 else 0.0
        result.metric_values["task_success"] = task_ok
        result.final_success = task_ok > 0
        if not result.final_success and result.failure_stage is None:
            result.failure_stage = "tool_selection"
        return result

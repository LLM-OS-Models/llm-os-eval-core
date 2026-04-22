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

        # Grade ALL tool calls, not just the first one
        n_gold = len(gold_calls)
        selection_hits = 0
        total_arg_score = 0.0
        total_schema_ok = 0.0

        for i, gold_call in enumerate(gold_calls):
            if i < len(predicted):
                pred_call = predicted[i]
                if pred_call.get("name") == gold_call.get("name"):
                    selection_hits += 1

                gold_args = gold_call.get("arguments", {})
                pred_args = pred_call.get("arguments", {}) or {}
                if gold_args:
                    matching = sum(1 for k, v in gold_args.items() if str(pred_args.get(k, "")) == str(v))
                    total_arg_score += matching / len(gold_args)
                else:
                    total_arg_score += 1.0

                if gold_tools_schema:
                    try:
                        schema = gold_tools_schema if isinstance(gold_tools_schema, dict) else {"type": "object", "properties": gold_tools_schema}
                        jsonschema.validate(instance=pred_args, schema=schema)
                        total_schema_ok += 1.0
                    except jsonschema.ValidationError:
                        pass
                else:
                    total_schema_ok += 1.0

        selection_ok = selection_hits / n_gold if n_gold > 0 else 0.0
        arg_score = total_arg_score / n_gold if n_gold > 0 else 0.0
        schema_ok = total_schema_ok / n_gold if n_gold > 0 else 0.0

        result.metric_values["tool_selection_accuracy"] = selection_ok
        result.metric_values["argument_validity"] = arg_score
        result.metric_values["schema_validity"] = schema_ok

        task_ok = 1.0 if selection_ok > 0 and arg_score >= 0.5 else 0.0
        result.metric_values["task_success"] = task_ok
        result.final_success = task_ok > 0
        if not result.final_success and result.failure_stage is None:
            result.failure_stage = "tool_selection"
        return result

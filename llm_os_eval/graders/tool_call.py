from __future__ import annotations
import json
import re

import jsonschema

from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult


def _strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think[^>]*>.*?</think\s*>", "", text, flags=re.DOTALL)
    first_brace = min(
        (i for i in (cleaned.find(c) for c in ["{", "["]) if i >= 0),
        default=len(cleaned),
    )
    if first_brace < len(cleaned):
        return cleaned[first_brace:]
    return cleaned


def _is_placeholder(value) -> bool:
    s = str(value)
    return bool(re.search(r"<[A-Z_]+>|[XY]\s*건|<\w+>", s))


# Korean-English city name equivalents for semantic argument matching
_CITY_ALIASES = {
    "서울": "Seoul", "Seoul": "서울",
    "부산": "Busan", "Busan": "부산",
    "도쿄": "Tokyo", "Tokyo": "도쿄",
    "뉴욕": "New York", "New York": "뉴욕",
}


def _arg_semantic_match(pred_val, gold_val) -> bool:
    """Check if two argument values are semantically equivalent."""
    sp = str(pred_val)
    sg = str(gold_val)
    if sp == sg:
        return True
    # City name aliases
    if sg in _CITY_ALIASES and _CITY_ALIASES.get(sg) == sp:
        return True
    if sp in _CITY_ALIASES and _CITY_ALIASES.get(sp) == sg:
        return True
    # Case-insensitive for English values
    if sp.lower() == sg.lower():
        return True
    return False


def _extract_json(text: str) -> dict | list | None:
    text = _strip_thinking(text)
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
        system_prompt = "너는 tool calling 평가용 어시스턴트다. 항상 JSON 형식의 tool call을 생성하라. 사고 과정 없이 바로 JSON만 출력하라. <think:> 태그를 절대 사용하지 마라. 사용자 요청에 따라 여러 tool_call이 필요할 수 있다. 반드시 tool_calls 배열에 모든 필요한 호출을 순서대로 나열하라. 인자 값은 한국어로 제공하라 (예: city='서울' not 'Seoul')."
        user_prompt = json.dumps(
            {
                "query": sample.user_query,
                "tools": tools,
                "output_format": {
                    "tool_calls": [{"name": "string", "arguments": {"key": "value"}}],
                    "note": "여러 단계가 필요하면 tool_calls 배열에 모두 포함하라."
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
                # Unwrap nested {"tool_call": {...}} wrapper used by some models
                if isinstance(pred_call, dict) and "tool_call" in pred_call and "name" not in pred_call:
                    pred_call = pred_call["tool_call"]
                if pred_call.get("name") == gold_call.get("name"):
                    selection_hits += 1

                gold_args = gold_call.get("arguments", {})
                pred_args = pred_call.get("arguments", {}) or {}
                if gold_args:
                    matching = 0
                    for k, v in gold_args.items():
                        if _is_placeholder(v):
                            matching += 1 if k in pred_args else 0
                        else:
                            matching += 1 if _arg_semantic_match(pred_args.get(k, ""), v) else 0
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

from __future__ import annotations
import re

from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult


def _field_match(gold_value: str, pred_text: str, match_type: str = "contains") -> float:
    gold_str = str(gold_value).strip()
    if match_type == "exact":
        return 1.0 if gold_str in pred_text else 0.0
    gold_lower = gold_str.lower()
    pred_lower = pred_text.lower()
    if gold_lower in pred_lower:
        return 1.0
    gold_digits = re.sub(r"[^\d]", "", gold_str)
    if gold_digits and gold_digits in pred_text:
        return 0.8
    gold_words = set(gold_lower.split())
    if len(gold_words) > 1:
        overlap = sum(1 for w in gold_words if w in pred_lower) / len(gold_words)
        return overlap
    return 0.0


class DocAIOCREvaluator(BaseEvaluator):
    task_type = "docai_ocr"

    def build_prompt(self, sample: EvalSample) -> tuple[str, str]:
        system_prompt = "너는 문서 OCR/이해 평가용 모델이다."
        user_prompt = f"""문서:
{sample.artifacts.get('document_path')}

질문:
{sample.user_query}
"""
        return system_prompt, user_prompt

    def grade(self, sample: EvalSample, result: EvalResult) -> EvalResult:
        text = result.raw_output or ""
        gold_fields = sample.gold.get("fields", {})
        gold_table = sample.gold.get("table_headers", [])
        target_fields = sample.gold.get("target_fields", [])

        if gold_fields:
            total_score = 0.0
            for key, value in gold_fields.items():
                if target_fields and key not in target_fields:
                    continue
                total_score += _field_match(value, text)
            denom = len(target_fields) if target_fields else len(gold_fields)
            field_acc = total_score / denom if denom > 0 else 0.0
        else:
            field_acc = 0.0
        result.metric_values["field_extraction_accuracy"] = field_acc

        if gold_table:
            table_hits = sum(1 for h in gold_table if h.lower() in text.lower())
            table_acc = table_hits / len(gold_table) if gold_table else 0.0
        else:
            table_acc = 1.0
        result.metric_values["table_parse_accuracy"] = table_acc

        overall = 0.6 * field_acc + 0.4 * table_acc
        result.metric_values["document_understanding_accuracy"] = overall
        result.final_success = field_acc > 0.5
        if not result.final_success:
            result.failure_stage = "document_understanding"
        return result

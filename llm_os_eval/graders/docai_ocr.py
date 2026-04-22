from __future__ import annotations
from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult

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
        score = 0.0
        for value in gold_fields.values():
            if str(value) in text:
                score += 1.0
        if gold_fields:
            score /= len(gold_fields)
        result.metric_values["field_extraction_accuracy"] = score
        result.final_success = score > 0.5
        if not result.final_success:
            result.failure_stage = "document_understanding"
        return result

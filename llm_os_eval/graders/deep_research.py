from __future__ import annotations
from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult

class DeepResearchEvaluator(BaseEvaluator):
    task_type = "deep_research"

    def build_prompt(self, sample: EvalSample) -> tuple[str, str]:
        system_prompt = "너는 리서치 에이전트 평가용 모델이다. 반드시 근거와 함께 답하라."
        user_prompt = f"""질문:
{sample.user_query}

출력 형식:
ANSWER: ...
CITATIONS: [...]
"""
        return system_prompt, user_prompt

    def grade(self, sample: EvalSample, result: EvalResult) -> EvalResult:
        text = result.raw_output or ""
        has_answer = 1.0 if "ANSWER:" in text else 0.0
        has_citations = 1.0 if "CITATIONS:" in text else 0.0
        result.metric_values["answer_accuracy_proxy"] = has_answer
        result.metric_values["citation_support_proxy"] = has_citations
        result.final_success = has_answer > 0 and has_citations > 0
        if not result.final_success:
            result.failure_stage = "citation_or_answer"
        return result

from __future__ import annotations
from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult

class MDRetrievalEvaluator(BaseEvaluator):
    task_type = "md_retrieval"

    def build_prompt(self, sample: EvalSample) -> tuple[str, str]:
        doc_list = "\n".join(
            f"- {d['doc_id']}: {d['path']}" for d in sample.artifacts.get("documents", [])
        )
        system_prompt = "너는 MD 문서 검색과 근거 기반 응답을 수행하는 평가용 어시스턴트다."
        user_prompt = f"""질문:
{sample.user_query}

문서 목록:
{doc_list}

반드시 다음 형식으로 답하라:
DOC_IDS: [...]
ANSWER: ...
"""
        return system_prompt, user_prompt

    def grade(self, sample: EvalSample, result: EvalResult) -> EvalResult:
        text = result.raw_output or ""
        gold_docs = set(sample.gold.get("relevant_doc_ids", []))
        predicted_docs = []
        if "DOC_IDS:" in text:
            try:
                part = text.split("DOC_IDS:", 1)[1].split("ANSWER:", 1)[0].strip()
                predicted_docs = [x.strip(" [],'\"") for x in part.split(",") if x.strip()]
            except Exception:
                predicted_docs = []
        hit = 1.0 if any(doc in gold_docs for doc in predicted_docs[:3]) else 0.0
        result.metric_values["file_hit_at_3"] = hit
        result.final_success = hit > 0
        if not result.final_success:
            result.failure_stage = "retrieval"
        return result

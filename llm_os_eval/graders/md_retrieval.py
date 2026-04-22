from __future__ import annotations
import re

from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult


def _parse_doc_ids(text: str, max_k: int = 10) -> list[str]:
    predicted = []
    if "DOC_IDS:" in text:
        try:
            part = text.split("DOC_IDS:", 1)[1].split("\n", 1)[0].strip()
            bracket_match = re.search(r"\[(.*?)\]", part)
            if bracket_match:
                items = bracket_match.group(1)
            else:
                items = part
            predicted = [x.strip(" [],'\"") for x in items.split(",") if x.strip(" [],'\"")]
        except Exception:
            predicted = []
    return predicted[:max_k]


def _parse_answer(text: str) -> str:
    if "ANSWER:" in text:
        return text.split("ANSWER:", 1)[1].strip()
    return ""


def _span_recall(pred_text: str, gold_spans: list[str]) -> float:
    if not gold_spans:
        return 1.0
    hits = sum(1 for span in gold_spans if span.lower() in pred_text.lower())
    return hits / len(gold_spans)


class MDRetrievalEvaluator(BaseEvaluator):
    task_type = "md_retrieval"

    def build_prompt(self, sample: EvalSample) -> tuple[str, str]:
        doc_list = "\n".join(
            f"- {d['doc_id']}: {d['path']}" for d in sample.artifacts.get("documents", [])
        )
        system_prompt = "너는 MD 문서 검색과 근거 기반 응답을 수행하는 평가용 어시스턴트다. 사고 과정 없이 바로 최종 답변만 출력하라."
        user_prompt = f"""질문:
{sample.user_query}

문서 목록:
{doc_list}

반드시 다음 형식으로만 답하라 (다른 텍스트는 출력하지 마라):
DOC_IDS: [doc_id1, doc_id2]
ANSWER: 질문에 대한 답변

주의: DOC_IDS에는 문서 목록의 정확한 doc_id를 사용하라. ANSWER에는 문서에서 찾은 근거를 포함하라.
"""
        return system_prompt, user_prompt

    def grade(self, sample: EvalSample, result: EvalResult) -> EvalResult:
        text = result.raw_output or ""
        gold_docs = set(sample.gold.get("relevant_doc_ids", []))
        gold_spans = sample.gold.get("relevant_spans", [])
        gold_answer = sample.gold.get("expected_answer", "")

        predicted_docs = _parse_doc_ids(text)

        hit_at_3 = 1.0 if any(doc in gold_docs for doc in predicted_docs[:3]) else 0.0
        hit_at_1 = 1.0 if predicted_docs[:1] and predicted_docs[0] in gold_docs else 0.0
        result.metric_values["file_hit_at_1"] = hit_at_1
        result.metric_values["file_hit_at_3"] = hit_at_3

        answer_text = _parse_answer(text)
        span_score = _span_recall(answer_text, gold_spans)
        result.metric_values["span_recall"] = span_score

        if gold_answer:
            answer_f1 = 1.0 if gold_answer.lower() in answer_text.lower() else 0.0
        else:
            answer_f1 = 1.0 if answer_text else 0.0
        result.metric_values["answer_f1"] = answer_f1

        result.final_success = hit_at_3 > 0 and span_score >= 0.5
        if not result.final_success:
            result.failure_stage = "retrieval" if hit_at_3 == 0 else "answer"
        return result

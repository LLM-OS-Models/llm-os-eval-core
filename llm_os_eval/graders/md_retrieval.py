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


def _resolve_doc_ids(raw_ids: list[str], documents: list[dict]) -> list[str]:
    """Resolve numeric/positional doc IDs to actual doc_id strings."""
    resolved = []
    for rid in raw_ids:
        if rid in [d.get("doc_id", "") for d in documents]:
            resolved.append(rid)
            continue
        # Try 1-based positional index
        try:
            idx = int(rid) - 1
            if 0 <= idx < len(documents):
                resolved.append(documents[idx].get("doc_id", rid))
                continue
        except (ValueError, TypeError):
            pass
        resolved.append(rid)
    return resolved


def _parse_answer(text: str) -> str:
    if "ANSWER:" not in text:
        return ""
    # Take the LAST occurrence of ANSWER: (models often repeat it after thinking)
    answer_part = text.rsplit("ANSWER:", 1)[1].strip()
    # Remove thinking blocks that might follow
    if "<think" in answer_part:
        answer_part = answer_part.split("<think")[0].strip()
    return answer_part


def _char_bigrams(text: str) -> set[str]:
    return set(text[i:i+2] for i in range(max(0, len(text) - 1)))


def _span_recall(pred_text: str, gold_spans: list[str]) -> float:
    if not gold_spans:
        return 1.0
    pred_lower = pred_text.lower()
    pred_bigrams = _char_bigrams(pred_lower)
    hits = 0
    for span in gold_spans:
        span_lower = span.lower()
        if span_lower in pred_lower:
            hits += 1
            continue
        span_bigrams = _char_bigrams(span_lower)
        if span_bigrams:
            overlap = len(span_bigrams & pred_bigrams) / len(span_bigrams)
            if overlap >= 0.5:
                hits += 1
    return hits / len(gold_spans)


def _faithfulness_score(answer: str, documents: list[dict]) -> float:
    if not answer or not documents:
        return 0.0
    # Split on sentence-ending punctuation AND newlines/markdown breaks
    raw_parts = re.split(r'[.!?。\n]+', answer)
    sentences = [s.strip() for s in raw_parts if len(s.strip()) > 5]
    if not sentences:
        return 0.0
    doc_text = " ".join(d.get("content", "") for d in documents).lower()
    doc_words = set(doc_text.split())
    if not doc_words:
        return 0.0
    supported = 0
    for sent in sentences:
        sent_words = set(sent.lower().split())
        if not sent_words:
            continue
        overlap = len(sent_words & doc_words) / len(sent_words)
        if overlap >= 0.5:
            supported += 1
    return supported / len(sentences)


class MDRetrievalEvaluator(BaseEvaluator):
    task_type = "md_retrieval"

    def build_prompt(self, sample: EvalSample) -> tuple[str, str]:
        docs = sample.artifacts.get("documents", [])
        doc_sections = []
        for d in docs:
            content = d.get("content", "")
            if content:
                doc_sections.append(f"### {d['doc_id']}\n{content}")
            else:
                doc_sections.append(f"### {d['doc_id']} ({d.get('path', '')})")
        doc_text = "\n\n".join(doc_sections)
        system_prompt = "너는 MD 문서 검색과 근거 기반 응답을 수행하는 평가용 어시스턴트다. 사고 과정 없이 바로 최종 답변만 출력하라. <think:> 태그를 절대 사용하지 마라."
        user_prompt = f"""질문:
{sample.user_query}

문서:
{doc_text}

반드시 다음 형식으로만 답하라 (다른 텍스트는 출력하지 마라):
DOC_IDS: [doc_id1, doc_id2]
ANSWER: 문서에서 찾은 근거를 그대로 인용하여 답변

주의: DOC_IDS에는 문서의 정확한 doc_id를 사용하라. ANSWER에는 문서의 원문을 최대한 그대로 포함하라.
"""
        return system_prompt, user_prompt

    def grade(self, sample: EvalSample, result: EvalResult) -> EvalResult:
        text = result.raw_output or ""
        gold_docs = set(sample.gold.get("relevant_doc_ids", []))
        gold_spans = sample.gold.get("relevant_spans", [])
        gold_answer = sample.gold.get("expected_answer", "")

        raw_doc_ids = _parse_doc_ids(text)
        docs = sample.artifacts.get("documents", [])
        predicted_docs = _resolve_doc_ids(raw_doc_ids, docs)

        # Fuzzy doc ID matching: strip extensions for comparison
        def _doc_match(pred, gold):
            if pred == gold:
                return True
            pred_base = pred.rsplit(".", 1)[0] if "." in pred else pred
            gold_base = gold.rsplit(".", 1)[0] if "." in gold else gold
            return pred_base == gold_base

        hit_at_3 = 1.0 if any(any(_doc_match(p, g) for g in gold_docs) for p in predicted_docs[:3]) else 0.0
        hit_at_1 = 1.0 if predicted_docs[:1] and any(_doc_match(predicted_docs[0], g) for g in gold_docs) else 0.0
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

        faithfulness = _faithfulness_score(answer_text, docs)
        result.metric_values["faithfulness"] = faithfulness

        result.final_success = hit_at_3 > 0 and span_score >= 0.5 and faithfulness >= 0.5
        if not result.final_success:
            if hit_at_3 == 0:
                result.failure_stage = "retrieval"
            elif faithfulness < 0.5:
                result.failure_stage = "faithfulness"
            else:
                result.failure_stage = "answer"
        return result

from __future__ import annotations
import re

from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult


def _extract_answer(text: str) -> str:
    if "ANSWER:" in text:
        parts = text.split("ANSWER:", 1)
        answer = parts[1].split("CITATIONS:")[0] if "CITATIONS:" in parts[1] else parts[1]
        return answer.strip()
    return ""


def _extract_citations(text: str) -> list[str]:
    citations = []
    if "CITATIONS:" in text:
        try:
            part = text.split("CITATIONS:", 1)[1].strip().split("\n")[0]
            bracket_match = re.search(r"\[(.*?)\]", part)
            if bracket_match:
                items = bracket_match.group(1)
            else:
                items = part
            citations = [x.strip(" [],'\"") for x in items.split(",") if x.strip(" [],'\"")]
        except Exception:
            pass
    url_pattern = re.findall(r"https?://[^\s)\]<>]+", text)
    citations.extend(url_pattern)
    return list(dict.fromkeys(citations))


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
        gold_content = sample.gold.get("required_content", [])
        gold_citations = sample.gold.get("required_citations", [])

        answer = _extract_answer(text)
        citations = _extract_citations(text)

        has_answer = 1.0 if len(answer) >= 20 else 0.0
        result.metric_values["answer_accuracy_proxy"] = has_answer

        if gold_content:
            content_hits = sum(1 for kw in gold_content if kw.lower() in answer.lower())
            result.metric_values["answer_accuracy"] = content_hits / len(gold_content)
        else:
            result.metric_values["answer_accuracy"] = has_answer

        has_citations = 1.0 if citations else 0.0
        result.metric_values["citation_support_proxy"] = has_citations

        if gold_citations:
            cit_hits = sum(1 for gc in gold_citations if any(gc in c for c in citations))
            result.metric_values["citation_support"] = cit_hits / len(gold_citations)
        else:
            result.metric_values["citation_support"] = has_citations

        result.final_success = (
            result.metric_values["answer_accuracy"] >= 0.5
            and result.metric_values["citation_support"] > 0
        )
        if not result.final_success:
            if result.metric_values["answer_accuracy"] < 0.5:
                result.failure_stage = "answer"
            else:
                result.failure_stage = "citation"
        return result

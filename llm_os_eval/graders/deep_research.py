from __future__ import annotations
import re

from duckduckgo_search import DDGS

from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult


def _web_search(query: str, max_results: int = 5) -> list[dict]:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception:
        return []


def _decompose_query(query: str) -> list[str]:
    subqueries = [query]
    separators = ["그리고", "또한", "및", "또는", ",", ";"]
    for sep in separators:
        expanded = []
        for sq in subqueries:
            parts = [p.strip() for p in sq.split(sep) if p.strip()]
            if len(parts) > 1:
                expanded.extend(parts)
            else:
                expanded.append(sq)
        subqueries = expanded
    entities = re.findall(r'[\w]+\s*(?:벤치마크|모델|평가|프레임워크|논문|성능|비교|순위|리더보드)', query)
    for ent in entities:
        if ent not in subqueries:
            subqueries.append(ent)
    return list(dict.fromkeys(subqueries))[:4]


def _extract_answer(text: str) -> str:
    if "ANSWER:" not in text:
        return ""
    answer_part = text.rsplit("ANSWER:", 1)[1].strip()
    if "CITATIONS:" in answer_part:
        answer_part = answer_part.split("CITATIONS:")[0].strip()
    if "<think" in answer_part:
        answer_part = answer_part.split("<think")[0].strip()
    return answer_part


def _extract_citations(text: str) -> list[str]:
    citations = []
    if "CITATIONS:" in text:
        try:
            part = text.split("CITATIONS:", 1)[1].strip()
            bracket_match = re.search(r"\[(.*?)\]", part, re.DOTALL)
            if bracket_match:
                items = bracket_match.group(1)
            else:
                items = part.split("\n")[0]
            citations = [x.strip(" [],'\"") for x in items.split(",") if x.strip(" [],'\"")]
        except Exception:
            pass
    url_pattern = re.findall(r"https?://[^\s)\]<>]+", text)
    citations.extend(url_pattern)
    return list(dict.fromkeys(citations))


class DeepResearchEvaluator(BaseEvaluator):
    task_type = "deep_research"

    MAX_SEARCH_CHARS = 800

    def build_prompt(self, sample: EvalSample) -> tuple[str, str]:
        subqueries = _decompose_query(sample.user_query)
        all_results = []
        seen_urls = set()
        for sq in subqueries:
            for r in _web_search(sq, max_results=3):
                href = r.get("href", "")
                if href not in seen_urls:
                    seen_urls.add(href)
                    all_results.append(r)
        search_context = ""
        if all_results:
            search_lines = []
            char_count = 0
            for i, r in enumerate(all_results[:5], 1):
                title = r.get("title", "")
                href = r.get("href", "")
                body = r.get("body", "")[:100]
                line = f"{i}. [{title}]({href})\n   {body}"
                if char_count + len(line) > self.MAX_SEARCH_CHARS:
                    break
                search_lines.append(line)
                char_count += len(line)
            search_context = "\n".join(search_lines)

        system_prompt = "너는 리서치 에이전트 평가용 모델이다. 사고 과정 없이 바로 최종 답변만 출력하라. <think:> 태그를 절대 사용하지 마라. 검색 결과를 바탕으로 정확한 정보와 출처 URL을 포함하라."
        context_block = f"""
검색 결과:
{search_context}
""" if search_context else "\n(검색 결과 없음 — 사전 지식으로 답변)\n"

        user_prompt = f"""질문:
{sample.user_query}
{context_block}
반드시 다음 형식으로만 답하라:
ANSWER: 질문에 대한 상세한 답변 (검색 결과의 구체적 내용을 포함)
CITATIONS: [https://example.com/source1, https://example.org/source2]

주의: ANSWER에는 검색 결과에서 확인한 구체적인 벤치마크/논문 이름을 포함하라. CITATIONS에는 검색 결과에 있는 실제 URL을 사용하라.
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
            and (result.metric_values["citation_support"] > 0 or not gold_citations)
        )
        if not result.final_success:
            if result.metric_values["answer_accuracy"] < 0.5:
                result.failure_stage = "answer"
            else:
                result.failure_stage = "citation"
        return result

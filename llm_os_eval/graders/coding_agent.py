from __future__ import annotations
import re

from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult


def _strip_thinking(text: str) -> str:
    # Remove <think:...>...</think:> and <think ...>...</think/> closed blocks
    cleaned = re.sub(r"<think[:\s][^>]*>.*?</think\s*>", "", text, flags=re.DOTALL)
    # Find first structured keyword
    first_structured = len(cleaned)
    for marker in ("FILES:", "PATCH:", "TESTS:"):
        idx = cleaned.find(marker)
        if idx >= 0 and idx < first_structured:
            first_structured = idx
    if first_structured < len(cleaned):
        return cleaned[first_structured:].strip()
    # No structured output found - return full cleaned text (might have file refs in prose)
    return cleaned.strip()


def _extract_file_list(text: str) -> list[str]:
    files = []
    # Parse FILES: [...] bracket notation
    if "FILES:" in text:
        try:
            part = text.split("FILES:", 1)[1].split("\n", 1)[0].strip()
            bracket_match = re.search(r"\[(.*?)\]", part)
            if bracket_match:
                items = bracket_match.group(1)
            else:
                items = part
            files = [x.strip(" [],'\"") for x in items.split(",") if x.strip(" [],'\"")]
        except Exception:
            pass
    # Scan all lines for file-path patterns (handles -, *, numbered lists, bare paths)
    path_pattern = re.compile(
        r"^(?:[-*]\s+|\d+[.)]\s+|)"          # optional list prefix
        r"(`{1,3})?"                          # optional opening backtick
        r"((?:src|lib|pkg|cmd|app|tests?|test)/[\w./_-]+\.\w+)"  # file path
        r"(?:\1)?$",                           # optional closing backtick
        re.IGNORECASE
    )
    for line in text.split("\n"):
        m = path_pattern.match(line.strip())
        if m:
            files.append(m.group(2))
    # Fallback: extract backticked .py file references in prose
    if not files:
        for m in re.finditer(r"`([\w/]+\.(?:py|js|ts|go|rs|java))`", text):
            files.append(m.group(1))
    # Fallback: extract any word.py pattern in text
    if not files:
        for m in re.finditer(r"(?:^|\s|['\"(\[])([\w]+\.py)(?:\s|$|['\")\],.])", text):
            files.append(m.group(1))
    return list(dict.fromkeys(files))


def _extract_patch(text: str) -> str:
    diff_match = re.search(r"```diff\s*(.*?)```", text, re.DOTALL)
    if diff_match:
        return diff_match.group(1).strip()
    patch_match = re.search(r"(---.*?\+\+\+.*?)(?=```|\Z)", text, re.DOTALL)
    if patch_match:
        return patch_match.group(1).strip()
    return ""


def _extract_test_list(text: str) -> list[str]:
    tests = []
    # Parse TESTS: [...] bracket notation
    if "TESTS:" in text:
        try:
            part = text.split("TESTS:", 1)[1].split("\n", 1)[0].strip()
            bracket_match = re.search(r"\[(.*?)\]", part)
            if bracket_match:
                items = bracket_match.group(1)
            else:
                items = part
            tests = [x.strip(" [],'\"") for x in items.split(",") if x.strip(" [],'\"")]
        except Exception:
            pass
    # Scan lines for test paths (handles various list formats and ::test_name)
    test_pattern = re.compile(
        r"^(?:[-*]\s+|\d+[.)]\s+|)`?"
        r"(tests?/[\w./_-]+(?:\s*::\s*\w+)?|test_\w+\.py(?:\s*::\s*\w+)?)"
        r"`?$",
        re.IGNORECASE
    )
    for line in text.split("\n"):
        m = test_pattern.match(line.strip())
        if m:
            tests.append(m.group(1).replace(" ", ""))
    # Fallback: extract bare test filenames from prose text
    if not tests:
        for m in re.finditer(r"\b(test_\w+\.py)\b", text):
            tests.append(m.group(1))
    return list(dict.fromkeys(tests))


def _normalize_test_name(name: str) -> list[str]:
    normalized = re.sub(r"[/.:_]", " ", name)
    return [w for w in normalized.lower().split() if w]


def _fuzzy_test_match(pred_tests: list[str], gold_tests: set[str], threshold: float = 0.6) -> int:
    hits = 0
    for gold in gold_tests:
        gold_words = set(_normalize_test_name(gold))
        if not gold_words:
            continue
        best_score = 0.0
        for pred in pred_tests:
            pred_words = set(_normalize_test_name(pred))
            if not pred_words:
                continue
            overlap = len(gold_words & pred_words)
            score = overlap / max(len(gold_words), len(pred_words))
            if score > best_score:
                best_score = score
        if best_score >= threshold:
            hits += 1
    return hits


class CodingAgentEvaluator(BaseEvaluator):
    task_type = "coding_agent"

    def build_prompt(self, sample: EvalSample) -> tuple[str, str]:
        system_prompt = "너는 repo-grounded coding agent 평가용 모델이다. 사고 과정 없이 바로 최종 답변만 출력하라. <think:> 태그를 절대 사용하지 마라."
        repo_tree = sample.artifacts.get("repo_tree", "")
        tree_section = f"\n저장소 구조:\n{repo_tree}\n" if repo_tree else ""
        user_prompt = f"""이슈:
{sample.artifacts.get('issue_text')}

작업 경로:
{sample.artifacts.get('workspace_path')}{tree_section}
요청:
{sample.user_query}

반드시 다음 형식으로만 답하라:
FILES: [file1.py, file2.py]
PATCH:
```diff
--- a/file1.py
+++ b/file1.py
@@ ... @@
-context line
+changed line
```
TESTS: [test_file.py::test_name]

주의:
- FILES에는 반드시 src/... 형태의 전체 경로를 사용하라. file1.py 같은 가짜 이름 금지.
- TESTS에는 버그와 직접 관련된 테스트 이름을 작성하라. 예: test_attachment_included (O), test_send_email (X).
- 세 섹션(FILES, PATCH, TESTS) 모두 반드시 출력하라.
"""
        return system_prompt, user_prompt

    def grade(self, sample: EvalSample, result: EvalResult) -> EvalResult:
        text = _strip_thinking(result.raw_output or "")
        gold_files = sample.gold.get("target_files", sample.gold.get("target_tests", []))
        gold_tests = set(sample.gold.get("target_tests", []))

        pred_files = _extract_file_list(text)
        pred_tests = _extract_test_list(text)
        pred_patch = _extract_patch(text)

        if gold_files:
            gold_set = set(gold_files)
            hits = sum(1 for f in pred_files if f in gold_set)
            if hits == 0 and pred_files:
                for gf in gold_set:
                    basename = gf.split("/")[-1]
                    if any(basename == pf.split("/")[-1] for pf in pred_files):
                        hits += 1
            recall = hits / len(gold_set) if gold_set else 0.0
        else:
            recall = 1.0 if pred_files else 0.0
        result.metric_values["file_selection_recall"] = recall

        has_patch = 1.0 if pred_patch else 0.0
        result.metric_values["patch_present"] = has_patch

        if gold_tests:
            pred_set = set(pred_tests)
            exact_hits = sum(1 for t in gold_tests if t in pred_set)
            test_recall = exact_hits / len(gold_tests) if gold_tests else 0.0
            if test_recall == 0 and pred_tests:
                fuzzy_hits = _fuzzy_test_match(pred_tests, gold_tests)
                test_recall = fuzzy_hits / len(gold_tests) if gold_tests else 0.0
        else:
            test_recall = 1.0 if pred_tests else 0.0
        result.metric_values["test_plan_recall"] = test_recall

        result.final_success = recall > 0 and test_recall > 0 and has_patch > 0
        if not result.final_success:
            if recall == 0:
                result.failure_stage = "file_selection"
            elif has_patch == 0:
                result.failure_stage = "patch_generation"
            elif test_recall == 0:
                result.failure_stage = "test_planning"
            else:
                result.failure_stage = "repo_reasoning"
        return result

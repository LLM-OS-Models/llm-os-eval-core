from __future__ import annotations
import re

from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult


def _extract_file_list(text: str) -> list[str]:
    files = []
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
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("- ") and "." in line:
            files.append(line[2:].strip())
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
    for line in text.split("\n"):
        line = line.strip()
        if re.match(r"(tests/|test_)", line):
            tests.append(line)
    return list(dict.fromkeys(tests))


class CodingAgentEvaluator(BaseEvaluator):
    task_type = "coding_agent"

    def build_prompt(self, sample: EvalSample) -> tuple[str, str]:
        system_prompt = "너는 repo-grounded coding agent 평가용 모델이다."
        user_prompt = f"""이슈:
{sample.artifacts.get('issue_text')}

작업 경로:
{sample.artifacts.get('workspace_path')}

요청:
{sample.user_query}

출력 형식:
FILES: [...]
PATCH_PLAN: ...
TESTS: [...]
"""
        return system_prompt, user_prompt

    def grade(self, sample: EvalSample, result: EvalResult) -> EvalResult:
        text = result.raw_output or ""
        gold_files = sample.gold.get("target_files", sample.gold.get("target_tests", []))
        gold_tests = set(sample.gold.get("target_tests", []))

        pred_files = _extract_file_list(text)
        pred_tests = _extract_test_list(text)
        pred_patch = _extract_patch(text)

        if gold_files:
            gold_set = set(gold_files)
            hits = sum(1 for f in pred_files if f in gold_set)
            recall = hits / len(gold_set) if gold_set else 0.0
        else:
            recall = 1.0 if pred_files else 0.0
        result.metric_values["file_selection_recall"] = recall

        has_patch = 1.0 if pred_patch else 0.0
        result.metric_values["patch_present"] = has_patch

        if gold_tests:
            pred_set = set(pred_tests)
            test_hits = sum(1 for t in gold_tests if t in pred_set)
            test_recall = test_hits / len(gold_tests) if gold_tests else 0.0
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

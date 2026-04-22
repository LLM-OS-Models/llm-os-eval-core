from __future__ import annotations
from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult

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
        has_files = 1.0 if "FILES:" in text else 0.0
        has_tests = 1.0 if "TESTS:" in text else 0.0
        result.metric_values["file_selection_recall_proxy"] = has_files
        result.metric_values["test_plan_present"] = has_tests
        result.final_success = has_files > 0 and has_tests > 0
        if not result.final_success:
            result.failure_stage = "repo_reasoning"
        return result

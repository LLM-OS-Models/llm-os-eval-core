from __future__ import annotations
from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult

class TerminalEvaluator(BaseEvaluator):
    task_type = "terminal"

    def build_prompt(self, sample: EvalSample) -> tuple[str, str]:
        system_prompt = "너는 터미널 에이전트 평가용 모델이다. 명령을 단계적으로 제안하라."
        user_prompt = f"""작업:
{sample.user_query}

환경:
- container_image: {sample.artifacts.get('container_image')}
- workspace_path: {sample.artifacts.get('workspace_path')}

출력 형식:
COMMANDS:
1. ...
2. ...
FINAL: ...
"""
        return system_prompt, user_prompt

    def grade(self, sample: EvalSample, result: EvalResult) -> EvalResult:
        text = result.raw_output or ""
        has_commands = 1.0 if "COMMANDS:" in text else 0.0
        result.metric_values["command_plan_present"] = has_commands
        result.final_success = has_commands > 0
        if not result.final_success:
            result.failure_stage = "planning"
        return result

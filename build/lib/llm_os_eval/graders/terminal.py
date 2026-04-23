from __future__ import annotations
import re

from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult


def _extract_commands(text: str) -> list[str]:
    commands = []
    patterns = [
        r"```(?:bash|sh|shell|zsh)?\s*\n(.*?)```",
        r"`([^`\n]+)`",
        r"^\s*\d+\.\s+(.+)$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        for m in matches:
            for line in m.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    commands.append(line)
    return commands


def _command_overlap(predicted: list[str], reference: list[str]) -> float:
    if not reference:
        return 0.0
    hits = 0
    for ref_cmd in reference:
        for pred_cmd in predicted:
            if ref_cmd.strip() in pred_cmd.strip() or pred_cmd.strip() in ref_cmd.strip():
                hits += 1
                break
    return hits / len(reference)


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
        gold_commands = sample.gold.get("commands", [])

        has_commands = 1.0 if "COMMANDS:" in text or bool(_extract_commands(text)) else 0.0
        result.metric_values["command_plan_present"] = has_commands

        predicted = _extract_commands(text)
        result.metric_values["command_count"] = float(len(predicted))

        if gold_commands:
            overlap = _command_overlap(predicted, gold_commands)
            result.metric_values["cmd_overlap"] = overlap
            result.final_success = overlap > 0.3
        else:
            result.metric_values["cmd_overlap"] = 0.0
            result.final_success = has_commands > 0

        if not result.final_success:
            result.failure_stage = "planning"
        return result

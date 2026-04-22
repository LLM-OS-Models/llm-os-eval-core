from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult
from llm_os_eval.runners.base import BaseRunner

class BaseEvaluator(ABC):
    task_type: str = "base"

    def __init__(self, runner: BaseRunner, model_name: str, checkpoint_name: str, prompt_version: str = "v1"):
        self.runner = runner
        self.model_name = model_name
        self.checkpoint_name = checkpoint_name
        self.prompt_version = prompt_version

    def load_jsonl(self, path: str | Path) -> list[EvalSample]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(EvalSample.model_validate_json(line))
        return samples

    @abstractmethod
    def build_prompt(self, sample: EvalSample) -> tuple[str, str]:
        raise NotImplementedError

    @abstractmethod
    def grade(self, sample: EvalSample, result: EvalResult) -> EvalResult:
        raise NotImplementedError

    def run_one(self, sample: EvalSample) -> EvalResult:
        system_prompt, user_prompt = self.build_prompt(sample)
        output = self.runner.generate(system_prompt, user_prompt)

        result = EvalResult(
            run_id=str(uuid.uuid4()),
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            model_name=self.model_name,
            checkpoint_name=self.checkpoint_name,
            prompt_version=self.prompt_version,
            raw_output=output.get("text", ""),
            parsed_output={},
            tool_calls=output.get("tool_calls", []),
            latency_ms=output.get("latency_ms", 0),
            input_tokens=output.get("input_tokens", 0),
            output_tokens=output.get("output_tokens", 0),
        )
        return self.grade(sample, result)

    def save_results(self, results: list[EvalResult], out_path: str | Path) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(r.model_dump_json() + "\n")

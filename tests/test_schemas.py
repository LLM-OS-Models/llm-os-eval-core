from __future__ import annotations
import json
import pytest
from pydantic import ValidationError
from llm_os_eval.schemas.sample import EvalSample, TaskType
from llm_os_eval.schemas.result import EvalResult


class TestEvalSample:
    def test_valid_sample_minimal(self):
        s = EvalSample(
            sample_id="test_001",
            task_type="tool_call",
            difficulty="medium",
            user_query="날씨를 조회해줘",
        )
        assert s.sample_id == "test_001"
        assert s.task_type == "tool_call"
        assert s.artifacts == {}
        assert s.gold == {}

    def test_valid_sample_full(self):
        s = EvalSample(
            sample_id="code_0001",
            task_type="coding_agent",
            difficulty="hard",
            user_query="로그인 버그 수정",
            artifacts={"repo_snapshot": "repos/auth.tar.gz"},
            gold={"target_tests": ["tests/test_login.py"]},
            grader={"timeout_sec": 1800},
            metadata={"language": "python"},
        )
        assert s.artifacts["repo_snapshot"] == "repos/auth.tar.gz"

    def test_all_task_types(self):
        for tt in [
            "md_retrieval", "tool_call", "terminal", "text2sql",
            "coding_agent", "docai_ocr", "deep_research",
        ]:
            s = EvalSample(sample_id="x", task_type=tt, difficulty="easy", user_query="q")
            assert s.task_type == tt

    def test_invalid_task_type_rejected(self):
        with pytest.raises(ValidationError):
            EvalSample(
                sample_id="x",
                task_type="invalid_type",
                difficulty="easy",
                user_query="q",
            )

    def test_invalid_difficulty_rejected(self):
        with pytest.raises(ValidationError):
            EvalSample(
                sample_id="x",
                task_type="terminal",
                difficulty="impossible",
                user_query="q",
            )

    def test_jsonl_roundtrip(self):
        original = EvalSample(
            sample_id="tool_0001",
            task_type="tool_call",
            difficulty="medium",
            user_query="API 호출",
            artifacts={"tools": [{"name": "get_weather"}]},
            gold={"tool_calls": [{"name": "get_weather"}]},
        )
        json_str = original.model_dump_json()
        restored = EvalSample.model_validate_json(json_str)
        assert restored.sample_id == original.sample_id
        assert restored.task_type == original.task_type
        assert restored.artifacts == original.artifacts


class TestEvalResult:
    def test_defaults(self):
        r = EvalResult(
            run_id="r1",
            sample_id="s1",
            task_type="terminal",
            model_name="test-model",
            checkpoint_name="base",
            prompt_version="v1",
        )
        assert r.raw_output is None
        assert r.final_success is False
        assert r.failure_stage is None
        assert r.metric_values == {}

    def test_full_result(self):
        r = EvalResult(
            run_id="r1",
            sample_id="s1",
            task_type="tool_call",
            model_name="Qwen3.5-4B",
            checkpoint_name="base",
            prompt_version="v1",
            raw_output='{"tool_calls": [{"name": "search"}]}',
            parsed_output={"tool_calls": [{"name": "search"}]},
            final_success=True,
            metric_values={"tool_selection_accuracy": 1.0},
            latency_ms=150,
            input_tokens=50,
            output_tokens=20,
        )
        assert r.final_success is True
        assert r.metric_values["tool_selection_accuracy"] == 1.0

    def test_jsonl_roundtrip(self):
        r = EvalResult(
            run_id="r2",
            sample_id="s2",
            task_type="md_retrieval",
            model_name="gemma-4-E2B-it",
            checkpoint_name="base",
            prompt_version="v1",
            raw_output="DOC_IDS: [doc_a]",
            final_success=True,
            metric_values={"file_hit_at_3": 1.0},
        )
        json_str = r.model_dump_json()
        restored = EvalResult.model_validate_json(json_str)
        assert restored.run_id == r.run_id
        assert restored.metric_values == r.metric_values

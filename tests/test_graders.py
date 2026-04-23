from __future__ import annotations
import pytest
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult
from llm_os_eval.graders.coding_agent import CodingAgentEvaluator
from llm_os_eval.graders.deep_research import DeepResearchEvaluator
from llm_os_eval.graders.docai_ocr import DocAIOCREvaluator
from llm_os_eval.graders.md_retrieval import MDRetrievalEvaluator
from llm_os_eval.graders.terminal import TerminalEvaluator
from llm_os_eval.graders.text2sql import Text2SQLEvaluator
from llm_os_eval.graders.tool_call import ToolCallEvaluator
from unittest.mock import MagicMock


def _make_runner_mock(response_text=""):
    runner = MagicMock()
    runner.generate.return_value = {
        "text": response_text,
        "tool_calls": [],
        "latency_ms": 100,
        "input_tokens": 10,
        "output_tokens": 20,
    }
    return runner


class TestCodingAgentEvaluator:
    def setup_method(self):
        self.runner = _make_runner_mock()
        self.evaluator = CodingAgentEvaluator(
            runner=self.runner, model_name="test", checkpoint_name="base"
        )
        self.sample = EvalSample(
            sample_id="code_001",
            task_type="coding_agent",
            difficulty="hard",
            user_query="로그인 버그 수정",
            artifacts={
                "issue_text": "Login fails with empty message",
                "workspace_path": "/workspace/auth",
            },
            gold={
                "target_files": ["auth.py"],
                "target_tests": ["tests/test_login.py"],
            },
        )

    def test_build_prompt(self):
        sys_prompt, user_prompt = self.evaluator.build_prompt(self.sample)
        assert "로그인 버그 수정" in user_prompt
        assert "Login fails" in user_prompt
        assert "/workspace/auth" in user_prompt

    def test_grade_success(self):
        result = EvalResult(
            run_id="r1", sample_id="code_001", task_type="coding_agent",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output="FILES: [auth.py]\nPATCH:\n```diff\n--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-old\n+new\n```\nTESTS: [tests/test_login.py]",
        )
        graded = self.evaluator.grade(self.sample, result)
        assert graded.final_success is True
        assert graded.metric_values["file_selection_recall"] == 1.0
        assert graded.metric_values["test_plan_recall"] == 1.0

    def test_grade_failure_no_files(self):
        result = EvalResult(
            run_id="r2", sample_id="code_001", task_type="coding_agent",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output="TESTS: [tests/test_login.py]",
        )
        graded = self.evaluator.grade(self.sample, result)
        assert graded.final_success is False
        assert graded.failure_stage == "file_selection"

    def test_run_one(self):
        self.runner.generate.return_value = {
            "text": "FILES: [auth.py]\nPATCH:\n```diff\n--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-old\n+new\n```\nTESTS: [tests/test_login.py]",
            "tool_calls": [], "latency_ms": 100, "input_tokens": 10, "output_tokens": 20,
        }
        result = self.evaluator.run_one(self.sample)
        assert result.final_success is True


class TestText2SQLEvaluator:
    def setup_method(self):
        self.runner = _make_runner_mock()
        self.evaluator = Text2SQLEvaluator(
            runner=self.runner, model_name="test", checkpoint_name="base"
        )
        self.sample = EvalSample(
            sample_id="sql_001",
            task_type="text2sql",
            difficulty="medium",
            user_query="환불률이 5%를 넘은 상품군 보여줘",
            artifacts={"db_path": "db/test.sqlite", "schema_docs": ["schema.md"]},
            gold={
                "result_hash": "abc123",
                "acceptable_sql_patterns": ["SELECT.*category.*FROM", "refund.*rate.*>.*0\\.05"],
            },
        )

    def test_build_prompt(self):
        sys_prompt, user_prompt = self.evaluator.build_prompt(self.sample)
        assert "환불률" in user_prompt
        assert "schema.md" in user_prompt

    def test_grade_parse_success(self):
        result = EvalResult(
            run_id="r1", sample_id="sql_001", task_type="text2sql",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output="```sql\nSELECT category FROM products WHERE refund_rate > 0.05\n```",
        )
        graded = self.evaluator.grade(self.sample, result)
        assert graded.metric_values["parse_success"] == 1.0

    def test_grade_parse_failure_empty(self):
        result = EvalResult(
            run_id="r2", sample_id="sql_001", task_type="text2sql",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output="",
        )
        graded = self.evaluator.grade(self.sample, result)
        assert graded.final_success is False
        assert graded.failure_stage == "sql_empty"

    def test_grade_schema_link(self):
        result = EvalResult(
            run_id="r3", sample_id="sql_001", task_type="text2sql",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output="SELECT category FROM products WHERE refund_rate > 0.05",
        )
        graded = self.evaluator.grade(self.sample, result)
        assert graded.metric_values["parse_success"] == 1.0
        assert graded.metric_values["schema_link_error"] == 0.0

    def test_grade_no_db_path(self):
        sample_no_db = EvalSample(
            sample_id="sql_002",
            task_type="text2sql",
            difficulty="easy",
            user_query="간단한 쿼리",
            artifacts={"db_path": "", "schema_docs": []},
            gold={"result_hash": "", "acceptable_sql_patterns": []},
        )
        result = EvalResult(
            run_id="r4", sample_id="sql_002", task_type="text2sql",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output="SELECT 1",
        )
        graded = self.evaluator.grade(sample_no_db, result)
        assert graded.final_success is True


class TestToolCallEvaluator:
    def setup_method(self):
        self.runner = _make_runner_mock()
        self.evaluator = ToolCallEvaluator(
            runner=self.runner, model_name="test", checkpoint_name="base"
        )
        self.sample = EvalSample(
            sample_id="tool_001",
            task_type="tool_call",
            difficulty="medium",
            user_query="서울 날씨 알려줘",
            artifacts={"tools": [{"name": "get_weather", "description": "Get weather"}]},
            gold={"tool_calls": [{"name": "get_weather", "arguments": {"city": "서울"}}]},
        )

    def test_build_prompt(self):
        sys_prompt, user_prompt = self.evaluator.build_prompt(self.sample)
        assert "서울 날씨" in user_prompt
        assert "get_weather" in user_prompt

    def test_grade_correct_tool(self):
        result = EvalResult(
            run_id="r1", sample_id="tool_001", task_type="tool_call",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output='{"tool_calls": [{"name": "get_weather", "arguments": {"city": "서울"}}]}',
        )
        graded = self.evaluator.grade(self.sample, result)
        assert graded.final_success is True
        assert graded.metric_values["tool_selection_accuracy"] == 1.0

    def test_grade_wrong_tool(self):
        result = EvalResult(
            run_id="r2", sample_id="tool_001", task_type="tool_call",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output='{"tool_calls": [{"name": "search_web", "arguments": {"q": "서울 날씨"}}]}',
        )
        graded = self.evaluator.grade(self.sample, result)
        assert graded.final_success is False
        assert graded.metric_values["tool_selection_accuracy"] == 0.0

    def test_grade_invalid_json(self):
        result = EvalResult(
            run_id="r3", sample_id="tool_001", task_type="tool_call",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output="this is not json",
        )
        graded = self.evaluator.grade(self.sample, result)
        assert graded.final_success is False
        assert graded.failure_stage == "parse"


class TestMDRetrievalEvaluator:
    def setup_method(self):
        self.runner = _make_runner_mock()
        self.evaluator = MDRetrievalEvaluator(
            runner=self.runner, model_name="test", checkpoint_name="base"
        )
        self.sample = EvalSample(
            sample_id="md_001",
            task_type="md_retrieval",
            difficulty="medium",
            user_query="배포 절차 설명",
            artifacts={"documents": [
                {"doc_id": "doc_a", "path": "docs/deploy.md", "content": "배포 절차는 다음과 같다. 코드 빌드 후 테스트를 거쳐 배포한다."},
                {"doc_id": "doc_b", "path": "docs/api.md", "content": "API 사용법 안내"},
            ]},
            gold={"relevant_doc_ids": ["doc_a"]},
        )

    def test_build_prompt(self):
        sys_prompt, user_prompt = self.evaluator.build_prompt(self.sample)
        assert "배포 절차" in user_prompt
        assert "doc_a" in user_prompt

    def test_grade_hit(self):
        result = EvalResult(
            run_id="r1", sample_id="md_001", task_type="md_retrieval",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output="DOC_IDS: [doc_a]\nANSWER: 배포 절차는...",
        )
        graded = self.evaluator.grade(self.sample, result)
        assert graded.final_success is True
        assert graded.metric_values["file_hit_at_3"] == 1.0

    def test_grade_miss(self):
        result = EvalResult(
            run_id="r2", sample_id="md_001", task_type="md_retrieval",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output="DOC_IDS: [doc_c, doc_d]\nANSWER: ...",
        )
        graded = self.evaluator.grade(self.sample, result)
        assert graded.final_success is False
        assert graded.failure_stage == "retrieval"


class TestDeepResearchEvaluator:
    def setup_method(self):
        self.runner = _make_runner_mock()
        self.evaluator = DeepResearchEvaluator(
            runner=self.runner, model_name="test", checkpoint_name="base"
        )
        self.sample = EvalSample(
            sample_id="res_001",
            task_type="deep_research",
            difficulty="hard",
            user_query="2025년 한국 경제 전망",
            gold={
                "required_content": ["GDP", "성장률"],
                "required_citations": ["https://"],
            },
        )

    def test_grade_success(self):
        result = EvalResult(
            run_id="r1", sample_id="res_001", task_type="deep_research",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output="ANSWER: 2025년 한국의 GDP 성장률은 약 2.3%로 전망됩니다. 이는 작년 대비 소폭 상승한 수치입니다.\nCITATIONS: [https://example.com/report]",
        )
        graded = self.evaluator.grade(self.sample, result)
        assert graded.final_success is True
        assert graded.metric_values["answer_accuracy"] >= 0.5
        assert graded.metric_values["citation_support"] > 0

    def test_grade_no_citations(self):
        result = EvalResult(
            run_id="r2", sample_id="res_001", task_type="deep_research",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output="ANSWER: 2025년 한국의 GDP 성장률은 약 2.3%로 전망됩니다.",
        )
        graded = self.evaluator.grade(self.sample, result)
        assert graded.final_success is False
        assert graded.failure_stage == "citation"


class TestDocAIOCREvaluator:
    def setup_method(self):
        self.runner = _make_runner_mock()
        self.evaluator = DocAIOCREvaluator(
            runner=self.runner, model_name="test", checkpoint_name="base"
        )
        self.sample = EvalSample(
            sample_id="ocr_001",
            task_type="docai_ocr",
            difficulty="medium",
            user_query="계약서에서 계약금을 추출하라",
            artifacts={"document_path": "docs/contract.pdf"},
            gold={"fields": {"계약금": "50,000,000원", "계약일": "2025-01-15"}},
        )

    def test_grade_partial_match(self):
        result = EvalResult(
            run_id="r1", sample_id="ocr_001", task_type="docai_ocr",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output="계약금: 50,000,000원입니다. 다른 내용...",
        )
        graded = self.evaluator.grade(self.sample, result)
        assert graded.metric_values["field_extraction_accuracy"] == 0.5
        assert graded.final_success is False

    def test_grade_full_match(self):
        result = EvalResult(
            run_id="r2", sample_id="ocr_001", task_type="docai_ocr",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output="계약금 50,000,000원, 계약일 2025-01-15",
        )
        graded = self.evaluator.grade(self.sample, result)
        assert graded.metric_values["field_extraction_accuracy"] == 1.0
        assert graded.final_success is True


class TestTerminalEvaluator:
    def setup_method(self):
        self.runner = _make_runner_mock()
        self.evaluator = TerminalEvaluator(
            runner=self.runner, model_name="test", checkpoint_name="base"
        )
        self.sample = EvalSample(
            sample_id="term_001",
            task_type="terminal",
            difficulty="medium",
            user_query="nginx 설치하고 설정",
            artifacts={"container_image": "ubuntu:22.04", "workspace_path": "/workspace"},
        )

    def test_build_prompt(self):
        sys_prompt, user_prompt = self.evaluator.build_prompt(self.sample)
        assert "nginx" in user_prompt
        assert "ubuntu:22.04" in user_prompt

    def test_grade_success(self):
        result = EvalResult(
            run_id="r1", sample_id="term_001", task_type="terminal",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output="COMMANDS:\n1. apt update\n2. apt install nginx\nFINAL: done",
        )
        graded = self.evaluator.grade(self.sample, result)
        assert graded.final_success is True
        assert graded.metric_values["command_plan_present"] == 1.0

    def test_grade_failure(self):
        result = EvalResult(
            run_id="r2", sample_id="term_001", task_type="terminal",
            model_name="test", checkpoint_name="base", prompt_version="v1",
            raw_output="nginx는 웹 서버입니다.",
        )
        graded = self.evaluator.grade(self.sample, result)
        assert graded.final_success is False
        assert graded.failure_stage == "planning"

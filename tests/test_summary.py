from __future__ import annotations
import json
import pytest
from llm_os_eval.reporters.summary import summarize_jsonl


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestSummarizeJsonl:
    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        result = summarize_jsonl(str(p))
        assert result["total"] == 0
        assert result["success_rate"] == 0.0

    def test_all_success(self, tmp_path):
        records = [
            {"final_success": True, "latency_ms": 100, "metric_values": {"acc": 1.0}},
            {"final_success": True, "latency_ms": 200, "metric_values": {"acc": 0.8}},
        ]
        p = tmp_path / "ok.jsonl"
        _write_jsonl(str(p), records)
        result = summarize_jsonl(str(p))
        assert result["total"] == 2
        assert result["success_rate"] == 1.0
        assert result["avg_latency_ms"] == 150.0
        assert result["metric_averages"]["acc"] == pytest.approx(0.9)

    def test_mixed(self, tmp_path):
        records = [
            {"final_success": True, "latency_ms": 50, "metric_values": {"a": 1.0, "b": 0.5}},
            {"final_success": False, "latency_ms": 150, "metric_values": {"a": 0.0}},
        ]
        p = tmp_path / "mix.jsonl"
        _write_jsonl(str(p), records)
        result = summarize_jsonl(str(p))
        assert result["total"] == 2
        assert result["success_rate"] == 0.5
        assert result["avg_latency_ms"] == 100.0
        assert result["metric_averages"]["a"] == pytest.approx(0.5)
        assert result["metric_averages"]["b"] == pytest.approx(0.5)

    def test_missing_fields_graceful(self, tmp_path):
        records = [
            {"final_success": True},
            {"final_success": False},
        ]
        p = tmp_path / "sparse.jsonl"
        _write_jsonl(str(p), records)
        result = summarize_jsonl(str(p))
        assert result["total"] == 2
        assert result["success_rate"] == 0.5
        assert result["avg_latency_ms"] == 0.0

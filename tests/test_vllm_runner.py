from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock
from llm_os_eval.runners.vllm_runner import VLLMRunner


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


class TestVLLMRunner:
    def test_generate_basic(self):
        runner = VLLMRunner(base_url="http://localhost:8000", model_name="test-model")
        mock_resp = _mock_response({
            "choices": [{
                "message": {
                    "content": "Hello",
                    "tool_calls": None,
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })
        with patch("llm_os_eval.runners.vllm_runner.httpx.post", return_value=mock_resp):
            result = runner.generate("sys", "user")

        assert result["text"] == "Hello"
        assert result["tool_calls"] == []
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 5
        assert result["latency_ms"] >= 0

    def test_generate_with_tools(self):
        runner = VLLMRunner(base_url="http://localhost:8000", model_name="test-model")
        mock_resp = _mock_response({
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": "{}"}}],
                }
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        })
        with patch("llm_os_eval.runners.vllm_runner.httpx.post", return_value=mock_resp) as mock_post:
            result = runner.generate("sys", "user", tools=[{"name": "search"}])

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "tools" in payload
        assert result["tool_calls"][0]["function"]["name"] == "search"

    def test_generate_empty_content(self):
        runner = VLLMRunner(base_url="http://localhost:8000", model_name="test-model")
        mock_resp = _mock_response({
            "choices": [{"message": {"content": None}}],
            "usage": {},
        })
        with patch("llm_os_eval.runners.vllm_runner.httpx.post", return_value=mock_resp):
            result = runner.generate("sys", "user")
        assert result["text"] == ""

    def test_base_url_trailing_slash_stripped(self):
        runner = VLLMRunner(base_url="http://localhost:8000/", model_name="test-model")
        mock_resp = _mock_response({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        })
        with patch("llm_os_eval.runners.vllm_runner.httpx.post", return_value=mock_resp) as mock_post:
            runner.generate("sys", "user")

        url = mock_post.call_args.kwargs.get("url") or mock_post.call_args[0][0]
        assert "//" not in url.replace("http://", "")

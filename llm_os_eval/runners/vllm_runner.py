from __future__ import annotations
import time
import httpx
from .base import BaseRunner

class VLLMRunner(BaseRunner):
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def generate(self, system_prompt, user_prompt, tools=None, max_tokens=1024, temperature=0.0):
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        t0 = time.time()
        resp = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=300.0)
        resp.raise_for_status()
        data = resp.json()
        dt = int((time.time() - t0) * 1000)
        msg = data["choices"][0]["message"]
        usage = data.get("usage", {})
        return {
            "text": msg.get("content", "") or "",
            "tool_calls": msg.get("tool_calls", []) or [],
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "latency_ms": dt,
        }

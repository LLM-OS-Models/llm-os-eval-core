#!/usr/bin/env python3
"""Generic single-GPU evaluator using transformers + eval-core graders.

Usage:
    python eval_runner.py --task-type tool_call --model Qwen/Qwen3.5-4B \
        --gpu 0 --eval-path eval/internal/v1.jsonl --output-dir results
"""
import json
import os
import sys
import time
import argparse

import torch
from datetime import datetime

# Add eval-core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "llm-os-eval-core"))

from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult

GRADER_MAP = {
    "coding_agent": "llm_os_eval.graders.coding_agent.CodingAgentEvaluator",
    "deep_research": "llm_os_eval.graders.deep_research.DeepResearchEvaluator",
    "docai_ocr": "llm_os_eval.graders.docai_ocr.DocAIOCREvaluator",
    "md_retrieval": "llm_os_eval.graders.md_retrieval.MDRetrievalEvaluator",
    "terminal": "llm_os_eval.graders.terminal.TerminalEvaluator",
    "text2sql": "llm_os_eval.graders.text2sql.Text2SQLEvaluator",
    "tool_call": "llm_os_eval.graders.tool_call.ToolCallEvaluator",
}


def _import_grader(task_type: str):
    import importlib
    if task_type not in GRADER_MAP:
        raise ValueError(f"Unknown task_type: {task_type}")
    module_path, class_name = GRADER_MAP[task_type].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class TransformersRunner:
    def __init__(self, model_name: str, gpu_id: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.device = "cuda:0"
        self.model_name = model_name
        print(f"Loading {model_name}...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map={"": self.device}, trust_remote_code=True,
        )
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        self.load_time = time.time() - t0
        print(f"Loaded {self.n_params:.1f}B params in {self.load_time:.0f}s")

    def generate(self, system_prompt: str, user_prompt: str,
                 tools=None, max_tokens=1024, temperature=0.0) -> dict:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        pred = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        latency = int((time.time() - t0) * 1000)
        return {
            "text": pred,
            "tool_calls": [],
            "input_tokens": inputs["input_ids"].shape[1],
            "output_tokens": out.shape[1] - inputs["input_ids"].shape[1],
            "latency_ms": latency,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", required=True, choices=list(GRADER_MAP.keys()))
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load samples
    samples = []
    with open(args.eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(EvalSample.model_validate_json(line))
    print(f"Loaded {len(samples)} samples for {args.task_type}")

    # Init runner + grader
    runner = TransformersRunner(args.model, args.gpu)
    GraderClass = _import_grader(args.task_type)
    evaluator = GraderClass(runner=runner, model_name=args.model, checkpoint_name="base")

    # Evaluate
    results = []
    t1 = time.time()
    for i, sample in enumerate(samples):
        sys_p, user_p = evaluator.build_prompt(sample)
        output = runner.generate(sys_p, user_p, max_tokens=args.max_tokens)
        result = EvalResult(
            run_id=f"eval_{i}", sample_id=sample.sample_id,
            task_type=sample.task_type, model_name=args.model,
            checkpoint_name="base", prompt_version="v1",
            raw_output=output.get("text", ""),
            latency_ms=output.get("latency_ms", 0),
            input_tokens=output.get("input_tokens", 0),
            output_tokens=output.get("output_tokens", 0),
        )
        result = evaluator.grade(sample, result)
        results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)} done")

    gen_time = time.time() - t1

    # Aggregate
    n = len(results)
    success = sum(1 for r in results if r.final_success)
    avg_latency = sum(r.latency_ms for r in results) / n if n else 0
    metric_avgs = {}
    for r in results:
        for k, v in r.metric_values.items():
            metric_avgs.setdefault(k, []).append(v)
    metric_avgs = {k: round(sum(v) / len(v), 4) for k, v in metric_avgs.items()}

    model_short = args.model.split("/")[-1]
    summary = {
        "model": args.model, "model_short": model_short,
        "task_type": args.task_type,
        "params_B": round(runner.n_params, 1),
        "gpu": args.gpu, "samples": n,
        "load_time_sec": round(runner.load_time),
        "gen_time_sec": round(gen_time, 1),
        "avg_sec_per_sample": round(gen_time / n, 2) if n else 0,
        "timestamp": datetime.now().isoformat(),
        "success_rate": round(success / n, 4) if n else 0,
        "avg_latency_ms": round(avg_latency, 1),
        "metric_averages": metric_avgs,
        "per_sample": [
            {
                "sample_id": r.sample_id,
                "difficulty": samples[i].difficulty,
                "final_success": r.final_success,
                "failure_stage": r.failure_stage,
                "metric_values": r.metric_values,
                "latency_ms": r.latency_ms,
                "pred_preview": (r.raw_output or "")[:200],
            }
            for i, r in enumerate(results)
        ],
    }

    out_path = os.path.join(args.output_dir, f"{model_short}_{args.task_type}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDONE {model_short} / {args.task_type}")
    print(f"  Load:{runner.load_time:.0f}s Gen:{gen_time:.1f}s {gen_time/n:.2f}s/sample" if n else "  No samples")
    print(f"  Success:{success}/{n} ({success/n:.1%})" if n else "  No samples")
    print(f"  Metrics: {metric_avgs}")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()

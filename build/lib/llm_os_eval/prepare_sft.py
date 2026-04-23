#!/usr/bin/env python3
"""Generic SFT data preparation: converts eval samples to training format.

Reads eval/internal/v1.jsonl and creates data/sft/train.jsonl and val.jsonl
in the format expected by sft_train.py:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    python prepare_sft.py --task-type tool_call --input eval/internal/v1.jsonl --output-dir data/sft --val-split 0.1
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "llm-os-eval-core"))

from llm_os_eval.schemas.sample import EvalSample

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
    module_path, class_name = GRADER_MAP[task_type].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _build_gold_response(sample: EvalSample) -> str:
    tt = sample.task_type
    gold = sample.gold

    if tt == "coding_agent":
        files = gold.get("target_files", gold.get("target_tests", []))
        tests = gold.get("target_tests", [])
        return f"FILES: {json.dumps(files, ensure_ascii=False)}\nPATCH_PLAN: 적절한 수정을 수행합니다.\nTESTS: {json.dumps(tests, ensure_ascii=False)}"

    if tt == "tool_call":
        return json.dumps({"tool_calls": gold.get("tool_calls", [])}, ensure_ascii=False)

    if tt == "md_retrieval":
        docs = gold.get("relevant_doc_ids", [])
        spans = gold.get("relevant_spans", [])
        answer = gold.get("expected_answer", "")
        return f"DOC_IDS: {json.dumps(docs, ensure_ascii=False)}\nANSWER: {answer}\n\n근거: {', '.join(spans)}"

    if tt == "deep_research":
        content = gold.get("required_content", [])
        citations = gold.get("required_citations", [])
        return f"ANSWER: {', '.join(content)}에 대한 분석 결과입니다. (상세 내용은 실제 학습 데이터로 대체 필요)\nCITATIONS: {json.dumps(citations, ensure_ascii=False)}"

    if tt == "docai_ocr":
        fields = gold.get("fields", {})
        lines = [f"{k}: {v}" for k, v in fields.items()]
        return "\n".join(lines)

    if tt == "text2sql":
        patterns = gold.get("acceptable_sql_patterns", [])
        if patterns:
            return f"SELECT * FROM table WHERE condition;"
        return "SELECT 1;"

    if tt == "terminal":
        commands = gold.get("commands", [])
        cmd_lines = "\n".join(f"{i+1}. {c}" for i, c in enumerate(commands))
        return f"COMMANDS:\n{cmd_lines}\nFINAL: done"

    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", required=True, choices=list(GRADER_MAP.keys()))
    parser.add_argument("--input", required=True, help="Input JSONL (eval samples)")
    parser.add_argument("--output-dir", default="data/sft")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", type=int, default=1, help="Augmentation factor")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    GraderClass = _import_grader(args.task_type)
    dummy_runner = None
    evaluator = GraderClass(runner=dummy_runner, model_name="dummy", checkpoint_name="base")

    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(EvalSample.model_validate_json(line))
    print(f"Loaded {len(samples)} samples")

    records = []
    for sample in samples:
        try:
            sys_prompt, user_prompt = evaluator.build_prompt(sample)
        except Exception:
            continue

        gold_response = _build_gold_response(sample)
        for _ in range(args.augment):
            records.append({
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": gold_response},
                ]
            })

    random.seed(args.seed)
    random.shuffle(records)

    val_count = max(1, int(len(records) * args.val_split))
    val_records = records[:val_count]
    train_records = records[val_count:]

    train_path = os.path.join(args.output_dir, "train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    val_path = os.path.join(args.output_dir, "val.jsonl")
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Train: {len(train_records)} -> {train_path}")
    print(f"Val: {len(val_records)} -> {val_path}")


if __name__ == "__main__":
    main()

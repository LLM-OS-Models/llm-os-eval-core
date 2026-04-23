from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from llm_os_eval.reporters.summary import summarize_jsonl
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult

app = typer.Typer(help="LLM-OS shared evaluation toolkit")
console = Console()

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
        raise ValueError(f"Unknown task_type: {task_type}. Choose from: {list(GRADER_MAP.keys())}")
    module_path, class_name = GRADER_MAP[task_type].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@app.command()
def hello():
    console.print("[green]llm-os-eval-core ready[/green]")


@app.command()
def summarize(result_path: str):
    result = summarize_jsonl(Path(result_path))
    console.print(json.dumps(result, indent=2, ensure_ascii=False))


@app.command()
def validate(samples_path: str):
    errors = 0
    total = 0
    with open(samples_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            total += 1
            try:
                EvalSample.model_validate_json(line)
            except Exception as e:
                console.print(f"[red]Line {i}:[/red] {e}")
                errors += 1
    console.print(f"[green]Validated {total} samples, {errors} errors[/green]")
    if errors > 0:
        raise typer.Exit(code=1)


@app.command()
def run(
    task_type: str = typer.Argument(..., help="Task type to evaluate"),
    model: str = typer.Option(..., "--model", "-m", help="Model name for vLLM API"),
    samples: str = typer.Option(..., "--samples", "-s", help="Path to samples JSONL"),
    output: str = typer.Option("results.jsonl", "--output", "-o", help="Output path"),
    base_url: str = typer.Option("http://localhost:8000", "--base-url", "-b", help="vLLM API base URL"),
    max_tokens: int = typer.Option(1024, "--max-tokens", help="Max generation tokens"),
    temperature: float = typer.Option(0.0, "--temperature", help="Sampling temperature"),
):
    from llm_os_eval.runners.vllm_runner import VLLMRunner

    GraderClass = _import_grader(task_type)
    runner = VLLMRunner(base_url=base_url, model_name=model)
    evaluator = GraderClass(runner=runner, model_name=model, checkpoint_name="base")

    samples_list = evaluator.load_jsonl(samples)
    results = []
    for i, sample in enumerate(samples_list):
        console.print(f"[{i+1}/{len(samples_list)}] {sample.sample_id}...", end=" ")
        try:
            result = evaluator.run_one(sample)
            status = "[green]OK[/green]" if result.final_success else "[yellow]FAIL[/yellow]"
            console.print(status)
            results.append(result)
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")

    evaluator.save_results(results, output)
    summary = summarize_jsonl(output)
    console.print(f"\nResults saved to {output}")
    console.print(f"Total: {summary['total']}, Success: {summary['success_rate']:.1%}")


@app.command()
def compare(
    baseline: str = typer.Argument(..., help="Baseline results JSONL"),
    sft: str = typer.Argument(..., help="SFT results JSONL"),
):
    def _load(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    base_rows = _load(baseline)
    sft_rows = _load(sft)

    base_by_id = {r["sample_id"]: r for r in base_rows}
    sft_by_id = {r["sample_id"]: r for r in sft_rows}

    common_ids = sorted(set(base_by_id) & set(sft_by_id))
    if not common_ids:
        console.print("[red]No common sample_ids found between files.[/red]")
        raise typer.Exit(code=1)

    base_success = sum(1 for sid in common_ids if base_by_id[sid].get("final_success"))
    sft_success = sum(1 for sid in common_ids if sft_by_id[sid].get("final_success"))

    all_metrics = set()
    for sid in common_ids:
        all_metrics.update(base_by_id[sid].get("metric_values", {}).keys())
        all_metrics.update(sft_by_id[sid].get("metric_values", {}).keys())

    table = Table(title=f"T_base vs T_sft ({len(common_ids)} samples)")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("SFT", justify="right")
    table.add_column("Delta", justify="right")

    table.add_row(
        "success_rate",
        f"{base_success / len(common_ids):.1%}",
        f"{sft_success / len(common_ids):.1%}",
        f"{(sft_success - base_success) / len(common_ids):+.1%}",
    )

    for metric in sorted(all_metrics):
        base_vals = [base_by_id[sid].get("metric_values", {}).get(metric, 0) for sid in common_ids]
        sft_vals = [sft_by_id[sid].get("metric_values", {}).get(metric, 0) for sid in common_ids]
        base_avg = sum(base_vals) / len(base_vals)
        sft_avg = sum(sft_vals) / len(sft_vals)
        delta = sft_avg - base_avg
        table.add_row(metric, f"{base_avg:.3f}", f"{sft_avg:.3f}", f"{delta:+.3f}")

    console.print(table)


if __name__ == "__main__":
    app()

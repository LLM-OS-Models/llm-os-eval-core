#!/usr/bin/env python3
"""Summarize evaluation results from multiple model runs.

Usage: python3 summarize.py [--results-dir DIR] [--output FILE]
"""
import json
import os
import argparse
from datetime import datetime


def summarize(results_dir="results", output_file=None):
    files = sorted(f for f in os.listdir(results_dir) if f.endswith(".json"))
    if not files:
        print("No result files found.")
        return

    rows = []
    for fname in files:
        path = os.path.join(results_dir, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metric_averages", {})
        rows.append({
            "model": data.get("model_short", fname.replace(".json", "")),
            "model_id": data.get("model", ""),
            "task_type": data.get("task_type", ""),
            "params_B": data.get("params_B", 0),
            "samples": data.get("samples", 0),
            "success_rate": data.get("success_rate", 0),
            "avg_latency_ms": data.get("avg_latency_ms", 0),
            "metrics": metrics,
        })

    rows.sort(key=lambda r: r["success_rate"], reverse=True)

    # Console output
    print(f"\n{'Model':<35} {'B':>5} {'Samples':>7} {'Success':>8} {'Latency':>8}")
    print("-" * 70)
    for r in rows:
        print(
            f"{r['model']:<35} {r['params_B']:>5.1f} {r['samples']:>7} "
            f"{r['success_rate']:>7.1%} {r['avg_latency_ms']:>6.0f}ms"
        )
    if rows and rows[0].get("metrics"):
        print(f"\nMetric averages:")
        all_keys = sorted(set(k for r in rows for k in r["metrics"]))
        header = f"{'Model':<35}" + "".join(f"{k:>12}" for k in all_keys)
        print(header)
        print("-" * len(header))
        for r in rows:
            vals = "".join(f"{r['metrics'].get(k, 0):>12.4f}" for k in all_keys)
            print(f"{r['model']:<35}{vals}")

    # Save markdown
    md_path = output_file or os.path.join(
        os.path.dirname(results_dir) or ".", "EVAL_SUMMARY.md"
    )
    task_type = rows[0]["task_type"] if rows else "unknown"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Evaluation Summary — {task_type}\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Sorted by success_rate.\n\n")
        f.write(f"| # | Model | Params | Samples | Success Rate |")
        if rows and rows[0].get("metrics"):
            for k in sorted(rows[0]["metrics"]):
                f.write(f" {k} |")
        f.write("\n")
        f.write(f"|---|-------|--------|---------|-------------|")
        if rows and rows[0].get("metrics"):
            f.write("".join(["---|" for _ in rows[0]["metrics"]]))
        f.write("\n")
        for i, r in enumerate(rows, 1):
            f.write(f"| {i} | `{r['model']}` | {r['params_B']:.1f}B | {r['samples']} | {r['success_rate']:.1%} |")
            if r.get("metrics"):
                for k in sorted(r["metrics"]):
                    f.write(f" {r['metrics'][k]:.4f} |")
            f.write("\n")
    print(f"\nSaved: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    summarize(args.results_dir, args.output)

from __future__ import annotations
import json
from pathlib import Path

def summarize_jsonl(path: str | Path) -> dict:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    total = len(rows)
    success = sum(1 for r in rows if r.get("final_success"))
    avg_latency = sum(r.get("latency_ms", 0) for r in rows) / total if total else 0
    metric_sums = {}
    metric_counts = {}
    for r in rows:
        for k, v in r.get("metric_values", {}).items():
            metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
            metric_counts[k] = metric_counts.get(k, 0) + 1
    metric_avgs = {k: metric_sums[k] / metric_counts[k] for k in metric_sums}
    return {
        "total": total,
        "success_rate": success / total if total else 0.0,
        "avg_latency_ms": avg_latency,
        "metric_averages": metric_avgs,
    }

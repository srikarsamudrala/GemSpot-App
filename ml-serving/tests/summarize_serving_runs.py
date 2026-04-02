#!/usr/bin/env python3
"""
Summarize latency and Locust outputs into a single JSON/markdown-friendly blob.

Usage:
    python tests/summarize_serving_runs.py \
      --label baseline_http \
      --benchmark benchmark_results.json \
      --locust-csv-prefix tests/loadtest
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_benchmark(path: Path) -> dict:
    payload = json.loads(path.read_text())
    by_candidates = payload.get("results", {})
    representative = by_candidates.get("50") or by_candidates.get(50) or {}
    return {
        "benchmark_url": payload.get("url"),
        "benchmark_rounds": payload.get("rounds"),
        "representative_candidates": 50,
        "latency_ms": {
            "p50": representative.get("p50"),
            "p95": representative.get("p95"),
            "p99": representative.get("p99"),
        },
    }


def load_locust(prefix: Path) -> dict:
    stats_path = prefix.parent / f"{prefix.name}_stats.csv"
    summary = {
        "throughput_rps": None,
        "error_rate_pct": None,
        "aggregated_p50_ms": None,
        "aggregated_p95_ms": None,
    }
    if not stats_path.exists():
        return summary

    with stats_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("Type") == "" and row.get("Name") == "Aggregated":
                summary["throughput_rps"] = row.get("Requests/s")
                summary["error_rate_pct"] = row.get("Failure Count")
                summary["aggregated_p50_ms"] = row.get("50%")
                summary["aggregated_p95_ms"] = row.get("95%")
                break
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GemSpot serving experiments")
    parser.add_argument("--label", required=True, help="Experiment label")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark_results.json")
    parser.add_argument("--locust-csv-prefix", required=True, help="Prefix passed to locust --csv")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    benchmark = load_benchmark(Path(args.benchmark))
    locust = load_locust(Path(args.locust_csv_prefix))
    summary = {
        "label": args.label,
        "benchmark": benchmark,
        "load_test": locust,
    }

    rendered = json.dumps(summary, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

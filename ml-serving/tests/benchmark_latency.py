#!/usr/bin/env python3
"""
GemSpot ML Service — Latency Benchmark

Measures P50/P95/P99 latencies for the /recommend endpoint
with varying candidate counts to characterize inference performance.

Usage:
    python benchmark_latency.py [--url URL] [--rounds ROUNDS]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time

import requests

ML_URL = "http://localhost:8050"

VIBE_TAGS = [
    "scenic", "relaxing", "adventurous", "cultural", "romantic",
    "family-friendly", "nightlife", "foodie", "historic", "trendy",
]

CATEGORIES = [
    "restaurant", "park", "museum", "cafe", "bar",
    "hotel", "attraction", "shopping", "beach", "temple",
]


def make_candidates(n: int) -> list[dict]:
    """Generate n synthetic candidate destinations."""
    candidates = []
    for i in range(n):
        tags = random.sample(VIBE_TAGS, 3)
        candidates.append({
            "gmap_id": f"bench_{i}",
            "name": f"Place {i}",
            "latitude": 40.7128 + random.uniform(-0.1, 0.1),
            "longitude": -74.006 + random.uniform(-0.1, 0.1),
            "category": random.choice(CATEGORIES),
            "avg_rating": round(random.uniform(1.0, 5.0), 1),
            "num_reviews": random.randint(10, 50000),
            "price": random.randint(0, 4),
            "vibe_tags": tags,
        })
    return candidates


def benchmark_request(url: str, candidates: list[dict]) -> float:
    """Send a single request and return latency in ms."""
    payload = {
        "user_id": "benchmark",
        "latitude": 40.7128,
        "longitude": -74.006,
        "candidates": candidates,
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{url}/recommend", json=payload, timeout=30)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    return elapsed_ms


def percentile(data: list[float], pct: float) -> float:
    """Calculate the given percentile from sorted data."""
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (pct / 100)
    f = int(k)
    c = f + 1
    if c >= len(data_sorted):
        return data_sorted[f]
    return data_sorted[f] + (k - f) * (data_sorted[c] - data_sorted[f])


def run_benchmark(url: str, rounds: int):
    # Check health first
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        health = resp.json()
        print(f"✅ ML Service: {health['status']} | Model: {health.get('model_version', 'N/A')}")
    except Exception as e:
        print(f"❌ ML Service unreachable: {e}")
        return

    candidate_counts = [1, 5, 10, 25, 50, 100, 200]
    print(f"\n{'='*75}")
    print(f"GemSpot ML Latency Benchmark — {rounds} rounds per candidate count")
    print(f"{'='*75}")
    print(f"{'Candidates':>12} {'Mean (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
    print(f"{'-'*75}")

    results = {}
    for n_candidates in candidate_counts:
        candidates = make_candidates(n_candidates)
        latencies = []

        # Warmup: 2 requests
        for _ in range(2):
            try:
                benchmark_request(url, candidates)
            except Exception:
                pass

        # Benchmark
        for r in range(rounds):
            try:
                ms = benchmark_request(url, candidates)
                latencies.append(ms)
            except Exception as e:
                print(f"  ⚠️  Request failed for {n_candidates} candidates, round {r}: {e}")

        if latencies:
            mean = statistics.mean(latencies)
            p50 = percentile(latencies, 50)
            p95 = percentile(latencies, 95)
            p99 = percentile(latencies, 99)
            mn = min(latencies)
            mx = max(latencies)
            results[n_candidates] = {
                "mean": mean, "p50": p50, "p95": p95, "p99": p99,
                "min": mn, "max": mx, "count": len(latencies),
            }
            print(f"{n_candidates:>12} {mean:>10.1f} {p50:>10.1f} {p95:>10.1f} {p99:>10.1f} {mn:>10.1f} {mx:>10.1f}")
        else:
            print(f"{n_candidates:>12} {'FAILED':>10}")

    print(f"{'='*75}")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "url": url,
        "rounds": rounds,
        "results": results,
    }
    outfile = "benchmark_results.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n📊 Results saved to {outfile}")

    # Print summary
    if results:
        fastest = min(results.values(), key=lambda x: x["p50"])
        slowest = max(results.values(), key=lambda x: x["p50"])
        print(f"\n📈 Summary:")
        print(f"   Fastest P50: {fastest['p50']:.1f}ms")
        print(f"   Slowest P50: {slowest['p50']:.1f}ms")
        if any(v["p95"] > 100 for v in results.values()):
            print(f"   ⚠️  Some P95 latencies exceed 100ms — consider optimizing for those candidate counts")
        else:
            print(f"   ✅ All P95 latencies under 100ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GemSpot ML Latency Benchmark")
    parser.add_argument("--url", default=ML_URL, help="ML service URL")
    parser.add_argument("--rounds", type=int, default=20, help="Number of rounds per test")
    args = parser.parse_args()
    run_benchmark(args.url, args.rounds)

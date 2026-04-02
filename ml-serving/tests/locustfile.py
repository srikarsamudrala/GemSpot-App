"""
GemSpot ML Service — Locust Load Test

Simulates concurrent users hitting the /recommend endpoint
to measure throughput, error rate, and latency under load.

Usage:
    locust -f locustfile.py --host http://localhost:8050

    # Headless mode (no UI):
    locust -f locustfile.py --host http://localhost:8050 \
        --headless -u 50 -r 5 --run-time 60s
"""
from __future__ import annotations

import random

from locust import HttpUser, between, task

VIBE_TAGS = [
    "scenic", "relaxing", "adventurous", "cultural", "romantic",
    "family-friendly", "nightlife", "foodie", "historic", "trendy",
]

CATEGORIES = [
    "restaurant", "park", "museum", "cafe", "bar",
    "hotel", "attraction", "shopping", "beach", "temple",
]


def make_candidates(n: int) -> list[dict]:
    candidates = []
    for i in range(n):
        candidates.append({
            "gmap_id": f"load_{i}_{random.randint(0, 99999)}",
            "name": f"Place {i}",
            "latitude": 40.7128 + random.uniform(-0.1, 0.1),
            "longitude": -74.006 + random.uniform(-0.1, 0.1),
            "category": random.choice(CATEGORIES),
            "avg_rating": round(random.uniform(1.0, 5.0), 1),
            "num_reviews": random.randint(10, 50000),
            "price": random.randint(0, 4),
            "vibe_tags": random.sample(VIBE_TAGS, 3),
        })
    return candidates


class GemSpotUser(HttpUser):
    """Simulates a user making recommendation requests."""

    wait_time = between(0.5, 2.0)  # seconds between requests

    @task(3)
    def recommend_small(self):
        """Most common: 10 candidates (typical search)."""
        payload = {
            "user_id": f"user_{random.randint(1, 1000)}",
            "latitude": 40.7128 + random.uniform(-0.5, 0.5),
            "longitude": -74.006 + random.uniform(-0.5, 0.5),
            "candidates": make_candidates(10),
        }
        self.client.post("/recommend", json=payload, name="/recommend [10 candidates]")

    @task(2)
    def recommend_medium(self):
        """Medium load: 50 candidates."""
        payload = {
            "user_id": f"user_{random.randint(1, 1000)}",
            "latitude": 40.7128 + random.uniform(-0.5, 0.5),
            "longitude": -74.006 + random.uniform(-0.5, 0.5),
            "candidates": make_candidates(50),
        }
        self.client.post("/recommend", json=payload, name="/recommend [50 candidates]")

    @task(1)
    def recommend_large(self):
        """Heavy load: 100 candidates."""
        payload = {
            "user_id": f"user_{random.randint(1, 1000)}",
            "latitude": 40.7128 + random.uniform(-0.5, 0.5),
            "longitude": -74.006 + random.uniform(-0.5, 0.5),
            "candidates": make_candidates(100),
        }
        self.client.post("/recommend", json=payload, name="/recommend [100 candidates]")

    @task(5)
    def health_check(self):
        """Lightweight health probes (most frequent)."""
        self.client.get("/health", name="/health")

    @task(1)
    def model_info(self):
        """Occasional model info check."""
        self.client.get("/model/info", name="/model/info")

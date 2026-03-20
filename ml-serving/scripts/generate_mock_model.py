#!/usr/bin/env python3
"""
Generate a mock XGBoost model for GemSpot development / demo.

Creates synthetic training data that matches the GemSpot feature schema
and trains a binary classifier (will_visit: 0 or 1).

Usage:
    python generate_mock_model.py [output_path]
    Defaults to /models/gemspot_xgb.json
"""
from __future__ import annotations

import os
import sys

import numpy as np
import xgboost as xgb

# Vibe tag vocabulary must match app/vibe_tags.py
VIBE_TAGS = [
    "scenic", "relaxing", "adventurous", "cultural", "romantic",
    "family-friendly", "nightlife", "foodie", "historic", "trendy",
]
NUM_VIBE_TAGS = len(VIBE_TAGS)
SCALAR_FEATURES = [
    "category_encoded",
    "avg_rating",
    "num_reviews",
    "price",
    "user_total_visits",
]
VIBE_FEATURE_NAMES = [f"vibe_{tag}" for tag in VIBE_TAGS]
PREF_FEATURE_NAMES = [f"pref_{tag}" for tag in VIBE_TAGS]
ALL_FEATURE_NAMES = SCALAR_FEATURES + VIBE_FEATURE_NAMES + PREF_FEATURE_NAMES
NUM_FEATURES = len(ALL_FEATURE_NAMES)  # 5 + 10 + 10 = 25


def _random_3hot(n: int) -> np.ndarray:
    """Return an (n, NUM_VIBE_TAGS) array with exactly 3 ones per row."""
    out = np.zeros((n, NUM_VIBE_TAGS), dtype=np.float32)
    for i in range(n):
        idx = np.random.choice(NUM_VIBE_TAGS, size=3, replace=False)
        out[i, idx] = 1.0
    return out


def generate_data(n: int = 5000):
    """Create synthetic training samples."""
    rng = np.random.default_rng(42)

    category = rng.integers(0, 15, size=(n, 1)).astype(np.float32)
    avg_rating = rng.uniform(1.0, 5.0, size=(n, 1)).astype(np.float32)
    num_reviews = rng.integers(0, 10000, size=(n, 1)).astype(np.float32)
    price = rng.choice([0, 1, 2, 3, 4], size=(n, 1)).astype(np.float32)
    user_visits = rng.integers(0, 200, size=(n, 1)).astype(np.float32)

    vibe = _random_3hot(n)
    prefs = rng.uniform(0, 50, size=(n, NUM_VIBE_TAGS)).astype(np.float32)

    X = np.hstack([category, avg_rating, num_reviews, price, user_visits, vibe, prefs])
    assert X.shape[1] == NUM_FEATURES

    # Synthetic label logic:
    # Higher rating + more reviews + vibe-preference overlap → higher chance of will_visit=1
    rating_signal = (avg_rating.flatten() - 3.0) / 2.0  # [-1, 1]
    review_signal = np.clip(np.log1p(num_reviews.flatten()) / 10, 0, 1)
    overlap = np.sum(vibe * (prefs > 10).astype(float), axis=1) / 3.0
    prob = 1 / (1 + np.exp(-(rating_signal + review_signal + overlap - 0.5) * 2))
    y = (rng.random(n) < prob).astype(np.float32)

    return X, y


def train_and_save(output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    X, y = generate_data()

    dtrain = xgb.DMatrix(X, label=y, feature_names=ALL_FEATURE_NAMES)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    bst = xgb.train(params, dtrain, num_boost_round=100)
    bst.save_model(output_path)
    print(f"✅ Mock XGBoost model saved to {output_path}")
    print(f"   Features: {NUM_FEATURES}  |  Samples: {len(y)}  |  Positive rate: {y.mean():.2%}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/models/gemspot_xgb.json"
    train_and_save(out)

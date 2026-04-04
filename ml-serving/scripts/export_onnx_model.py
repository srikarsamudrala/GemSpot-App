#!/usr/bin/env python3
"""
Export a GemSpot XGBoost booster to ONNX for serving experiments.

Usage:
    python scripts/export_onnx_model.py \
        --input /models/gemspot_xgb.json \
        --output /models/gemspot_xgb.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnxmltools
import xgboost as xgb
from onnxmltools.convert.common.data_types import FloatTensorType

VIBE_TAGS = [
    "scenic",
    "relaxing",
    "adventurous",
    "cultural",
    "romantic",
    "family-friendly",
    "nightlife",
    "foodie",
    "historic",
    "trendy",
]
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


def export_model(input_path: str, output_path: str) -> None:
    booster = xgb.Booster()
    booster.load_model(input_path)
    initial_types = [("input", FloatTensorType([None, len(ALL_FEATURE_NAMES)]))]
    onnx_model = onnxmltools.convert_xgboost(booster, initial_types=initial_types, target_opset=13)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(onnx_model.SerializeToString())
    print(f"Exported ONNX model to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GemSpot XGBoost booster to ONNX")
    parser.add_argument("--input", default="/models/gemspot_xgb.json", help="Input XGBoost model path")
    parser.add_argument("--output", default="/models/gemspot_xgb.onnx", help="Output ONNX model path")
    args = parser.parse_args()
    export_model(args.input, args.output)

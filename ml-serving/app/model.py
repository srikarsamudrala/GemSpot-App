"""
GemSpot ML Serving — Backend-aware model loader and scoring.

Supports multiple inference backends so we can benchmark meaningful serving
options while keeping the request/response contract stable.
"""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import numpy as np
import requests
import xgboost as xgb

from .config import MODEL_PATH, ONNX_MODEL_PATH, SERVING_BACKEND, TRITON_MODEL_NAME, TRITON_URL
from .inference_utils import ALL_FEATURE_NAMES, build_feature_matrix, build_gem_cards
from .schemas import CandidateDestination, GemCard
from .vibe_tags import NUM_VIBE_TAGS

logger = logging.getLogger("gemspot.model")

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency at runtime
    ort = None

MODEL_VERSION = "mock-v1.0"
SUPPORTED_BACKENDS = {"xgboost", "onnx", "triton"}


class GemSpotModel:
    """Wraps multiple inference backends behind one scoring interface."""

    def __init__(self) -> None:
        self._booster: Optional[xgb.Booster] = None
        self._onnx_session = None
        self._triton_output_name: Optional[str] = None
        self._loaded = False
        self.backend = SERVING_BACKEND if SERVING_BACKEND in SUPPORTED_BACKENDS else "xgboost"
        if self.backend != SERVING_BACKEND:
            logger.warning("Unknown serving backend '%s', defaulting to xgboost", SERVING_BACKEND)

    def load(self, path: str | None = None) -> None:
        if self.backend == "triton":
            self._load_triton()
        elif self.backend == "onnx":
            self._load_onnx(path or ONNX_MODEL_PATH)
        else:
            self._load_xgboost(path or MODEL_PATH)

    def _load_xgboost(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning("XGBoost model file not found at %s — predictions will be random", path)
            return
        self._booster = xgb.Booster()
        self._booster.load_model(path)
        self._loaded = True
        logger.info("Loaded XGBoost model from %s", path)

    def _load_onnx(self, path: str) -> None:
        if ort is None:
            logger.warning("onnxruntime is unavailable — predictions will be random")
            return
        if not os.path.exists(path):
            logger.warning("ONNX model file not found at %s — predictions will be random", path)
            return
        self._onnx_session = ort.InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
        )
        self._loaded = True
        logger.info("Loaded ONNX model from %s", path)

    def _load_triton(self) -> None:
        metadata_url = f"{TRITON_URL}/v2/models/{TRITON_MODEL_NAME}"
        try:
            resp = requests.get(metadata_url, timeout=5)
            resp.raise_for_status()
            metadata = resp.json()
            float_outputs = [
                output["name"]
                for output in metadata.get("outputs", [])
                if output.get("datatype", "").startswith("FP")
            ]
            self._triton_output_name = float_outputs[0] if float_outputs else None
            self._loaded = True
            logger.info(
                "Connected to Triton model %s at %s (output=%s)",
                TRITON_MODEL_NAME,
                TRITON_URL,
                self._triton_output_name,
            )
        except Exception as exc:
            logger.warning("Triton model unavailable at %s: %s", metadata_url, exc)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _predict_scores(self, features: np.ndarray) -> list[float]:
        if self.backend == "triton" and self._loaded:
            return self._predict_scores_triton(features)
        if self.backend == "onnx" and self._onnx_session is not None:
            input_name = self._onnx_session.get_inputs()[0].name
            outputs = self._onnx_session.run(None, {input_name: features})
            primary = outputs[0]
            scores = np.asarray(primary, dtype=np.float32).reshape(-1)
            return scores.tolist()

        if self._booster is not None:
            dmat = xgb.DMatrix(features, feature_names=ALL_FEATURE_NAMES)
            return self._booster.predict(dmat).tolist()

        return np.random.uniform(0.3, 0.95, size=len(features)).tolist()

    def _predict_scores_triton(self, features: np.ndarray) -> list[float]:
        infer_url = f"{TRITON_URL}/v2/models/{TRITON_MODEL_NAME}/infer"
        requested_outputs = [{"name": self._triton_output_name}] if self._triton_output_name else []
        payload = {
            "inputs": [
                {
                    "name": "input",
                    "shape": list(features.shape),
                    "datatype": "FP32",
                    "data": features.reshape(-1).tolist(),
                }
            ],
            "outputs": requested_outputs,
        }
        response = requests.post(infer_url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        outputs = data.get("outputs", [])
        if not outputs:
            raise RuntimeError("Triton response contained no outputs")

        primary_output = outputs[0]
        raw = primary_output.get("data", [])
        arr = np.asarray(raw, dtype=np.float32)
        if arr.size == len(features):
            return arr.reshape(-1).tolist()
        if arr.size == len(features) * 2:
            return arr.reshape(len(features), 2)[:, 1].tolist()
        if arr.size > len(features):
            return arr.reshape(len(features), -1)[:, -1].tolist()
        return arr.reshape(-1).tolist()

    def score_candidates(
        self,
        candidates: List[CandidateDestination],
        user_lat: float,
        user_lon: float,
        user_total_visits: int = 0,
        user_personal_preferences: Optional[List[float]] = None,
    ) -> List[GemCard]:
        if not candidates:
            return []

        t0 = time.perf_counter()
        features = build_feature_matrix(
            candidates=candidates,
            user_total_visits=user_total_visits,
            user_personal_preferences=user_personal_preferences,
        )
        scores = self._predict_scores(features)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        gem_cards = build_gem_cards(
            candidates=candidates,
            scores=scores,
            user_lat=user_lat,
            user_lon=user_lon,
        )
        logger.info(
            "Scored %d candidates in %.1f ms (backend=%s loaded=%s)",
            len(candidates),
            elapsed_ms,
            self.backend,
            self._loaded,
        )
        return gem_cards


model = GemSpotModel()

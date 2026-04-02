"""
GemSpot ML Serving — Backend-aware model loader and scoring.

Supports multiple inference backends so we can benchmark meaningful serving
options while keeping the request/response contract stable.
"""
from __future__ import annotations

import logging
import os
import time
from math import radians, cos, sin, asin, sqrt
from typing import List, Optional

import numpy as np
import xgboost as xgb

from .config import MODEL_PATH, ONNX_MODEL_PATH, SERVING_BACKEND
from .schemas import CandidateDestination, GemCard
from .vibe_tags import VIBE_TAGS, NUM_VIBE_TAGS, encode_vibe_tags, get_mock_vibe_tags

logger = logging.getLogger("gemspot.model")

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency at runtime
    ort = None


_CATEGORY_CODES: dict[str, int] = {
    "general": 0,
    "park": 1,
    "museum": 2,
    "restaurant": 3,
    "cafe": 4,
    "bar": 5,
    "hotel": 6,
    "hostel": 7,
    "attraction": 8,
    "beach": 9,
    "hiking_trail": 10,
    "nightclub": 11,
    "gallery": 12,
    "temple": 13,
    "market": 14,
    "tourist_attraction": 8,
    "art_gallery": 12,
    "lodging": 6,
    "food": 3,
    "tourism": 8,
}


def _encode_category(cat: str) -> int:
    return _CATEGORY_CODES.get(cat.lower().strip(), 0)


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

MODEL_VERSION = "mock-v1.0"
SUPPORTED_BACKENDS = {"xgboost", "onnx"}


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in km between two (lat, lon) pairs."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


class GemSpotModel:
    """Wraps multiple inference backends behind one scoring interface."""

    def __init__(self) -> None:
        self._booster: Optional[xgb.Booster] = None
        self._onnx_session = None
        self._loaded = False
        self.backend = SERVING_BACKEND if SERVING_BACKEND in SUPPORTED_BACKENDS else "xgboost"
        if self.backend != SERVING_BACKEND:
            logger.warning("Unknown serving backend '%s', defaulting to xgboost", SERVING_BACKEND)

    def load(self, path: str | None = None) -> None:
        if self.backend == "onnx":
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

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _build_feature_matrix(
        self,
        candidates: List[CandidateDestination],
        user_total_visits: int = 0,
        user_personal_preferences: Optional[List[float]] = None,
    ) -> np.ndarray:
        prefs: list[float]
        if user_personal_preferences is None:
            prefs = [0.0] * NUM_VIBE_TAGS
        else:
            prefs = [float(x) for x in user_personal_preferences[:NUM_VIBE_TAGS]]
        while len(prefs) < NUM_VIBE_TAGS:
            prefs.append(0.0)

        rows: list[list[float]] = []
        for candidate in candidates:
            vibe = candidate.vibe_tags if candidate.vibe_tags else get_mock_vibe_tags(candidate.category)
            vibe_vec = encode_vibe_tags(vibe)
            row = [
                float(_encode_category(candidate.category)),
                candidate.avg_rating,
                float(candidate.num_reviews),
                candidate.price,
                float(user_total_visits),
            ]
            row.extend(vibe_vec)
            row.extend(prefs)
            rows.append(row)

        return np.array(rows, dtype=np.float32)

    def _predict_scores(self, features: np.ndarray) -> list[float]:
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
        features = self._build_feature_matrix(
            candidates=candidates,
            user_total_visits=user_total_visits,
            user_personal_preferences=user_personal_preferences,
        )
        scores = self._predict_scores(features)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        gem_cards: list[GemCard] = []
        for candidate, raw_score in zip(candidates, scores):
            score = float(raw_score)
            vibe = candidate.vibe_tags if candidate.vibe_tags else get_mock_vibe_tags(candidate.category)
            distance = float(_haversine(user_lat, user_lon, candidate.latitude, candidate.longitude))
            gem_cards.append(
                GemCard(
                    gmap_id=candidate.gmap_id,
                    name=candidate.name,
                    latitude=candidate.latitude,
                    longitude=candidate.longitude,
                    score=round(score, 4),
                    confidence_pct=f"{int(score * 100)}%",
                    vibe_tags=list(vibe[:3]),
                    category=candidate.category,
                    distance_km=round(distance, 2),
                    avg_rating=candidate.avg_rating,
                    num_reviews=candidate.num_reviews,
                )
            )

        gem_cards.sort(key=lambda card: card.score, reverse=True)
        logger.info(
            "Scored %d candidates in %.1f ms (backend=%s loaded=%s)",
            len(candidates),
            elapsed_ms,
            self.backend,
            self._loaded,
        )
        return gem_cards


model = GemSpotModel()

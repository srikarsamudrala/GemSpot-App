"""
GemSpot ML Serving — XGBoost Model Loader & Scoring

Loads a trained XGBoost classifier and provides a scoring function that
accepts candidate destinations + user features and returns sorted Gem Cards.
"""
from __future__ import annotations

import logging
import os
import time
from math import radians, cos, sin, asin, sqrt
from typing import List, Optional

import numpy as np
import xgboost as xgb

from .config import MODEL_PATH
from .schemas import CandidateDestination, GemCard
from .vibe_tags import (
    VIBE_TAGS,
    NUM_VIBE_TAGS,
    encode_vibe_tags,
    get_mock_vibe_tags,
)

logger = logging.getLogger("gemspot.model")

# ── Category encoding (simple ordinal for XGBoost) ──────────────────────────
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


# ── Feature names (must match training order) ───────────────────────────────
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


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in km between two (lat, lon) pairs."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


class GemSpotModel:
    """Wraps an XGBoost Booster for GemSpot inference."""

    def __init__(self) -> None:
        self._booster: Optional[xgb.Booster] = None
        self._loaded = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def load(self, path: str | None = None) -> None:
        path = path or MODEL_PATH
        if not os.path.exists(path):
            logger.warning("Model file not found at %s — predictions will be random", path)
            return
        self._booster = xgb.Booster()
        self._booster.load_model(path)
        self._loaded = True
        logger.info("Loaded XGBoost model from %s", path)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── scoring ──────────────────────────────────────────────────────────────

    def score_candidates(
        self,
        candidates: List[CandidateDestination],
        user_lat: float,
        user_lon: float,
        user_total_visits: int = 0,
        user_personal_preferences: Optional[List[float]] = None,
    ) -> List[GemCard]:
        """
        Build the feature matrix, run inference, and return scored GemCards
        sorted descending by probability.
        """
        if not candidates:
            return []

        t0 = time.perf_counter()

        # Default user prefs
        prefs: list[float]
        if user_personal_preferences is None:
            prefs = [0.0] * NUM_VIBE_TAGS
        else:
            prefs = [float(x) for x in user_personal_preferences[:NUM_VIBE_TAGS]]
        while len(prefs) < NUM_VIBE_TAGS:
            prefs.append(0.0)

        rows: list[list[float]] = []
        for c in candidates:
            vibe = c.vibe_tags if c.vibe_tags else get_mock_vibe_tags(c.category)
            vibe_vec = encode_vibe_tags(vibe)

            row: list[float] = [
                float(_encode_category(c.category)),
                c.avg_rating,
                float(c.num_reviews),
                c.price,
                float(user_total_visits),
            ]
            row.extend(vibe_vec)
            row.extend(prefs)

            rows.append(row)

        X = np.array(rows, dtype=np.float32)

        # Predict
        if self._booster is not None:
            dmat = xgb.DMatrix(X, feature_names=ALL_FEATURE_NAMES)
            scores = self._booster.predict(dmat).tolist()
        else:
            # Fallback: random scores for demo
            scores = np.random.uniform(0.3, 0.95, size=len(candidates)).tolist()

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Build GemCards
        gem_cards: list[GemCard] = []
        for cand, raw_score in zip(candidates, scores):
            s = float(raw_score)
            vibe = cand.vibe_tags if cand.vibe_tags else get_mock_vibe_tags(cand.category)
            dist = float(_haversine(user_lat, user_lon, cand.latitude, cand.longitude))
            gem_cards.append(
                GemCard(
                    gmap_id=cand.gmap_id,
                    name=cand.name,
                    latitude=cand.latitude,
                    longitude=cand.longitude,
                    score=round(s, 4),
                    confidence_pct=f"{int(s * 100)}%",
                    vibe_tags=list(vibe[:3]),
                    category=cand.category,
                    distance_km=round(dist, 2),
                    avg_rating=cand.avg_rating,
                    num_reviews=cand.num_reviews,
                )
            )

        # Sort descending by score
        gem_cards.sort(key=lambda g: g.score, reverse=True)

        logger.info(
            "Scored %d candidates in %.1f ms (model_loaded=%s)",
            len(candidates),
            elapsed_ms,
            self._loaded,
        )
        return gem_cards


# ── Singleton ────────────────────────────────────────────────────────────────
model = GemSpotModel()

"""
Shared feature-building and response-formatting helpers for GemSpot serving.
"""
from __future__ import annotations

from math import radians, cos, sin, asin, sqrt
from typing import List, Optional

import numpy as np

from .schemas import CandidateDestination, GemCard
from .vibe_tags import VIBE_TAGS, NUM_VIBE_TAGS, encode_vibe_tags, get_mock_vibe_tags

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


def encode_category(category: str) -> int:
    return _CATEGORY_CODES.get(category.lower().strip(), 0)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def build_feature_matrix(
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
            float(encode_category(candidate.category)),
            candidate.avg_rating,
            float(candidate.num_reviews),
            candidate.price,
            float(user_total_visits),
        ]
        row.extend(vibe_vec)
        row.extend(prefs)
        rows.append(row)

    return np.array(rows, dtype=np.float32)


def build_gem_cards(
    candidates: List[CandidateDestination],
    scores: List[float],
    user_lat: float,
    user_lon: float,
) -> List[GemCard]:
    gem_cards: list[GemCard] = []
    for candidate, raw_score in zip(candidates, scores):
        score = float(raw_score)
        vibe = candidate.vibe_tags if candidate.vibe_tags else get_mock_vibe_tags(candidate.category)
        distance = float(haversine_km(user_lat, user_lon, candidate.latitude, candidate.longitude))
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
    return gem_cards

"""
GemSpot ML Serving — Pydantic Request / Response Schemas
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Request schemas ──────────────────────────────────────────────────────────

class CandidateDestination(BaseModel):
    """A single candidate destination to be scored by the model."""
    gmap_id: str
    name: str
    latitude: float
    longitude: float
    category: str = "general"
    avg_rating: float = 0.0
    num_reviews: int = 0
    price: float = 0.0
    vibe_tags: List[str] = Field(default_factory=list)


class RecommendRequest(BaseModel):
    """
    Recommendation request payload.

    The caller may supply pre-fetched candidates (e.g. from PostGIS) or
    omit them to let the service generate mock candidates for demo purposes.
    """
    user_id: str
    latitude: float = Field(..., description="User's current latitude")
    longitude: float = Field(..., description="User's current longitude")
    radius_km: float = Field(default=10.0, ge=0.1, le=100.0)
    top_k: int = Field(default=10, ge=1, le=50)
    category_filter: Optional[str] = None
    candidates: Optional[List[CandidateDestination]] = None

    # --- Mock user features (would come from Redis in production) ---
    user_total_visits: int = Field(default=0, description="Total visits by user")
    user_personal_preferences: Optional[List[float]] = Field(
        default=None,
        description="User preference vector (length = NUM_VIBE_TAGS)",
    )


# ── Response schemas ─────────────────────────────────────────────────────────

class GemCard(BaseModel):
    """A single scored destination returned to the frontend."""
    gmap_id: str
    name: str
    latitude: float
    longitude: float
    score: float = Field(..., description="Model probability (0-1)")
    confidence_pct: str = Field(..., description='Human-readable, e.g. "87%"')
    vibe_tags: List[str]
    category: str
    distance_km: float
    avg_rating: float
    num_reviews: int


class RecommendResponse(BaseModel):
    """Full recommendation response."""
    gem_cards: List[GemCard]
    model_version: str
    inference_time_ms: float
    candidates_scored: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_version: str
    serving_backend: str
    model_loaded: bool
    feature_names: List[str]
    num_vibe_tags: int
    vibe_tag_vocabulary: List[str]

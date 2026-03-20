"""
GemSpot ML Serving — FastAPI Application

Endpoints:
  GET  /health       — liveness / readiness probe
  POST /recommend    — score candidates and return sorted Gem Cards
  GET  /model/info   — model metadata
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import LOG_LEVEL, DEFAULT_TOP_K
from .model import model, MODEL_VERSION, ALL_FEATURE_NAMES
from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    RecommendRequest,
    RecommendResponse,
)
from .vibe_tags import VIBE_TAGS, NUM_VIBE_TAGS

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("gemspot")


# ── Lifespan (load model on startup) ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading GemSpot XGBoost model …")
    model.load()
    yield
    logger.info("Shutting down GemSpot ML service")


app = FastAPI(
    title="GemSpot ML Serving",
    description="Real-time XGBoost recommendation scoring for the GemSpot travel agent",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=model.is_loaded,
        model_version=MODEL_VERSION if model.is_loaded else None,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    return ModelInfoResponse(
        model_version=MODEL_VERSION,
        feature_names=ALL_FEATURE_NAMES,
        num_vibe_tags=NUM_VIBE_TAGS,
        vibe_tag_vocabulary=VIBE_TAGS,
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    t0 = time.perf_counter()

    candidates = req.candidates or []

    # Score candidates through the XGBoost model
    gem_cards = model.score_candidates(
        candidates=candidates,
        user_lat=req.latitude,
        user_lon=req.longitude,
        user_total_visits=req.user_total_visits,
        user_personal_preferences=req.user_personal_preferences,
    )

    # Apply top-k limit
    top_k = min(req.top_k, DEFAULT_TOP_K, len(gem_cards)) if gem_cards else 0
    gem_cards = gem_cards[:top_k] if top_k else gem_cards[:req.top_k]

    elapsed = (time.perf_counter() - t0) * 1000

    return RecommendResponse(
        gem_cards=gem_cards,
        model_version=MODEL_VERSION,
        inference_time_ms=round(elapsed, 2),
        candidates_scored=len(candidates),
    )

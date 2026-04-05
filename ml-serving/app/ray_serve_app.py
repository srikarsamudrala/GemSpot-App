"""
GemSpot Ray Serve bonus path.

Serves the same /recommend contract using Ray Serve and ONNX Runtime so it can
be benchmarked as a distinct serving framework on Chameleon.
"""
from __future__ import annotations

import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from ray import serve

from .config import ONNX_MODEL_PATH
from .inference_utils import ALL_FEATURE_NAMES, build_feature_matrix, build_gem_cards
from .model import MODEL_VERSION
from .schemas import HealthResponse, ModelInfoResponse, RecommendRequest, RecommendResponse
from .vibe_tags import NUM_VIBE_TAGS, VIBE_TAGS

fastapi_app = FastAPI(title="GemSpot Ray Serve", version=MODEL_VERSION)


@serve.deployment(ray_actor_options={"num_cpus": 1})
@serve.ingress(fastapi_app)
class GemSpotRayServe:
    def __init__(self) -> None:
        try:
            self.session = ort.InferenceSession(
                ONNX_MODEL_PATH,
                providers=["CPUExecutionProvider"],
            )
            self.model_loaded = True
            self.load_error = None
        except Exception as exc:  # pragma: no cover - runtime-only failure mode
            self.session = None
            self.model_loaded = False
            self.load_error = str(exc)

    @fastapi_app.get("/health", response_model=HealthResponse)
    async def health(self):
        return HealthResponse(
            status="healthy",
            model_loaded=self.model_loaded,
            model_version=MODEL_VERSION if self.model_loaded else None,
        )

    @fastapi_app.get("/model/info", response_model=ModelInfoResponse)
    async def model_info(self):
        return ModelInfoResponse(
            model_version=MODEL_VERSION,
            serving_backend="ray_serve",
            model_loaded=self.model_loaded,
            feature_names=ALL_FEATURE_NAMES,
            num_vibe_tags=NUM_VIBE_TAGS,
            vibe_tag_vocabulary=VIBE_TAGS,
        )

    @fastapi_app.post("/recommend", response_model=RecommendResponse)
    async def recommend(self, req: RecommendRequest):
        if self.session is None:
            raise HTTPException(status_code=503, detail=self.load_error or "Model unavailable")

        candidates = req.candidates or []
        t0 = time.perf_counter()
        features = build_feature_matrix(
            candidates=candidates,
            user_total_visits=req.user_total_visits,
            user_personal_preferences=req.user_personal_preferences,
        )
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: features})
        scores = np.asarray(outputs[0], dtype=np.float32).reshape(-1).tolist()
        gem_cards = build_gem_cards(
            candidates=candidates,
            scores=scores,
            user_lat=req.latitude,
            user_lon=req.longitude,
        )

        top_k = min(req.top_k, len(gem_cards)) if gem_cards else 0
        if top_k:
            gem_cards = gem_cards[:top_k]
        elapsed = (time.perf_counter() - t0) * 1000
        return RecommendResponse(
            gem_cards=gem_cards,
            model_version=MODEL_VERSION,
            inference_time_ms=round(elapsed, 2),
            candidates_scored=len(candidates),
        )


app = GemSpotRayServe.bind()

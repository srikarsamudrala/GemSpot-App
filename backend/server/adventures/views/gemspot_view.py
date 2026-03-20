"""
GemSpot ML — Django proxy view

Proxies recommendation requests to the GemSpot ML serving micro-service.
Stage 1 (PostGIS candidate generation) happens here; Stage 2 (XGBoost
scoring) is delegated to the FastAPI service.
"""
from __future__ import annotations

import logging
import time

import requests as http_requests
from django.conf import settings
from django.contrib.gis.db.models.functions import Distance
from django.contrib.gis.geos import Point
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from adventures.models import Location

logger = logging.getLogger(__name__)

# Sensible defaults
DEFAULT_RADIUS_KM = 10.0
DEFAULT_TOP_K = 10
ML_SERVICE_TIMEOUT = 10  # seconds


def _get_ml_url() -> str:
    return getattr(settings, "ML_SERVING_URL", "http://localhost:8050")


class GemSpotViewSet(viewsets.ViewSet):
    """
    GemSpot AI-powered recommendations.

    GET  /api/gemspot/           → list (proximity + ML scoring)
    GET  /api/gemspot/health/    → ML service health
    POST /api/gemspot/rate/      → feedback for retraining
    """

    permission_classes = [IsAuthenticated]

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_candidate_payload(locations):
        """Convert Django Location queryset into ML candidate dicts."""
        candidates = []
        for loc in locations:
            # Determine vibe tags from existing tags field
            vibe_tags = loc.tags[:3] if loc.tags else []
            candidates.append(
                {
                    "gmap_id": str(loc.id),
                    "name": loc.name,
                    "latitude": float(loc.latitude),
                    "longitude": float(loc.longitude),
                    "category": loc.category.name if loc.category else "general",
                    "avg_rating": float(loc.rating or 0),
                    "num_reviews": 0,  # Not tracked per-location in AdventureLog
                    "price": 0,
                    "vibe_tags": vibe_tags,
                }
            )
        return candidates

    # ── endpoints ────────────────────────────────────────────────────────

    def list(self, request):
        """
        Two-stage recommendation pipeline:
          Stage 1 — PostGIS spatial filter (this view)
          Stage 2 — XGBoost scoring (ML micro-service)

        Query params:
            lat       (required)  User latitude
            lon       (required)  User longitude
            radius_km (optional)  Search radius, default 10
            top_k     (optional)  Max results, default 10
            category  (optional)  Filter by category name
        """
        lat = request.query_params.get("lat")
        lon = request.query_params.get("lon")
        radius_km = float(request.query_params.get("radius_km", DEFAULT_RADIUS_KM))
        top_k = int(request.query_params.get("top_k", DEFAULT_TOP_K))
        category_filter = request.query_params.get("category")

        if not lat or not lon:
            return Response(
                {"error": "lat and lon query parameters are required."}, status=400
            )

        try:
            lat, lon = float(lat), float(lon)
        except ValueError:
            return Response({"error": "Invalid lat/lon values."}, status=400)

        # ── Stage 1: PostGIS candidate generation ────────────────────────
        user_point = Point(lon, lat, srid=4326)
        radius_m = radius_km * 1000

        candidates_qs = (
            Location.objects.filter(
                is_public=True,
                latitude__isnull=False,
                longitude__isnull=False,
            )
            .exclude(user=request.user)  # Don't recommend own locations
        )

        if category_filter:
            candidates_qs = candidates_qs.filter(category__name=category_filter)

        # Bounding-box pre-filter then distance annotation
        # PostGIS ST_DWithin for fast spatial query
        from django.contrib.gis.measure import D

        candidates_qs = candidates_qs.filter(
            latitude__range=(lat - radius_km / 111.0, lat + radius_km / 111.0),
            longitude__range=(
                lon - radius_km / (111.0 * max(abs(lat) * 0.0174533, 0.01)),
                lon + radius_km / (111.0 * max(abs(lat) * 0.0174533, 0.01)),
            ),
        )

        # Limit to a reasonable candidate pool
        candidates_qs = candidates_qs[:200]
        candidates = list(candidates_qs)

        if not candidates:
            return Response(
                {
                    "gem_cards": [],
                    "model_version": "n/a",
                    "inference_time_ms": 0,
                    "candidates_scored": 0,
                    "stage1_candidates": 0,
                    "message": "No nearby public locations found. Try increasing the search radius.",
                }
            )

        candidate_payload = self._build_candidate_payload(candidates)

        # ── Stage 2: ML scoring ──────────────────────────────────────────
        ml_url = f"{_get_ml_url()}/recommend"
        payload = {
            "user_id": str(request.user.uuid),
            "latitude": lat,
            "longitude": lon,
            "radius_km": radius_km,
            "top_k": top_k,
            "category_filter": category_filter,
            "candidates": candidate_payload,
            "user_total_visits": 0,  # Would come from Redis in production
            "user_personal_preferences": None,
        }

        t0 = time.perf_counter()
        try:
            resp = http_requests.post(ml_url, json=payload, timeout=ML_SERVICE_TIMEOUT)
            resp.raise_for_status()
            ml_data = resp.json()
        except http_requests.exceptions.ConnectionError:
            logger.warning("ML service unreachable at %s — returning unsorted candidates", ml_url)
            # Graceful fallback: return candidates without ML scoring
            fallback_cards = [
                {
                    "gmap_id": c["gmap_id"],
                    "name": c["name"],
                    "latitude": c["latitude"],
                    "longitude": c["longitude"],
                    "score": 0.5,
                    "confidence_pct": "N/A",
                    "vibe_tags": c.get("vibe_tags", []),
                    "category": c.get("category", "general"),
                    "distance_km": 0,
                    "avg_rating": c.get("avg_rating", 0),
                    "num_reviews": c.get("num_reviews", 0),
                }
                for c in candidate_payload[:top_k]
            ]
            return Response(
                {
                    "gem_cards": fallback_cards,
                    "model_version": "fallback",
                    "inference_time_ms": 0,
                    "candidates_scored": len(candidate_payload),
                    "stage1_candidates": len(candidates),
                    "warning": "ML service unavailable — results are unranked.",
                }
            )
        except Exception as e:
            logger.error("ML service error: %s", e)
            return Response({"error": "ML scoring failed. Please try again later."}, status=502)

        elapsed = (time.perf_counter() - t0) * 1000
        ml_data["stage1_candidates"] = len(candidates)
        ml_data["total_time_ms"] = round(elapsed, 2)

        return Response(ml_data)

    @action(detail=False, methods=["get"])
    def health(self, request):
        """Check if the ML serving micro-service is reachable."""
        try:
            resp = http_requests.get(
                f"{_get_ml_url()}/health", timeout=5
            )
            return Response(resp.json())
        except Exception as e:
            return Response(
                {"status": "unreachable", "error": str(e)}, status=503
            )

    @action(detail=False, methods=["post"])
    def rate(self, request):
        """
        Capture user feedback for ML retraining.

        Body:
            gmap_id  (str)   Destination ID
            rating   (int)   1-5 star rating
        """
        gmap_id = request.data.get("gmap_id")
        rating = request.data.get("rating")
        if not gmap_id or rating is None:
            return Response({"error": "gmap_id and rating are required."}, status=400)

        will_visit = 1 if int(rating) >= 4 else 0

        # In production this would push to Redpanda / Redis.
        # For now, just log it.
        logger.info(
            "GemSpot feedback: user=%s gmap_id=%s rating=%s will_visit=%s",
            request.user.uuid,
            gmap_id,
            rating,
            will_visit,
        )
        return Response(
            {"status": "recorded", "will_visit": will_visit}
        )

"""
GemSpot ML Serving — Vibe Tag Definitions & Encoding

Defines the fixed vibe-tag vocabulary used across training and inference.
Each destination gets a 3-hot binary vector of size V (len(VIBE_TAGS)).
"""
from __future__ import annotations

import random
from typing import List

# ── Fixed vibe-tag vocabulary (order matters — matches model feature indices) ──
VIBE_TAGS: list[str] = [
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

NUM_VIBE_TAGS = len(VIBE_TAGS)

# Tag-to-index lookup
_TAG_INDEX = {tag: idx for idx, tag in enumerate(VIBE_TAGS)}


def encode_vibe_tags(tags: List[str]) -> List[int]:
    """
    Encode a list of vibe-tag strings into a binary vector of length V.
    Only the top-3 tags (by order given) are set to 1; rest are 0.
    Unknown tags are silently ignored.
    """
    vec = [0] * NUM_VIBE_TAGS
    count = 0
    for tag in tags:
        idx = _TAG_INDEX.get(tag.lower().strip())
        if idx is not None and vec[idx] == 0:
            vec[idx] = 1
            count += 1
            if count >= 3:
                break
    return vec


def decode_vibe_tags(vec: List[int]) -> List[str]:
    """Convert a binary vibe-tag vector back to a list of tag strings."""
    return [VIBE_TAGS[i] for i, v in enumerate(vec) if v == 1]


# ── Category → typical vibe-tag mapping (used for mock data / cold-start) ──
_CATEGORY_VIBE_MAP: dict[str, list[str]] = {
    "park": ["scenic", "relaxing", "family-friendly"],
    "museum": ["cultural", "historic", "family-friendly"],
    "restaurant": ["foodie", "romantic", "trendy"],
    "cafe": ["relaxing", "foodie", "trendy"],
    "bar": ["nightlife", "trendy", "romantic"],
    "hotel": ["relaxing", "romantic", "family-friendly"],
    "hostel": ["adventurous", "trendy", "nightlife"],
    "attraction": ["scenic", "cultural", "family-friendly"],
    "beach": ["scenic", "relaxing", "romantic"],
    "hiking_trail": ["adventurous", "scenic", "relaxing"],
    "nightclub": ["nightlife", "trendy", "adventurous"],
    "gallery": ["cultural", "romantic", "historic"],
    "temple": ["cultural", "historic", "scenic"],
    "market": ["foodie", "cultural", "trendy"],
}


def get_mock_vibe_tags(category: str | None) -> List[str]:
    """
    Return simulated vibe tags for a given category.
    Used for demo/dev when no LLM pipeline is available.
    """
    if category and category.lower() in _CATEGORY_VIBE_MAP:
        return _CATEGORY_VIBE_MAP[category.lower()]
    # Random fallback
    return random.sample(VIBE_TAGS, 3)

"""The proposed AI Hybrid Policy (offline, versioned): P(fire) + one corroboration guardrail -> 3 states.

    FireDetectionCandidate + FireDetectionEventHistory
        -> FeatureExtractorV5 -> HGB -> P(fire)
        -> this policy -> NO_EVENT / SUSPECTED / CONFIRMED

    P(fire) <  0.40                                            NO_EVENT
    P(fire) >= 0.40 and CONFIRMED not satisfied                SUSPECTED   (possible wildfire: keep observing)
    P(fire) >= 0.80 and current satellite pixels >= 2          CONFIRMED

Pure and deterministic: it sees only a probability and the CURRENT candidate's hotspot count. Since Task 9B it is used at
runtime ONLY through FireDetectionAIHybridClassifierV5, and only when the explicit ai_hybrid_v5 decision mode is selected; it does
not change what rule_only / shadow / hybrid mean. Its thresholds are fixed in fire_detection_policy_config_v5 and were locked
before any confirmatory evaluation - runtime imports them, it never copies them.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from src.ml.fire_detection.fire_detection_policy_config_v5 import (
    CONFIRM_THRESHOLD,
    MULTI_PIXEL_COUNT_FEATURES,
    MULTI_PIXEL_MIN_SATELLITE_PIXELS,
    STATUS_CONFIRMED,
    STATUS_NO_EVENT,
    STATUS_SUSPECTED,
    SUSPECT_THRESHOLD,
)

STATUS_CODES = {STATUS_NO_EVENT: 0, STATUS_SUSPECTED: 1, STATUS_CONFIRMED: 2}
STATUS_NAMES = {code: name for name, code in STATUS_CODES.items()}


def current_satellite_pixel_count(features: Mapping[str, float]) -> int:
    """Hotspots of the CURRENT candidate = low + nominal + high confidence counts (history is never counted)."""
    return int(sum(features[name] for name in MULTI_PIXEL_COUNT_FEATURES))


def is_multi_pixel(satellite_pixel_count: float) -> bool:
    return satellite_pixel_count >= MULTI_PIXEL_MIN_SATELLITE_PIXELS


def decide_status(probability: float, satellite_pixel_count: float) -> str:
    """Map one (P(fire), current satellite pixel count) to NO_EVENT / SUSPECTED / CONFIRMED."""
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or math.isnan(probability):
        raise ValueError(f"probability must be a number in [0, 1], got {probability!r}")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be within [0, 1], got {probability!r}")
    if satellite_pixel_count < 0:
        raise ValueError(f"satellite_pixel_count must be >= 0, got {satellite_pixel_count!r}")
    if probability >= CONFIRM_THRESHOLD and is_multi_pixel(satellite_pixel_count):
        return STATUS_CONFIRMED
    if probability >= SUSPECT_THRESHOLD:
        return STATUS_SUSPECTED
    return STATUS_NO_EVENT


def decide_statuses(probabilities: Sequence[float], satellite_pixel_counts: Sequence[float]) -> np.ndarray:
    """Vectorised `decide_status`; returns integer codes (0 NO_EVENT, 1 SUSPECTED, 2 CONFIRMED)."""
    p = np.asarray(probabilities, dtype=float)
    counts = np.asarray(satellite_pixel_counts, dtype=float)
    if p.shape != counts.shape:
        raise ValueError("probabilities and satellite_pixel_counts must have the same length.")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must all be within [0, 1].")
    if (counts < 0).any():
        raise ValueError("satellite_pixel_counts must be >= 0.")
    codes = np.where(p >= SUSPECT_THRESHOLD, STATUS_CODES[STATUS_SUSPECTED], STATUS_CODES[STATUS_NO_EVENT])
    confirmed = (p >= CONFIRM_THRESHOLD) & (counts >= MULTI_PIXEL_MIN_SATELLITE_PIXELS)
    return np.where(confirmed, STATUS_CODES[STATUS_CONFIRMED], codes).astype(int)

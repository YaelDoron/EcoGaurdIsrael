"""V3 feature-group definitions for the satellite-physical vs news-semantic ablation (Part 37).

This is an experiment only - it does not change FIRE_DETECTION_FEATURE_NAMES_V3
(the official V3 schema, which equals FEATURE_GROUP_FULL_V3 below).
"""
from __future__ import annotations

from src.ml.fire_detection.fire_detection_features_v3 import FIRE_DETECTION_FEATURE_NAMES_V3

# Not part of the official V3 schema - derived, on the fly, only for the BASE
# and BASE+SATELLITE_PHYSICAL ablation configurations below, as the sum of
# the 5 V3 news_*_count features. It stands in for V2's plain `news_count`
# (a coarse "news exists" presence signal, no semantic detail) so BASE is a
# fair like-for-like comparison point rather than silently missing all news
# information. It is never stored as its own CSV/schema column because it is
# exactly reconstructable from the 5 semantic counts - the same redundancy
# reasoning that already excluded satellite_count from V3.
_DERIVED_NEWS_TOTAL_COUNT = "news_total_count"

BASE_FEATURE_NAMES: tuple[str, ...] = (
    "satellite_low_count",
    "satellite_nominal_count",
    "satellite_high_count",
    _DERIVED_NEWS_TOTAL_COUNT,
    "time_span_minutes",
    "max_pairwise_distance_km",
)

SATELLITE_PHYSICAL_FEATURE_NAMES: tuple[str, ...] = (
    "satellite_frp_available_ratio",
    "satellite_frp_mean",
    "satellite_frp_max",
    "satellite_brightness_available_ratio",
    "satellite_brightness_mean",
    "satellite_brightness_max",
)

NEWS_SEMANTIC_FEATURE_NAMES: tuple[str, ...] = (
    "news_none_count",
    "news_weak_count",
    "news_moderate_count",
    "news_strong_count",
    "news_unknown_count",
)

FEATURE_GROUP_BASE = "base"
FEATURE_GROUP_BASE_PLUS_SATELLITE_PHYSICAL = "base_plus_satellite_physical"
FEATURE_GROUP_BASE_PLUS_NEWS_SEMANTIC = "base_plus_news_semantic"
FEATURE_GROUP_FULL_V3 = "full_v3"

_SATELLITE_CONFIDENCE_NAMES = ("satellite_low_count", "satellite_nominal_count", "satellite_high_count")
_GEOMETRY_NAMES = ("time_span_minutes", "max_pairwise_distance_km")

# FEATURE_GROUP_FULL_V3 is, by construction, exactly FIRE_DETECTION_FEATURE_NAMES_V3 -
# the official V3 schema is the "everything" ablation configuration.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    FEATURE_GROUP_BASE: BASE_FEATURE_NAMES,
    FEATURE_GROUP_BASE_PLUS_SATELLITE_PHYSICAL: (
        _SATELLITE_CONFIDENCE_NAMES
        + SATELLITE_PHYSICAL_FEATURE_NAMES
        + (_DERIVED_NEWS_TOTAL_COUNT,)
        + _GEOMETRY_NAMES
    ),
    FEATURE_GROUP_BASE_PLUS_NEWS_SEMANTIC: (
        _SATELLITE_CONFIDENCE_NAMES + NEWS_SEMANTIC_FEATURE_NAMES + _GEOMETRY_NAMES
    ),
    FEATURE_GROUP_FULL_V3: FIRE_DETECTION_FEATURE_NAMES_V3,
}

_V3_INDEX = {name: index for index, name in enumerate(FIRE_DETECTION_FEATURE_NAMES_V3)}
_NEWS_COUNT_NAMES = NEWS_SEMANTIC_FEATURE_NAMES


def _feature_value(full_row: tuple[float, ...], name: str) -> float:
    if name == _DERIVED_NEWS_TOTAL_COUNT:
        return sum(full_row[_V3_INDEX[news_name]] for news_name in _NEWS_COUNT_NAMES)
    return full_row[_V3_INDEX[name]]


def build_feature_matrix(full_feature_rows: list[tuple[float, ...]], group_name: str) -> list[list[float]]:
    """Select/derive one ablation feature group's columns from full V3-ordered rows.

    full_feature_rows: each row must be in FIRE_DETECTION_FEATURE_NAMES_V3 order
    (e.g. FireDetectionFeaturesV3.as_tuple()). Returns row-aligned columns for
    the named feature group.
    """
    if group_name not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group {group_name!r}; expected one of {sorted(FEATURE_GROUPS)}.")
    feature_names = FEATURE_GROUPS[group_name]
    return [[_feature_value(row, name) for name in feature_names] for row in full_feature_rows]

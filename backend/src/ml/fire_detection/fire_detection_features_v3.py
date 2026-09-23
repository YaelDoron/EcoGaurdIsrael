"""V3 numeric feature representation for Fire Detection ML classifiers.

V3 is a NEW, separately versioned feature schema - it does not replace or
redefine FIRE_DETECTION_FEATURE_NAMES (V1/V2's 7-feature schema in
fire_detection_features.py), which remains unchanged so training_v1.csv and
training_v2.csv, and their saved models, stay reproducible.

V3 exists to fix the representation limitation V2's grouped evaluation
exposed (see backend/docs/fire_detection_feature_representation_v3.md):
V1/V2 evidence carried only satellite confidence label and evidence counts,
so an isolated single-item candidate (e.g. one lone news report, or one
satellite hotspot) was nearly indistinguishable between a real early fire
and a false alarm. V3 exposes selected existing satellite physical
measurements (FRP, brightness) and a structured news semantic signal
(NONE/WEAK/MODERATE/STRONG) that Task 4 added to FireDetectionEvidence,
without redesigning correlation or the geometry features.

`satellite_count` is deliberately NOT a V3 feature: Task 3 showed it is
exactly reconstructable from the three confidence-specific counts below, and
its removal had negligible effect on model performance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

FIRE_DETECTION_FEATURE_NAMES_V3: tuple[str, ...] = (
    "satellite_low_count",
    "satellite_nominal_count",
    "satellite_high_count",
    "satellite_frp_available_ratio",
    "satellite_frp_mean",
    "satellite_frp_max",
    "satellite_brightness_available_ratio",
    "satellite_brightness_mean",
    "satellite_brightness_max",
    "news_none_count",
    "news_weak_count",
    "news_moderate_count",
    "news_strong_count",
    "news_unknown_count",
    "time_span_minutes",
    "max_pairwise_distance_km",
)

SAMPLE_ID_COLUMN = "sample_id"
SCENARIO_FAMILY_COLUMN = "scenario_family"
# Higher-level evidence-composition grouping (Part 26), metadata only - see
# fire_detection_training_data_generator_v3.ScenarioArchetype. Never a feature.
SCENARIO_ARCHETYPE_COLUMN = "scenario_archetype"
LABEL_COLUMN = "label"

TRAINING_DATA_METADATA_COLUMNS_V3: tuple[str, ...] = (
    SAMPLE_ID_COLUMN,
    SCENARIO_FAMILY_COLUMN,
    SCENARIO_ARCHETYPE_COLUMN,
)

TRAINING_DATA_CSV_COLUMNS_V3: tuple[str, ...] = (
    TRAINING_DATA_METADATA_COLUMNS_V3 + FIRE_DETECTION_FEATURE_NAMES_V3 + (LABEL_COLUMN,)
)

_COUNT_FIELD_NAMES = (
    "satellite_low_count",
    "satellite_nominal_count",
    "satellite_high_count",
    "news_none_count",
    "news_weak_count",
    "news_moderate_count",
    "news_strong_count",
    "news_unknown_count",
)
_RATIO_FIELD_NAMES = ("satellite_frp_available_ratio", "satellite_brightness_available_ratio")
_NON_NEGATIVE_FLOAT_FIELD_NAMES = (
    "satellite_frp_mean",
    "satellite_frp_max",
    "satellite_brightness_mean",
    "satellite_brightness_max",
    "time_span_minutes",
    "max_pairwise_distance_km",
)


@dataclass(frozen=True)
class FireDetectionFeaturesV3:
    """One Fire Detection evidence candidate reduced to the V3 tabular feature vector.

    `*_available_ratio` fields explicitly distinguish "measurement missing"
    from "measurement is low" - see FireDetectionFeatureExtractorV3. Field
    order matches FIRE_DETECTION_FEATURE_NAMES_V3.
    """

    satellite_low_count: int
    satellite_nominal_count: int
    satellite_high_count: int

    satellite_frp_available_ratio: float
    satellite_frp_mean: float
    satellite_frp_max: float

    satellite_brightness_available_ratio: float
    satellite_brightness_mean: float
    satellite_brightness_max: float

    news_none_count: int
    news_weak_count: int
    news_moderate_count: int
    news_strong_count: int
    news_unknown_count: int

    time_span_minutes: float
    max_pairwise_distance_km: float

    def __post_init__(self) -> None:
        for field_name in _COUNT_FIELD_NAMES:
            self._validate_count(field_name, getattr(self, field_name))
        for field_name in _RATIO_FIELD_NAMES:
            self._validate_ratio(field_name, getattr(self, field_name))
        for field_name in _NON_NEGATIVE_FLOAT_FIELD_NAMES:
            self._validate_non_negative_float(field_name, getattr(self, field_name))

    def as_tuple(self) -> tuple[float, ...]:
        """Return the feature vector in FIRE_DETECTION_FEATURE_NAMES_V3 order."""
        return tuple(getattr(self, name) for name in FIRE_DETECTION_FEATURE_NAMES_V3)

    def as_dict(self) -> dict[str, float]:
        """Return the feature vector as a name -> value mapping."""
        return {name: getattr(self, name) for name in FIRE_DETECTION_FEATURE_NAMES_V3}

    @staticmethod
    def _validate_count(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")

    @staticmethod
    def _validate_ratio(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name} must be within [0, 1], got {value!r}")

    @staticmethod
    def _validate_non_negative_float(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{field_name} must be a finite non-negative number, got {value!r}")

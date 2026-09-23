"""Model-agnostic numeric feature representation for Fire Detection ML classifiers.

FIRE_DETECTION_FEATURE_NAMES is the single central definition of the ML
feature set and its column order. Both the synthetic training-data generator
script and the training script must import this constant rather than
declaring their own copy of the feature list, so a future runtime inference
path stays compatible with whatever classifier (Logistic Regression, Random
Forest, ...) is trained on this representation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

FIRE_DETECTION_FEATURE_NAMES: tuple[str, ...] = (
    "satellite_count",
    "news_count",
    "satellite_low_count",
    "satellite_nominal_count",
    "satellite_high_count",
    "time_span_minutes",
    "max_pairwise_distance_km",
)

SAMPLE_ID_COLUMN = "sample_id"
SCENARIO_FAMILY_COLUMN = "scenario_family"
LABEL_COLUMN = "label"

TRAINING_DATA_METADATA_COLUMNS: tuple[str, ...] = (SAMPLE_ID_COLUMN, SCENARIO_FAMILY_COLUMN)

# Central CSV schema: metadata columns, then features in FIRE_DETECTION_FEATURE_NAMES
# order, then the label. sample_id/scenario_family are metadata only and must
# never be fed to a classifier as a feature.
TRAINING_DATA_CSV_COLUMNS: tuple[str, ...] = (
    TRAINING_DATA_METADATA_COLUMNS + FIRE_DETECTION_FEATURE_NAMES + (LABEL_COLUMN,)
)

_COUNT_FIELD_NAMES = (
    "satellite_count",
    "news_count",
    "satellite_low_count",
    "satellite_nominal_count",
    "satellite_high_count",
)
_NON_NEGATIVE_FLOAT_FIELD_NAMES = ("time_span_minutes", "max_pairwise_distance_km")


@dataclass(frozen=True)
class FireDetectionFeatures:
    """One Fire Detection evidence candidate reduced to a tabular feature vector.

    Deliberately excludes raw latitude/longitude, database/evidence ids,
    location names, scenario/sample metadata, FFWI, fire danger, fire
    severity, and FRP (see backend/docs/fire_detection_ml.md for why). Field
    order matches FIRE_DETECTION_FEATURE_NAMES.
    """

    satellite_count: int
    news_count: int
    satellite_low_count: int
    satellite_nominal_count: int
    satellite_high_count: int
    time_span_minutes: float
    max_pairwise_distance_km: float

    def __post_init__(self) -> None:
        for field_name in _COUNT_FIELD_NAMES:
            self._validate_count(field_name, getattr(self, field_name))
        for field_name in _NON_NEGATIVE_FLOAT_FIELD_NAMES:
            self._validate_non_negative_float(field_name, getattr(self, field_name))

        confidence_count_sum = self.satellite_low_count + self.satellite_nominal_count + self.satellite_high_count
        if confidence_count_sum != self.satellite_count:
            raise ValueError(
                "satellite confidence counts must sum to satellite_count: "
                f"{confidence_count_sum} != {self.satellite_count}"
            )

    def as_tuple(self) -> tuple[float, ...]:
        """Return the feature vector in FIRE_DETECTION_FEATURE_NAMES order."""
        return tuple(getattr(self, name) for name in FIRE_DETECTION_FEATURE_NAMES)

    def as_dict(self) -> dict[str, float]:
        """Return the feature vector as a name -> value mapping."""
        return {name: getattr(self, name) for name in FIRE_DETECTION_FEATURE_NAMES}

    @staticmethod
    def _validate_count(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")

    @staticmethod
    def _validate_non_negative_float(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{field_name} must be a finite non-negative number, got {value!r}")

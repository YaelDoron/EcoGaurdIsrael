"""V5 feature schema and dataset columns for Fire Detection ML: ML features vs metadata vs target.

V5 answers: given the CURRENT wildfire evidence AND the event's persistence history, how likely is it
that an actual wildfire exists? It uses only

    current satellite evidence  +  satellite persistence (passes)  +  news evidence / corroboration

Fire Danger / FFWI is deliberately NOT part of V5: it stays a separate risk context.

This module is the single home of the V5 FEATURE CONTRACT used identically by offline training and
(later) runtime inference:

  * FIRE_DETECTION_FEATURE_NAMES_V5 - the one canonical ordered 25-name list;
  * FireDetectionFeaturesV5         - a validated, named, ordered feature vector;
  * NULLABLE_FEATURE_NAMES_V5       - features whose "not computable" state is NaN, never 0.

V4 (FIRE_DETECTION_FEATURE_NAMES_V4, training_v4.csv, its extractor and reports) is frozen and untouched.

MISSING-VALUE SEMANTICS (NaN means "cannot be computed", which is different from a real zero)

  feature                              NaN when
  ---------------------------------    ---------------------------------------------------------
  satellite_night_fraction             the current candidate has no satellite hotspot with a known day/night
  satellite_cluster_radius_km          the current candidate has no satellite hotspot
  news_satellite_lag_minutes           the current candidate lacks news OR satellite evidence
  satellite_history_span_minutes       the effective history holds no satellite pass
  satellite_centroid_stability_km      fewer than 2 satellite passes (one pass cannot show movement)
  satellite_frp_trend_per_hour         fewer than 3 passes carry an FRP value
  satellite_brightness_trend_per_hour  fewer than 3 passes carry a brightness value

`satellite_pass_count == 0` <=> there is no satellite evidence in the effective history, in which case
every satellite-derived nullable feature above is NaN. `satellite_frp_sum`, `satellite_frp_std` and
`satellite_brightness_std` follow the V3 convention instead (0.0 with the matching *_available_ratio
preserving the distinction between "nothing measured" and "measured as zero"), as the task specifies.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable, Mapping

from src.ml.fire_detection.fire_detection_features_v3 import FIRE_DETECTION_FEATURE_NAMES_V3

# The 14 V3/V4 evidence features V5 keeps, in their original relative order. Derived from the V3
# list (never re-typed) by removing the two geometry features V5 drops.
REMOVED_FROM_V4_EVIDENCE_FEATURE_NAMES: tuple[str, ...] = ("time_span_minutes", "max_pairwise_distance_km")
RETAINED_FEATURE_NAMES_V5: tuple[str, ...] = tuple(
    name for name in FIRE_DETECTION_FEATURE_NAMES_V3 if name not in REMOVED_FROM_V4_EVIDENCE_FEATURE_NAMES
)
REMOVED_FROM_V4_FIRE_DANGER_FEATURE_NAMES: tuple[str, ...] = (
    "fire_danger_available",
    "fire_danger_score",
    "fire_danger_age_minutes",
)
REMOVED_V4_FEATURE_NAMES: tuple[str, ...] = (
    REMOVED_FROM_V4_EVIDENCE_FEATURE_NAMES + REMOVED_FROM_V4_FIRE_DANGER_FEATURE_NAMES
)

CURRENT_EVIDENCE_FEATURE_NAMES_V5: tuple[str, ...] = (
    "satellite_frp_sum",
    "satellite_frp_std",
    "satellite_brightness_std",
    "satellite_night_fraction",
    "satellite_cluster_radius_km",
    "news_satellite_lag_minutes",
)

HISTORY_FEATURE_NAMES_V5: tuple[str, ...] = (
    "satellite_pass_count",
    "satellite_history_span_minutes",
    "satellite_centroid_stability_km",
    "satellite_frp_trend_per_hour",
    "satellite_brightness_trend_per_hour",
)

FIRE_DETECTION_FEATURE_NAMES_V5: tuple[str, ...] = (
    RETAINED_FEATURE_NAMES_V5 + CURRENT_EVIDENCE_FEATURE_NAMES_V5 + HISTORY_FEATURE_NAMES_V5
)

NULLABLE_FEATURE_NAMES_V5: tuple[str, ...] = (
    "satellite_night_fraction",
    "satellite_cluster_radius_km",
    "news_satellite_lag_minutes",
    "satellite_history_span_minutes",
    "satellite_centroid_stability_km",
    "satellite_frp_trend_per_hour",
    "satellite_brightness_trend_per_hour",
)

FEATURE_COUNT_V5 = len(FIRE_DETECTION_FEATURE_NAMES_V5)
MISSING_VALUE_V5 = float("nan")

# --- dataset columns -------------------------------------------------------------------------

SAMPLE_ID_COLUMN = "sample_id"
SEED_COLUMN = "seed"
ENVIRONMENT_ID_COLUMN = "environment_id"
REGIME_COLUMN = "regime"
LATENT_SUBTYPE_COLUMN = "latent_subtype"
PAIR_ID_COLUMN = "pair_id"
PAIR_TYPE_COLUMN = "pair_type"
AS_OF_UTC_COLUMN = "as_of_utc"
LABEL_COLUMN = "label"

TRAINING_DATA_METADATA_COLUMNS_V5: tuple[str, ...] = (
    SAMPLE_ID_COLUMN,
    SEED_COLUMN,
    ENVIRONMENT_ID_COLUMN,
    REGIME_COLUMN,
    LATENT_SUBTYPE_COLUMN,
    PAIR_ID_COLUMN,
    PAIR_TYPE_COLUMN,
    AS_OF_UTC_COLUMN,
)

TRAINING_DATA_CSV_COLUMNS_V5: tuple[str, ...] = (
    TRAINING_DATA_METADATA_COLUMNS_V5 + FIRE_DETECTION_FEATURE_NAMES_V5 + (LABEL_COLUMN,)
)

# Grouping fields for Task 7's evaluation (metadata only; never model input).
GROUP_FIELD_ENVIRONMENT = ENVIRONMENT_ID_COLUMN
GROUP_FIELD_REGIME = REGIME_COLUMN
GROUP_FIELD_PAIR = PAIR_ID_COLUMN
GROUP_FIELD_LATENT_SUBTYPE = LATENT_SUBTYPE_COLUMN

FORBIDDEN_FEATURE_NAMES_V5: frozenset[str] = frozenset(
    TRAINING_DATA_METADATA_COLUMNS_V5
    + (
        LABEL_COLUMN,
        "ground_truth",
        "fire_danger_level",
        "rule_status",
        "rule_confidence",
        "rule_decision",
        "detection_confidence",
        "ml_probability",
        "model_score",
        "event_status",
        "status",
    )
)

# Name fragments that betray a leakage column. Fire Danger fragments are included: V5 has none.
# ("hour_of"/"time_of_day" catch clock time, but `..._per_hour` is a legitimate trend unit.)
FORBIDDEN_FEATURE_NAME_FRAGMENTS_V5: tuple[str, ...] = (
    "scenario",
    "family",
    "archetype",
    "regime",
    "subtype",
    "latent",
    "environment",
    "pair_",
    "ground",
    "truth",
    "label",
    "preset",
    "expected",
    "rule_",
    "ml_",
    "probability",
    "suspected",
    "confirmed",
    "status",
    "event",
    "sample_id",
    "seed",
    "level",
    "band",
    "timestamp",
    "as_of",
    "utc",
    "hour_of",
    "time_of_day",
    "date",
    "fire_danger",
    "ffwi",
)


def forbidden_feature_names(feature_names: tuple[str, ...] = FIRE_DETECTION_FEATURE_NAMES_V5) -> tuple[str, ...]:
    """Return every name in feature_names that is (or looks like) a leakage column."""
    offenders = []
    for name in feature_names:
        lowered = name.lower()
        if name in FORBIDDEN_FEATURE_NAMES_V5 or any(fragment in lowered for fragment in FORBIDDEN_FEATURE_NAME_FRAGMENTS_V5):
            offenders.append(name)
    return tuple(offenders)


def is_missing_value(value: object) -> bool:
    """True for the ML-facing missing marker (NaN) and for None."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def validate_feature_names_v5(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Raise unless feature_names is exactly the canonical V5 list, in order. Returns it as a tuple."""
    names = tuple(feature_names)
    if names != FIRE_DETECTION_FEATURE_NAMES_V5:
        missing = [name for name in FIRE_DETECTION_FEATURE_NAMES_V5 if name not in names]
        unexpected = [name for name in names if name not in FIRE_DETECTION_FEATURE_NAMES_V5]
        raise ValueError(
            f"V5 feature schema mismatch: expected {FEATURE_COUNT_V5} features in canonical order, got "
            f"{len(names)} (missing={missing}, unexpected={unexpected}, "
            f"same_set_different_order={not missing and not unexpected and len(names) == FEATURE_COUNT_V5})."
        )
    return names


_INDEX = {name: index for index, name in enumerate(FIRE_DETECTION_FEATURE_NAMES_V5)}
_INTEGER_FEATURES = frozenset(
    name for name in FIRE_DETECTION_FEATURE_NAMES_V5 if name.endswith("_count")
)
_RATIO_FEATURES = frozenset(("satellite_frp_available_ratio", "satellite_brightness_available_ratio", "satellite_night_fraction"))
_SIGNED_FEATURES = frozenset(("news_satellite_lag_minutes", "satellite_frp_trend_per_hour", "satellite_brightness_trend_per_hour"))
_SATELLITE_COUNT_FEATURES = ("satellite_low_count", "satellite_nominal_count", "satellite_high_count")
_TREND_FEATURES = ("satellite_frp_trend_per_hour", "satellite_brightness_trend_per_hour")
MIN_PASSES_FOR_TREND_V5 = 3
MIN_PASSES_FOR_CENTROID_STABILITY_V5 = 2


@dataclass(frozen=True)
class FireDetectionFeaturesV5:
    """One candidate's 25 V5 features: named, ordered and validated.

    `values` follows FIRE_DETECTION_FEATURE_NAMES_V5. Counts are ints, everything else floats. NaN is
    accepted ONLY in NULLABLE_FEATURE_NAMES_V5, and only in the states the module docstring lists:
    the constructor rejects internally inconsistent vectors (e.g. a trend from 2 passes, a span for
    zero passes, satellite counts without a pass), so a malformed row can never be trained on silently.
    Construct via FireDetectionFeatureExtractorV5 (runtime and training) or the from_* helpers.
    """

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        raw = tuple(self.values)
        if len(raw) != FEATURE_COUNT_V5:
            raise ValueError(f"V5 features must contain exactly {FEATURE_COUNT_V5} values, got {len(raw)}.")
        for name, value in zip(FIRE_DETECTION_FEATURE_NAMES_V5, raw):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"Feature {name!r} must be numeric, got {value!r}.")
        normalized = tuple(self._canonical(name, value) for name, value in zip(FIRE_DETECTION_FEATURE_NAMES_V5, raw))
        self._validate_ranges(normalized)
        self._validate_consistency(normalized)
        object.__setattr__(self, "values", normalized)

    @staticmethod
    def _canonical(name: str, value: Real) -> float:
        if name in _INTEGER_FEATURES:
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"Feature {name!r} must be finite, got {value!r}.")
            if value != int(value):
                raise ValueError(f"Feature {name!r} must be a whole number, got {value!r}.")
            return int(value)
        return float(value)

    @staticmethod
    def _validate_ranges(values: tuple[float, ...]) -> None:
        for name, value in zip(FIRE_DETECTION_FEATURE_NAMES_V5, values):
            if name in _INTEGER_FEATURES:
                if value < 0:
                    raise ValueError(f"{name} must be non-negative, got {value!r}.")
                continue
            if math.isnan(value):
                if name not in NULLABLE_FEATURE_NAMES_V5:
                    raise ValueError(f"Feature {name!r} must not be NaN (only {NULLABLE_FEATURE_NAMES_V5} may be).")
                continue
            if math.isinf(value):
                raise ValueError(f"Feature {name!r} must be finite, got {value!r}.")
            if name in _RATIO_FEATURES and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value!r}.")
            if name not in _SIGNED_FEATURES and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value!r}.")

    @staticmethod
    def _validate_consistency(values: tuple[float, ...]) -> None:
        def get(name: str) -> float:
            return values[_INDEX[name]]

        passes = get("satellite_pass_count")
        satellite_total = sum(get(name) for name in _SATELLITE_COUNT_FEATURES)
        news_total = sum(get(name) for name in ("news_none_count", "news_weak_count", "news_moderate_count", "news_strong_count", "news_unknown_count"))
        if satellite_total + news_total < 1:
            raise ValueError("A candidate must contain at least one evidence item.")
        if satellite_total > 0 and passes < 1:
            raise ValueError("Current satellite evidence requires satellite_pass_count >= 1.")
        if passes > 0 and math.isnan(get("satellite_history_span_minutes")):
            raise ValueError("satellite_history_span_minutes must be computed when a satellite pass exists.")
        if passes == 0:
            for name in ("satellite_history_span_minutes", "satellite_centroid_stability_km", *_TREND_FEATURES):
                if not math.isnan(get(name)):
                    raise ValueError(f"{name} must be NaN when there is no satellite pass, got {get(name)!r}.")
        if passes == 1 and get("satellite_history_span_minutes") != 0.0:
            raise ValueError("satellite_history_span_minutes must be 0.0 for a single pass.")
        if passes < MIN_PASSES_FOR_CENTROID_STABILITY_V5 and not math.isnan(get("satellite_centroid_stability_km")):
            raise ValueError("satellite_centroid_stability_km requires at least 2 passes.")
        if passes >= MIN_PASSES_FOR_CENTROID_STABILITY_V5 and math.isnan(get("satellite_centroid_stability_km")):
            raise ValueError("satellite_centroid_stability_km must be computed from >= 2 passes.")
        if passes < MIN_PASSES_FOR_TREND_V5:
            for name in _TREND_FEATURES:
                if not math.isnan(get(name)):
                    raise ValueError(f"{name} requires at least {MIN_PASSES_FOR_TREND_V5} passes, got {passes}.")
        if satellite_total == 0:
            for name in ("satellite_night_fraction", "satellite_cluster_radius_km"):
                if not math.isnan(get(name)):
                    raise ValueError(f"{name} must be NaN without current satellite evidence.")
            if get("satellite_frp_sum") != 0.0:
                raise ValueError("satellite_frp_sum must be 0.0 without current satellite evidence.")
        elif math.isnan(get("satellite_cluster_radius_km")):
            raise ValueError("satellite_cluster_radius_km must be computed for current satellite evidence.")
        if (satellite_total == 0 or news_total == 0) and not math.isnan(get("news_satellite_lag_minutes")):
            raise ValueError("news_satellite_lag_minutes requires BOTH current satellite and news evidence.")
        if satellite_total > 0 and news_total > 0 and math.isnan(get("news_satellite_lag_minutes")):
            raise ValueError("news_satellite_lag_minutes must be computed when both source families are present.")

    # --- construction from stored/loose representations ---

    @classmethod
    def from_optional_values(cls, values: Iterable[float | None]) -> FireDetectionFeaturesV5:
        """From an ordered sequence where None marks a missing nullable value (e.g. a loaded CSV row)."""
        raw = tuple(values)
        if len(raw) != FEATURE_COUNT_V5:
            raise ValueError(f"V5 features must contain exactly {FEATURE_COUNT_V5} values, got {len(raw)}.")
        converted = []
        for name, value in zip(FIRE_DETECTION_FEATURE_NAMES_V5, raw):
            if value is None:
                if name not in NULLABLE_FEATURE_NAMES_V5:
                    raise ValueError(f"Feature {name!r} must not be missing.")
                value = MISSING_VALUE_V5
            converted.append(value)
        return cls(tuple(converted))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float | None]) -> FireDetectionFeaturesV5:
        """From a name -> value mapping. The keys must be EXACTLY the 25 canonical names (any order)."""
        missing = [name for name in FIRE_DETECTION_FEATURE_NAMES_V5 if name not in mapping]
        unexpected = [name for name in mapping if name not in FIRE_DETECTION_FEATURE_NAMES_V5]
        if missing or unexpected:
            raise ValueError(f"V5 feature mapping mismatch: missing={missing}, unexpected={unexpected}.")
        return cls.from_optional_values(mapping[name] for name in FIRE_DETECTION_FEATURE_NAMES_V5)

    # --- representations ---

    @property
    def names(self) -> tuple[str, ...]:
        return FIRE_DETECTION_FEATURE_NAMES_V5

    def __getitem__(self, name: str) -> float:
        return self.values[_INDEX[name]]

    def as_tuple(self) -> tuple[float, ...]:
        """Ordered ML-facing values (NaN = not computable). Convert to a float array for sklearn."""
        return self.values

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FIRE_DETECTION_FEATURE_NAMES_V5, self.values))

    def to_nullable_tuple(self) -> tuple[float | None, ...]:
        """Ordered storage/traceability form: None instead of NaN."""
        return tuple(None if is_missing_value(value) else value for value in self.values)

    def to_nullable_dict(self) -> dict[str, float | None]:
        return dict(zip(FIRE_DETECTION_FEATURE_NAMES_V5, self.to_nullable_tuple()))

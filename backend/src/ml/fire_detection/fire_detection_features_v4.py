"""V4 dataset schema for Fire Detection ML: ML features vs metadata vs target.

V4 is a NEW, separately versioned schema. It does not change
FIRE_DETECTION_FEATURE_NAMES_V3 (the runtime model's schema) or anything V3
produced; it only builds on it.

Every CSV column is exactly one of:

  - an ML FEATURE  (FIRE_DETECTION_FEATURE_NAMES_V4): the 16 unchanged V3
    evidence features plus the 3 Fire Danger context features
    (see FireDetectionContext / backend/docs/fire_detection_context.md);
  - METADATA (TRAINING_DATA_METADATA_COLUMNS_V4): for splitting, debugging and
    analysis only - NEVER model input;
  - the TARGET (LABEL_COLUMN): whether an actual wildfire existed, taken from
    simulation ground truth, never from any detector output.

Missing Fire Danger is preserved, not imputed: `fire_danger_available` is 0 and
`fire_danger_score` / `fire_danger_age_minutes` are EMPTY in the CSV (None when
loaded). A real FFWI of 0 stays distinguishable from "no assessment".

This module is also the single home of the V4 FEATURE CONTRACT used identically
by offline training and (later) runtime inference:

  - FIRE_DETECTION_FEATURE_NAMES_V4: the one canonical ordered 19-name list
    (built from the V3 evidence names + the 3 context names; never re-typed);
  - FireDetectionFeaturesV4: a validated, named, ordered feature vector.
    ML-facing missing Fire Danger is NaN; the storage/traceability forms
    (`to_nullable_*`) use None. Produced by FireDetectionFeatureExtractorV4.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable, Mapping

from src.calculators.fire_danger.ffwi_config import FFWI_MAX_SCORE, FFWI_MIN_SCORE
from src.ml.fire_detection.fire_detection_features_v3 import (
    FIRE_DETECTION_FEATURE_NAMES_V3,
    FireDetectionFeaturesV3,
)

FIRE_DANGER_CONTEXT_FEATURE_NAMES_V4: tuple[str, ...] = (
    "fire_danger_available",
    "fire_danger_score",
    "fire_danger_age_minutes",
)

FIRE_DETECTION_FEATURE_NAMES_V4: tuple[str, ...] = (
    FIRE_DETECTION_FEATURE_NAMES_V3 + FIRE_DANGER_CONTEXT_FEATURE_NAMES_V4
)

# Features that are legitimately empty (None) when Fire Danger is unavailable.
NULLABLE_FEATURE_NAMES_V4: tuple[str, ...] = ("fire_danger_score", "fire_danger_age_minutes")

SAMPLE_ID_COLUMN = "sample_id"
SEED_COLUMN = "seed"
SCENARIO_FAMILY_COLUMN = "scenario_family"
SCENARIO_ARCHETYPE_COLUMN = "scenario_archetype"
GROUND_TRUTH_SCENARIO_TYPE_COLUMN = "ground_truth_scenario_type"
FIRE_DANGER_BAND_COLUMN = "fire_danger_band"
AS_OF_UTC_COLUMN = "as_of_utc"
LABEL_COLUMN = "label"

FIRE_DANGER_BAND_MISSING = "missing"
FIRE_DANGER_BANDS: tuple[str, ...] = ("low", "moderate", "high", "very_high", "extreme", FIRE_DANGER_BAND_MISSING)

# Grouping fields for later grouped evaluation (see docs/fire_detection_dataset_v4.md):
#   scenario_family    -> GroupKFold-style "unseen scenario family" evaluation
#   scenario_archetype -> leave-one-archetype-out evaluation
GROUP_FIELD_FAMILY = SCENARIO_FAMILY_COLUMN
GROUP_FIELD_ARCHETYPE = SCENARIO_ARCHETYPE_COLUMN

TRAINING_DATA_METADATA_COLUMNS_V4: tuple[str, ...] = (
    SAMPLE_ID_COLUMN,
    SEED_COLUMN,
    SCENARIO_FAMILY_COLUMN,
    SCENARIO_ARCHETYPE_COLUMN,
    GROUND_TRUTH_SCENARIO_TYPE_COLUMN,
    FIRE_DANGER_BAND_COLUMN,
    AS_OF_UTC_COLUMN,
)

TRAINING_DATA_CSV_COLUMNS_V4: tuple[str, ...] = (
    TRAINING_DATA_METADATA_COLUMNS_V4 + FIRE_DETECTION_FEATURE_NAMES_V4 + (LABEL_COLUMN,)
)

# Anything that describes ground truth, the scenario, a detector's output, or
# absolute time must never be an ML feature. Exact names, plus name fragments
# that catch near-variants. (`time_span_minutes` is a legitimate duration
# feature, which is why the time fragments below are specific.)
FORBIDDEN_FEATURE_NAMES_V4: frozenset[str] = frozenset(
    TRAINING_DATA_METADATA_COLUMNS_V4
    + (
        LABEL_COLUMN,
        "ground_truth",
        "fire_danger_level",
        "fire_danger_assessment_id",
        "fire_danger_assessed_at",
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

FORBIDDEN_FEATURE_NAME_FRAGMENTS_V4: tuple[str, ...] = (
    "scenario",
    "family",
    "archetype",
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
    "hour",
    "date",
)


def forbidden_feature_names(feature_names: tuple[str, ...] = FIRE_DETECTION_FEATURE_NAMES_V4) -> tuple[str, ...]:
    """Return every name in feature_names that is (or looks like) a leakage column."""
    offenders = []
    for name in feature_names:
        lowered = name.lower()
        if name in FORBIDDEN_FEATURE_NAMES_V4 or any(fragment in lowered for fragment in FORBIDDEN_FEATURE_NAME_FRAGMENTS_V4):
            offenders.append(name)
    return tuple(offenders)


# --- feature contract (shared by training and runtime inference) ---

FEATURE_COUNT_V4 = len(FIRE_DETECTION_FEATURE_NAMES_V4)
# The ML-facing representation of "Fire Danger unavailable". NaN, never 0.0:
# a real FFWI of 0.0 is a valid "very low danger" observation and a real age of
# 0.0 means "assessed just now". Task 4's preprocessing decides how to impute.
MISSING_VALUE_V4 = float("nan")

_EVIDENCE_FEATURE_COUNT = len(FIRE_DETECTION_FEATURE_NAMES_V3)
_AVAILABLE_INDEX = FIRE_DETECTION_FEATURE_NAMES_V4.index("fire_danger_available")
_SCORE_INDEX = FIRE_DETECTION_FEATURE_NAMES_V4.index("fire_danger_score")
_AGE_INDEX = FIRE_DETECTION_FEATURE_NAMES_V4.index("fire_danger_age_minutes")
_INTEGER_FEATURE_INDICES = frozenset(
    index for index, name in enumerate(FIRE_DETECTION_FEATURE_NAMES_V4) if name.endswith("_count")
) | {_AVAILABLE_INDEX}


def is_missing_value(value: object) -> bool:
    """True for the ML-facing missing marker (NaN) and for None."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def validate_feature_names_v4(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Raise unless feature_names is exactly the canonical V4 list, in order. Returns it as a tuple.

    Used to validate anything that claims to be a V4 feature schema (extractor
    output, dataset headers and - in Task 4 - saved model metadata) so a
    reordered or extended feature list can never be fed to the model silently.
    """
    names = tuple(feature_names)
    if names != FIRE_DETECTION_FEATURE_NAMES_V4:
        missing = [name for name in FIRE_DETECTION_FEATURE_NAMES_V4 if name not in names]
        unexpected = [name for name in names if name not in FIRE_DETECTION_FEATURE_NAMES_V4]
        raise ValueError(
            f"V4 feature schema mismatch: expected {FEATURE_COUNT_V4} features in canonical order, got "
            f"{len(names)} (missing={missing}, unexpected={unexpected}, "
            f"same_set_different_order={not missing and not unexpected and len(names) == FEATURE_COUNT_V4})."
        )
    return names


@dataclass(frozen=True)
class FireDetectionFeaturesV4:
    """One candidate's 19 V4 features: named, ordered and validated.

    `values` follows FIRE_DETECTION_FEATURE_NAMES_V4. Counts and
    `fire_danger_available` are ints, everything else floats. When Fire Danger
    is unavailable, `fire_danger_available` is 0 and `fire_danger_score` /
    `fire_danger_age_minutes` are NaN - and NaN is accepted ONLY there. The 16
    evidence features are validated with the unchanged V3 rules.
    Construct via FireDetectionFeatureExtractorV4 (runtime and training) or the
    from_* helpers (loading stored rows / mappings).
    """

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        raw = tuple(self.values)
        if len(raw) != FEATURE_COUNT_V4:
            raise ValueError(f"V4 features must contain exactly {FEATURE_COUNT_V4} values, got {len(raw)}.")
        for name, value in zip(FIRE_DETECTION_FEATURE_NAMES_V4, raw):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"Feature {name!r} must be numeric, got {value!r}.")

        normalized = tuple(self._canonical(index, value) for index, value in enumerate(raw))
        self._validate_evidence_features(normalized)
        self._validate_fire_danger_features(normalized)
        object.__setattr__(self, "values", normalized)

    @staticmethod
    def _canonical(index: int, value: Real) -> float:
        name = FIRE_DETECTION_FEATURE_NAMES_V4[index]
        if index in _INTEGER_FEATURE_INDICES:
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"Feature {name!r} must be finite, got {value!r}.")
            if value != int(value):
                raise ValueError(f"Feature {name!r} must be a whole number, got {value!r}.")
            return int(value)
        return float(value)

    @staticmethod
    def _validate_evidence_features(values: tuple[float, ...]) -> None:
        # Reuse the V3 dataclass's own validation (counts, [0, 1] ratios, non-negative finite
        # measurements) instead of re-implementing it: same rules, one definition.
        FireDetectionFeaturesV3(**dict(zip(FIRE_DETECTION_FEATURE_NAMES_V3, values[:_EVIDENCE_FEATURE_COUNT])))

    @staticmethod
    def _validate_fire_danger_features(values: tuple[float, ...]) -> None:
        available, score, age = values[_AVAILABLE_INDEX], values[_SCORE_INDEX], values[_AGE_INDEX]
        if available not in (0, 1):
            raise ValueError(f"fire_danger_available must be 0 or 1, got {available!r}.")
        if available == 1:
            if math.isnan(score) or math.isnan(age):
                raise ValueError("Available Fire Danger requires a real score and age (got NaN).")
            if not FFWI_MIN_SCORE <= score <= FFWI_MAX_SCORE:
                raise ValueError(f"fire_danger_score must be within [{FFWI_MIN_SCORE}, {FFWI_MAX_SCORE}], got {score!r}.")
            if math.isinf(age) or age < 0:
                raise ValueError(f"fire_danger_age_minutes must be a finite value >= 0, got {age!r}.")
        elif not (math.isnan(score) and math.isnan(age)):
            raise ValueError(
                "Unavailable Fire Danger must have NaN score and age, never a real value such as 0.0 "
                f"(got score={score!r}, age={age!r})."
            )

    # --- construction from stored/loose representations ---

    @classmethod
    def from_optional_values(cls, values: Iterable[float | None]) -> FireDetectionFeaturesV4:
        """From an ordered sequence where None marks a missing nullable Fire Danger value (e.g. a loaded CSV row)."""
        raw = tuple(values)
        if len(raw) != FEATURE_COUNT_V4:
            raise ValueError(f"V4 features must contain exactly {FEATURE_COUNT_V4} values, got {len(raw)}.")
        converted = []
        for name, value in zip(FIRE_DETECTION_FEATURE_NAMES_V4, raw):
            if value is None:
                if name not in NULLABLE_FEATURE_NAMES_V4:
                    raise ValueError(f"Feature {name!r} must not be missing.")
                value = MISSING_VALUE_V4
            converted.append(value)
        return cls(tuple(converted))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float | None]) -> FireDetectionFeaturesV4:
        """From a name -> value mapping. The keys must be EXACTLY the 19 canonical names (any order)."""
        missing = [name for name in FIRE_DETECTION_FEATURE_NAMES_V4 if name not in mapping]
        unexpected = [name for name in mapping if name not in FIRE_DETECTION_FEATURE_NAMES_V4]
        if missing or unexpected:
            raise ValueError(f"V4 feature mapping mismatch: missing={missing}, unexpected={unexpected}.")
        return cls.from_optional_values(mapping[name] for name in FIRE_DETECTION_FEATURE_NAMES_V4)

    # --- representations ---

    @property
    def names(self) -> tuple[str, ...]:
        return FIRE_DETECTION_FEATURE_NAMES_V4

    @property
    def fire_danger_available(self) -> bool:
        return self.values[_AVAILABLE_INDEX] == 1

    def __getitem__(self, name: str) -> float:
        return self.values[FIRE_DETECTION_FEATURE_NAMES_V4.index(name)]

    def as_tuple(self) -> tuple[float, ...]:
        """Ordered ML-facing values (NaN = Fire Danger unavailable). Convert to a float array for sklearn."""
        return self.values

    def as_dict(self) -> dict[str, float]:
        """name -> value, ML-facing (NaN for unavailable Fire Danger score/age)."""
        return dict(zip(FIRE_DETECTION_FEATURE_NAMES_V4, self.values))

    def to_nullable_tuple(self) -> tuple[float | None, ...]:
        """Ordered storage/traceability form: None instead of NaN."""
        return tuple(None if is_missing_value(value) else value for value in self.values)

    def to_nullable_dict(self) -> dict[str, float | None]:
        return dict(zip(FIRE_DETECTION_FEATURE_NAMES_V4, self.to_nullable_tuple()))

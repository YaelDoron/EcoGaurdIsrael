"""Tests for the V4 feature contract: FireDetectionFeaturesV4 validation, canonical schema, no second name list."""
from __future__ import annotations

import ast
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.ml.fire_detection.fire_detection_features_v3 import FIRE_DETECTION_FEATURE_NAMES_V3
from src.ml.fire_detection.fire_detection_features_v4 import (
    FEATURE_COUNT_V4,
    FIRE_DETECTION_FEATURE_NAMES_V4,
    MISSING_VALUE_V4,
    FireDetectionFeaturesV4,
    is_missing_value,
    validate_feature_names_v4,
)

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
NAN = float("nan")

# A valid available-Fire-Danger vector, named for readability.
VALID = {
    "satellite_low_count": 0,
    "satellite_nominal_count": 1,
    "satellite_high_count": 1,
    "satellite_frp_available_ratio": 1.0,
    "satellite_frp_mean": 45.0,
    "satellite_frp_max": 60.0,
    "satellite_brightness_available_ratio": 0.5,
    "satellite_brightness_mean": 320.0,
    "satellite_brightness_max": 320.0,
    "news_none_count": 0,
    "news_weak_count": 0,
    "news_moderate_count": 1,
    "news_strong_count": 0,
    "news_unknown_count": 0,
    "time_span_minutes": 12.5,
    "max_pairwise_distance_km": 1.2,
    "fire_danger_available": 1,
    "fire_danger_score": 33.0,
    "fire_danger_age_minutes": 15.0,
}
MISSING_DANGER = {**VALID, "fire_danger_available": 0, "fire_danger_score": NAN, "fire_danger_age_minutes": NAN}


def values_with(base=VALID, **overrides):
    merged = {**base, **overrides}
    return tuple(merged[name] for name in FIRE_DETECTION_FEATURE_NAMES_V4)


# --- canonical schema ---


def test_there_are_exactly_19_features_in_canonical_order():
    assert FEATURE_COUNT_V4 == 19 == len(FIRE_DETECTION_FEATURE_NAMES_V4)
    assert tuple(VALID) == FIRE_DETECTION_FEATURE_NAMES_V4  # the test fixture itself follows the canonical order
    assert FIRE_DETECTION_FEATURE_NAMES_V4[:16] == FIRE_DETECTION_FEATURE_NAMES_V3
    assert FIRE_DETECTION_FEATURE_NAMES_V4[16:] == ("fire_danger_available", "fire_danger_score", "fire_danger_age_minutes")


@pytest.mark.parametrize(
    "forbidden",
    [
        "fire_danger_level",
        "scenario_family",
        "scenario_archetype",
        "scenario_type",
        "ground_truth",
        "as_of_utc",
        "timestamp",
        "rule_confidence",
        "rule_status",
        "ml_probability",
        "event_status",
        "label",
    ],
)
def test_no_leakage_or_metadata_name_is_a_feature(forbidden):
    assert forbidden not in FIRE_DETECTION_FEATURE_NAMES_V4


def test_the_ordered_feature_list_is_defined_in_exactly_one_place():
    """No module other than the two schema modules may spell out (a copy of) the ordered feature list."""
    allowed = {"fire_detection_features_v3.py", "fire_detection_features_v4.py"}
    offenders = []
    for path in SRC_ROOT.rglob("*.py"):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Tuple, ast.List)):
                names = [element.value for element in node.elts if isinstance(element, ast.Constant) and isinstance(element.value, str)]
                if len(set(names) & set(FIRE_DETECTION_FEATURE_NAMES_V4)) >= 16:
                    offenders.append(str(path.relative_to(SRC_ROOT)))
    assert not offenders, offenders


@pytest.mark.parametrize(
    "names",
    [
        FIRE_DETECTION_FEATURE_NAMES_V4[:-1],  # a feature dropped
        FIRE_DETECTION_FEATURE_NAMES_V4 + ("label",),  # a leakage column appended
        FIRE_DETECTION_FEATURE_NAMES_V4[1:] + FIRE_DETECTION_FEATURE_NAMES_V4[:1],  # rotated order
        tuple(reversed(FIRE_DETECTION_FEATURE_NAMES_V4)),  # reversed
        FIRE_DETECTION_FEATURE_NAMES_V3,  # the V3 schema is not the V4 schema
        (),
    ],
    ids=["dropped", "leakage-appended", "rotated", "reversed", "v3-schema", "empty"],
)
def test_schema_mismatch_is_detected(names):
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_feature_names_v4(names)


def test_the_canonical_schema_validates():
    assert validate_feature_names_v4(list(FIRE_DETECTION_FEATURE_NAMES_V4)) == FIRE_DETECTION_FEATURE_NAMES_V4


def test_a_reordered_but_complete_schema_is_reported_as_an_order_problem():
    reordered = (FIRE_DETECTION_FEATURE_NAMES_V4[1], FIRE_DETECTION_FEATURE_NAMES_V4[0]) + FIRE_DETECTION_FEATURE_NAMES_V4[2:]

    with pytest.raises(ValueError, match="same_set_different_order=True"):
        validate_feature_names_v4(reordered)


# --- the vector: valid cases ---


def test_a_valid_available_vector_is_accepted_and_ordered():
    features = FireDetectionFeaturesV4(values_with())

    assert features.as_tuple() == values_with()
    assert features.as_dict() == VALID
    assert features.fire_danger_available is True


def test_nan_is_accepted_for_unavailable_score_and_age_only():
    features = FireDetectionFeaturesV4(values_with(MISSING_DANGER))

    assert features.fire_danger_available is False
    assert math.isnan(features["fire_danger_score"]) and math.isnan(features["fire_danger_age_minutes"])
    assert features["fire_danger_available"] == 0


def test_counts_and_availability_are_canonicalised_to_ints_and_the_rest_to_floats():
    features = FireDetectionFeaturesV4(values_with(satellite_nominal_count=1.0, satellite_frp_max=60, fire_danger_available=1.0))

    assert type(features["satellite_nominal_count"]) is int
    assert type(features["fire_danger_available"]) is int
    assert type(features["satellite_frp_max"]) is float


def test_the_vector_is_immutable():
    features = FireDetectionFeaturesV4(values_with())

    with pytest.raises(FrozenInstanceError):
        features.values = ()


# --- the vector: schema validation fails loudly ---


@pytest.mark.parametrize("length", [0, 1, 16, 18, 20, 25])
def test_wrong_feature_count_is_rejected(length):
    with pytest.raises(ValueError, match="exactly 19"):
        FireDetectionFeaturesV4(tuple(range(length)))


@pytest.mark.parametrize("bad", ["1", None, True, [1], object()])
def test_non_numeric_values_are_rejected(bad):
    for index in (0, 4, 16, 17):  # a count, a measurement, availability, the score
        values = list(values_with())
        values[index] = bad
        with pytest.raises(ValueError, match="numeric"):
            FireDetectionFeaturesV4(tuple(values))


@pytest.mark.parametrize("bad", [2, -1, 0.5, 1.5])
def test_fire_danger_available_must_be_binary(bad):
    with pytest.raises(ValueError):
        FireDetectionFeaturesV4(values_with(fire_danger_available=bad))


@pytest.mark.parametrize("missing", ["fire_danger_score", "fire_danger_age_minutes"])
def test_available_fire_danger_without_score_or_age_is_rejected(missing):
    with pytest.raises(ValueError, match="requires a real score and age"):
        FireDetectionFeaturesV4(values_with(**{missing: NAN}))


@pytest.mark.parametrize(
    "overrides",
    [
        {"fire_danger_score": 0.0, "fire_danger_age_minutes": 0.0},  # both silently turned into "real" zeros
        {"fire_danger_score": 0.0},  # score zero-filled, age NaN
        {"fire_danger_age_minutes": 0.0},  # age zero-filled, score NaN
        {"fire_danger_score": 30.0},
    ],
)
def test_unavailable_fire_danger_converted_to_real_numbers_is_rejected(overrides):
    base = {**MISSING_DANGER}
    base.update(overrides)

    with pytest.raises(ValueError, match="Unavailable Fire Danger must have NaN"):
        FireDetectionFeaturesV4(values_with(base))


@pytest.mark.parametrize("score", [-0.1, 100.1, 500.0, float("inf")])
def test_out_of_range_fire_danger_score_is_rejected(score):
    with pytest.raises(ValueError):
        FireDetectionFeaturesV4(values_with(fire_danger_score=score))


@pytest.mark.parametrize("age", [-1.0, float("inf")])
def test_invalid_fire_danger_age_is_rejected(age):
    with pytest.raises(ValueError):
        FireDetectionFeaturesV4(values_with(fire_danger_age_minutes=age))


@pytest.mark.parametrize(
    "name, bad",
    [
        ("satellite_low_count", -1),
        ("satellite_high_count", 1.5),
        ("news_strong_count", NAN),
        ("satellite_frp_available_ratio", 1.5),
        ("satellite_brightness_available_ratio", -0.1),
        ("satellite_frp_mean", -5.0),
        ("satellite_frp_max", NAN),
        ("satellite_brightness_max", float("inf")),
        ("time_span_minutes", NAN),
        ("max_pairwise_distance_km", -0.5),
    ],
)
def test_invalid_evidence_features_are_rejected(name, bad):
    with pytest.raises(ValueError):
        FireDetectionFeaturesV4(values_with(**{name: bad}))


# --- construction from stored/loose forms ---


def test_from_mapping_accepts_any_key_order():
    shuffled = dict(reversed(list(VALID.items())))

    assert FireDetectionFeaturesV4.from_mapping(shuffled).as_tuple() == values_with()


@pytest.mark.parametrize(
    "extra",
    ["label", "fire_danger_level", "scenario_family", "scenario_archetype", "ground_truth_scenario_type", "rule_confidence", "as_of_utc"],
)
def test_no_leakage_metadata_is_accepted_by_the_feature_contract(extra):
    with pytest.raises(ValueError, match=extra):
        FireDetectionFeaturesV4.from_mapping({**VALID, extra: 1})


def test_from_mapping_rejects_a_missing_required_feature():
    incomplete = {name: value for name, value in VALID.items() if name != "satellite_frp_mean"}

    with pytest.raises(ValueError, match="satellite_frp_mean"):
        FireDetectionFeaturesV4.from_mapping(incomplete)


def test_from_optional_values_turns_none_into_nan_only_for_the_nullable_fire_danger_features():
    stored = values_with(MISSING_DANGER)
    stored = tuple(None if is_missing_value(value) else value for value in stored)

    features = FireDetectionFeaturesV4.from_optional_values(stored)

    assert math.isnan(features["fire_danger_score"]) and math.isnan(features["fire_danger_age_minutes"])
    assert features.to_nullable_tuple() == stored

    broken = list(stored)
    broken[0] = None
    with pytest.raises(ValueError, match="must not be missing"):
        FireDetectionFeaturesV4.from_optional_values(broken)


def test_none_and_nan_round_trip_without_becoming_zero():
    for source in (VALID, MISSING_DANGER):
        features = FireDetectionFeaturesV4(values_with(source))
        again = FireDetectionFeaturesV4.from_optional_values(features.to_nullable_tuple())
        assert [is_missing_value(v) for v in again.as_tuple()] == [is_missing_value(v) for v in features.as_tuple()]
        assert features.to_nullable_dict()["fire_danger_score"] == (None if source is MISSING_DANGER else 33.0)


def test_the_missing_marker_is_nan_and_is_recognised_consistently():
    assert math.isnan(MISSING_VALUE_V4)
    assert is_missing_value(MISSING_VALUE_V4) and is_missing_value(None)
    assert not is_missing_value(0.0) and not is_missing_value(0)

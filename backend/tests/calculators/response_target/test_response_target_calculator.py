"""Tests for pure response-target generation."""
from __future__ import annotations

import math

import pytest

from src.calculators.response_target import ResponseTargetCalculator
from src.calculators.response_target.response_target_config import (
    ACTIVE_FIRE_BASE_PRIORITY,
    MIN_PREDICTED_TARGET_RISK_SCORE,
    PREDICTION_HORIZON_FACTORS,
    TARGET_DEDUP_DISTANCE_METERS,
)
from src.models import PredictedRiskTargetCandidate, ResponseTargetType
from src.utils.geo import destination_point, haversine_distance_km

FIRE_EVENT_ID = 1
FIRE_LAT = 32.731
FIRE_LON = 35.046


def make_candidate(**overrides) -> PredictedRiskTargetCandidate:
    defaults = dict(
        fire_event_id=FIRE_EVENT_ID,
        latitude=32.74,
        longitude=35.06,
        risk_score=80.0,
        prediction_horizon_minutes=30,
        spread_prediction_id=10,
        spread_prediction_cell_id=100,
    )
    defaults.update(overrides)
    return PredictedRiskTargetCandidate(**defaults)


def build_targets(*, severity_score=None, predicted_candidates=()):
    return ResponseTargetCalculator().build_targets(
        fire_event_id=FIRE_EVENT_ID,
        fire_latitude=FIRE_LAT,
        fire_longitude=FIRE_LON,
        severity_score=severity_score,
        predicted_candidates=predicted_candidates,
    )


def predicted_targets(targets):
    return tuple(target for target in targets if target.target_type is ResponseTargetType.PREDICTED_RISK)


def candidate_at_distance(distance_meters: float, **overrides) -> PredictedRiskTargetCandidate:
    latitude, longitude = destination_point(FIRE_LAT, FIRE_LON, 90.0, distance_meters / 1000.0)
    return make_candidate(latitude=latitude, longitude=longitude, **overrides)


# ---------------------------------------------------------------------------
# Active fire
# ---------------------------------------------------------------------------


def test_active_fire_always_produces_one_target_without_predictions():
    targets = build_targets()

    assert len(targets) == 1
    assert targets[0].target_type is ResponseTargetType.ACTIVE_FIRE
    assert targets[0].latitude == FIRE_LAT
    assert targets[0].longitude == FIRE_LON
    assert targets[0].priority_score == ACTIVE_FIRE_BASE_PRIORITY


def test_active_fire_survives_when_severity_score_is_none():
    targets = build_targets(severity_score=None)

    assert len(targets) == 1
    assert targets[0].priority_score == 100.0


def test_higher_severity_produces_higher_active_priority():
    low = build_targets(severity_score=10.0)[0]
    high = build_targets(severity_score=75.0)[0]

    assert high.priority_score > low.priority_score


@pytest.mark.parametrize("severity_score,expected_priority", [(0.0, 100.0), (100.0, 200.0)])
def test_severity_boundaries(severity_score, expected_priority):
    targets = build_targets(severity_score=severity_score)

    assert targets[0].priority_score == expected_priority


@pytest.mark.parametrize("severity_score", [-0.1, 100.1, math.nan, math.inf, -math.inf, True])
def test_invalid_severity_rejected(severity_score):
    with pytest.raises(ValueError):
        build_targets(severity_score=severity_score)


# ---------------------------------------------------------------------------
# Prediction inclusion and traceability
# ---------------------------------------------------------------------------


def test_risk_below_threshold_does_not_become_target():
    targets = build_targets(predicted_candidates=(make_candidate(risk_score=59.999),))

    assert predicted_targets(targets) == ()


def test_risk_at_threshold_becomes_target():
    targets = build_targets(predicted_candidates=(make_candidate(risk_score=MIN_PREDICTED_TARGET_RISK_SCORE),))

    assert len(predicted_targets(targets)) == 1


def test_high_risk_prediction_becomes_predicted_risk_target():
    targets = build_targets(predicted_candidates=(make_candidate(risk_score=90.0),))

    assert predicted_targets(targets)[0].target_type is ResponseTargetType.PREDICTED_RISK


def test_predicted_target_preserves_source_fields():
    candidate = make_candidate(
        latitude=32.75,
        longitude=35.07,
        prediction_horizon_minutes=60,
        spread_prediction_id=123,
        spread_prediction_cell_id=456,
    )
    target = predicted_targets(build_targets(predicted_candidates=(candidate,)))[0]

    assert target.fire_event_id == candidate.fire_event_id
    assert target.latitude == candidate.latitude
    assert target.longitude == candidate.longitude
    assert target.prediction_horizon_minutes == candidate.prediction_horizon_minutes
    assert target.spread_prediction_id == candidate.spread_prediction_id
    assert target.spread_prediction_cell_id == candidate.spread_prediction_cell_id


# ---------------------------------------------------------------------------
# Horizon priority
# ---------------------------------------------------------------------------


def test_30_minute_target_has_higher_priority_than_equivalent_60_minute_target():
    near_30 = make_candidate(latitude=32.74, longitude=35.06, risk_score=80.0, prediction_horizon_minutes=30)
    far_60 = make_candidate(
        latitude=32.75,
        longitude=35.07,
        risk_score=80.0,
        prediction_horizon_minutes=60,
        spread_prediction_id=11,
        spread_prediction_cell_id=101,
    )
    targets = predicted_targets(build_targets(predicted_candidates=(far_60, near_30)))

    assert targets[0].priority_score == 80.0 * PREDICTION_HORIZON_FACTORS[30]
    assert targets[1].priority_score == 80.0 * PREDICTION_HORIZON_FACTORS[60]
    assert targets[0].priority_score > targets[1].priority_score


def test_unsupported_horizon_fails_explicitly():
    with pytest.raises(ValueError):
        build_targets(predicted_candidates=(make_candidate(prediction_horizon_minutes=45),))


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_identical_predicted_coordinates_produce_one_target():
    first = make_candidate(spread_prediction_id=10, spread_prediction_cell_id=100)
    second = make_candidate(spread_prediction_id=11, spread_prediction_cell_id=101)

    targets = predicted_targets(build_targets(predicted_candidates=(first, second)))

    assert len(targets) == 1


def test_predicted_candidates_within_100m_produce_one_target():
    first = candidate_at_distance(250.0, spread_prediction_id=10, spread_prediction_cell_id=100)
    latitude, longitude = destination_point(first.latitude, first.longitude, 90.0, 0.050)
    second = make_candidate(latitude=latitude, longitude=longitude, spread_prediction_id=11, spread_prediction_cell_id=101)

    assert haversine_distance_km(first.latitude, first.longitude, second.latitude, second.longitude) * 1000 <= (
        TARGET_DEDUP_DISTANCE_METERS
    )
    targets = predicted_targets(build_targets(predicted_candidates=(second, first)))

    assert len(targets) == 1


def test_higher_priority_duplicate_is_retained():
    lower = make_candidate(risk_score=75.0, spread_prediction_id=10, spread_prediction_cell_id=100)
    higher = make_candidate(risk_score=90.0, spread_prediction_id=11, spread_prediction_cell_id=101)

    target = predicted_targets(build_targets(predicted_candidates=(lower, higher)))[0]

    assert target.spread_prediction_id == higher.spread_prediction_id
    assert target.priority_score == 90.0


def test_equal_priority_duplicates_use_deterministic_tie_breakers_regardless_of_order():
    first = make_candidate(risk_score=80.0, spread_prediction_id=10, spread_prediction_cell_id=200)
    second = make_candidate(risk_score=80.0, spread_prediction_id=11, spread_prediction_cell_id=100)

    forward = build_targets(predicted_candidates=(first, second))
    reversed_order = build_targets(predicted_candidates=(second, first))

    assert forward == reversed_order
    assert predicted_targets(forward)[0].spread_prediction_id == 10


def test_predicted_target_within_100m_of_active_fire_is_removed():
    candidate = candidate_at_distance(99.9)

    targets = predicted_targets(build_targets(predicted_candidates=(candidate,)))

    assert targets == ()


def test_predicted_target_outside_100m_of_active_fire_is_preserved():
    candidate = candidate_at_distance(100.1)

    targets = predicted_targets(build_targets(predicted_candidates=(candidate,)))

    assert len(targets) == 1


# ---------------------------------------------------------------------------
# Event isolation, ordering, determinism
# ---------------------------------------------------------------------------


def test_candidate_from_another_fire_event_is_rejected():
    with pytest.raises(ValueError):
        build_targets(predicted_candidates=(make_candidate(fire_event_id=2),))


def test_higher_priority_appears_before_lower_priority():
    lower = make_candidate(latitude=32.75, longitude=35.08, risk_score=70.0, spread_prediction_id=10)
    higher = make_candidate(latitude=32.76, longitude=35.09, risk_score=90.0, spread_prediction_id=11)

    targets = predicted_targets(build_targets(predicted_candidates=(lower, higher)))

    assert [target.spread_prediction_id for target in targets] == [11, 10]


def test_ties_are_resolved_deterministically():
    east = make_candidate(latitude=32.75, longitude=35.08, risk_score=80.0, spread_prediction_id=20)
    west = make_candidate(latitude=32.75, longitude=35.07, risk_score=80.0, spread_prediction_id=10)

    targets = predicted_targets(build_targets(predicted_candidates=(east, west)))

    assert [target.longitude for target in targets] == [35.07, 35.08]


def test_shuffled_input_produces_same_final_tuple():
    candidate_a = make_candidate(latitude=32.75, longitude=35.08, risk_score=80.0, spread_prediction_id=10)
    candidate_b = make_candidate(latitude=32.76, longitude=35.09, risk_score=90.0, spread_prediction_id=11)
    candidate_c = make_candidate(latitude=32.77, longitude=35.10, risk_score=70.0, spread_prediction_id=12)

    first = build_targets(predicted_candidates=(candidate_a, candidate_b, candidate_c))
    second = build_targets(predicted_candidates=(candidate_c, candidate_a, candidate_b))

    assert first == second


def test_repeated_calculation_with_identical_input_is_equal():
    candidates = (
        make_candidate(latitude=32.75, longitude=35.08, risk_score=80.0, spread_prediction_id=10),
        make_candidate(latitude=32.76, longitude=35.09, risk_score=90.0, spread_prediction_id=11),
    )

    first = build_targets(predicted_candidates=candidates)
    second = build_targets(predicted_candidates=candidates)

    assert first == second


@pytest.mark.parametrize("fire_event_id", [0, -1, True, "1"])
def test_invalid_fire_event_id_rejected(fire_event_id):
    with pytest.raises(ValueError):
        ResponseTargetCalculator().build_targets(
            fire_event_id=fire_event_id,
            fire_latitude=FIRE_LAT,
            fire_longitude=FIRE_LON,
            severity_score=None,
            predicted_candidates=(),
        )


@pytest.mark.parametrize("latitude", [-90.1, 90.1, math.nan, math.inf, -math.inf, True])
def test_invalid_fire_latitude_rejected(latitude):
    with pytest.raises(ValueError):
        ResponseTargetCalculator().build_targets(
            fire_event_id=FIRE_EVENT_ID,
            fire_latitude=latitude,
            fire_longitude=FIRE_LON,
            severity_score=None,
            predicted_candidates=(),
        )


@pytest.mark.parametrize("longitude", [-180.1, 180.1, math.nan, math.inf, -math.inf, True])
def test_invalid_fire_longitude_rejected(longitude):
    with pytest.raises(ValueError):
        ResponseTargetCalculator().build_targets(
            fire_event_id=FIRE_EVENT_ID,
            fire_latitude=FIRE_LAT,
            fire_longitude=longitude,
            severity_score=None,
            predicted_candidates=(),
        )


def test_non_candidate_sequence_member_rejected():
    with pytest.raises(ValueError):
        build_targets(predicted_candidates=("not-a-candidate",))

"""Tests for FireSpreadEffectiveState fingerprint semantics."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.calculators.fire_spread.fire_spread_config import METHODOLOGY_NAME, METHODOLOGY_VERSION
from src.models import (
    FireSpreadEffectiveState,
    FireSpreadFuelClass,
    FireSpreadInput,
    FireSpreadInputResult,
    FireSpreadInputStatus,
)

FIRE_EVENT_ID = 42
AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def make_input(**overrides) -> FireSpreadInput:
    defaults = dict(
        origin_latitude=32.731,
        origin_longitude=35.046,
        wind_speed_kmh=21.6,
        wind_direction_deg=270.0,
        fuel_moisture_percent=9.5,
        fuel_class=FireSpreadFuelClass.SHRUBS,
        horizon_minutes=30,
    )
    defaults.update(overrides)
    return FireSpreadInput(**defaults)


def make_state(**overrides) -> FireSpreadEffectiveState:
    fire_event_id = overrides.pop("fire_event_id", FIRE_EVENT_ID)
    spread_input = make_input(**overrides)
    return FireSpreadEffectiveState.from_input(fire_event_id=fire_event_id, spread_input=spread_input)


def test_ready_prepared_input_can_construct_effective_state():
    spread_input = make_input()

    state = FireSpreadEffectiveState.from_input(fire_event_id=FIRE_EVENT_ID, spread_input=spread_input)

    assert state.fire_event_id == FIRE_EVENT_ID
    assert state.origin_latitude == spread_input.origin_latitude
    assert state.origin_longitude == spread_input.origin_longitude
    assert state.wind_speed_kmh == spread_input.wind_speed_kmh
    assert state.wind_direction_deg == spread_input.wind_direction_deg
    assert state.fuel_moisture_percent == spread_input.fuel_moisture_percent
    assert state.fuel_class is spread_input.fuel_class
    assert state.horizon_minutes == spread_input.horizon_minutes
    assert state.methodology == METHODOLOGY_NAME
    assert state.methodology_version == METHODOLOGY_VERSION


def test_identical_prepared_input_produces_equal_state_and_fingerprint():
    first = make_state()
    second = make_state()

    assert first == second
    assert first.fingerprint == second.fingerprint


def test_fingerprint_is_stable_across_repeated_calls():
    state = make_state()

    assert state.fingerprint == state.fingerprint


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("horizon_minutes", 60),
        ("origin_latitude", 32.732),
        ("origin_longitude", 35.047),
        ("wind_speed_kmh", 22.1),
        ("wind_direction_deg", 280.0),
        ("fuel_moisture_percent", 10.0),
        ("fuel_class", FireSpreadFuelClass.GRASSLAND),
    ],
)
def test_calculator_relevant_field_change_changes_state_and_fingerprint(field_name, value):
    baseline = make_state()
    changed = make_state(**{field_name: value})

    assert changed != baseline
    assert changed.fingerprint != baseline.fingerprint


def test_fire_event_id_is_part_of_effective_identity_for_isolation():
    baseline = make_state(fire_event_id=10)
    same_environment_other_event = make_state(fire_event_id=20)

    assert same_environment_other_event != baseline
    assert same_environment_other_event.fingerprint != baseline.fingerprint


def test_weather_source_ids_are_not_part_of_effective_fingerprint():
    first = make_state_from_trace_context(weather_observation_id=100)
    second = make_state_from_trace_context(weather_observation_id=101)

    assert first == second
    assert first.fingerprint == second.fingerprint


def test_severity_assessment_id_alone_is_not_part_of_effective_fingerprint():
    first = make_state_from_trace_context(severity_assessment_id=10)
    second = make_state_from_trace_context(severity_assessment_id=11)

    assert first == second
    assert first.fingerprint == second.fingerprint


def test_severity_score_alone_is_not_part_of_effective_fingerprint():
    first = make_state_from_trace_context(severity_score=50.0)
    second = make_state_from_trace_context(severity_score=90.0)

    assert first == second
    assert first.fingerprint == second.fingerprint


def test_fire_detection_confidence_alone_is_not_part_of_effective_fingerprint():
    first = make_state_from_trace_context(fire_detection_confidence=0.81)
    second = make_state_from_trace_context(fire_detection_confidence=0.90)

    assert first == second
    assert first.fingerprint == second.fingerprint


def test_processing_as_of_timestamp_alone_is_not_part_of_effective_fingerprint():
    first = make_state_from_trace_context(as_of=AS_OF)
    second = make_state_from_trace_context(as_of=datetime(2026, 9, 14, 12, 5, tzinfo=timezone.utc))

    assert first == second
    assert first.fingerprint == second.fingerprint


def test_same_environmental_values_and_same_methodology_identity_match():
    spread_input = make_input()
    first = FireSpreadEffectiveState.from_input(
        fire_event_id=FIRE_EVENT_ID,
        spread_input=spread_input,
        methodology=METHODOLOGY_NAME,
        methodology_version=METHODOLOGY_VERSION,
    )
    second = FireSpreadEffectiveState.from_input(
        fire_event_id=FIRE_EVENT_ID,
        spread_input=spread_input,
        methodology=METHODOLOGY_NAME,
        methodology_version=METHODOLOGY_VERSION,
    )

    assert first == second
    assert first.fingerprint == second.fingerprint


@pytest.mark.parametrize(
    ("methodology", "methodology_version"),
    [
        ("ECOGUARD_PROPAGATOR_CA_NEXT", METHODOLOGY_VERSION),
        (METHODOLOGY_NAME, "2.0"),
    ],
)
def test_methodology_identity_difference_changes_fingerprint(methodology, methodology_version):
    baseline = make_state()
    changed = FireSpreadEffectiveState.from_input(
        fire_event_id=FIRE_EVENT_ID,
        spread_input=make_input(),
        methodology=methodology,
        methodology_version=methodology_version,
    )

    assert changed != baseline
    assert changed.fingerprint != baseline.fingerprint


@pytest.mark.parametrize(
    "status",
    [FireSpreadInputStatus.INSUFFICIENT_DATA, FireSpreadInputStatus.INACTIVE_EVENT],
)
def test_non_ready_input_results_do_not_provide_fake_effective_state(status):
    result = FireSpreadInputResult(status=status, input_data=None, fire_event_id=FIRE_EVENT_ID)

    with pytest.raises(ValueError):
        FireSpreadEffectiveState.from_input(fire_event_id=FIRE_EVENT_ID, spread_input=result.input_data)


def test_invalid_fire_event_id_rejected():
    with pytest.raises(ValueError):
        FireSpreadEffectiveState.from_input(fire_event_id=0, spread_input=make_input())


def test_invalid_spread_input_rejected():
    with pytest.raises(ValueError):
        FireSpreadEffectiveState.from_input(fire_event_id=FIRE_EVENT_ID, spread_input="not-input")


def test_architecture_guard_for_task_1_production_modules():
    forbidden_fragments = (
        "repositories",
        "database",
        "simulation",
        "scripts",
        "external",
        "road_network",
        "RoadNetworkRepository",
        "routing",
        "Dijkstra",
        "resource_allocation",
    )
    production_files = [
        Path("backend/src/models/fire_spread_effective_state.py"),
        Path("backend/src/models/operational_refresh_trigger_type.py"),
        Path("backend/src/services/operational_refresh/operational_refresh_policy.py"),
    ]

    violations = []
    for path in production_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = ",".join(alias.name for alias in node.names)
            if module and any(fragment in module for fragment in forbidden_fragments):
                violations.append((str(path), module))

    assert violations == []


def make_state_from_trace_context(
    *,
    weather_observation_id: int = 100,
    severity_assessment_id: int = 10,
    severity_score: float = 50.0,
    fire_detection_confidence: float = 0.85,
    as_of: datetime = AS_OF,
) -> FireSpreadEffectiveState:
    """Mimic later orchestration context while only passing prepared input into Task 1 state."""
    assert weather_observation_id > 0
    assert severity_assessment_id > 0
    assert 0.0 <= severity_score <= 100.0
    assert 0.0 <= fire_detection_confidence <= 1.0
    assert as_of.tzinfo is not None
    return FireSpreadEffectiveState.from_input(fire_event_id=FIRE_EVENT_ID, spread_input=make_input())

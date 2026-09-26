"""Tests for US 6.1 Task 2 active-FireEvents read models (DTOs).

These models are a pure, read-only presentation contract: they must not
recompute detection, severity, or any other analysis, so this suite also
asserts the module stays free of imports into those subsystems (and of
FastAPI/Pydantic, which belong to the Task 3 HTTP layer instead).
"""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.models import (
    ActiveFireEventSeveritySummary,
    ActiveFireEventsResult,
    ActiveFireEventSummary,
    FireEventStatus,
    FireSeverityAssessmentStatus,
    FireSeverityLevel,
)

DETECTED_AT = datetime(2026, 9, 17, 10, 0, tzinfo=timezone.utc)
UPDATED_AT = datetime(2026, 9, 17, 10, 10, tzinfo=timezone.utc)
ASSESSED_AT = datetime(2026, 9, 17, 10, 20, tzinfo=timezone.utc)
AS_OF = datetime(2026, 9, 17, 11, 0, tzinfo=timezone.utc)


def severity(**overrides) -> ActiveFireEventSeveritySummary:
    values = {
        "assessment_id": 1,
        "status": FireSeverityAssessmentStatus.VALID,
        "score": 70.0,
        "level": FireSeverityLevel.HIGH,
        "assessed_at": ASSESSED_AT,
    }
    values.update(overrides)
    return ActiveFireEventSeveritySummary(**values)


def event(**overrides) -> ActiveFireEventSummary:
    values = {
        "fire_event_id": 1,
        "status": FireEventStatus.SUSPECTED,
        "latitude": 32.731,
        "longitude": 35.046,
        "detection_confidence": 0.6,
        "detected_at": DETECTED_AT,
        "updated_at": UPDATED_AT,
        "created_at": UPDATED_AT,
        "severity": None,
    }
    values.update(overrides)
    return ActiveFireEventSummary(**values)


# ---------------------------------------------------------------------------
# ActiveFireEventSeveritySummary
# ---------------------------------------------------------------------------


def test_severity_summary_accepts_valid_result():
    result = severity()

    assert result.assessment_id == 1
    assert result.status is FireSeverityAssessmentStatus.VALID
    assert result.score == pytest.approx(70.0)
    assert result.level is FireSeverityLevel.HIGH


def test_severity_summary_accepts_no_score_or_level_for_non_valid_status():
    result = severity(status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA, score=None, level=None)

    assert result.score is None
    assert result.level is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"assessment_id": 0},
        {"assessment_id": -1},
        {"assessment_id": True},
        {"status": "valid"},
        {"score": float("nan")},
        {"level": "high"},
        {"assessed_at": ASSESSED_AT.replace(tzinfo=None)},
    ],
)
def test_severity_summary_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        severity(**overrides)


def test_severity_summary_is_frozen():
    result = severity()
    with pytest.raises(FrozenInstanceError):
        result.score = 10.0


# ---------------------------------------------------------------------------
# ActiveFireEventSummary
# ---------------------------------------------------------------------------


def test_event_summary_accepts_no_severity():
    result = event(severity=None)

    assert result.severity is None


def test_event_summary_accepts_severity():
    result = event(severity=severity())

    assert result.severity is not None
    assert result.severity.assessment_id == 1


@pytest.mark.parametrize("status", [FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED])
def test_event_summary_accepts_active_statuses(status):
    result = event(status=status)

    assert result.status is status


@pytest.mark.parametrize(
    "overrides",
    [
        {"fire_event_id": 0},
        {"status": "suspected"},
        {"latitude": 91.0},
        {"longitude": -181.0},
        {"detection_confidence": 1.5},
        {"detection_confidence": -0.1},
        {"detected_at": DETECTED_AT.replace(tzinfo=None)},
        {"updated_at": UPDATED_AT.replace(tzinfo=None)},
        {"severity": object()},
    ],
)
def test_event_summary_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        event(**overrides)


def test_event_summary_is_frozen():
    result = event()
    with pytest.raises(FrozenInstanceError):
        result.latitude = 0.0


def test_event_summary_has_no_area_name_field():
    """Task 6: FireEvent persists no trustworthy area/location label, so this
    read model must not fabricate one - the field is simply absent here."""
    assert not hasattr(event(), "area_name")


# ---------------------------------------------------------------------------
# ActiveFireEventsResult
# ---------------------------------------------------------------------------


def test_result_accepts_empty_items():
    result = ActiveFireEventsResult(as_of=AS_OF, items=())

    assert result.items == ()


def test_result_accepts_multiple_items_and_preserves_order():
    first = event(fire_event_id=5)
    second = event(fire_event_id=2)
    result = ActiveFireEventsResult(as_of=AS_OF, items=(first, second))

    assert result.items == (first, second)


def test_result_coerces_iterable_items_to_tuple():
    result = ActiveFireEventsResult(as_of=AS_OF, items=[event(fire_event_id=1)])

    assert isinstance(result.items, tuple)


@pytest.mark.parametrize(
    "overrides",
    [
        {"as_of": AS_OF.replace(tzinfo=None)},
        {"items": (object(),)},
    ],
)
def test_result_rejects_invalid_values(overrides):
    values = {"as_of": AS_OF, "items": ()}
    values.update(overrides)
    with pytest.raises(ValueError):
        ActiveFireEventsResult(**values)


def test_result_is_frozen():
    result = ActiveFireEventsResult(as_of=AS_OF, items=())
    with pytest.raises(FrozenInstanceError):
        result.as_of = AS_OF


# ---------------------------------------------------------------------------
# Architecture guard: pure read/presentation module only
# ---------------------------------------------------------------------------


def test_active_fire_events_does_not_import_calculation_persistence_or_http_modules():
    forbidden_fragments = (
        "FireDetectionAgent",
        "FireSeverityAssessmentAgent",
        "FireSpreadPredictionAgent",
        "ResponseTargetGenerationAgent",
        "OperationalRefreshOrchestrator",
        "fastapi",
        "pydantic",
        "simulation",
        "external",
        "sqlalchemy",
        "repositories",
        "database",
        "calculators",
        "services",
    )
    path = (Path(__file__).resolve().parents[3] / "backend/src/models/active_fire_events.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        module = ""
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
        elif isinstance(node, ast.Import):
            module = ",".join(alias.name for alias in node.names)
        if any(fragment in module for fragment in forbidden_fragments):
            violations.append(module)

    assert violations == []

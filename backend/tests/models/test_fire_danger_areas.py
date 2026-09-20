"""Tests for Task A4 Fire Danger areas/latest read models (DTOs).

Pure, read-only presentation contract: must not recompute FFWI or import
calculation/persistence/HTTP modules, matching test_active_fire_events.py's
own precedent for the same kind of read model.
"""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.models.fire_danger_areas import (
    FireDangerAreaAssessmentSummary,
    FireDangerAreaSnapshot,
    FireDangerAreasResult,
)
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel

ASSESSED_AT = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
AS_OF = datetime(2026, 9, 19, 12, 30, tzinfo=timezone.utc)


def assessment_summary(**overrides) -> FireDangerAreaAssessmentSummary:
    values = {
        "assessment_id": 1,
        "status": FireDangerAssessmentStatus.VALID,
        "score": 42.5,
        "level": FireDangerLevel.VERY_HIGH,
        "assessed_at": ASSESSED_AT,
        "methodology": "FOSBERG_FFWI",
        "methodology_version": "1.0",
    }
    values.update(overrides)
    return FireDangerAreaAssessmentSummary(**values)


def area_snapshot(**overrides) -> FireDangerAreaSnapshot:
    values = {
        "area_id": "area-carmel",
        "area_name": "Carmel",
        "area_latitude": 32.731,
        "area_longitude": 35.046,
        "area_radius_km": 5.0,
        "assessment": None,
    }
    values.update(overrides)
    return FireDangerAreaSnapshot(**values)


# ---------------------------------------------------------------------------
# FireDangerAreaAssessmentSummary
# ---------------------------------------------------------------------------


def test_assessment_summary_accepts_valid_result():
    result = assessment_summary()

    assert result.score == pytest.approx(42.5)
    assert result.level is FireDangerLevel.VERY_HIGH
    assert result.status is FireDangerAssessmentStatus.VALID


def test_assessment_summary_accepts_no_score_or_level_for_insufficient_data():
    result = assessment_summary(status=FireDangerAssessmentStatus.INSUFFICIENT_DATA, score=None, level=None)

    assert result.score is None
    assert result.level is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"assessment_id": 0},
        {"assessment_id": -1},
        {"status": "valid"},
        {"score": float("nan")},
        {"level": "very_high"},
        {"assessed_at": ASSESSED_AT.replace(tzinfo=None)},
        {"methodology": ""},
        {"methodology_version": ""},
    ],
)
def test_assessment_summary_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        assessment_summary(**overrides)


def test_assessment_summary_is_frozen():
    result = assessment_summary()
    with pytest.raises(FrozenInstanceError):
        result.score = 10.0


# ---------------------------------------------------------------------------
# FireDangerAreaSnapshot
# ---------------------------------------------------------------------------


def test_area_snapshot_accepts_no_assessment():
    result = area_snapshot(assessment=None)

    assert result.assessment is None


def test_area_snapshot_accepts_assessment():
    result = area_snapshot(assessment=assessment_summary())

    assert result.assessment is not None
    assert result.assessment.score == pytest.approx(42.5)


@pytest.mark.parametrize(
    "overrides",
    [
        {"area_id": ""},
        {"area_name": ""},
        {"area_latitude": 91.0},
        {"area_longitude": -181.0},
        {"area_radius_km": 0},
        {"area_radius_km": -1},
        {"assessment": object()},
    ],
)
def test_area_snapshot_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        area_snapshot(**overrides)


def test_area_snapshot_is_frozen():
    result = area_snapshot()
    with pytest.raises(FrozenInstanceError):
        result.area_name = "Other"


# ---------------------------------------------------------------------------
# FireDangerAreasResult
# ---------------------------------------------------------------------------


def test_result_accepts_empty_areas():
    result = FireDangerAreasResult(as_of=AS_OF, areas=())

    assert result.areas == ()


def test_result_accepts_multiple_areas_and_preserves_order():
    first = area_snapshot(area_id="area-a")
    second = area_snapshot(area_id="area-b")
    result = FireDangerAreasResult(as_of=AS_OF, areas=(first, second))

    assert result.areas == (first, second)


def test_result_coerces_iterable_areas_to_tuple():
    result = FireDangerAreasResult(as_of=AS_OF, areas=[area_snapshot()])

    assert isinstance(result.areas, tuple)


@pytest.mark.parametrize(
    "overrides",
    [
        {"as_of": AS_OF.replace(tzinfo=None)},
        {"areas": (object(),)},
    ],
)
def test_result_rejects_invalid_values(overrides):
    values = {"as_of": AS_OF, "areas": ()}
    values.update(overrides)
    with pytest.raises(ValueError):
        FireDangerAreasResult(**values)


def test_result_is_frozen():
    result = FireDangerAreasResult(as_of=AS_OF, areas=())
    with pytest.raises(FrozenInstanceError):
        result.as_of = AS_OF


# ---------------------------------------------------------------------------
# Architecture guard: pure read/presentation module only
# ---------------------------------------------------------------------------


def test_fire_danger_areas_does_not_import_calculation_persistence_or_http_modules():
    forbidden_fragments = (
        "FFWICalculator",
        "FireDangerAssessmentAgent",
        "FireDangerInputService",
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
    path = Path(__file__).resolve().parents[2] / "src" / "models" / "fire_danger_areas.py"
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

"""Tests for Task A5 Operations Activity domain read models (DTOs).

Pure, read-only presentation contracts: must not recompute FFWI/severity/
detection/spread or import business-execution/persistence/HTTP modules,
matching test_active_fire_events.py's / test_fire_danger_areas.py's own
precedent for the same kind of read model.
"""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.models.fire_danger_assessment_detail import FireDangerAssessmentDetail
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.models.operations_activity import (
    FireDangerActivityDetail,
    FireEventActivityDetail,
    FireEventActivityDetails,
    FireEventEvidenceRefs,
    OperationsActivityLocation,
    OperationsActivityType,
)

ASSESSED_AT = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def make_fire_danger_detail(**overrides) -> FireDangerAssessmentDetail:
    values = dict(
        assessment_id=1,
        area_id="area-carmel",
        area_name="Carmel",
        area_latitude=32.731,
        area_longitude=35.046,
        area_radius_km=5.0,
        status=FireDangerAssessmentStatus.VALID,
        score=42.5,
        level=FireDangerLevel.VERY_HIGH,
        assessed_at=ASSESSED_AT,
        methodology="FOSBERG_FFWI",
        methodology_version="1.0",
        weather_inputs=(),
    )
    values.update(overrides)
    return FireDangerAssessmentDetail(**values)


def make_fire_event(**overrides) -> FireEvent:
    values = dict(
        latitude=32.7,
        longitude=35.0,
        detected_at=ASSESSED_AT,
        updated_at=ASSESSED_AT,
        status=FireEventStatus.CONFIRMED,
        detection_confidence=0.9,
        methodology="ECOGUARD_MULTI_SOURCE_DETECTION",
        methodology_version="1.0",
    )
    values.update(overrides)
    return FireEvent(**values)


# ---------------------------------------------------------------------------
# OperationsActivityLocation
# ---------------------------------------------------------------------------


def test_location_accepts_valid_coordinates():
    location = OperationsActivityLocation(latitude=32.7, longitude=35.0)

    assert location.latitude == 32.7


@pytest.mark.parametrize("overrides", [{"latitude": 91.0}, {"longitude": -181.0}])
def test_location_rejects_out_of_range_coordinates(overrides):
    values = {"latitude": 0.0, "longitude": 0.0}
    values.update(overrides)
    with pytest.raises(ValueError):
        OperationsActivityLocation(**values)


def test_location_is_frozen():
    location = OperationsActivityLocation(latitude=32.7, longitude=35.0)
    with pytest.raises(FrozenInstanceError):
        location.latitude = 0.0


# ---------------------------------------------------------------------------
# FireDangerActivityDetail
# ---------------------------------------------------------------------------


def test_fire_danger_activity_detail_fixes_its_own_discriminator():
    detail = FireDangerActivityDetail(
        entity_id=1,
        occurred_at=ASSESSED_AT,
        title="Fire Danger Assessment — Carmel",
        location=OperationsActivityLocation(latitude=32.731, longitude=35.046),
        details=make_fire_danger_detail(),
    )

    assert detail.activity_type is OperationsActivityType.FIRE_DANGER


def test_fire_danger_activity_detail_rejects_wrong_discriminator():
    with pytest.raises(ValueError):
        FireDangerActivityDetail(
            entity_id=1,
            occurred_at=ASSESSED_AT,
            title="x",
            location=None,
            details=make_fire_danger_detail(),
            activity_type=OperationsActivityType.NEWS_REPORT,
        )


def test_fire_danger_activity_detail_rejects_wrong_details_type():
    with pytest.raises(ValueError):
        FireDangerActivityDetail(
            entity_id=1,
            occurred_at=ASSESSED_AT,
            title="x",
            location=None,
            details=object(),
        )


# ---------------------------------------------------------------------------
# FireEventActivityDetail - no fabricated area name
# ---------------------------------------------------------------------------


def test_fire_event_activity_detail_title_has_no_area_name():
    detail = FireEventActivityDetail(
        entity_id=42,
        occurred_at=ASSESSED_AT,
        title="Fire Event #42",
        location=OperationsActivityLocation(latitude=32.7, longitude=35.0),
        details=FireEventActivityDetails(
            fire_event=make_fire_event(),
            evidence=FireEventEvidenceRefs(satellite_hotspot_ids=(), news_report_ids=()),
            latest_severity=None,
            created_at=ASSESSED_AT,
        ),
    )

    assert detail.title == "Fire Event #42"
    assert not hasattr(detail.details.fire_event, "area_name")


def test_fire_event_evidence_refs_coerces_to_tuple():
    refs = FireEventEvidenceRefs(satellite_hotspot_ids=[1, 2], news_report_ids=[3])

    assert refs.satellite_hotspot_ids == (1, 2)
    assert isinstance(refs.news_report_ids, tuple)


# ---------------------------------------------------------------------------
# Architecture guard: pure read/presentation module only
# ---------------------------------------------------------------------------


def test_operations_activity_does_not_import_business_execution_or_persistence_modules():
    forbidden_fragments = (
        "FFWICalculator",
        "FireDangerAssessmentAgent",
        "FireDetectionCalculator",
        "FireDetectionAgent",
        "FireSeverityAssessmentAgent",
        "FireSpreadPredictionAgent",
        "SimulationRefreshCoordinator",
        "GlobalPlanningOrchestrator",
        "genetic_optimizer",
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
    path = Path(__file__).resolve().parents[2] / "src" / "models" / "operations_activity.py"
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

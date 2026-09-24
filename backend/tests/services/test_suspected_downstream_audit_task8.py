"""Task 8 audit, updated by Task 9A: which downstream subsystems see a SUSPECTED FireEvent?

Task 8 found that nothing distinguished SUSPECTED from CONFIRMED. Task 9A introduced the response-eligible concept
({CONFIRMED}, `src.models.fire_event_response_eligibility`) and gated every response-side entry point on it, while the
monitoring side (repository matching, history, evidence, severity/spread INPUT primitives, dashboard) keeps the wider
active set {SUSPECTED, CONFIRMED}. These tests pin that split so it cannot drift back.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import importlib
import inspect

import pytest

from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.repositories.fire_event_repository import FireEventRepository

ACTIVE = {FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED}
ELIGIBLE = {FireEventStatus.CONFIRMED}
T = datetime(2026, 9, 14, 6, 0, tzinfo=timezone.utc)

# (subsystem, module, attribute, expected set)
ACTIVE_SETS = [
    ("event repository (active queries)", "src.repositories.fire_event_repository", "_ACTIVE_STATUSES", ACTIVE),
    ("severity assessment input", "src.services.fire_severity.fire_severity_input_service", "_ACTIVE_EVENT_STATUSES", ACTIVE),
    ("spread prediction input", "src.services.fire_spread.fire_spread_input_service", "_ACTIVE_EVENT_STATUSES", ACTIVE),
    ("response-target input", "src.services.response_target.response_target_input_service", "_ACTIVE_EVENT_STATUSES", ACTIVE),
    ("operational refresh (severity/spread/targets/planning trigger)", "src.services.operational_refresh.operational_refresh_orchestrator", "_ACTIVE_FIRE_EVENT_STATUSES", ELIGIBLE),
    ("response-plan activation (resource commitment)", "src.services.resource_reservation.response_plan_activation_service", "_RESPONSE_ELIGIBLE_STATUSES", ELIGIBLE),
    ("global response-plan activation (resource commitment)", "src.services.global_planning.global_response_plan_activation_service", "_RESPONSE_ELIGIBLE_STATUSES", ELIGIBLE),
    ("resource-commitment queries", "src.repositories.resource_commitment_repository", "_ACTIVE_STATUSES", ACTIVE),
]


@pytest.mark.parametrize("subsystem,module,attribute,expected", ACTIVE_SETS, ids=[row[0] for row in ACTIVE_SETS])
def test_monitoring_sets_keep_suspected_and_response_side_sets_are_confirmed_only(subsystem, module, attribute, expected):
    statuses = set(getattr(importlib.import_module(module), attribute))
    assert statuses == expected, f"{subsystem}: {statuses}"


def test_only_resolved_and_dismissed_are_inactive_for_the_planning_pipeline():
    for module in (
        "src.services.fire_severity.fire_severity_input_service",
        "src.services.fire_spread.fire_spread_input_service",
        "src.services.response_target.response_target_input_service",
    ):
        inactive = set(importlib.import_module(module)._INACTIVE_EVENT_STATUSES)
        assert inactive == {FireEventStatus.RESOLVED, FireEventStatus.DISMISSED}
        assert FireEventStatus.SUSPECTED not in inactive


def test_the_simulation_refresh_coordinator_selects_events_through_the_response_eligible_query_only():
    """Weather/resource-triggered refresh must use the response-eligible query, never the wider monitoring one."""
    module = importlib.import_module("src.simulation.analysis.simulation_refresh_coordinator")
    tree = ast.parse(inspect.getsource(module))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "get_response_eligible_events_near" in names
    assert "get_active_events_near" not in names


def test_a_suspected_event_is_returned_by_the_active_event_queries_that_feed_the_pipeline(sqlite_session_factory):
    repository = FireEventRepository(session_factory=sqlite_session_factory)
    from src.models.fire_evidence_ref import FireEvidenceRef
    from src.models.fire_evidence_type import FireEvidenceType
    from src.models import SatelliteHotspot
    from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository

    satellites = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    satellites.save_hotspot(SatelliteHotspot(latitude=32.7, longitude=35.0, detected_at=T, confidence="n", satellite="NOAA-20", instrument="VIIRS"))
    ref = FireEvidenceRef(FireEvidenceType.SATELLITE, satellites.get_recent_hotspots(T, 60)[0].id)
    stored = repository.create_event(
        FireEvent(latitude=32.7, longitude=35.0, detected_at=T, updated_at=T, status=FireEventStatus.SUSPECTED,
                  detection_confidence=0.5, methodology="test", methodology_version="1"),
        supporting_evidence=(ref,),
    )
    assert [e.id for e in repository.get_active_events()] == [stored.id]
    assert [e.id for e in repository.get_active_events_near(32.7, 35.0, 10.0, T)] == [stored.id]
    assert stored.id in repository.get_active_fire_event_ids()
    assert repository.get_by_id(stored.id).event.status is FireEventStatus.SUSPECTED


def test_the_backend_has_no_status_specific_dispatch_gate():
    """CONFIRMED appears in services only inside the shared 'active statuses' set literals - never as a branch condition."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "services"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "CONFIRMED" and getattr(node.value, "id", "") == "FireEventStatus":
                inside_set, ancestor = False, parents.get(node)
                while ancestor is not None:
                    inside_set = inside_set or isinstance(ancestor, ast.Set)
                    ancestor = parents.get(ancestor)
                if not inside_set:
                    offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == [], f"a service now branches on CONFIRMED (update this audit): {offenders}"

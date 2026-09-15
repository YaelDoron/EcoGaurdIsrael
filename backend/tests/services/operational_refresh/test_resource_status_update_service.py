"""Tests for ResourceStatusUpdateService operational availability updates."""
from __future__ import annotations

from datetime import datetime, timezone
import ast
from pathlib import Path

import pytest

from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.models.fire_event_status import FireEventStatus
from src.models.firefighting_resource import FirefightingResource
from src.models.resource_status import ResourceStatus
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.services.operational_refresh import (
    ResourceStatusUpdateService,
    ResourceStatusUpdateStatus,
)


class FakeResourceRepository:
    def __init__(self, resources) -> None:
        self.resources = dict(resources)
        self.get_calls = []
        self.update_calls = []
        self.fail_update = False

    def get_by_id(self, resource_id):
        self.get_calls.append(resource_id)
        return self.resources.get(resource_id)

    def update_status(self, resource_id, status):
        self.update_calls.append((resource_id, status))
        if self.fail_update:
            raise RuntimeError("database unavailable")
        resource = self.resources.get(resource_id)
        if resource is None:
            return None
        updated = FirefightingResource(id=resource.id, station_id=resource.station_id, status=status)
        self.resources[resource_id] = updated
        return updated


class CallRecorder:
    def __init__(self) -> None:
        self.calls = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return None

        return _record


def make_resource(status: ResourceStatus = ResourceStatus.AVAILABLE) -> FirefightingResource:
    return FirefightingResource(id="TRUCK-A", station_id="STATION-1", status=status)


@pytest.mark.parametrize(
    ("previous_status", "new_status"),
    [
        (ResourceStatus.AVAILABLE, ResourceStatus.ASSIGNED),
        (ResourceStatus.AVAILABLE, ResourceStatus.UNAVAILABLE),
        (ResourceStatus.ASSIGNED, ResourceStatus.AVAILABLE),
        (ResourceStatus.UNAVAILABLE, ResourceStatus.AVAILABLE),
    ],
)
def test_required_status_transitions_return_updated(previous_status, new_status):
    repository = FakeResourceRepository({"TRUCK-A": make_resource(previous_status)})
    service = ResourceStatusUpdateService(repository)

    result = service.update_status(resource_id="TRUCK-A", new_status=new_status)

    assert result.status is ResourceStatusUpdateStatus.UPDATED
    assert result.changed is True
    assert result.resource_id == "TRUCK-A"
    assert result.previous_status is previous_status
    assert result.current_status is new_status
    assert result.resource.id == "TRUCK-A"
    assert result.resource.station_id == "STATION-1"
    assert repository.update_calls == [("TRUCK-A", new_status)]


@pytest.mark.parametrize(
    "status",
    [ResourceStatus.AVAILABLE, ResourceStatus.ASSIGNED, ResourceStatus.UNAVAILABLE],
)
def test_same_state_transition_returns_no_op_without_persistence_update(status):
    repository = FakeResourceRepository({"TRUCK-A": make_resource(status)})
    service = ResourceStatusUpdateService(repository)

    result = service.update_status(resource_id="TRUCK-A", new_status=status)

    assert result.status is ResourceStatusUpdateStatus.NO_OP
    assert result.changed is False
    assert result.previous_status is status
    assert result.current_status is status
    assert result.resource.id == "TRUCK-A"
    assert result.resource.station_id == "STATION-1"
    assert repository.update_calls == []


def test_missing_resource_returns_not_found_without_creating_resource():
    repository = FakeResourceRepository({})
    service = ResourceStatusUpdateService(repository)

    result = service.update_status(resource_id="MISSING", new_status=ResourceStatus.AVAILABLE)

    assert result.status is ResourceStatusUpdateStatus.NOT_FOUND
    assert result.changed is False
    assert result.previous_status is None
    assert result.current_status is None
    assert result.resource is None
    assert repository.resources == {}


@pytest.mark.parametrize("bad_status", ["available", None, object()])
def test_invalid_status_is_rejected(bad_status):
    repository = FakeResourceRepository({"TRUCK-A": make_resource()})
    service = ResourceStatusUpdateService(repository)

    with pytest.raises(ValueError):
        service.update_status(resource_id="TRUCK-A", new_status=bad_status)


def test_repository_failure_returns_failed_result():
    repository = FakeResourceRepository({"TRUCK-A": make_resource()})
    repository.fail_update = True
    service = ResourceStatusUpdateService(repository)

    result = service.update_status(resource_id="TRUCK-A", new_status=ResourceStatus.UNAVAILABLE)

    assert result.status is ResourceStatusUpdateStatus.FAILED
    assert result.changed is False
    assert "database unavailable" in result.error_message


def insert_station_resource_and_fire_event(sqlite_session_factory) -> int:
    session = sqlite_session_factory()
    session.add(FireStationDB(id="STATION-1", name="Station 1", latitude=32.1, longitude=35.1))
    session.add(
        FirefightingResourceDB(id="TRUCK-A", station_id="STATION-1", status=ResourceStatus.AVAILABLE)
    )
    fire_event = FireEventDB(
        latitude=32.731,
        longitude=35.046,
        detected_at=datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 9, 15, 10, 5, tzinfo=timezone.utc),
        status=FireEventStatus.CONFIRMED.value,
        detection_confidence=0.82,
        methodology="test-detection",
        methodology_version="1.0",
    )
    session.add(fire_event)
    session.commit()
    fire_event_id = fire_event.id
    session.close()
    return fire_event_id


def test_available_resource_set_refreshes_through_existing_repository_query(sqlite_session_factory):
    insert_station_resource_and_fire_event(sqlite_session_factory)
    repository = FirefightingResourceRepository(sqlite_session_factory)
    service = ResourceStatusUpdateService(repository)

    assert [resource.id for resource in repository.get_available_resources(["STATION-1"])] == ["TRUCK-A"]

    service.update_status(resource_id="TRUCK-A", new_status=ResourceStatus.UNAVAILABLE)
    assert repository.get_available_resources(["STATION-1"]) == []

    service.update_status(resource_id="TRUCK-A", new_status=ResourceStatus.AVAILABLE)
    assert [resource.id for resource in repository.get_available_resources(["STATION-1"])] == ["TRUCK-A"]

    service.update_status(resource_id="TRUCK-A", new_status=ResourceStatus.ASSIGNED)
    assert repository.get_available_resources(["STATION-1"]) == []

    service.update_status(resource_id="TRUCK-A", new_status=ResourceStatus.AVAILABLE)
    assert [resource.id for resource in repository.get_available_resources(["STATION-1"])] == ["TRUCK-A"]


def test_resource_status_update_does_not_modify_fire_event(sqlite_session_factory):
    fire_event_id = insert_station_resource_and_fire_event(sqlite_session_factory)
    repository = FirefightingResourceRepository(sqlite_session_factory)
    service = ResourceStatusUpdateService(repository)

    service.update_status(resource_id="TRUCK-A", new_status=ResourceStatus.UNAVAILABLE)

    session = sqlite_session_factory()
    fire_event = session.get(FireEventDB, fire_event_id)
    assert fire_event.latitude == 32.731
    assert fire_event.longitude == 35.046
    assert fire_event.status == FireEventStatus.CONFIRMED.value
    assert fire_event.detection_confidence == 0.82
    session.close()


def test_resource_status_update_has_no_analysis_agent_dependencies():
    repository = FakeResourceRepository({"TRUCK-A": make_resource()})
    service = ResourceStatusUpdateService(repository)
    fire_detection_agent = CallRecorder()
    severity_agent = CallRecorder()
    spread_orchestrator = CallRecorder()
    spread_agent = CallRecorder()
    spread_input_service = CallRecorder()
    response_target_agent = CallRecorder()

    service.update_status(resource_id="TRUCK-A", new_status=ResourceStatus.UNAVAILABLE)

    assert fire_detection_agent.calls == []
    assert severity_agent.calls == []
    assert spread_orchestrator.calls == []
    assert spread_agent.calls == []
    assert spread_input_service.calls == []
    assert response_target_agent.calls == []


def test_resource_status_update_modules_do_not_import_out_of_scope_systems():
    forbidden_fragments = (
        "RoadNetworkRepository",
        "road_network",
        "GraphNode",
        "GraphEdge",
        "osmnx",
        "seed_road_networks",
        "FireDetectionAgent",
        "FireSeverityAssessmentAgent",
        "FireSpreadRefreshOrchestrator",
        "FireSpreadPredictionAgent",
        "FireSpreadInputService",
        "ResponseTargetGenerationAgent",
        "simulation",
        "routing",
        "Dijkstra",
        "genetic",
        "ResponsePlan",
    )
    production_files = [
        Path("backend/src/services/operational_refresh/resource_status_update_service.py"),
        Path("backend/src/services/operational_refresh/resource_status_update_result.py"),
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

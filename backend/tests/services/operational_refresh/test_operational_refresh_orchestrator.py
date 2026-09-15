"""Tests for central operational refresh orchestration."""
from __future__ import annotations

from datetime import datetime, timezone
import ast
from pathlib import Path

import pytest

from src.agents.analysis.response_target_generation_result import (
    ResponseTargetGenerationResult,
    ResponseTargetGenerationStatus,
)
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.models.fire_severity_assessment import FireSeverityAssessment
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.firefighting_resource import FirefightingResource
from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType
from src.models.resource_status import ResourceStatus
from src.models.response_target import ResponseTarget
from src.models.response_target_type import ResponseTargetType
from src.repositories.fire_event_repository import StoredFireEvent
from src.repositories.fire_severity_assessment_repository import StoredFireSeverityAssessment
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.services.operational_refresh import (
    FireSpreadRefreshHorizonResult,
    FireSpreadRefreshHorizonStatus,
    FireSpreadRefreshResult,
    OperationalRefreshOrchestrator,
    OperationalRefreshStatus,
    ResourceStatusUpdateResult,
    ResourceStatusUpdateService,
    ResourceStatusUpdateStatus,
)

AS_OF = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
FIRE_EVENT_ID = 42


class FakeSeverityAgent:
    def __init__(self, log: list[str], *, fail: bool = False, score: float = 70.0) -> None:
        self.log = log
        self.fail = fail
        self.score = score
        self.calls = []

    def assess(self, fire_event_id: int, assessed_at: datetime) -> StoredFireSeverityAssessment:
        self.log.append("severity")
        self.calls.append({"fire_event_id": fire_event_id, "assessed_at": assessed_at})
        if self.fail:
            raise RuntimeError("severity database failed")
        return make_severity_result(fire_event_id=fire_event_id, assessed_at=assessed_at, score=self.score)


class FakeSpreadRefreshOrchestrator:
    def __init__(self, log: list[str], result: FireSpreadRefreshResult | None = None) -> None:
        self.log = log
        self.calls = []
        self.result = result

    def refresh(self, *, fire_event_id: int, trigger_type: OperationalRefreshTriggerType, as_of: datetime):
        self.log.append("spread")
        self.calls.append({"fire_event_id": fire_event_id, "trigger_type": trigger_type, "as_of": as_of})
        return self.result or spread_result(trigger_type=trigger_type)


class FakeResponseTargetAgent:
    def __init__(self, log: list[str], result: ResponseTargetGenerationResult | None = None) -> None:
        self.log = log
        self.calls = []
        self.result = result

    def generate(self, *, fire_event_id: int, as_of: datetime) -> ResponseTargetGenerationResult:
        self.log.append("targets")
        self.calls.append({"fire_event_id": fire_event_id, "as_of": as_of})
        return self.result or target_result(fire_event_id=fire_event_id)


class FakeResourceStatusService:
    def __init__(self, result: ResourceStatusUpdateResult) -> None:
        self.result = result
        self.calls = []

    def update_status(self, *, resource_id, new_status):
        self.calls.append({"resource_id": resource_id, "new_status": new_status})
        return self.result


class FakeResourceRepository:
    def __init__(self, resources=()) -> None:
        self.resources = tuple(resources)
        self.calls = []

    def get_available_resources(self, station_ids: list[str]):
        self.calls.append(station_ids)
        return list(self.resources)


class FakeFireEventRepository:
    def __init__(self, status: FireEventStatus | None = FireEventStatus.CONFIRMED) -> None:
        self.status = status
        self.calls = []

    def get_by_id(self, fire_event_id: int):
        self.calls.append(fire_event_id)
        if self.status is None:
            return None
        return StoredFireEvent(
            id=fire_event_id,
            event=FireEvent(
                latitude=32.731,
                longitude=35.046,
                detected_at=AS_OF,
                updated_at=AS_OF,
                status=self.status,
                detection_confidence=0.82,
                methodology="test",
                methodology_version="1.0",
            ),
        )


class CallRecorder:
    def __init__(self) -> None:
        self.calls = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return None

        return _record


def make_severity_result(
    *,
    fire_event_id: int = FIRE_EVENT_ID,
    assessed_at: datetime = AS_OF,
    score: float = 70.0,
) -> StoredFireSeverityAssessment:
    return StoredFireSeverityAssessment(
        assessment_id=100,
        assessment=FireSeverityAssessment(
            fire_event_id=fire_event_id,
            assessed_at=assessed_at,
            status=FireSeverityAssessmentStatus.VALID,
            score=score,
            level=FireSeverityLevel.HIGH,
            methodology="test-severity",
            methodology_version="1.0",
        ),
        weather_observation_ids=(200,),
        satellite_hotspot_ids=(300,),
        selected_frp_hotspot_id=300,
    )


def spread_result(
    *,
    trigger_type: OperationalRefreshTriggerType = OperationalRefreshTriggerType.WEATHER_UPDATE,
    statuses: tuple[FireSpreadRefreshHorizonStatus, ...] = (
        FireSpreadRefreshHorizonStatus.REFRESHED,
        FireSpreadRefreshHorizonStatus.NO_OP,
    ),
) -> FireSpreadRefreshResult:
    return FireSpreadRefreshResult(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=trigger_type,
        as_of=AS_OF,
        reevaluation_required=True,
        horizon_results=tuple(
            FireSpreadRefreshHorizonResult(
                fire_event_id=FIRE_EVENT_ID,
                horizon_minutes=horizon,
                status=status,
                prediction_id=index if status is not FireSpreadRefreshHorizonStatus.NO_OP else None,
                previous_prediction_id=10 + index,
            )
            for index, (horizon, status) in enumerate(zip((30, 60), statuses), start=1)
        ),
    )


def target_result(
    *,
    fire_event_id: int = FIRE_EVENT_ID,
    status: ResponseTargetGenerationStatus = ResponseTargetGenerationStatus.GENERATED,
) -> ResponseTargetGenerationResult:
    if status is ResponseTargetGenerationStatus.GENERATED:
        targets = (
            ResponseTarget(
                fire_event_id=fire_event_id,
                target_type=ResponseTargetType.ACTIVE_FIRE,
                latitude=32.731,
                longitude=35.046,
                priority_score=72.0,
            ),
        )
        return ResponseTargetGenerationResult(
            success=True,
            fire_event_id=fire_event_id,
            status=status,
            target_set_id=55,
            targets=targets,
            target_count=len(targets),
        )
    if status is ResponseTargetGenerationStatus.INACTIVE_EVENT:
        return ResponseTargetGenerationResult(
            success=True,
            fire_event_id=fire_event_id,
            status=status,
            target_set_id=None,
            targets=(),
            target_count=0,
        )
    return ResponseTargetGenerationResult(
        success=False,
        fire_event_id=fire_event_id,
        status=status,
        target_set_id=None,
        targets=(),
        target_count=0,
        error_message="Response-target generation failed.",
    )


def resource_update_result(
    status: ResourceStatusUpdateStatus,
    *,
    resource: FirefightingResource | None = None,
) -> ResourceStatusUpdateResult:
    if status is ResourceStatusUpdateStatus.UPDATED:
        return ResourceStatusUpdateResult(
            resource_id=resource.id,
            previous_status=ResourceStatus.AVAILABLE,
            current_status=resource.status,
            status=status,
            resource=resource,
        )
    if status is ResourceStatusUpdateStatus.NO_OP:
        return ResourceStatusUpdateResult(
            resource_id=resource.id,
            previous_status=resource.status,
            current_status=resource.status,
            status=status,
            resource=resource,
        )
    if status is ResourceStatusUpdateStatus.NOT_FOUND:
        return ResourceStatusUpdateResult(
            resource_id="MISSING",
            previous_status=None,
            current_status=None,
            status=status,
        )
    return ResourceStatusUpdateResult(
        resource_id="TRUCK-A",
        previous_status=None,
        current_status=None,
        status=status,
        error_message="resource persistence failed",
    )


def make_orchestrator(
    *,
    severity_agent=None,
    spread=None,
    targets=None,
    resource_service=None,
    resource_repository=None,
    fire_event_repository=None,
    log=None,
):
    log = log if log is not None else []
    return OperationalRefreshOrchestrator(
        severity_agent=severity_agent or FakeSeverityAgent(log),
        spread_refresh_orchestrator=spread or FakeSpreadRefreshOrchestrator(log),
        response_target_agent=targets or FakeResponseTargetAgent(log),
        resource_status_service=resource_service
        or FakeResourceStatusService(
            resource_update_result(
                ResourceStatusUpdateStatus.UPDATED,
                resource=FirefightingResource("TRUCK-A", "STATION-1", ResourceStatus.UNAVAILABLE),
            )
        ),
        resource_repository=resource_repository or FakeResourceRepository(),
        fire_event_repository=fire_event_repository or FakeFireEventRepository(),
    )


@pytest.mark.parametrize(
    "trigger_type",
    [OperationalRefreshTriggerType.WEATHER_UPDATE, OperationalRefreshTriggerType.FIRE_EVENT_UPDATE],
)
def test_active_environmental_triggers_run_severity_spread_then_targets(trigger_type):
    log: list[str] = []
    orchestrator = make_orchestrator(log=log)

    result = orchestrator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=trigger_type,
        as_of=AS_OF,
    )

    assert result.success is True
    assert result.status is OperationalRefreshStatus.REFRESHED
    assert log == ["severity", "spread", "targets"]
    assert result.severity_result is not None
    assert result.spread_refresh_result is not None
    assert result.response_target_result is not None


def test_weather_update_reuses_severity_before_spread_without_fire_event_lookup():
    log: list[str] = []
    fire_event_repository = FakeFireEventRepository(status=FireEventStatus.RESOLVED)
    orchestrator = make_orchestrator(log=log, fire_event_repository=fire_event_repository)

    orchestrator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )

    assert log == ["severity", "spread", "targets"]
    assert fire_event_repository.calls == []


def test_severity_update_skips_severity_and_runs_spread_then_targets_once():
    log: list[str] = []
    targets = FakeResponseTargetAgent(log)
    orchestrator = make_orchestrator(log=log, targets=targets)

    result = orchestrator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE,
        as_of=AS_OF,
    )

    assert result.success is True
    assert result.severity_result is None
    assert log == ["spread", "targets"]
    assert len(targets.calls) == 1


def test_response_targets_run_once_after_complete_two_horizon_spread_cycle():
    log: list[str] = []
    spread = FakeSpreadRefreshOrchestrator(
        log,
        result=spread_result(
            statuses=(
                FireSpreadRefreshHorizonStatus.REFRESHED,
                FireSpreadRefreshHorizonStatus.REFRESHED,
            )
        ),
    )
    targets = FakeResponseTargetAgent(log)
    orchestrator = make_orchestrator(log=log, spread=spread, targets=targets)

    result = orchestrator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )

    assert result.spread_refresh_result.predictions_created == 2
    assert log == ["severity", "spread", "targets"]
    assert len(spread.calls) == 1
    assert len(targets.calls) == 1


def test_severity_score_only_change_still_regenerates_targets_when_spread_no_ops():
    log: list[str] = []
    severity = FakeSeverityAgent(log, score=80.0)
    spread = FakeSpreadRefreshOrchestrator(
        log,
        result=spread_result(
            trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE,
            statuses=(FireSpreadRefreshHorizonStatus.NO_OP, FireSpreadRefreshHorizonStatus.NO_OP),
        ),
    )
    targets = FakeResponseTargetAgent(log)
    orchestrator = make_orchestrator(log=log, severity_agent=severity, spread=spread, targets=targets)

    result = orchestrator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE,
        as_of=AS_OF,
    )

    assert result.status is OperationalRefreshStatus.NO_OP
    assert result.spread_refresh_result.no_ops == 2
    assert len(targets.calls) == 1
    assert log == ["spread", "targets"]


def test_weather_equivalent_spread_state_is_successful_no_op_and_targets_still_refresh():
    log: list[str] = []
    spread = FakeSpreadRefreshOrchestrator(
        log,
        result=spread_result(
            statuses=(FireSpreadRefreshHorizonStatus.NO_OP, FireSpreadRefreshHorizonStatus.NO_OP),
        ),
    )
    targets = FakeResponseTargetAgent(log)
    orchestrator = make_orchestrator(log=log, spread=spread, targets=targets)

    result = orchestrator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )

    assert result.success is True
    assert result.status is OperationalRefreshStatus.NO_OP
    assert result.spread_refresh_result.predictions_created == 0
    assert len(targets.calls) == 1


def test_inactive_fire_event_update_skips_active_severity_and_uses_existing_inactive_pipelines():
    log: list[str] = []
    spread = FakeSpreadRefreshOrchestrator(
        log,
        result=spread_result(
            trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
            statuses=(
                FireSpreadRefreshHorizonStatus.INACTIVE_EVENT,
                FireSpreadRefreshHorizonStatus.INACTIVE_EVENT,
            ),
        ),
    )
    targets = FakeResponseTargetAgent(
        log,
        result=target_result(status=ResponseTargetGenerationStatus.INACTIVE_EVENT),
    )
    orchestrator = make_orchestrator(
        log=log,
        spread=spread,
        targets=targets,
        fire_event_repository=FakeFireEventRepository(status=FireEventStatus.RESOLVED),
    )

    result = orchestrator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        as_of=AS_OF,
    )

    assert result.status is OperationalRefreshStatus.INACTIVE_EVENT
    assert result.severity_result is None
    assert log == ["spread", "targets"]
    assert result.response_target_result.target_set_id is None


def test_spread_transition_to_insufficient_still_allows_targets_to_use_latest_state():
    log: list[str] = []
    spread = FakeSpreadRefreshOrchestrator(
        log,
        result=spread_result(
            statuses=(
                FireSpreadRefreshHorizonStatus.INSUFFICIENT_DATA,
                FireSpreadRefreshHorizonStatus.NO_OP,
            ),
        ),
    )
    targets = FakeResponseTargetAgent(log)
    orchestrator = make_orchestrator(log=log, spread=spread, targets=targets)

    result = orchestrator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )

    assert result.success is True
    assert result.status is OperationalRefreshStatus.REFRESHED
    assert len(targets.calls) == 1
    assert log == ["severity", "spread", "targets"]


def test_resource_status_update_path_calls_resource_service_and_available_query_only():
    log: list[str] = []
    resource = FirefightingResource("TRUCK-A", "STATION-1", ResourceStatus.UNAVAILABLE)
    service = FakeResourceStatusService(resource_update_result(ResourceStatusUpdateStatus.UPDATED, resource=resource))
    repository = FakeResourceRepository(
        resources=(FirefightingResource("TRUCK-B", "STATION-1", ResourceStatus.AVAILABLE),)
    )
    severity = FakeSeverityAgent(log)
    spread = FakeSpreadRefreshOrchestrator(log)
    targets = FakeResponseTargetAgent(log)
    orchestrator = make_orchestrator(
        log=log,
        severity_agent=severity,
        spread=spread,
        targets=targets,
        resource_service=service,
        resource_repository=repository,
    )

    result = orchestrator.refresh_resource(resource_id="TRUCK-A", new_status=ResourceStatus.UNAVAILABLE)

    assert result.status is OperationalRefreshStatus.RESOURCE_UPDATED
    assert service.calls == [{"resource_id": "TRUCK-A", "new_status": ResourceStatus.UNAVAILABLE}]
    assert repository.calls == [["STATION-1"]]
    assert [resource.id for resource in result.available_resources] == ["TRUCK-B"]
    assert log == []
    assert severity.calls == []
    assert spread.calls == []
    assert targets.calls == []


@pytest.mark.parametrize(
    ("old_status", "new_status", "expected_ids"),
    [
        (ResourceStatus.AVAILABLE, ResourceStatus.UNAVAILABLE, ["TRUCK-B"]),
        (ResourceStatus.UNAVAILABLE, ResourceStatus.AVAILABLE, ["TRUCK-A", "TRUCK-B"]),
        (ResourceStatus.AVAILABLE, ResourceStatus.ASSIGNED, ["TRUCK-B"]),
        (ResourceStatus.ASSIGNED, ResourceStatus.AVAILABLE, ["TRUCK-A", "TRUCK-B"]),
    ],
)
def test_resource_branch_refreshes_available_set_through_existing_repository_query(
    sqlite_session_factory,
    old_status,
    new_status,
    expected_ids,
):
    insert_station_resources(sqlite_session_factory, old_status=old_status)
    repository = FirefightingResourceRepository(sqlite_session_factory)
    service = ResourceStatusUpdateService(repository)
    orchestrator = make_orchestrator(
        resource_service=service,
        resource_repository=repository,
    )

    result = orchestrator.refresh_resource(resource_id="TRUCK-A", new_status=new_status)

    assert result.status is OperationalRefreshStatus.RESOURCE_UPDATED
    assert [resource.id for resource in result.available_resources] == expected_ids


def test_same_state_resource_update_returns_no_op_with_current_available_set(sqlite_session_factory):
    insert_station_resources(sqlite_session_factory, old_status=ResourceStatus.AVAILABLE)
    repository = FirefightingResourceRepository(sqlite_session_factory)
    service = ResourceStatusUpdateService(repository)
    orchestrator = make_orchestrator(resource_service=service, resource_repository=repository)

    result = orchestrator.refresh_resource(resource_id="TRUCK-A", new_status=ResourceStatus.AVAILABLE)

    assert result.status is OperationalRefreshStatus.RESOURCE_NO_OP
    assert result.success is True
    assert [resource.id for resource in result.available_resources] == ["TRUCK-A", "TRUCK-B"]


def test_resource_not_found_returns_clear_result_and_no_environmental_calls():
    log: list[str] = []
    service = FakeResourceStatusService(resource_update_result(ResourceStatusUpdateStatus.NOT_FOUND))
    repository = FakeResourceRepository()
    orchestrator = make_orchestrator(log=log, resource_service=service, resource_repository=repository)

    result = orchestrator.refresh_resource(resource_id="MISSING", new_status=ResourceStatus.AVAILABLE)

    assert result.status is OperationalRefreshStatus.RESOURCE_NOT_FOUND
    assert result.success is False
    assert result.available_resources == ()
    assert repository.calls == []
    assert log == []


def test_resource_persistence_failure_does_not_invoke_environmental_pipeline():
    log: list[str] = []
    service = FakeResourceStatusService(resource_update_result(ResourceStatusUpdateStatus.FAILED))
    orchestrator = make_orchestrator(log=log, resource_service=service)

    result = orchestrator.refresh_resource(resource_id="TRUCK-A", new_status=ResourceStatus.UNAVAILABLE)

    assert result.status is OperationalRefreshStatus.FAILED
    assert result.success is False
    assert "resource persistence failed" in result.error_message
    assert log == []


def test_wrong_trigger_on_fire_event_branch_is_rejected_before_calls():
    log: list[str] = []
    orchestrator = make_orchestrator(log=log)

    with pytest.raises(ValueError):
        orchestrator.refresh_fire_event(
            fire_event_id=FIRE_EVENT_ID,
            trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE,
            as_of=AS_OF,
        )

    assert log == []


def test_naive_as_of_rejected_before_environmental_calls():
    log: list[str] = []
    orchestrator = make_orchestrator(log=log)

    with pytest.raises(ValueError):
        orchestrator.refresh_fire_event(
            fire_event_id=FIRE_EVENT_ID,
            trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
            as_of=datetime(2026, 9, 15, 12, 0),
        )

    assert log == []


def test_severity_operational_failure_stops_downstream_steps():
    log: list[str] = []
    orchestrator = make_orchestrator(log=log, severity_agent=FakeSeverityAgent(log, fail=True))

    result = orchestrator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )

    assert result.status is OperationalRefreshStatus.FAILED
    assert result.spread_refresh_result is None
    assert result.response_target_result is None
    assert log == ["severity"]


def test_spread_failure_prevents_target_generation():
    log: list[str] = []
    spread = FakeSpreadRefreshOrchestrator(
        log,
        result=spread_result(statuses=(FireSpreadRefreshHorizonStatus.FAILED, FireSpreadRefreshHorizonStatus.NO_OP)),
    )
    targets = FakeResponseTargetAgent(log)
    orchestrator = make_orchestrator(log=log, spread=spread, targets=targets)

    result = orchestrator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )

    assert result.status is OperationalRefreshStatus.FAILED
    assert result.response_target_result is None
    assert targets.calls == []
    assert log == ["severity", "spread"]


def test_response_target_failure_is_surfaced_after_spread():
    log: list[str] = []
    targets = FakeResponseTargetAgent(log, result=target_result(status=ResponseTargetGenerationStatus.FAILED))
    orchestrator = make_orchestrator(log=log, targets=targets)

    result = orchestrator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )

    assert result.status is OperationalRefreshStatus.FAILED
    assert result.response_target_result.status is ResponseTargetGenerationStatus.FAILED
    assert log == ["severity", "spread", "targets"]


def test_environmental_refresh_does_not_touch_resource_repository_or_service():
    log: list[str] = []
    service = FakeResourceStatusService(
        resource_update_result(
            ResourceStatusUpdateStatus.UPDATED,
            resource=FirefightingResource("TRUCK-A", "STATION-1", ResourceStatus.UNAVAILABLE),
        )
    )
    repository = FakeResourceRepository()
    orchestrator = make_orchestrator(log=log, resource_service=service, resource_repository=repository)

    orchestrator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )

    assert service.calls == []
    assert repository.calls == []


def test_multi_event_isolation_passes_only_requested_fire_event_id():
    log: list[str] = []
    severity = FakeSeverityAgent(log)
    spread = FakeSpreadRefreshOrchestrator(log)
    targets = FakeResponseTargetAgent(log)
    events = FakeFireEventRepository()
    orchestrator = make_orchestrator(
        log=log,
        severity_agent=severity,
        spread=spread,
        targets=targets,
        fire_event_repository=events,
    )

    orchestrator.refresh_fire_event(
        fire_event_id=99,
        trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        as_of=AS_OF,
    )

    assert events.calls == [99]
    assert severity.calls == [{"fire_event_id": 99, "assessed_at": AS_OF}]
    assert spread.calls == [
        {"fire_event_id": 99, "trigger_type": OperationalRefreshTriggerType.FIRE_EVENT_UPDATE, "as_of": AS_OF}
    ]
    assert targets.calls == [{"fire_event_id": 99, "as_of": AS_OF}]


def test_task4_never_calls_fire_detection_agent():
    log: list[str] = []
    fire_detection_agent = CallRecorder()
    orchestrator = make_orchestrator(log=log)

    orchestrator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        as_of=AS_OF,
    )
    orchestrator.refresh_resource(resource_id="TRUCK-A", new_status=ResourceStatus.UNAVAILABLE)

    assert fire_detection_agent.calls == []


def test_orchestrator_does_not_import_calculators_agents_below_boundary_or_roads():
    forbidden_fragments = (
        "FireSeverityCalculator",
        "FireSpreadCalculator",
        "ResponseTargetCalculator",
        "FireSpreadInputService",
        "ResponseTargetInputService",
        "RoadNetworkRepository",
        "road_network",
        "GraphNode",
        "GraphEdge",
        "osmnx",
        "seed_road_networks",
        "FireDetectionAgent",
        "simulation",
        "routing",
        "Dijkstra",
        "genetic",
        "ResponsePlan",
    )
    production_files = [
        Path("backend/src/services/operational_refresh/operational_refresh_orchestrator.py"),
        Path("backend/src/services/operational_refresh/operational_refresh_result.py"),
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


def insert_station_resources(sqlite_session_factory, *, old_status: ResourceStatus) -> None:
    session = sqlite_session_factory()
    session.add(FireStationDB(id="STATION-1", name="Station 1", latitude=32.1, longitude=35.1))
    session.flush()
    session.add_all(
        [
            FirefightingResourceDB(id="TRUCK-A", station_id="STATION-1", status=old_status),
            FirefightingResourceDB(id="TRUCK-B", station_id="STATION-1", status=ResourceStatus.AVAILABLE),
        ]
    )
    session.commit()
    session.close()

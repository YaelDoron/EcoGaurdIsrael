"""Tests for PlanningEffectiveStateBuilder, using fakes for every dependency (no real DB)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.calculators.response_optimization.response_optimization_config import (
    METHODOLOGY as OPTIMIZATION_METHODOLOGY_NAME,
    METHODOLOGY_VERSION as OPTIMIZATION_METHODOLOGY_VERSION,
)
from src.calculators.response_target.response_target_config import (
    RESPONSE_TARGET_METHODOLOGY_NAME,
    RESPONSE_TARGET_METHODOLOGY_VERSION,
)
from src.calculators.routing.routing_config import ROUTING_METHODOLOGY_NAME, ROUTING_METHODOLOGY_VERSION
from src.models import (
    PlanningEffectiveStateResult,
    PlanningEffectiveStateStatus,
    PlanningResourceState,
    PlanningTargetState,
    ResponseTarget,
    ResponseTargetSet,
    ResponseTargetType,
)
from src.repositories.response_target_repository import StoredResponseTarget, StoredResponseTargetSet
from src.services.response_planning import PlanningEffectiveStateBuilder

AS_OF = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
FIRE_EVENT_ID = 42
OTHER_FIRE_EVENT_ID = 99

ACTIVE_TARGET = ResponseTarget(
    fire_event_id=FIRE_EVENT_ID,
    target_type=ResponseTargetType.ACTIVE_FIRE,
    latitude=32.731,
    longitude=35.046,
    priority_score=150.0,
)
PREDICTED_TARGET = ResponseTarget(
    fire_event_id=FIRE_EVENT_ID,
    target_type=ResponseTargetType.PREDICTED_RISK,
    latitude=32.75,
    longitude=35.07,
    priority_score=80.0,
    prediction_horizon_minutes=30,
    spread_prediction_id=10,
    spread_prediction_cell_id=100,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeResponseTargetRepository:
    def __init__(self, stored_set: StoredResponseTargetSet | None, exc: Exception | None = None) -> None:
        self.stored_set = stored_set
        self.exc = exc
        self.calls = []

    def get_latest_for_event_as_of(self, fire_event_id, as_of):
        self.calls.append({"fire_event_id": fire_event_id, "as_of": as_of})
        if self.exc is not None:
            raise self.exc
        return self.stored_set


class FakeOperationalContextService:
    def __init__(self, stations=None, resources=None, exc: Exception | None = None) -> None:
        self.stations = stations if stations is not None else [make_station()]
        self.resources = resources if resources is not None else [make_resource()]
        self.exc = exc
        self.calls = []

    def get_available_operational_context(
        self,
        latitude,
        longitude,
        min_resources: int = 1,
        excluded_fire_event_id: int | None = None,
    ):
        self.calls.append(
            {
                "latitude": latitude,
                "longitude": longitude,
                "min_resources": min_resources,
                "excluded_fire_event_id": excluded_fire_event_id,
            }
        )
        if self.exc is not None:
            raise self.exc
        return self.stations, self.resources


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_target_set(targets=(ACTIVE_TARGET,), set_id: int = 501, fire_event_id: int = FIRE_EVENT_ID):
    domain_set = ResponseTargetSet(
        fire_event_id=fire_event_id,
        generated_at=AS_OF - timedelta(minutes=5),
        methodology=RESPONSE_TARGET_METHODOLOGY_NAME,
        methodology_version=RESPONSE_TARGET_METHODOLOGY_VERSION,
        targets=targets,
    )
    stored_targets = tuple(
        StoredResponseTarget(id=900 + index, target_order=index, target=target)
        for index, target in enumerate(targets)
    )
    return StoredResponseTargetSet(id=set_id, target_set=domain_set, targets=stored_targets)


def make_station(station_id="station-1", latitude=32.700, longitude=35.000):
    return SimpleNamespace(id=station_id, latitude=latitude, longitude=longitude)


def make_resource(resource_id="truck-1", station_id="station-1"):
    return SimpleNamespace(id=resource_id, station_id=station_id)


def make_builder(response_target_repository=None, operational_context_service=None) -> PlanningEffectiveStateBuilder:
    return PlanningEffectiveStateBuilder(
        response_target_repository=response_target_repository or FakeResponseTargetRepository(make_target_set()),
        operational_context_service=operational_context_service or FakeOperationalContextService(),
    )


# ---------------------------------------------------------------------------
# 1-2: Successful build from a valid target set and available resources
# ---------------------------------------------------------------------------


def test_builds_state_from_valid_target_set_and_available_resources():
    result = make_builder().build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert isinstance(result, PlanningEffectiveStateResult)
    assert result.status is PlanningEffectiveStateStatus.BUILT
    assert result.state.fire_event_id == FIRE_EVENT_ID
    assert len(result.state.targets) == 1
    assert len(result.state.resources) == 1


def test_persisted_target_fields_are_mapped_correctly():
    result = make_builder(
        response_target_repository=FakeResponseTargetRepository(make_target_set((ACTIVE_TARGET, PREDICTED_TARGET)))
    ).build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    predicted = next(t for t in result.state.targets if t.target_type is ResponseTargetType.PREDICTED_RISK)
    assert predicted.latitude == PREDICTED_TARGET.latitude
    assert predicted.longitude == PREDICTED_TARGET.longitude
    assert predicted.priority_score == PREDICTED_TARGET.priority_score
    assert predicted.prediction_horizon_minutes == PREDICTED_TARGET.prediction_horizon_minutes

    active = next(t for t in result.state.targets if t.target_type is ResponseTargetType.ACTIVE_FIRE)
    assert active.prediction_horizon_minutes is None


def test_methodology_fields_come_from_existing_constants():
    result = make_builder().build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.state.routing_methodology == ROUTING_METHODOLOGY_NAME
    assert result.state.routing_methodology_version == ROUTING_METHODOLOGY_VERSION
    assert result.state.optimization_methodology == OPTIMIZATION_METHODOLOGY_NAME
    assert result.state.optimization_methodology_version == OPTIMIZATION_METHODOLOGY_VERSION


# ---------------------------------------------------------------------------
# 3: Provenance/persistence IDs are not copied
# ---------------------------------------------------------------------------


def test_provenance_ids_are_not_copied_into_planning_target_state():
    predicted_target_set = make_target_set((ACTIVE_TARGET, PREDICTED_TARGET))
    result = make_builder(response_target_repository=FakeResponseTargetRepository(predicted_target_set)).build(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
    )

    for target in result.state.targets:
        assert not hasattr(target, "response_target_id")
        assert not hasattr(target, "response_target_set_id")
        assert not hasattr(target, "spread_prediction_id")
        assert not hasattr(target, "spread_prediction_cell_id")


# ---------------------------------------------------------------------------
# 4: Resource id / FireStation origin mapping
# ---------------------------------------------------------------------------


def test_resource_and_station_origin_are_mapped_correctly():
    station = make_station("station-9", latitude=32.8, longitude=35.2)
    resource = make_resource("truck-9", "station-9")

    result = make_builder(
        operational_context_service=FakeOperationalContextService(stations=[station], resources=[resource])
    ).build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.state.resources == (
        PlanningResourceState(
            resource_id="truck-9",
            station_id="station-9",
            station_latitude=32.8,
            station_longitude=35.2,
        ),
    )


# ---------------------------------------------------------------------------
# 5-7: Availability selection is reused, not redefined
# ---------------------------------------------------------------------------


def test_only_resources_returned_by_operational_context_service_are_included():
    """ASSIGNED/UNAVAILABLE filtering happens inside OperationalContextService (reused), not here."""
    operational_context_service = FakeOperationalContextService(
        stations=[make_station()],
        resources=[make_resource("truck-available")],
    )

    result = make_builder(operational_context_service=operational_context_service).build(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
    )

    assert [r.resource_id for r in result.state.resources] == ["truck-available"]


def test_no_available_resources_produces_empty_resource_tuple():
    operational_context_service = FakeOperationalContextService(stations=[], resources=[])

    result = make_builder(operational_context_service=operational_context_service).build(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
    )

    assert result.status is PlanningEffectiveStateStatus.BUILT
    assert result.state is not None
    assert result.state.resources == ()


def test_available_operational_context_is_queried_with_active_fire_target_coordinates():
    operational_context_service = FakeOperationalContextService()
    target_set = make_target_set((ACTIVE_TARGET, PREDICTED_TARGET))

    make_builder(
        response_target_repository=FakeResponseTargetRepository(target_set),
        operational_context_service=operational_context_service,
    ).build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert operational_context_service.calls == [
        {
            "latitude": ACTIVE_TARGET.latitude,
            "longitude": ACTIVE_TARGET.longitude,
            "min_resources": 1,
            "excluded_fire_event_id": FIRE_EVENT_ID,
        }
    ]


# ---------------------------------------------------------------------------
# 8-9: Deterministic ordering
# ---------------------------------------------------------------------------


def test_different_repository_order_produces_equal_state():
    stations = [make_station("station-1"), make_station("station-2", latitude=32.71, longitude=35.02)]
    resources_a = [make_resource("truck-1", "station-1"), make_resource("truck-2", "station-2")]
    resources_b = [make_resource("truck-2", "station-2"), make_resource("truck-1", "station-1")]
    target_set_a = make_target_set((ACTIVE_TARGET, PREDICTED_TARGET))
    target_set_b = make_target_set((PREDICTED_TARGET, ACTIVE_TARGET), set_id=777)

    result_a = make_builder(
        response_target_repository=FakeResponseTargetRepository(target_set_a),
        operational_context_service=FakeOperationalContextService(stations=stations, resources=resources_a),
    ).build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)
    result_b = make_builder(
        response_target_repository=FakeResponseTargetRepository(target_set_b),
        operational_context_service=FakeOperationalContextService(stations=stations, resources=resources_b),
    ).build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result_a.state == result_b.state


def test_different_target_set_ids_with_identical_semantic_targets_produce_equal_state():
    first = make_builder(
        response_target_repository=FakeResponseTargetRepository(make_target_set(set_id=501))
    ).build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)
    second = make_builder(
        response_target_repository=FakeResponseTargetRepository(make_target_set(set_id=999))
    ).build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert first.state.targets == second.state.targets


# ---------------------------------------------------------------------------
# 10-13: Semantic changes change the state
# ---------------------------------------------------------------------------


def test_changed_priority_score_produces_different_state():
    baseline = make_builder().build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)
    changed_target = ResponseTarget(
        fire_event_id=FIRE_EVENT_ID,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        latitude=ACTIVE_TARGET.latitude,
        longitude=ACTIVE_TARGET.longitude,
        priority_score=999.0,
    )
    changed = make_builder(
        response_target_repository=FakeResponseTargetRepository(make_target_set((changed_target,)))
    ).build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert changed.state != baseline.state


def test_changed_target_coordinate_produces_different_state():
    baseline = make_builder().build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)
    changed_target = ResponseTarget(
        fire_event_id=FIRE_EVENT_ID,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        latitude=33.0,
        longitude=ACTIVE_TARGET.longitude,
        priority_score=ACTIVE_TARGET.priority_score,
    )
    changed = make_builder(
        response_target_repository=FakeResponseTargetRepository(make_target_set((changed_target,)))
    ).build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert changed.state != baseline.state


def test_adding_available_resource_produces_different_state():
    baseline = make_builder(
        operational_context_service=FakeOperationalContextService(resources=[make_resource("truck-1")])
    ).build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)
    changed = make_builder(
        operational_context_service=FakeOperationalContextService(
            stations=[make_station("station-1"), make_station("station-2", latitude=32.8, longitude=35.1)],
            resources=[make_resource("truck-1"), make_resource("truck-2", "station-2")],
        )
    ).build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert changed.state != baseline.state


def test_changed_resource_station_origin_produces_different_state():
    baseline = make_builder(
        operational_context_service=FakeOperationalContextService(
            stations=[make_station("station-1", latitude=32.7, longitude=35.0)],
            resources=[make_resource("truck-1", "station-1")],
        )
    ).build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)
    changed = make_builder(
        operational_context_service=FakeOperationalContextService(
            stations=[make_station("station-1", latitude=33.5, longitude=36.0)],
            resources=[make_resource("truck-1", "station-1")],
        )
    ).build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert changed.state != baseline.state


# ---------------------------------------------------------------------------
# 14: FireEvent isolation
# ---------------------------------------------------------------------------


def test_fire_event_a_does_not_consume_fire_event_b_target_set():
    response_target_repository = FakeResponseTargetRepository(make_target_set(fire_event_id=FIRE_EVENT_ID))

    with pytest.raises(ValueError):
        make_builder(response_target_repository=response_target_repository).build(
            fire_event_id=OTHER_FIRE_EVENT_ID, as_of=AS_OF
        )

    assert response_target_repository.calls == [{"fire_event_id": OTHER_FIRE_EVENT_ID, "as_of": AS_OF}]


def test_repository_is_queried_with_requested_fire_event_id():
    response_target_repository = FakeResponseTargetRepository(make_target_set())

    make_builder(response_target_repository=response_target_repository).build(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
    )

    assert response_target_repository.calls == [{"fire_event_id": FIRE_EVENT_ID, "as_of": AS_OF}]


# ---------------------------------------------------------------------------
# 15: Missing current target state is explicit, not fabricated
# ---------------------------------------------------------------------------


def test_no_current_target_set_returns_explicit_status_without_fabricating_state():
    response_target_repository = FakeResponseTargetRepository(stored_set=None)
    operational_context_service = FakeOperationalContextService()

    result = make_builder(
        response_target_repository=response_target_repository,
        operational_context_service=operational_context_service,
    ).build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is PlanningEffectiveStateStatus.NO_CURRENT_TARGETS
    assert result.state is None
    assert operational_context_service.calls == []


def test_resource_referencing_unknown_station_is_rejected_explicitly():
    operational_context_service = FakeOperationalContextService(
        stations=[make_station("station-1")],
        resources=[make_resource("truck-1", "station-ghost")],
    )

    with pytest.raises(ValueError):
        make_builder(operational_context_service=operational_context_service).build(
            fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
        )


# ---------------------------------------------------------------------------
# 16-17: Scope guard - no analysis/routing/optimization/persistence side effects
# ---------------------------------------------------------------------------


def test_only_the_two_expected_dependencies_are_consulted():
    response_target_repository = FakeResponseTargetRepository(make_target_set())
    operational_context_service = FakeOperationalContextService()

    make_builder(
        response_target_repository=response_target_repository,
        operational_context_service=operational_context_service,
    ).build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert len(response_target_repository.calls) == 1
    assert len(operational_context_service.calls) == 1


def test_result_has_no_routing_optimization_or_baseline_attributes():
    """PlanningEffectiveState.fingerprint (Task 3) is expected; routing/optimization/baseline outputs are not."""
    result = make_builder().build(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    for forbidden in ("route", "routes", "optimization_result", "baseline"):
        assert not hasattr(result, forbidden)
        assert not hasattr(result.state, forbidden)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_fire_event_id", [0, -1, True, "42"])
def test_invalid_fire_event_id_rejected(invalid_fire_event_id):
    with pytest.raises(ValueError):
        make_builder().build(fire_event_id=invalid_fire_event_id, as_of=AS_OF)


def test_naive_as_of_rejected():
    with pytest.raises(ValueError):
        make_builder().build(fire_event_id=FIRE_EVENT_ID, as_of=datetime(2026, 9, 16, 12, 0))


def test_repository_exception_propagates():
    response_target_repository = FakeResponseTargetRepository(None, exc=RuntimeError("db exploded"))

    with pytest.raises(RuntimeError):
        make_builder(response_target_repository=response_target_repository).build(
            fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
        )


def test_operational_context_exception_propagates():
    operational_context_service = FakeOperationalContextService(exc=RuntimeError("context blew up"))

    with pytest.raises(RuntimeError):
        make_builder(operational_context_service=operational_context_service).build(
            fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
        )

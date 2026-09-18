"""Tests for OperationalContextService's Stage 0 cross-event resource exclusion
(Global Multi-Incident Optimizer refactor - see the architecture audit).

Only exercises get_available_resources() directly - the lowest-level point
where "status == AVAILABLE" (already applied inside
FirefightingResourceRepository.get_available_resources) and "not reserved by
another active event's current plan" are composed (Task 4/12). Fakes stand
in for FirefightingResourceRepository and CrossEventReservedResourceResolver;
FireStationRepository/RoadNetworkRepository/RoadNetworkFetcher are left at
their real process-wide defaults since nothing here calls anything that
touches them.
"""
from __future__ import annotations

from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.models.resource_status import ResourceStatus
from src.services.operational.operational_context_service import OperationalContextService

STATION = FireStationDB(id="STATION-1", name="Station 1", latitude=32.7, longitude=35.0)


def make_resource(resource_id: str) -> FirefightingResourceDB:
    return FirefightingResourceDB(id=resource_id, station_id=STATION.id, status=ResourceStatus.AVAILABLE)


class FakeFirefightingResourceRepository:
    def __init__(self, resources):
        self.resources = list(resources)
        self.calls: list[tuple[str, ...]] = []

    def get_available_resources(self, station_ids):
        self.calls.append(tuple(station_ids))
        return list(self.resources)


class FakeCrossEventReservedResourceResolver:
    def __init__(self, reserved_ids: frozenset[str] = frozenset()):
        self.reserved_ids = reserved_ids
        self.calls: list[int] = []

    def get_resource_ids_reserved_by_other_active_plans(self, *, excluded_fire_event_id: int) -> frozenset[str]:
        self.calls.append(excluded_fire_event_id)
        return self.reserved_ids


def make_service(resources, reserved_ids: frozenset[str] = frozenset()):
    firefighting_resource_repository = FakeFirefightingResourceRepository(resources)
    resolver = FakeCrossEventReservedResourceResolver(reserved_ids)
    service = OperationalContextService(
        firefighting_resource_repository=firefighting_resource_repository,
        cross_event_reserved_resource_resolver=resolver,
    )
    return service, firefighting_resource_repository, resolver


def test_no_excluded_fire_event_id_preserves_prior_behavior_exactly():
    """Task 17 regression: omitting excluded_fire_event_id (the default,
    None) must not consult the resolver at all, and must return every
    AVAILABLE resource unchanged - the exact single-event prior behavior."""
    resources = [make_resource("TRUCK-A"), make_resource("TRUCK-B")]
    service, _, resolver = make_service(resources)

    found = service.get_available_resources([STATION])

    assert [resource.id for resource in found] == ["TRUCK-A", "TRUCK-B"]
    assert resolver.calls == []


def test_excludes_resources_reserved_by_other_active_events_current_plans():
    resources = [make_resource("TRUCK-A"), make_resource("TRUCK-B")]
    service, _, _ = make_service(resources, reserved_ids=frozenset({"TRUCK-A"}))

    found = service.get_available_resources([STATION], excluded_fire_event_id=99)

    assert [resource.id for resource in found] == ["TRUCK-B"]


def test_existing_available_status_filter_is_narrowed_further_not_replaced():
    """Task 12: the new exclusion is additional. AVAILABLE filtering already
    happened one layer down (the fake here stands in for
    FirefightingResourceRepository.get_available_resources, which never
    returns non-AVAILABLE resources) - a resource that was never in that
    AVAILABLE candidate pool stays absent regardless of reservation state."""
    resources = [make_resource("TRUCK-A")]
    service, _, _ = make_service(resources, reserved_ids=frozenset())

    found = service.get_available_resources([STATION], excluded_fire_event_id=99)

    assert [resource.id for resource in found] == ["TRUCK-A"]


def test_resolver_is_queried_with_the_excluded_fire_event_id():
    resources = [make_resource("TRUCK-A")]
    service, _, resolver = make_service(resources)

    service.get_available_resources([STATION], excluded_fire_event_id=7)

    assert resolver.calls == [7]


def test_empty_reserved_set_returns_all_available_resources_unchanged():
    resources = [make_resource("TRUCK-A"), make_resource("TRUCK-B")]
    service, _, _ = make_service(resources, reserved_ids=frozenset())

    found = service.get_available_resources([STATION], excluded_fire_event_id=1)

    assert [resource.id for resource in found] == ["TRUCK-A", "TRUCK-B"]


def test_no_stations_returns_empty_without_querying_resolver():
    service, firefighting_resource_repository, resolver = make_service([])

    found = service.get_available_resources([], excluded_fire_event_id=1)

    assert found == []
    assert firefighting_resource_repository.calls == []
    assert resolver.calls == []

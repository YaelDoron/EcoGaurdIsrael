"""Tests for FirefightingResourceRepository persistence behavior."""
from __future__ import annotations

import pytest

from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.models.resource_status import ResourceStatus
from src.repositories.exceptions import FirefightingResourceRepositoryError
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository


def insert_station_with_resources(sqlite_session_factory):
    session = sqlite_session_factory()
    session.add_all(
        [
            FireStationDB(id="S-1", name="Station 1", latitude=32.1, longitude=35.1),
            FireStationDB(id="S-2", name="Station 2", latitude=32.2, longitude=35.2),
        ]
    )
    session.flush()
    session.add_all(
        [
            FirefightingResourceDB(id="TRUCK-A", station_id="S-1", status=ResourceStatus.AVAILABLE),
            FirefightingResourceDB(id="TRUCK-B", station_id="S-1", status=ResourceStatus.AVAILABLE),
            FirefightingResourceDB(id="TRUCK-C", station_id="S-1", status=ResourceStatus.UNAVAILABLE),
            FirefightingResourceDB(id="TRUCK-D", station_id="S-2", status=ResourceStatus.ASSIGNED),
        ]
    )
    session.commit()
    session.close()


def available_ids(repository: FirefightingResourceRepository, station_ids: list[str]) -> list[str]:
    return [resource.id for resource in repository.get_available_resources(station_ids)]


def get_db_resource(sqlite_session_factory, resource_id: str) -> FirefightingResourceDB:
    session = sqlite_session_factory()
    resource = session.get(FirefightingResourceDB, resource_id)
    session.close()
    return resource


def test_get_available_resources_returns_only_available_resources_in_stable_order(sqlite_session_factory):
    insert_station_with_resources(sqlite_session_factory)
    repository = FirefightingResourceRepository(sqlite_session_factory)

    assert available_ids(repository, ["S-1", "S-2"]) == ["TRUCK-A", "TRUCK-B"]


def test_update_status_persists_existing_row_and_preserves_station(sqlite_session_factory):
    insert_station_with_resources(sqlite_session_factory)
    repository = FirefightingResourceRepository(sqlite_session_factory)

    updated = repository.update_status("TRUCK-A", ResourceStatus.UNAVAILABLE)

    assert updated.id == "TRUCK-A"
    assert updated.station_id == "S-1"
    assert updated.status is ResourceStatus.UNAVAILABLE
    persisted = get_db_resource(sqlite_session_factory, "TRUCK-A")
    assert persisted.id == "TRUCK-A"
    assert persisted.station_id == "S-1"
    assert persisted.status is ResourceStatus.UNAVAILABLE


def test_get_by_id_returns_domain_resource(sqlite_session_factory):
    insert_station_with_resources(sqlite_session_factory)
    repository = FirefightingResourceRepository(sqlite_session_factory)

    resource = repository.get_by_id("TRUCK-A")

    assert resource.id == "TRUCK-A"
    assert resource.station_id == "S-1"
    assert resource.status is ResourceStatus.AVAILABLE


def test_update_status_returns_none_for_missing_resource(sqlite_session_factory):
    insert_station_with_resources(sqlite_session_factory)
    repository = FirefightingResourceRepository(sqlite_session_factory)

    assert repository.update_status("NO-SUCH-TRUCK", ResourceStatus.AVAILABLE) is None


@pytest.mark.parametrize("bad_status", ["available", None, object()])
def test_update_status_rejects_invalid_status(sqlite_session_factory, bad_status):
    insert_station_with_resources(sqlite_session_factory)
    repository = FirefightingResourceRepository(sqlite_session_factory)

    with pytest.raises(FirefightingResourceRepositoryError):
        repository.update_status("TRUCK-A", bad_status)


def test_available_resource_query_reflects_status_changes(sqlite_session_factory):
    insert_station_with_resources(sqlite_session_factory)
    repository = FirefightingResourceRepository(sqlite_session_factory)

    assert available_ids(repository, ["S-1"]) == ["TRUCK-A", "TRUCK-B"]

    repository.update_status("TRUCK-A", ResourceStatus.UNAVAILABLE)
    assert available_ids(repository, ["S-1"]) == ["TRUCK-B"]

    repository.update_status("TRUCK-A", ResourceStatus.AVAILABLE)
    assert available_ids(repository, ["S-1"]) == ["TRUCK-A", "TRUCK-B"]

    repository.update_status("TRUCK-A", ResourceStatus.ASSIGNED)
    assert available_ids(repository, ["S-1"]) == ["TRUCK-B"]

    repository.update_status("TRUCK-A", ResourceStatus.AVAILABLE)
    assert available_ids(repository, ["S-1"]) == ["TRUCK-A", "TRUCK-B"]


def test_multiple_resource_updates_mutate_individual_resource_state(sqlite_session_factory):
    insert_station_with_resources(sqlite_session_factory)
    repository = FirefightingResourceRepository(sqlite_session_factory)

    repository.update_status("TRUCK-A", ResourceStatus.ASSIGNED)
    repository.update_status("TRUCK-C", ResourceStatus.AVAILABLE)

    assert available_ids(repository, ["S-1"]) == ["TRUCK-B", "TRUCK-C"]

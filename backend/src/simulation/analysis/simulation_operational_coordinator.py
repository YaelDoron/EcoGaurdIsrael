"""Simulate operational load by depleting firefighting resource availability.

SimulationOperationalCoordinator exists for simulation/demo scenarios only:
it lets a scenario recreate a "high load" incident by taking a fraction of
the real trucks near an incident out of service, so the real
OperationalContextService (and, downstream, Epic 5 routing) sees genuinely
depleted resources rather than mocked data.
"""
from __future__ import annotations

import random

from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.models.resource_status import ResourceStatus
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.services.operational.operational_context_service import OperationalContextService

# Depleted resources are split roughly evenly between UNAVAILABLE (out of
# service) and ASSIGNED (already dispatched elsewhere) to simulate a
# realistic mix of reasons a truck might be unavailable.
_DEPLETED_STATUSES = (ResourceStatus.UNAVAILABLE, ResourceStatus.ASSIGNED)


class SimulationOperationalCoordinator:
    """Randomly deplete firefighting resource availability near an incident."""

    def __init__(
        self,
        operational_context_service: OperationalContextService | None = None,
        firefighting_resource_repository: FirefightingResourceRepository | None = None,
        random_generator: random.Random | None = None,
    ) -> None:
        self._operational_context_service = operational_context_service or OperationalContextService()
        self._firefighting_resource_repository = (
            firefighting_resource_repository or FirefightingResourceRepository()
        )
        self._random = random_generator or random.Random()

    def scramble_resource_availability(
        self,
        incident_latitude: float,
        incident_longitude: float,
        availability_ratio: float = 0.9,
    ) -> list[FirefightingResourceDB]:
        """Randomly deplete a fraction of the trucks stationed near an incident.

        Finds the fire stations relevant to (incident_latitude,
        incident_longitude) via OperationalContextService, collects every
        firefighting resource attached to them (regardless of current
        status), then randomly keeps `availability_ratio` of them AVAILABLE
        and marks the rest UNAVAILABLE or ASSIGNED (chosen at random per
        depleted resource).

        Changes are committed to the database immediately, so subsequent
        reads (e.g. OperationalContextService.get_available_resources) see
        the depleted fleet. Returns the resources that were changed.
        """
        self._validate_availability_ratio(availability_ratio)

        stations = self._operational_context_service.get_stations_in_operational_area(
            incident_latitude, incident_longitude
        )
        if not stations:
            return []

        station_ids = [station.id for station in stations]
        resources = self._firefighting_resource_repository.get_resources_for_stations(station_ids)
        if not resources:
            return []

        resource_ids = [resource.id for resource in resources]
        self._random.shuffle(resource_ids)

        kept_available_count = round(len(resource_ids) * availability_ratio)
        depleted_ids = resource_ids[kept_available_count:]
        if not depleted_ids:
            return []

        new_status_by_id: dict[str, ResourceStatus] = {
            resource_id: self._random.choice(_DEPLETED_STATUSES) for resource_id in depleted_ids
        }

        ids_by_status: dict[ResourceStatus, list[str]] = {status: [] for status in _DEPLETED_STATUSES}
        for resource_id, status in new_status_by_id.items():
            ids_by_status[status].append(resource_id)
        for status, ids in ids_by_status.items():
            self._firefighting_resource_repository.set_statuses(ids, status)

        # The bulk UPDATE above ran in its own session, so these detached
        # objects (from the earlier read) still hold their pre-update
        # status in memory; reflect the new status explicitly before
        # returning them to the caller.
        depleted_resources = [resource for resource in resources if resource.id in new_status_by_id]
        for resource in depleted_resources:
            resource.status = new_status_by_id[resource.id]
        return depleted_resources

    def select_available_resource(
        self,
        incident_latitude: float,
        incident_longitude: float,
    ) -> FirefightingResourceDB | None:
        """Return the first AVAILABLE nearby resource in stable resource-id order."""
        stations = self._operational_context_service.get_stations_in_operational_area(
            incident_latitude,
            incident_longitude,
        )
        if not stations:
            return None

        station_ids = [station.id for station in stations]
        resources = self._firefighting_resource_repository.get_available_resources(station_ids)
        if not resources:
            return None
        return sorted(resources, key=lambda resource: str(resource.id))[0]

    @staticmethod
    def _validate_availability_ratio(availability_ratio: float) -> None:
        if (
            isinstance(availability_ratio, bool)
            or not isinstance(availability_ratio, (int, float))
            or not 0.0 <= availability_ratio <= 1.0
        ):
            raise ValueError(
                f"availability_ratio must be a number within [0, 1], got {availability_ratio!r}."
            )

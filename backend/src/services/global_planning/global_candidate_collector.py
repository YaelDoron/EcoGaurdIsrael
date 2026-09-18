"""Global candidate resource collection (Stage 3 of the Global
Multi-Incident Optimizer refactor, Tasks 4/7/8/9; demand-aware expansion
since Stage 5, Tasks 11-12).

Builds ONE global resource universe from ALL active FireEvents together,
instead of one independent candidate pool per event: each physical
resource appears exactly once, however many incidents it happens to be
geographically near.

Policy (Task 7):
  1. for every active FireEvent's anchor coordinate, reuse
     OperationalContextService.get_stations_in_operational_area unchanged
     (its own progressive 5 -> 20 -> 50 km widening + closest-station
     fallback already implements "nearby local resources first, farther
     stations still discoverable");
  2. union those stations across all events, by station id;
  3. optionally top up with next-nearest stations overall if the union is
     still smaller than `config.max_fallback_stations`;
  4. union all resources (ANY operational status - Task 4's core point:
     stop conflating AVAILABLE with globally unowned) at those stations,
     deduplicated by resource_id;
  5. attach current ResourceCommitment ownership (Task 9), read-only.

Stage 5 demand-aware reinforcement (Tasks 11-12): after the above, if the
caller passes `desired_assignable_supply` (the sum of every active
incident's desired_resources) and the assignable resource count so far is
smaller than that, next-nearest not-yet-selected stations are added ONE AT
A TIME (by minimum distance to any anchor), re-checking assignable supply
after each addition, until supply meets demand, no more eligible stations
remain, or `config.max_demand_driven_stations` is reached. Never fabricates
resources - only makes genuinely farther, real stations discoverable so
the Global GA can choose to use them.
"""
from __future__ import annotations

from collections.abc import Iterable

from src.database.models.fire_station_db import FireStationDB
from src.models.global_planning_resource import GlobalPlanningResource
from src.models.resource_status import ResourceStatus
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.services.global_planning.global_candidate_resource_config import GlobalCandidateResourceConfig
from src.services.operational.operational_context_service import OperationalContextService
from src.utils.geo import haversine_distance_km


class GlobalCandidateCollector:
    """Builds the global (event-agnostic) candidate resource universe for one GlobalPlanningRun."""

    def __init__(
        self,
        fire_station_repository: FireStationRepository | None = None,
        firefighting_resource_repository: FirefightingResourceRepository | None = None,
        resource_commitment_repository: ResourceCommitmentRepository | None = None,
        operational_context_service: OperationalContextService | None = None,
        config: GlobalCandidateResourceConfig | None = None,
    ) -> None:
        self._fire_station_repository = fire_station_repository or FireStationRepository()
        self._firefighting_resource_repository = (
            firefighting_resource_repository or FirefightingResourceRepository()
        )
        self._resource_commitment_repository = resource_commitment_repository or ResourceCommitmentRepository()
        self._operational_context_service = operational_context_service or OperationalContextService()
        self._config = config or GlobalCandidateResourceConfig()

    def collect(
        self,
        anchors: Iterable[tuple[int, float, float]],
        desired_assignable_supply: int | None = None,
        required_resource_ids: Iterable[str] | None = None,
    ) -> tuple[GlobalPlanningResource, ...]:
        """anchors: (fire_event_id, latitude, longitude) for every active event with a usable target set.

        `desired_assignable_supply` (Stage 5, Task 11): the sum of every
        active incident's desired_resources - when given, triggers
        progressive farther-station reinforcement if the geographic union
        alone cannot supply it.

        `required_resource_ids` (Stage 6): resource ids that MUST appear in
        the result regardless of geography - a hard-dispatched resource's
        own station is pulled in even if genuinely far outside every normal
        search radius, so its existing lock can still be represented (Task
        14). Never fabricates a resource: if a required id genuinely no
        longer exists, it is silently absent here and the Global GA's own
        problem-builder surfaces that honestly (GlobalHardDispatchLockInfeasible
        or, if it already went UNAVAILABLE, simple exclusion from supply).
        """
        anchor_list = list(anchors)
        stations_by_id: dict[str, FireStationDB] = {}
        for _fire_event_id, latitude, longitude in anchor_list:
            for station in self._operational_context_service.get_stations_in_operational_area(
                latitude, longitude, radius_km=self._config.search_radii_km[0]
            ):
                stations_by_id[station.id] = station

        if self._config.max_fallback_stations is not None and len(stations_by_id) < self._config.max_fallback_stations:
            self._top_up_with_nearest_stations(
                stations_by_id, anchor_list, target_station_count=self._config.max_fallback_stations
            )

        self._ensure_required_resource_stations(stations_by_id, required_resource_ids)

        resources = self._collect_resources(stations_by_id)

        if desired_assignable_supply is not None:
            resources = self._expand_for_demand(stations_by_id, anchor_list, resources, desired_assignable_supply)

        commitments = self._resource_commitment_repository.get_for_resource_ids(
            [resource.id for resource in resources]
        )
        commitment_by_resource_id = {commitment.resource_id: commitment for commitment in commitments}

        return tuple(
            self._to_global_resource(resource, stations_by_id[resource.station_id], commitment_by_resource_id)
            for resource in resources
        )

    def _collect_resources(self, stations_by_id: dict[str, FireStationDB]) -> list:
        return self._firefighting_resource_repository.get_resources_for_stations(sorted(stations_by_id))

    def _ensure_required_resource_stations(
        self, stations_by_id: dict[str, FireStationDB], required_resource_ids: Iterable[str] | None
    ) -> None:
        if not required_resource_ids:
            return
        missing_station_ids: set[str] = set()
        for resource_id in required_resource_ids:
            resource = self._firefighting_resource_repository.get_by_id(resource_id)
            if resource is None:
                continue
            station_id = str(resource.station_id)
            if station_id not in stations_by_id:
                missing_station_ids.add(station_id)
        if not missing_station_ids:
            return
        for station in self._fire_station_repository.get_all_stations():
            if station.id in missing_station_ids:
                stations_by_id[station.id] = station

    def _expand_for_demand(
        self,
        stations_by_id: dict[str, FireStationDB],
        anchors: list[tuple[int, float, float]],
        resources: list,
        desired_assignable_supply: int,
    ) -> list:
        if not anchors or desired_assignable_supply <= 0:
            return resources

        def _assignable_count(candidate_resources: list) -> int:
            return sum(1 for resource in candidate_resources if resource.status is not ResourceStatus.UNAVAILABLE)

        stations_added = 0
        max_additional = self._config.max_demand_driven_stations
        while _assignable_count(resources) < desired_assignable_supply:
            if max_additional is not None and stations_added >= max_additional:
                break
            remaining = self._next_nearest_stations(stations_by_id, anchors)
            if not remaining:
                break
            next_station = remaining[0]
            stations_by_id[next_station.id] = next_station
            stations_added += 1
            resources = self._collect_resources(stations_by_id)
        return resources

    def _next_nearest_stations(
        self,
        stations_by_id: dict[str, FireStationDB],
        anchors: list[tuple[int, float, float]],
    ) -> list[FireStationDB]:
        remaining = [
            station for station in self._fire_station_repository.get_all_stations() if station.id not in stations_by_id
        ]
        if not remaining:
            return []
        remaining.sort(key=lambda station: self._min_distance_to_any_anchor(station, anchors))
        return remaining

    def _top_up_with_nearest_stations(
        self,
        stations_by_id: dict[str, FireStationDB],
        anchors: list[tuple[int, float, float]],
        target_station_count: int,
    ) -> None:
        if not anchors:
            return
        remaining = self._next_nearest_stations(stations_by_id, anchors)
        if not remaining:
            return
        needed = target_station_count - len(stations_by_id)
        for station in remaining[:needed]:
            stations_by_id[station.id] = station

    @staticmethod
    def _min_distance_to_any_anchor(station: FireStationDB, anchors: list[tuple[int, float, float]]) -> float:
        return min(
            haversine_distance_km(latitude, longitude, station.latitude, station.longitude)
            for _fire_event_id, latitude, longitude in anchors
        )

    @staticmethod
    def _to_global_resource(resource, station: FireStationDB, commitment_by_resource_id) -> GlobalPlanningResource:
        commitment = commitment_by_resource_id.get(resource.id)
        return GlobalPlanningResource(
            resource_id=resource.id,
            station_id=station.id,
            station_name=station.name,
            station_latitude=station.latitude,
            station_longitude=station.longitude,
            operational_status=resource.status,
            current_commitment_fire_event_id=commitment.fire_event_id if commitment is not None else None,
            current_commitment_response_plan_id=commitment.response_plan_id if commitment is not None else None,
        )

"""Orchestrates Epic 5 route planning: targets -> operational context -> node mapping -> shortest paths -> persistence.

RoutePlanningAgent is pure orchestration. It does not itself compute
distances/times (DijkstraCalculator), decide which node a coordinate maps to
(NodeMappingService), resolve stations/resources/road network
(OperationalContextService), or perform persistence (ResponseTargetRepository,
RoutePlanningRepository) - it only sequences those calls and converts between
their shapes.
"""
from __future__ import annotations

from datetime import datetime
import logging

from sqlalchemy.orm import Session, sessionmaker

from src.agents.routing.route_planning_result import RoutePlanningResult, RoutePlanningStatus
from src.calculators.routing.dijkstra_calculator import DijkstraCalculator
from src.calculators.routing.haversine_fallback_calculator import (
    MIN_DISTINCT_DISTANCE_METERS,
    estimate_disconnected_graph_distance_and_eta,
    estimate_off_road_distance_and_eta,
    is_degenerate_zero_result,
)
from src.calculators.routing.routing_config import ROUTING_METHODOLOGY_NAME, ROUTING_METHODOLOGY_VERSION
from src.database.connection import get_session_factory
from src.models.graph_edge import GraphEdge
from src.models.response_target_type import ResponseTargetType
from src.models.routing import RoutePlanningRun, RouteResult, RouteStatus, RoutingResource, RoutingTarget
from src.repositories.response_target_repository import (
    ResponseTargetRepository,
    StoredResponseTarget,
    StoredResponseTargetSet,
)
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.operational.operational_context_service import OperationalContext, OperationalContextService
from src.services.routing.node_mapping_service import NodeMappingService
from src.utils.geo import haversine_distance_km

logger = logging.getLogger(__name__)


class RoutePlanningAgent:
    """Coordinates response-target retrieval, operational context, routing, and persistence."""

    def __init__(
        self,
        response_target_repository: ResponseTargetRepository,
        operational_context_service: OperationalContextService,
        node_mapping_service: NodeMappingService,
        dijkstra_calculator: DijkstraCalculator,
        route_planning_repository: RoutePlanningRepository,
        session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        self._response_target_repository = response_target_repository
        self._operational_context_service = operational_context_service
        self._node_mapping_service = node_mapping_service
        self._dijkstra_calculator = dijkstra_calculator
        self._route_planning_repository = route_planning_repository
        self._session_factory = session_factory or get_session_factory()

    def plan(self, *, fire_event_id: int, as_of: datetime) -> RoutePlanningResult:
        """Plan and persist routes from every available resource to every response target."""
        self._validate_request(fire_event_id, as_of)

        try:
            stored_target_set = self._response_target_repository.get_latest_for_event_as_of(fire_event_id, as_of)
            if stored_target_set is None:
                return RoutePlanningResult(
                    success=True,
                    fire_event_id=fire_event_id,
                    status=RoutePlanningStatus.NO_TARGETS,
                    run_id=None,
                    run=None,
                    route_count=0,
                    error_message=None,
                )

            fire_target = _find_active_fire_target(stored_target_set)
            routing_targets = _to_routing_targets(stored_target_set)

            context = self._build_operational_context(
                fire_target.target.latitude, fire_target.target.longitude, fire_event_id
            )
            routing_resources = _to_routing_resources(context)

            if not routing_resources:
                stored_run = self._route_planning_repository.save_run(
                    RoutePlanningRun(
                        fire_event_id=fire_event_id,
                        response_target_set_id=stored_target_set.id,
                        planned_at=as_of,
                        methodology=ROUTING_METHODOLOGY_NAME,
                        methodology_version=ROUTING_METHODOLOGY_VERSION,
                        resource_ids=(),
                        routes=(),
                    )
                )
                logger.info(
                    "Stored empty routing run %s for FireEvent %s: no available resources.",
                    stored_run.id,
                    fire_event_id,
                )
                return RoutePlanningResult(
                    success=True,
                    fire_event_id=fire_event_id,
                    status=RoutePlanningStatus.NO_RESOURCES_AVAILABLE,
                    run_id=stored_run.id,
                    run=stored_run,
                    route_count=0,
                    error_message=None,
                )

            resource_node_map = self._node_mapping_service.map_resources(
                routing_resources, context.road_nodes, context.road_edges
            )
            target_node_map = self._node_mapping_service.map_targets(
                routing_targets, context.road_nodes, context.road_edges
            )

            routes = tuple(
                self._compute_route(resource, target, resource_node_map, target_node_map, context.road_edges)
                for resource in routing_resources
                for target in routing_targets
            )

            stored_run = self._route_planning_repository.save_run(
                RoutePlanningRun(
                    fire_event_id=fire_event_id,
                    response_target_set_id=stored_target_set.id,
                    planned_at=as_of,
                    methodology=ROUTING_METHODOLOGY_NAME,
                    methodology_version=ROUTING_METHODOLOGY_VERSION,
                    resource_ids=tuple(resource.resource_id for resource in routing_resources),
                    routes=routes,
                )
            )
            logger.info(
                "Stored routing run %s for FireEvent %s with %s route result(s)",
                stored_run.id,
                fire_event_id,
                len(stored_run.routes),
            )
            return RoutePlanningResult(
                success=True,
                fire_event_id=fire_event_id,
                status=RoutePlanningStatus.PLANNED,
                run_id=stored_run.id,
                run=stored_run,
                route_count=len(stored_run.routes),
                error_message=None,
            )
        except Exception:
            logger.exception("Route planning failed for FireEvent %s", fire_event_id)
            return RoutePlanningResult(
                success=False,
                fire_event_id=fire_event_id,
                status=RoutePlanningStatus.FAILED,
                run_id=None,
                run=None,
                route_count=0,
                error_message="Route planning failed.",
            )

    def _build_operational_context(
        self, fire_latitude: float, fire_longitude: float, fire_event_id: int
    ) -> OperationalContext:
        """Build this fire's operational context, excluding resources reserved by
        another active FireEvent's current plan (Stage 0 - see
        OperationalContextService's module docstring)."""
        session = self._session_factory()
        try:
            return self._operational_context_service.build_context(
                session, fire_latitude, fire_longitude, excluded_fire_event_id=fire_event_id
            )
        finally:
            session.close()

    def _compute_route(
        self,
        resource: RoutingResource,
        target: RoutingTarget,
        resource_node_map: dict[str, int | None],
        target_node_map: dict[int, int | None],
        edges: list[GraphEdge],
    ) -> RouteResult:
        source_node_id = resource_node_map.get(resource.resource_id)
        target_node_id = target_node_map.get(target.response_target_id)

        if source_node_id is None or target_node_id is None:
            return RouteResult(
                resource_id=resource.resource_id,
                response_target_id=target.response_target_id,
                status=RouteStatus.UNMAPPABLE,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                node_path=(),
                distance_meters=None,
                travel_time_seconds=None,
            )

        dijkstra_result = self._dijkstra_calculator.calculate_shortest_path(source_node_id, target_node_id, edges)
        if dijkstra_result.status is RouteStatus.UNREACHABLE:
            # Disconnected graph: no road path between two mapped nodes. A
            # straight-line estimate (a two-node path) keeps the pair
            # assignable instead of leaving the fire with no response actions.
            estimate = estimate_disconnected_graph_distance_and_eta(
                resource.latitude, resource.longitude, target.latitude, target.longitude
            )
            if estimate is not None:
                distance_meters, travel_time_seconds = estimate
                logger.warning(
                    "Route %s -> target %s: no road path in the fetched graph (disconnected components); "
                    "using a straight-line estimate of %.2f km.",
                    resource.resource_id,
                    target.response_target_id,
                    distance_meters / 1000.0,
                )
                return RouteResult(
                    resource_id=resource.resource_id,
                    response_target_id=target.response_target_id,
                    status=RouteStatus.REACHABLE,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    node_path=(source_node_id, target_node_id),
                    distance_meters=distance_meters,
                    travel_time_seconds=travel_time_seconds,
                )
            # Too far apart for a straight line to be a plausible stand-in:
            # stays UNREACHABLE, exactly as before the fallback existed.
            return RouteResult(
                resource_id=resource.resource_id,
                response_target_id=target.response_target_id,
                status=RouteStatus.UNREACHABLE,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                node_path=(),
                distance_meters=None,
                travel_time_seconds=None,
            )

        distance_meters, travel_time_seconds = self._resolve_reachable_distance_and_eta(
            resource, target, dijkstra_result.distance_meters, dijkstra_result.travel_time_seconds
        )

        return RouteResult(
            resource_id=resource.resource_id,
            response_target_id=target.response_target_id,
            status=RouteStatus.REACHABLE,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            node_path=dijkstra_result.node_path,
            distance_meters=distance_meters,
            travel_time_seconds=travel_time_seconds,
        )

    @staticmethod
    def _resolve_reachable_distance_and_eta(
        resource: RoutingResource,
        target: RoutingTarget,
        distance_meters: float | None,
        travel_time_seconds: float | None,
    ) -> tuple[float | None, float | None]:
        """Substitute a straight-line (Haversine) off-road estimate for a
        REACHABLE result that graph-Dijkstra reports as exactly 0m/0s (see
        haversine_fallback_calculator's module docstring: this happens when
        the resource's and target's nearest mapped graph nodes coincide,
        which a sparse/coarse fetched road-network graph can produce even
        when the two real-world coordinates are far apart). Left completely
        untouched otherwise - including a genuine 0m/0s where the two
        coordinates really are the same point.
        """
        if distance_meters is None or travel_time_seconds is None:
            return distance_meters, travel_time_seconds
        if not is_degenerate_zero_result(distance_meters, travel_time_seconds):
            return distance_meters, travel_time_seconds

        real_distance_km = haversine_distance_km(
            resource.latitude, resource.longitude, target.latitude, target.longitude
        )
        if real_distance_km * 1000.0 <= MIN_DISTINCT_DISTANCE_METERS:
            return distance_meters, travel_time_seconds

        logger.warning(
            "Route %s -> target %s: graph Dijkstra reported a degenerate 0m/0s result "
            "(source/target nodes coincide) while the real coordinates are %.2f km apart - "
            "substituting a straight-line off-road estimate.",
            resource.resource_id,
            target.response_target_id,
            real_distance_km,
        )
        return estimate_off_road_distance_and_eta(
            resource.latitude, resource.longitude, target.latitude, target.longitude
        )

    @staticmethod
    def _validate_request(fire_event_id: int, as_of: datetime) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {fire_event_id!r}")
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")


def _to_routing_resources(context: OperationalContext) -> tuple[RoutingResource, ...]:
    stations_by_id = {station.id: station for station in context.stations}
    resources = []
    for resource in context.available_resources:
        station = stations_by_id.get(resource.station_id)
        if station is None:
            raise ValueError(
                f"Firefighting resource {resource.id!r} references station {resource.station_id!r}, "
                "which is not part of this operational context."
            )
        resources.append(
            RoutingResource(
                resource_id=resource.id,
                station_id=resource.station_id,
                latitude=station.latitude,
                longitude=station.longitude,
            )
        )
    return tuple(resources)


def _to_routing_targets(stored_target_set: StoredResponseTargetSet) -> tuple[RoutingTarget, ...]:
    return tuple(
        RoutingTarget(
            response_target_id=stored_target.id,
            target_order=stored_target.target_order,
            target_type=stored_target.target.target_type.value,
            latitude=stored_target.target.latitude,
            longitude=stored_target.target.longitude,
            priority_score=stored_target.target.priority_score,
        )
        for stored_target in stored_target_set.targets
    )


def _find_active_fire_target(stored_target_set: StoredResponseTargetSet) -> StoredResponseTarget:
    for stored_target in stored_target_set.targets:
        if stored_target.target.target_type is ResponseTargetType.ACTIVE_FIRE:
            return stored_target
    raise ValueError(
        f"ResponseTargetSet {stored_target_set.id!r} has no ACTIVE_FIRE target; this should be impossible."
    )

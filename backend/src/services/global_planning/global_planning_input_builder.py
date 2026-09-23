"""GlobalPlanningInputBuilder: the single read/orchestration service that
assembles GlobalPlanningInput (Stage 3 of the Global Multi-Incident
Optimizer refactor, Task 19).

No DB writes, no GA, no resource activation - purely a read pipeline over
already-persisted planning state:

    GlobalPlanningRun membership (Stage 2, read-only - Task 20)
        -> each member's current ResponseTargetSet (Task 5/6)
        -> global candidate resources (GlobalCandidateCollector, Task 7-9)
        -> road network covering every anchor + candidate station
        -> cross-event route matrix (GlobalRouteMatrixBuilder, Task 10-16)
        -> validate + fingerprint (GlobalPlanningInput, Task 17/18)

Mid-build consistency (Task 21, hardened in Stage 3.1): resource
operational status/commitment can change while the (comparatively slow)
route matrix is being computed. Stage 3 does not need full snapshot
isolation, but the returned input must be self-consistent: after building
the matrix, the candidate resource state is re-read and compared against
what routes were computed against. A material difference triggers one
bounded rebuild (_MAX_REBUILD_ATTEMPTS); if resource state has STILL
changed after that rebuild, `build()` raises GlobalPlanningInputUnstable
rather than returning a known-stale input - the future Global GA (Stage 4)
must never be handed an input the builder already knows drifted during its
own construction. Never an unbounded retry loop.

Target-set snapshot consistency (Task 6) is a deliberately DIFFERENT
guarantee and is intentionally left out of this revalidation: each active
event's current ResponseTargetSet is selected exactly once, at the start
of build(), and never re-read or rebuilt afterward - "one selected current
target set per FireEvent, captured at input-build time" is the policy
itself (not a race condition to correct for). Resource state, by
contrast, is genuinely volatile at route-matrix-computation timescales
(activation elsewhere can commit/release/disable a candidate resource
mid-build) and is what this revalidation protects against.

Performance pass (demo-simulation profiling, Task: "Implement the first
performance-optimization pass"): two additions, both scoped to THIS
builder INSTANCE's lifetime only (never a module-level/process-global
cache) - safe because production wires exactly one GlobalPlanningInputBuilder
per GlobalPlanningRefreshCoordinator, itself constructed once per running
process/simulation (see global_planning_input_production_factory.py,
global_planning_refresh_production_factory.py):

1. Road-network subgraph reuse (`_road_network_cache`): `_load_road_network`
   caches its single most recent (bbox -> nodes/edges) result. GraphNodeDB/
   GraphEdgeDB are intentionally never deleted mid-simulation (see
   DemoStateResetService) and this codebase has no live external writer
   that inserts into an ALREADY-explored bbox while a builder instance is
   alive - so "same exact bbox as last time" is a safe, conservative
   reuse key: any bbox change (a moved/added/removed active incident or
   candidate resource station) is a cache miss and reloads from
   PostgreSQL/OSM exactly as before. This is a documented assumption, not
   a proven invariant enforced elsewhere - if a future caller starts
   mutating already-cached graph regions out of band, this cache must be
   revisited. Returned/stored node and edge objects are always copied
   (`model_copy()`) in both directions so a consumer mutating a returned
   list/object can never corrupt the cached copy or a later cache hit.

2. Cheap pre-routing signature (`compute_pre_routing_bundle`): everything
   `_compute_fingerprint` hashes EXCEPT route-matrix content, extracted
   into a small reusable payload (`_pre_routing_payload`) shared by both
   functions so the cheap signature can never omit something the real
   fingerprint covers. GlobalPlanningRefreshCoordinator uses this to
   decide, BEFORE calling build(), whether a cycle can skip road-network
   loading/routing/GA entirely - see that module's docstring for the full
   correctness argument (this builder itself does not decide NO_OP; it
   only computes and returns the signature).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import logging

from sqlalchemy.orm import Session, sessionmaker

from src.calculators.routing.routing_config import ROUTING_METHODOLOGY_NAME, ROUTING_METHODOLOGY_VERSION
from src.database.connection import get_session_factory
from src.models.current_global_assignment import CurrentGlobalAssignment
from src.models.global_incident_demand import GlobalIncidentDemand
from src.models.global_planning_input import GlobalPlanningInput
from src.models.global_planning_resource import GlobalPlanningResource
from src.models.global_planning_target import GlobalPlanningTarget
from src.models.graph_edge import GraphEdge
from src.models.graph_node import GraphNode
from src.models.response_target_type import ResponseTargetType
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.response_target_repository import ResponseTargetRepository, StoredResponseTargetSet
from src.repositories.road_network_repository import RoadNetworkRepository
from src.services.global_planning.current_global_assignment_loader import CurrentGlobalAssignmentLoader
from src.services.global_planning.global_candidate_collector import GlobalCandidateCollector
from src.services.global_planning.global_incident_demand_builder import GlobalIncidentDemandBuilder
from src.services.global_planning.global_planning_input_unstable import GlobalPlanningInputUnstable
from src.services.global_planning.global_route_matrix_builder import GlobalRouteMatrixBuilder
from src.services.operational.operational_context_service import BUFFER_DEGREES
from src.services.operational.road_network_fetcher import RoadNetworkFetcher

logger = logging.getLogger(__name__)

# Bounded mid-build revalidation (Task 21, Stage 3.1): 1 retry = 2 attempts
# total. Never unbounded - if resource state still disagrees after the
# retry, build() raises GlobalPlanningInputUnstable rather than looping
# further or returning a known-stale input.
_MAX_REBUILD_ATTEMPTS = 1

_GLOBAL_INPUT_METHODOLOGY = "global_candidate_route_matrix"
_GLOBAL_INPUT_METHODOLOGY_VERSION = "1.0"


def _target_signature(target: GlobalPlanningTarget) -> tuple:
    return (
        target.target_type.value,
        target.latitude,
        target.longitude,
        target.priority_score,
        target.prediction_horizon_minutes,
    )


@dataclass(frozen=True)
class _RoadNetworkCacheEntry:
    """One cached (bbox -> subgraph) result, scoped to one builder instance."""

    bbox: tuple[float, float, float, float]
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


@dataclass(frozen=True)
class GlobalPlanningPreRoutingBundle:
    """Every pre-routing planning input, read once, plus its cheap signature.

    Returned by `compute_pre_routing_bundle()` and optionally fed back into
    `build()` via `precomputed=` so a cycle that turns out NOT to be a
    cheap NO_OP never re-reads targets/demand/assignments/candidates a
    second time.
    """

    active_fire_event_ids: tuple[int, ...]
    targets: tuple[GlobalPlanningTarget, ...]
    event_target_set_ids: dict[int, int]
    anchors: tuple[tuple[int, float, float], ...]
    incident_demands: tuple[GlobalIncidentDemand, ...]
    desired_assignable_supply: int
    current_assignments: tuple[CurrentGlobalAssignment, ...]
    required_resource_ids: tuple[str, ...]
    resources: tuple[GlobalPlanningResource, ...]
    pre_routing_signature: str


class GlobalPlanningInputBuilder:
    """Assembles one validated, fingerprinted GlobalPlanningInput for an existing GlobalPlanningRun."""

    def __init__(
        self,
        global_planning_run_repository: GlobalPlanningRunRepository | None = None,
        response_target_repository: ResponseTargetRepository | None = None,
        candidate_collector: GlobalCandidateCollector | None = None,
        incident_demand_builder: GlobalIncidentDemandBuilder | None = None,
        current_global_assignment_loader: CurrentGlobalAssignmentLoader | None = None,
        route_matrix_builder: GlobalRouteMatrixBuilder | None = None,
        road_network_repository: RoadNetworkRepository | None = None,
        road_network_fetcher: RoadNetworkFetcher | None = None,
        session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        self._global_planning_run_repository = global_planning_run_repository or GlobalPlanningRunRepository()
        self._response_target_repository = response_target_repository or ResponseTargetRepository()
        self._candidate_collector = candidate_collector or GlobalCandidateCollector()
        self._incident_demand_builder = incident_demand_builder or GlobalIncidentDemandBuilder()
        self._current_global_assignment_loader = current_global_assignment_loader or CurrentGlobalAssignmentLoader()
        self._route_matrix_builder = route_matrix_builder or GlobalRouteMatrixBuilder()
        self._road_network_repository = road_network_repository or RoadNetworkRepository()
        self._road_network_fetcher = road_network_fetcher or RoadNetworkFetcher()
        self._session_factory = session_factory or get_session_factory()
        self._road_network_cache: _RoadNetworkCacheEntry | None = None

    def compute_pre_routing_bundle(
        self, *, global_planning_run_id: int, as_of: datetime
    ) -> GlobalPlanningPreRoutingBundle:
        """Read every pre-routing planning input once and return them plus
        their cheap signature - never touches the road network, route
        matrix, Dijkstra, or GA. Pure/stateless: this method does not
        itself decide NO_OP or remember anything between calls (that
        decision belongs to GlobalPlanningRefreshCoordinator, which knows
        the run's actual terminal outcome). Pass the returned bundle into
        `build()` via `precomputed=` to avoid re-reading these same inputs.
        """
        self._validate_request(global_planning_run_id, as_of)

        members = self._global_planning_run_repository.get_members(global_planning_run_id)
        active_fire_event_ids = tuple(sorted(member.member.fire_event_id for member in members))

        targets, event_target_set_ids, anchors = self._load_targets(active_fire_event_ids, as_of)
        incident_demands = self._incident_demand_builder.build(active_fire_event_ids, as_of)
        desired_assignable_supply = sum(demand.desired_resources for demand in incident_demands)
        # Captured ONCE, like the target-set snapshot above (Task 6's own
        # precedent) - not re-read in the mid-build revalidation loop below,
        # which already protects against operational-status/commitment
        # drift for the resources it does track.
        current_assignments = self._current_global_assignment_loader.load(active_fire_event_ids)
        current_assignments = self._remap_stale_assignment_targets(current_assignments, targets)
        required_resource_ids = tuple(assignment.resource_id for assignment in current_assignments)

        resources = self._candidate_collector.collect(anchors, desired_assignable_supply, required_resource_ids)

        pre_routing_signature = self._compute_pre_routing_signature(
            active_fire_event_ids, targets, resources, event_target_set_ids, incident_demands, current_assignments
        )

        return GlobalPlanningPreRoutingBundle(
            active_fire_event_ids=active_fire_event_ids,
            targets=targets,
            event_target_set_ids=event_target_set_ids,
            anchors=anchors,
            incident_demands=incident_demands,
            desired_assignable_supply=desired_assignable_supply,
            current_assignments=current_assignments,
            required_resource_ids=required_resource_ids,
            resources=resources,
            pre_routing_signature=pre_routing_signature,
        )

    def build(
        self,
        *,
        global_planning_run_id: int,
        as_of: datetime,
        precomputed: GlobalPlanningPreRoutingBundle | None = None,
    ) -> GlobalPlanningInput:
        """Assemble one validated, fingerprinted GlobalPlanningInput.

        `precomputed` (Optimization 3): when the caller already has a
        GlobalPlanningPreRoutingBundle from `compute_pre_routing_bundle()`
        (e.g. because it just used it for a NO_OP precheck that came back
        "changed"), pass it here to skip re-reading targets/demand/
        assignments/candidates - identical semantics either way, since
        `build()` with no `precomputed` computes the exact same bundle
        itself before proceeding.
        """
        self._validate_request(global_planning_run_id, as_of)

        bundle = precomputed or self.compute_pre_routing_bundle(
            global_planning_run_id=global_planning_run_id, as_of=as_of
        )
        active_fire_event_ids = bundle.active_fire_event_ids
        targets = bundle.targets
        event_target_set_ids = bundle.event_target_set_ids
        anchors = bundle.anchors
        incident_demands = bundle.incident_demands
        desired_assignable_supply = bundle.desired_assignable_supply
        current_assignments = bundle.current_assignments
        required_resource_ids = bundle.required_resource_ids
        resources = bundle.resources

        road_nodes, road_edges = self._load_road_network(anchors, resources)
        matrix_result = self._route_matrix_builder.build(resources, targets, road_nodes, road_edges)

        for attempt in range(_MAX_REBUILD_ATTEMPTS + 1):
            revalidated_resources = self._candidate_collector.collect(
                anchors, desired_assignable_supply, required_resource_ids
            )
            if self._same_material_state(resources, revalidated_resources):
                resources = revalidated_resources
                break

            if attempt == _MAX_REBUILD_ATTEMPTS:
                logger.warning(
                    "Resource state kept changing for GlobalPlanningRun %s across %s attempt(s); "
                    "refusing to return a known-stale GlobalPlanningInput.",
                    global_planning_run_id,
                    _MAX_REBUILD_ATTEMPTS + 1,
                )
                raise GlobalPlanningInputUnstable(
                    global_planning_run_id=global_planning_run_id, attempts=_MAX_REBUILD_ATTEMPTS + 1
                )

            logger.warning(
                "Resource state changed mid-build for GlobalPlanningRun %s (attempt %s/%s); rebuilding route matrix.",
                global_planning_run_id,
                attempt + 1,
                _MAX_REBUILD_ATTEMPTS + 1,
            )
            resources = revalidated_resources
            road_nodes, road_edges = self._load_road_network(anchors, resources)
            matrix_result = self._route_matrix_builder.build(resources, targets, road_nodes, road_edges)

        input_fingerprint = self._compute_fingerprint(
            active_fire_event_ids,
            targets,
            resources,
            event_target_set_ids,
            matrix_result.matrix,
            incident_demands,
            current_assignments,
        )

        return GlobalPlanningInput(
            global_planning_run_id=global_planning_run_id,
            as_of=as_of,
            active_fire_event_ids=active_fire_event_ids,
            targets=targets,
            resources=resources,
            route_matrix=matrix_result.matrix,
            event_target_set_ids=event_target_set_ids,
            incident_demands=incident_demands,
            current_assignments=current_assignments,
            input_fingerprint=input_fingerprint,
        )

    def _load_targets(
        self, active_fire_event_ids: tuple[int, ...], as_of: datetime
    ) -> tuple[tuple[GlobalPlanningTarget, ...], dict[int, int], tuple[tuple[int, float, float], ...]]:
        targets: list[GlobalPlanningTarget] = []
        event_target_set_ids: dict[int, int] = {}
        anchors: list[tuple[int, float, float]] = []

        for fire_event_id in active_fire_event_ids:
            stored_target_set = self._response_target_repository.get_latest_for_event_as_of(fire_event_id, as_of)
            if stored_target_set is None:
                logger.info("FireEvent %s has no usable current ResponseTargetSet; contributing zero targets.", fire_event_id)
                continue

            active_fire_target = self._find_active_fire_target(stored_target_set)
            if active_fire_target is None:
                logger.warning(
                    "FireEvent %s's current ResponseTargetSet %s has no ACTIVE_FIRE target; contributing zero targets.",
                    fire_event_id,
                    stored_target_set.id,
                )
                continue

            event_target_set_ids[fire_event_id] = stored_target_set.id
            anchors.append((fire_event_id, active_fire_target.latitude, active_fire_target.longitude))
            for stored_target in stored_target_set.targets:
                targets.append(
                    GlobalPlanningTarget(
                        fire_event_id=fire_event_id,
                        response_target_id=stored_target.id,
                        target_order=stored_target.target_order,
                        target_type=stored_target.target.target_type,
                        latitude=stored_target.target.latitude,
                        longitude=stored_target.target.longitude,
                        priority_score=stored_target.target.priority_score,
                        prediction_horizon_minutes=stored_target.target.prediction_horizon_minutes,
                    )
                )

        return tuple(targets), event_target_set_ids, tuple(anchors)

    def _load_road_network(self, anchors, resources: tuple[GlobalPlanningResource, ...]):
        """Return the road-network subgraph covering every anchor + candidate
        station, reusing the single most-recently-loaded subgraph when the
        exact same bbox is requested again (Optimization 2 - see this
        module's docstring for the reuse-safety assumption). Every
        node/edge returned - on both a cache miss (stored copy) and a hit
        (served copy) - is `model_copy()`-d, so nothing a caller does to
        the returned lists can ever corrupt the cached copy or a later hit.
        """
        latitudes = [latitude for _fire_event_id, latitude, _longitude in anchors] + [
            resource.station_latitude for resource in resources
        ]
        longitudes = [longitude for _fire_event_id, _latitude, longitude in anchors] + [
            resource.station_longitude for resource in resources
        ]
        if not latitudes:
            return [], []

        min_lat = max(-90.0, min(latitudes) - BUFFER_DEGREES)
        max_lat = min(90.0, max(latitudes) + BUFFER_DEGREES)
        min_lon = max(-180.0, min(longitudes) - BUFFER_DEGREES)
        max_lon = min(180.0, max(longitudes) + BUFFER_DEGREES)
        bbox = (min_lat, max_lat, min_lon, max_lon)

        cached = self._road_network_cache
        if cached is not None and cached.bbox == bbox:
            return (
                [node.model_copy() for node in cached.nodes],
                [edge.model_copy() for edge in cached.edges],
            )

        session = self._session_factory()
        try:
            road_nodes, road_edges = self._road_network_repository.get_network_in_bbox(
                session, min_lat, max_lat, min_lon, max_lon
            )
            if not road_nodes:
                road_nodes, road_edges = self._road_network_fetcher.fetch_network_in_bbox(
                    min_lat, max_lat, min_lon, max_lon
                )
                if road_nodes:
                    self._road_network_repository.save_network(session, road_nodes, road_edges)
            self._road_network_cache = _RoadNetworkCacheEntry(
                bbox=bbox,
                nodes=tuple(node.model_copy() for node in road_nodes),
                edges=tuple(edge.model_copy() for edge in road_edges),
            )
            return road_nodes, road_edges
        finally:
            session.close()

    @staticmethod
    def _same_material_state(
        before: tuple[GlobalPlanningResource, ...], after: tuple[GlobalPlanningResource, ...]
    ) -> bool:
        def _signature(resources):
            return frozenset(
                (r.resource_id, r.operational_status, r.current_commitment_fire_event_id) for r in resources
            )

        return _signature(before) == _signature(after)

    @staticmethod
    def _remap_stale_assignment_targets(
        current_assignments: tuple[CurrentGlobalAssignment, ...],
        targets: tuple[GlobalPlanningTarget, ...],
    ) -> tuple[CurrentGlobalAssignment, ...]:
        """A dispatched/planned resource's recorded response_target_id can go
        stale between cycles: severity/spread refresh regenerates a
        FireEvent's ResponseTargetSet (new target ids) even though the
        resource's real-world commitment to that FireEvent is unchanged.
        Remap any assignment whose response_target_id no longer exists in
        the freshly-loaded `targets` to that event's CANONICAL ACTIVE_FIRE
        target (same "first by target_order/id" selection
        GlobalAllocationSlotFactory itself uses) - the operational fact
        being represented is "this resource serves this fire's suppression
        effort," never the exact slot/target row id, which is bookkeeping
        that naturally refreshes alongside the target set. Never fabricates
        a target: if the event genuinely has no ACTIVE_FIRE target anymore,
        the assignment is left as-is and GlobalPlanningInput's own
        validation surfaces that honestly.
        """
        current_target_ids = {target.response_target_id for target in targets}
        canonical_active_fire_target_id: dict[int, int] = {}
        for target in sorted(targets, key=lambda t: (t.fire_event_id, t.target_order, t.response_target_id)):
            if target.target_type is ResponseTargetType.ACTIVE_FIRE:
                canonical_active_fire_target_id.setdefault(target.fire_event_id, target.response_target_id)

        remapped = []
        for assignment in current_assignments:
            if assignment.response_target_id in current_target_ids:
                remapped.append(assignment)
                continue
            replacement_target_id = canonical_active_fire_target_id.get(assignment.fire_event_id)
            if replacement_target_id is None:
                remapped.append(assignment)
                continue
            remapped.append(replace(assignment, response_target_id=replacement_target_id))
        return tuple(remapped)

    @staticmethod
    def _find_active_fire_target(stored_target_set: StoredResponseTargetSet):
        for stored_target in stored_target_set.targets:
            if stored_target.target.target_type is ResponseTargetType.ACTIVE_FIRE:
                return stored_target.target
        return None

    @staticmethod
    def _pre_routing_payload(
        active_fire_event_ids: tuple[int, ...],
        targets: tuple[GlobalPlanningTarget, ...],
        resources: tuple[GlobalPlanningResource, ...],
        event_target_set_ids: dict[int, int],
        incident_demands: tuple[GlobalIncidentDemand, ...],
        current_assignments: tuple[CurrentGlobalAssignment, ...],
    ) -> tuple:
        """Every deterministic, order-independent planning input that does
        NOT require routing to know - shared verbatim by `_compute_fingerprint`
        (which extends it with route content) and `_compute_pre_routing_signature`
        (Optimization 3's cheap NO_OP precheck), so the cheap signature can
        never omit something the real fingerprint covers: it is always
        exactly this payload, hashed alone instead of hashed-plus-routes.
        """
        sorted_targets = tuple(
            sorted(
                ((target.fire_event_id,) + _target_signature(target) for target in targets),
                key=lambda item: item,
            )
        )
        sorted_resources = tuple(
            sorted(
                (
                    (
                        resource.resource_id,
                        resource.station_id,
                        resource.station_latitude,
                        resource.station_longitude,
                        resource.operational_status.value,
                        resource.current_commitment_fire_event_id,
                        # current_commitment_response_plan_id deliberately EXCLUDED
                        # (Stage 6, Task 17): it is a DB surrogate that changes on
                        # every activation - even one that re-affirms the exact
                        # same resource/event/dispatch-state assignment - so
                        # including it would make genuine NO_OP detection
                        # permanently impossible after a resource's first
                        # activation. Ownership (which FireEvent) is what is
                        # semantically material; current_assignments already
                        # separately carries the exact target/dispatch_state.
                    )
                    for resource in resources
                ),
                key=lambda item: item[0],
            )
        )

        sorted_demands = tuple(
            sorted(
                (
                    (
                        demand.fire_event_id,
                        demand.severity_level.value if demand.severity_level is not None else None,
                        demand.severity_score,
                        demand.minimum_resources,
                        demand.desired_resources,
                        demand.demand_source.value,
                        demand.policy_methodology,
                        demand.policy_version,
                    )
                    for demand in incident_demands
                ),
                key=lambda item: item[0],
            )
        )

        sorted_current_assignments = tuple(
            sorted(
                (
                    (
                        assignment.resource_id,
                        assignment.fire_event_id,
                        # response_target_id deliberately EXCLUDED (Stage 6,
                        # Task 17, same lesson as current_commitment_response_plan_id
                        # above): ResponseTargetGenerationAgent mints a brand
                        # new ResponseTargetSet/ResponseTarget row on every
                        # environmental refresh, even when target CONTENT is
                        # identical (see `targets`' own content-based
                        # signature, which already deliberately excludes the
                        # raw response_target_id for the same reason).
                        # Including the raw id here would make genuine NO_OP
                        # detection fail on every cycle after severity/spread
                        # regenerates targets. Which FireEvent owns the
                        # resource, and its dispatch_state, are what is
                        # semantically material.
                        assignment.dispatch_state.value,
                    )
                    for assignment in current_assignments
                ),
                key=lambda item: item[0],
            )
        )

        return (
            ("active_fire_event_ids", tuple(sorted(active_fire_event_ids))),
            # event_target_set_ids deliberately reduced to just its KEY set
            # (which events have a usable target set at all), not the raw
            # target_set_id values - same DB-surrogate-churn reasoning as
            # current_assignments above; `targets`' own content signature
            # already captures anything semantically material about them.
            ("events_with_target_sets", tuple(sorted(event_target_set_ids.keys()))),
            ("targets", sorted_targets),
            ("resources", sorted_resources),
            ("incident_demands", sorted_demands),
            ("current_assignments", sorted_current_assignments),
            ("methodology", _GLOBAL_INPUT_METHODOLOGY),
            ("methodology_version", _GLOBAL_INPUT_METHODOLOGY_VERSION),
        )

    @staticmethod
    def _compute_pre_routing_signature(
        active_fire_event_ids: tuple[int, ...],
        targets: tuple[GlobalPlanningTarget, ...],
        resources: tuple[GlobalPlanningResource, ...],
        event_target_set_ids: dict[int, int],
        incident_demands: tuple[GlobalIncidentDemand, ...],
        current_assignments: tuple[CurrentGlobalAssignment, ...],
    ) -> str:
        """Cheap NO_OP precheck signature (Optimization 3).

        Hashes `_pre_routing_payload` alone - every input `_compute_fingerprint`
        hashes EXCEPT route-matrix content, which requires the expensive
        road-network load + Dijkstra to produce. This is safe as a NO_OP
        precheck specifically because target and resource-station
        coordinates - which this payload DOES cover in full - are exactly
        the values `_load_road_network` uses to compute its bbox: if they
        are byte-identical to the last successfully-planned cycle's, the
        requested bbox is byte-identical too, and (given this builder's
        documented road-network reuse-safety assumption - see the module
        docstring) the resulting graph, and therefore every route's ETA/
        distance/feasibility, would also be unchanged. Because this payload
        is always a strict subset of what `_compute_fingerprint` hashes,
        two inputs that differ here are GUARANTEED to differ in the real
        fingerprint too - this can only ever fail open to a full build,
        never produce a false NO_OP.
        """
        payload = GlobalPlanningInputBuilder._pre_routing_payload(
            active_fire_event_ids, targets, resources, event_target_set_ids, incident_demands, current_assignments
        )
        serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_fingerprint(
        active_fire_event_ids: tuple[int, ...],
        targets: tuple[GlobalPlanningTarget, ...],
        resources: tuple[GlobalPlanningResource, ...],
        event_target_set_ids: dict[int, int],
        route_matrix,
        incident_demands: tuple[GlobalIncidentDemand, ...],
        current_assignments: tuple[CurrentGlobalAssignment, ...],
    ) -> str:
        """Deterministic fingerprint of the actual optimizer input (Task 18).

        Includes: active FireEvent ids, target-set ids, target CONTENT
        (not raw response_target_id - a DB surrogate with no semantic
        meaning of its own, mirroring PlanningEffectiveState.fingerprint's
        existing precedent), resource ids/station origins/statuses/
        commitment ownership, per-route feasibility+ETA+distance,
        (Stage 5, Task 6) per-event demand CONTENT - severity level/score
        (or explicit fallback state), minimum/desired resource counts, and
        the policy methodology/version that produced them, and (Stage 6,
        Task 15) each resource's current_assignment CONTENT - fire_event_id,
        response_target_id, and dispatch_state. A severity change, a
        demand-policy version bump, OR a dispatch-state transition
        (PLANNED -> DISPATCHED flips a resource from a soft preference to a
        hard constraint - a materially different planning problem) therefore
        changes the fingerprint even when targets/resources/routes are
        unchanged. Deliberately excludes node_path (two routes with
        identical ETA/distance/feasibility are operationally interchangeable
        inputs to the future GA; only cost, not the exact path taken, should
        force a re-optimization) and random seed (Stage 4's own
        optimization config owns that separately). Order-independent.

        Everything except `routes`/`routing_methodology*` is delegated to
        `_pre_routing_payload`, shared with `_compute_pre_routing_signature` -
        see that method for why this makes the cheap precheck safe.
        """
        target_by_id = {target.response_target_id: target for target in targets}

        sorted_routes = tuple(
            sorted(
                (
                    (option.resource_id, option.fire_event_id)
                    + _target_signature(target_by_id[option.response_target_id])
                    + (option.eta_seconds, option.route_distance_meters)
                    for option in route_matrix
                ),
                key=lambda item: item,
            )
        )

        payload = GlobalPlanningInputBuilder._pre_routing_payload(
            active_fire_event_ids, targets, resources, event_target_set_ids, incident_demands, current_assignments
        ) + (
            ("routes", sorted_routes),
            ("routing_methodology", ROUTING_METHODOLOGY_NAME),
            ("routing_methodology_version", ROUTING_METHODOLOGY_VERSION),
        )
        serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_request(global_planning_run_id: object, as_of: object) -> None:
        if isinstance(global_planning_run_id, bool) or not isinstance(global_planning_run_id, int) or global_planning_run_id <= 0:
            raise ValueError(f"global_planning_run_id must be a positive integer, got {global_planning_run_id!r}")
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")

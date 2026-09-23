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

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from dataclasses import dataclass, replace
from datetime import datetime
import functools
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
from src.services.operational.road_network_fetcher import GRAPH_FIDELITY_CUSTOM_FILTER, RoadNetworkFetcher
from src.services.routing.node_mapping_service import MAX_SNAP_DISTANCE_KM
from src.utils.geo import haversine_distance_km

logger = logging.getLogger(__name__)

# Bounded mid-build revalidation (Task 21, Stage 3.1): 1 retry = 2 attempts
# total. Never unbounded - if resource state still disagrees after the
# retry, build() raises GlobalPlanningInputUnstable rather than looping
# further or returning a known-stale input.
_MAX_REBUILD_ATTEMPTS = 1

_GLOBAL_INPUT_METHODOLOGY = "global_candidate_route_matrix"
_GLOBAL_INPUT_METHODOLOGY_VERSION = "1.0"

# Infrastructure fix: the combined-bbox road network fetch below covers
# every active event in ONE bounding box, and previously only fell back to
# a live OSM fetch when that WHOLE combined bbox came back with zero nodes
# (`if not road_nodes`). With several concurrent events spread across
# Israel, one event with dense coverage (e.g. Carmel) makes the combined
# bbox non-empty, so that fallback never triggers for OTHER events sharing
# the same build even when their own local area has NO road data at all -
# their targets/resources then simply never get a feasible route, which
# looks identical to "the GA assigned 0 resources" from the outside. The
# per-anchor check below closes that gap by verifying road coverage near
# EACH event's own anchor individually, regardless of what the combined
# bbox already contains.
#
# _PER_ANCHOR_COVERAGE_CHECK_RADIUS_KM: deliberately reuses
# NodeMappingService.MAX_SNAP_DISTANCE_KM rather than a wider number of its
# own. A wider check (an earlier version of this code used 15km) can find
# "some node vaguely nearby" that is still USELESS for actually routing
# this anchor, because NodeMappingService.map_target will refuse to snap to
# anything farther than MAX_SNAP_DISTANCE_KM anyway - a live test against
# Jerusalem Forest reproduced exactly that: 207 nodes existed at ~15km
# (a thin sliver spilling over from a NEIGHBORING event's own fetch, not
# real local coverage) while 0..12km had nothing, so the loose check wrongly
# skipped fetching and the target stayed unroutable. Checking against the
# SAME radius NodeMappingService itself uses to snap keeps this "does a
# fetch even help" check honest.
_PER_ANCHOR_COVERAGE_CHECK_RADIUS_KM = MAX_SNAP_DISTANCE_KM

# Progressive expansion (lazy loading): fetching the full
# _PER_ANCHOR_FETCH_RADIUS_KM (50km, tiled) is the expensive path - even
# with tiling and concurrency, it is several real Overpass requests. Most
# fires have enough operationally-available resources within a small local
# radius, so paying that cost every time is wasted I/O. _LOCAL_FETCH_RADIUS_KM
# is Step 1: one small, untiled, reliably-fast fetch (matches the size
# measured live to complete in ~19s - see RoadNetworkFetcher.DEFAULT_TILE_SIZE_KM's
# own docstring for the same number's provenance). Only if the candidates
# reachable within THIS radius fall short of the fire's own desired_resources
# does Step 3 pay for the wide, tiled fetch - see
# _locally_reachable_assignable_count and _ensure_per_anchor_coverage.
_LOCAL_FETCH_RADIUS_KM = 10.0

# _PER_ANCHOR_FETCH_RADIUS_KM: matches GlobalCandidateResourceConfig's own
# widest normal search radius (search_radii_km's last entry, 50.0) - any
# resource GlobalCandidateCollector could plausibly have gathered as a
# candidate for this anchor without invoking its closest-station-fallback
# path is already within this radius. Was temporarily capped at 10km
# because fetching this whole radius as ONE Overpass request reliably
# exceeded OSM_FETCH_TIMEOUT_SECONDS (60s) at anything past ~15km - a real
# incident (a genuinely closer, available station between 10km and 50km
# away was invisible to the GA solely because that radius was never
# fetched at all). RoadNetworkFetcher.fetch_network_in_bbox_tiled removes
# that constraint: this whole radius is now fetched as several small,
# independently-reliable tiles (see DEFAULT_TILE_SIZE_KM) rather than one
# large request, so the full candidate-search radius can be used here
# again without reintroducing the timeout.
_PER_ANCHOR_FETCH_RADIUS_KM = 50.0

# Same per-degree conversion convention already used elsewhere in this
# codebase (see news_data_generator.py/satellite_data_generator.py's own
# _KM_PER_LATITUDE_DEGREE) - no shared util exists for it, so this follows
# the same local-constant pattern rather than introducing one.
_KM_PER_DEGREE = 111.32

# Targeted origin fetching: an anchor-centered fetch (Steps 1/3 above)
# guarantees the FIRE's own area has road coverage, but says nothing about
# whether a specific CANDIDATE RESOURCE's station - which could be
# anywhere within that area, including right at its edge, or split across
# a tile boundary - has ITS OWN immediate local street grid present.
# NodeMappingService can only snap a resource to whatever node is nearest
# within the fetched graph; if the station's own short access road/local
# streets are missing, it snaps to a distant, disconnected node instead,
# producing a real, unnecessary detour even though the wider area "has
# roads" by the anchor-level check. This radius is deliberately small and
# guaranteed for every candidate station lacking coverage, independent of
# anchor-level decisions - it is Step 4, after Steps 1-3, and checks
# against whatever they already produced.
_STATION_MICRO_FETCH_RADIUS_KM = 2.0

# _STATION_COVERAGE_CHECK_RADIUS_KM: Step 4's OWN "does this station already
# have coverage, or does it need a micro-fetch" decision must NOT reuse
# _PER_ANCHOR_COVERAGE_CHECK_RADIUS_KM (5km). That radius answers "is
# ANYTHING snappable at all" for an anchor - the right question when the
# alternative is UNMAPPABLE. A station is a much stricter case: Steps 1/3
# already fetched a wide area around the FIRE, so a station within that wide
# area almost always has at least one node somewhere within 5km of it
# already (_has_nearby_coverage would report "covered"), even when that
# node is 1-2km away and nowhere near the station's own local street grid -
# NodeMappingService then snaps the resource to that distant node, producing
# a real, unnecessary detour (and the exact "route starts 1-2km from the
# station's blue dot" symptom) while Step 4 silently never fires, believing
# the station doesn't need it. A live repro (Gush Etzion) confirmed this:
# the station's nearest node was ~1.2km away, comfortably inside 5km, so the
# micro-fetch that should have supplied its actual local access road never
# ran. This radius is deliberately much tighter - close to
# _STATION_MICRO_FETCH_RADIUS_KM's own footprint - so "covered" here means
# "has a node close enough to plausibly BE the station's own street", not
# merely "has a node close enough to still be snappable in principle".
_STATION_COVERAGE_CHECK_RADIUS_KM = 0.3

# Performance fix: multiple anchors needing a dedicated OSM fetch in the
# same build used to fetch strictly one at a time (up to
# OSM_FETCH_TIMEOUT_SECONDS=60s EACH, sequentially - measured live this
# session at up to ~90s for two anchors in one build). Each fetch is
# genuinely independent I/O (a separate Overpass HTTP call via
# RoadNetworkFetcher, which releases the GIL while waiting), so running
# them concurrently on a small thread pool bounds the wall-clock cost at
# roughly the SLOWEST single fetch instead of their sum. Bounded (not
# unbounded) because a real cycle rarely has more than a handful of
# simultaneously-uncovered anchors, and an unbounded pool would let one
# unusually large build fire an unreasonable number of concurrent Overpass
# requests against a public, rate-limited API.
_MAX_CONCURRENT_ANCHOR_FETCHES = 4


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
        road_nodes, road_edges = self._load_road_network(anchors, resources, incident_demands)

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
            road_nodes, road_edges = self._load_road_network(anchors, resources, incident_demands)
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

    def _load_road_network(
        self,
        anchors,
        resources: tuple[GlobalPlanningResource, ...],
        incident_demands: tuple[GlobalIncidentDemand, ...],
    ):
        """Outer safety net around _load_road_network_inner.

        Observed live, twice, in production: a road-network-loading failure
        (an Overpass timeout that somehow still escaped RoadNetworkFetcher's
        own fail-safe contract; separately, a PostgreSQL bind-parameter
        limit tripped by a large combined bbox - see
        RoadNetworkRepository._BULK_OPERATION_BATCH_SIZE, now fixed at its
        source) propagating out of this step crashed the WHOLE
        GlobalPlanningInput build. That doesn't just fail one event's
        routing - it aborts build() entirely, so EVERY active event in this
        cycle silently drops out of the Global Response Plan (the read
        service then keeps serving the last plan that DID complete, which
        may be missing newer events altogether) until the next cycle
        succeeds. Fixing each specific cause as it's found is necessary but
        not sufficient; this call is also wrapped so that ANY failure here -
        known or not yet seen - degrades to "no road network data this
        cycle" (every target/resource still included in the build, simply
        with zero feasible Dijkstra routes) rather than ever again aborting
        the whole run. This never touches the GA or Dijkstra itself - it
        only decides what data they run against.
        """
        try:
            return self._load_road_network_inner(anchors, resources, incident_demands)
        except Exception:  # noqa: BLE001 - must never abort the whole GlobalPlanningInput build.
            logger.warning(
                "Road-network loading failed entirely for this build; continuing with NO road-network data "
                "(every active event's targets/resources still appear in the plan, but with zero feasible "
                "routes this cycle rather than the whole run being aborted).",
                exc_info=True,
            )
            return [], []

    def _load_road_network_inner(
        self,
        anchors,
        resources: tuple[GlobalPlanningResource, ...],
        incident_demands: tuple[GlobalIncidentDemand, ...],
    ):
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
                # Tiled (see RoadNetworkFetcher.fetch_network_in_bbox_tiled):
                # this combined bbox spans every active anchor + candidate
                # station, which can be arbitrarily large for several
                # dispersed concurrent events - the same "one huge request
                # reliably times out" risk _ensure_per_anchor_coverage's own
                # fetch faces below, fixed the same way. GRAPH_FIDELITY_CUSTOM_FILTER
                # (see its own docstring): this is also a "pay for the wide
                # picture" path, not the fast local one, so it gets the same
                # graph-fidelity relaxation as Step 3 below.
                road_nodes, road_edges = self._road_network_fetcher.fetch_network_in_bbox_tiled(
                    min_lat, max_lat, min_lon, max_lon, custom_filter=GRAPH_FIDELITY_CUSTOM_FILTER
                )
                if road_nodes:
                    self._road_network_repository.save_network(session, road_nodes, road_edges)
            road_nodes, road_edges = self._ensure_per_anchor_coverage(
                session, anchors, resources, incident_demands, road_nodes, road_edges
            self._road_network_cache = _RoadNetworkCacheEntry(
                bbox=bbox,
                nodes=tuple(node.model_copy() for node in road_nodes),
                edges=tuple(edge.model_copy() for edge in road_edges),
            )
            return road_nodes, road_edges
        finally:
            session.close()

    def _ensure_per_anchor_coverage(
        self,
        session: Session,
        anchors,
        resources: tuple[GlobalPlanningResource, ...],
        incident_demands: tuple[GlobalIncidentDemand, ...],
        road_nodes: list,
        road_edges: list,
    ) -> tuple[list, list]:
        """Top up `road_nodes`/`road_edges` with a live OSM fetch for any
        anchor whose own local area has no road coverage at all, even
        though the combined bbox above wasn't empty overall - see the
        module-level comment by _PER_ANCHOR_COVERAGE_CHECK_RADIUS_KM.

        Progressive expansion (lazy loading), per uncovered anchor:
          Step 1: one small, untiled, fast local fetch
                  (_LOCAL_FETCH_RADIUS_KM) - cheap enough to always attempt.
          Step 2: count operationally-assignable candidate `resources`
                  within that same local radius that the fetch just made
                  reachable (_locally_reachable_assignable_count). Only if
                  that falls short of the fire's own desired_resources does
                  this anchor proceed to:
          Step 3: the expensive wide, TILED fetch (_PER_ANCHOR_FETCH_RADIUS_KM,
                  see RoadNetworkFetcher.fetch_network_in_bbox_tiled) - paid
                  only for anchors with a genuine local shortage, not every
                  uncovered anchor unconditionally.
          Step 4 (targeted origin fetching): independent of the anchor-level
                  decision above, every CANDIDATE RESOURCE's own station
                  gets a small, guaranteed micro-fetch
                  (_STATION_MICRO_FETCH_RADIUS_KM) if it still lacks nearby
                  coverage after Steps 1-3 - see _stations_needing_fetch.
                  A station can sit at the edge of an otherwise-covered
                  area (or a tile boundary) with its own immediate local
                  streets missing, forcing NodeMappingService to snap it to
                  a distant, disconnected node and producing a real
                  unnecessary detour even though the anchor's own area
                  "has roads" by Steps 1-3's own check.

        Each of the 4 steps runs as its own concurrent batch (see
        _fetch_and_merge_concurrently) rather than pipelining one anchor's
        or station's full sequence in isolation - each step's own decision
        is cheap/in-memory, so batching by step keeps every fetch maximally
        parallel.

        Each anchor's fetch/save failure is caught independently (a
        ThreadPoolExecutor future's own exception, or save_network's) so one
        anchor never aborts another's result. A target left without any
        feasible Dijkstra route this cycle (either because it was never
        deemed to need expansion for a false reason, or because the wide
        fetch itself still didn't reach it) simply stays uncovered - Dijkstra
        is never given a fabricated air-distance route to fall back to (see
        test_input_builder_never_imports_the_haversine_route_fallback).
        """
        nodes_by_id = {node.id: node for node in road_nodes}
        edges_by_key = {(edge.source_node_id, edge.target_node_id): edge for edge in road_edges}

        anchors_needing_fetch = self._anchors_needing_fetch(anchors, nodes_by_id.values())
        # Deliberately NOT an early return when anchors_needing_fetch is
        # empty: Step 4 below must still run even when every anchor is
        # already covered - that is EXACTLY the reported scenario (the
        # anchor's own area "has roads" by this check, but a specific
        # candidate station's immediate local streets still don't).
        # _fetch_and_merge_concurrently itself already no-ops on an empty
        # anchor list, so Steps 1-3 correctly do nothing below when there
        # is nothing for them to do; only Step 4 is unconditional.
        for fire_event_id, latitude, longitude in anchors_needing_fetch:
            logger.warning(
                "FireEvent %s's anchor (%.5f, %.5f) has no road-network node within %.0fkm even though the "
                "combined build bbox was non-empty; fetching a local OSM bbox for this area.",
                fire_event_id,
                latitude,
                longitude,
                _PER_ANCHOR_COVERAGE_CHECK_RADIUS_KM,
            )

        # Step 1: fast, simple, untiled local fetch for every uncovered
        # anchor. Uses GRAPH_FIDELITY_CUSTOM_FILTER too - same filter as
        # Step 3, not the plain "drive" preset - so the local and wide
        # fetches have IDENTICAL road-class fidelity and only ever differ
        # in the AREA covered, never in which road types are visible once
        # covered. The relaxed filter costs no extra Overpass requests or
        # meaningfully more data at this small radius, so there is no real
        # reason to reserve it for the wide path only.
        self._fetch_and_merge_concurrently(
            anchors_needing_fetch,
            nodes_by_id,
            edges_by_key,
            session,
            bbox_fn=lambda latitude, longitude: self._per_anchor_bbox(latitude, longitude, _LOCAL_FETCH_RADIUS_KM),
            fetch_fn=functools.partial(
                self._road_network_fetcher.fetch_network_in_bbox, custom_filter=GRAPH_FIDELITY_CUSTOM_FILTER
            ),
            phase_label="local",
        )

        # Step 2: decide, per anchor, whether the local fetch already found
        # enough operationally-assignable supply - only a genuine shortage
        # proceeds to Step 3's expensive wide/tiled fetch.
        desired_resources_by_event = {demand.fire_event_id: demand.desired_resources for demand in incident_demands}
        anchors_needing_expansion: list[tuple[int, float, float]] = []
        for fire_event_id, latitude, longitude in anchors_needing_fetch:
            desired_resources = desired_resources_by_event.get(fire_event_id)
            if desired_resources is None or desired_resources <= 0:
                continue
            local_count = self._locally_reachable_assignable_count(latitude, longitude, resources, nodes_by_id.values())
            if local_count >= desired_resources:
                logger.info(
                    "FireEvent %s: %d locally-reachable assignable resource(s) within %.0fkm already meet its "
                    "desired_resources=%d; skipping the wider tiled OSM fetch for this anchor.",
                    fire_event_id,
                    local_count,
                    _LOCAL_FETCH_RADIUS_KM,
                    desired_resources,
                )
                continue
            logger.warning(
                "FireEvent %s: only %d locally-reachable assignable resource(s) within %.0fkm, short of its "
                "desired_resources=%d; expanding to a wider tiled OSM fetch (up to %.0fkm).",
                fire_event_id,
                local_count,
                _LOCAL_FETCH_RADIUS_KM,
                desired_resources,
                _PER_ANCHOR_FETCH_RADIUS_KM,
            )
            anchors_needing_expansion.append((fire_event_id, latitude, longitude))

        # Step 3: wide, tiled fetch - only for anchors with a genuine
        # shortage. Same GRAPH_FIDELITY_CUSTOM_FILTER as Step 1 above -
        # local and wide fetches now differ ONLY in the area they cover
        # (_LOCAL_FETCH_RADIUS_KM vs _PER_ANCHOR_FETCH_RADIUS_KM, plus
        # tiling), never in which road classes are visible once fetched.
        self._fetch_and_merge_concurrently(
            anchors_needing_expansion,
            nodes_by_id,
            edges_by_key,
            session,
            bbox_fn=lambda latitude, longitude: self._per_anchor_bbox(latitude, longitude, _PER_ANCHOR_FETCH_RADIUS_KM),
            fetch_fn=functools.partial(
                self._road_network_fetcher.fetch_network_in_bbox_tiled, custom_filter=GRAPH_FIDELITY_CUSTOM_FILTER
            ),
            phase_label="wide/tiled",
        )

        # Step 4: targeted origin fetching. Independent of anchor-level
        # decisions above - checked against whatever Steps 1-3 already
        # produced, for EVERY candidate resource's station (not just ones
        # near an anchor that needed expansion), since even a station well
        # inside an already-"covered" area can lack its own immediate
        # local streets. Deliberately small (_STATION_MICRO_FETCH_RADIUS_KM)
        # and cheap - most stations will already be covered by Steps 1-3's
        # wider fetches and are skipped here, so this typically only fires
        # for the genuine edge-of-coverage/tile-boundary cases it exists
        # for, not for every candidate unconditionally.
        stations_needing_fetch = self._stations_needing_fetch(resources, nodes_by_id.values())
        self._fetch_and_merge_concurrently(
            stations_needing_fetch,
            nodes_by_id,
            edges_by_key,
            session,
            bbox_fn=lambda latitude, longitude: self._per_anchor_bbox(latitude, longitude, _STATION_MICRO_FETCH_RADIUS_KM),
            fetch_fn=functools.partial(
                self._road_network_fetcher.fetch_network_in_bbox, custom_filter=GRAPH_FIDELITY_CUSTOM_FILTER
            ),
            phase_label="station-micro",
            entity_label="Station",
        )

        return list(nodes_by_id.values()), list(edges_by_key.values())

    def _fetch_and_merge_concurrently(
        self,
        anchors,
        nodes_by_id: dict,
        edges_by_key: dict,
        session: Session,
        *,
        bbox_fn,
        fetch_fn,
        phase_label: str,
        entity_label: str = "FireEvent",
    ) -> None:
        """Shared concurrency/merge machinery for the local (Step 1),
        wide/tiled (Step 3), and station-micro (Step 4) fetch batches: run
        `fetch_fn` for every anchor CONCURRENTLY on a small thread pool (see
        _MAX_CONCURRENT_ANCHOR_FETCHES) - genuinely independent I/O, each
        releasing the GIL while its HTTP call is in flight - then persist
        and merge results SEQUENTIALLY against the one shared `session`
        back on this thread, since SQLAlchemy Sessions are not thread-safe.

        Infrastructure resilience: `fetch_fn` (fetch_network_in_bbox /
        fetch_network_in_bbox_tiled) already never raises on its own (it
        swallows Overpass timeouts/errors and returns ([], []) - see their
        own docstrings), but consuming a future's .result() can still
        surface an unexpected error, and the follow-up save_network DB
        write runs against a `session` that may have been sitting idle
        while these fetches were in flight - an idle-too-long connection to
        a serverless Postgres backend (Neon) failing on the next write is a
        realistic failure mode this try/except is sized for. One anchor's
        failure must never cost another's result or abort the whole build
        (see GlobalPlanningRefreshCoordinator's narrow exception handling,
        module-level comment above _PER_ANCHOR_FETCH_RADIUS_KM) - only this
        anchor's targets stay unroutable via Dijkstra this cycle.

        Persistent cache, by construction, not a bolt-on: every successful
        fetch here is written straight to RoadNetworkRepository below, and
        anchor/station coordinates are static - so a bbox fetched and saved
        ONCE is real coverage forever. The NEXT cycle's own combined-bbox
        read at the top of _load_road_network_inner (which spans every
        anchor's AND every candidate station's coordinate by construction)
        picks it back up from the DB with no live OSM call at all, and
        _has_nearby_coverage (a tight, entity-centered radius check, not
        "any node anywhere in a big bbox") then correctly recognizes this
        entity as already covered and skips it before it ever reaches this
        method again. A deliberately-considered and REJECTED alternative:
        adding a DB-read "already cached" gate directly in THIS method,
        keyed off `bbox_fn`'s own (phase-specific) bbox. For Step 4
        (station-micro, ~2km) that would have been roughly sound; for Step 1
        (10km) and especially Step 3 (50km, tiled) it would silently
        reintroduce the exact "some node exists somewhere in this big bbox,
        therefore treat the whole area as covered" bug _ensure_per_anchor_coverage
        exists to fix (see _PER_ANCHOR_COVERAGE_CHECK_RADIUS_KM's own
        docstring) - a real per-anchor/station shortage could then be masked
        by unrelated cached data elsewhere in a wide bbox. The existing
        tight-radius check upstream is the correct place for that decision;
        this method's only job is fetch-if-asked, merge, and persist.
        """
        if not anchors:
            return

        worker_count = min(_MAX_CONCURRENT_ANCHOR_FETCHES, len(anchors))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=f"road-network-fetch-{phase_label}") as executor:
            future_to_anchor = {
                executor.submit(fetch_fn, *bbox_fn(latitude, longitude)): (entity_id, latitude, longitude)
                for entity_id, latitude, longitude in anchors
            }

            for future in as_completed(future_to_anchor):
                entity_id, latitude, longitude = future_to_anchor[future]
                try:
                    fetched_nodes, fetched_edges = future.result()
                    if fetched_nodes:
                        self._road_network_repository.save_network(session, fetched_nodes, fetched_edges)
                except Exception:  # noqa: BLE001 - one anchor's fetch/save failure must never abort the whole build.
                    logger.warning(
                        "%s %s's %s OSM fetch/save failed; continuing the planning run without this road "
                        "coverage.",
                        entity_label,
                        entity_id,
                        phase_label,
                        exc_info=True,
                    )
                    continue

                if not fetched_nodes:
                    logger.warning(
                        "%s %s's %s OSM fetch returned no road network for (%.5f, %.5f); this area will "
                        "remain unroutable until OSM coverage exists or the fetch succeeds on a later attempt.",
                        entity_label,
                        entity_id,
                        phase_label,
                        latitude,
                        longitude,
                    )
                    continue

                logger.info(
                    "%s %s's %s OSM fetch returned %d node(s) and %d edge(s).",
                    entity_label,
                    entity_id,
                    phase_label,
                    len(fetched_nodes),
                    len(fetched_edges),
                )
                for node in fetched_nodes:
                    nodes_by_id[node.id] = node
                for edge in fetched_edges:
                    edges_by_key[(edge.source_node_id, edge.target_node_id)] = edge

    @staticmethod
    def _anchors_needing_fetch(anchors, nodes):
        """Which anchors lack nearby coverage, deduped against EACH OTHER by
        coordinate alone (not by waiting on any fetch result) - two anchors
        within _PER_ANCHOR_COVERAGE_CHECK_RADIUS_KM of each other only ever
        get one fetch between them, matching the old sequential behavior's
        "two events sharing one genuinely uncovered region only trigger one
        live fetch" guarantee without requiring the fetches themselves to
        run in order."""
        node_list = list(nodes)
        needing_fetch: list[tuple[int, float, float]] = []
        for fire_event_id, latitude, longitude in anchors:
            if GlobalPlanningInputBuilder._has_nearby_coverage(latitude, longitude, node_list):
                continue
            if any(
                haversine_distance_km(latitude, longitude, other_lat, other_lon) <= _PER_ANCHOR_COVERAGE_CHECK_RADIUS_KM
                for _other_event_id, other_lat, other_lon in needing_fetch
            ):
                continue
            needing_fetch.append((fire_event_id, latitude, longitude))
        return needing_fetch

    @staticmethod
    def _stations_needing_fetch(
        resources: tuple[GlobalPlanningResource, ...], nodes
    ) -> list[tuple[str, float, float]]:
        """Which candidate resource stations lack nearby coverage (Step 4),
        deduped by coordinate - several resources sharing one station only
        trigger one micro-fetch between them. Only assignable resources
        (Task 4's exclusion rule - see GlobalPlanningResource.is_assignable)
        are considered: an UNAVAILABLE resource's station is never
        reachable by the GA regardless of graph quality, so fetching for it
        would be pure waste."""
        node_list = list(nodes)
        seen_coordinates: set[tuple[float, float]] = set()
        needing_fetch: list[tuple[str, float, float]] = []
        for resource in resources:
            if not resource.is_assignable:
                continue
            coordinate = (resource.station_latitude, resource.station_longitude)
            if coordinate in seen_coordinates:
                continue
            seen_coordinates.add(coordinate)
            if GlobalPlanningInputBuilder._has_nearby_coverage(
                resource.station_latitude,
                resource.station_longitude,
                node_list,
                radius_km=_STATION_COVERAGE_CHECK_RADIUS_KM,
            ):
                continue
            needing_fetch.append((resource.station_id, resource.station_latitude, resource.station_longitude))
        return needing_fetch

    @staticmethod
    def _locally_reachable_assignable_count(
        latitude: float,
        longitude: float,
        resources: tuple[GlobalPlanningResource, ...],
        nodes,
    ) -> int:
        """How many operationally-assignable candidate `resources` (Task 4's
        one exclusion rule - see GlobalPlanningResource.is_assignable, never
        UNAVAILABLE) sit within _LOCAL_FETCH_RADIUS_KM of this anchor AND
        have a nearby road-network node to snap to in `nodes` (the SAME
        "would NodeMappingService even consider this reachable" proxy
        _has_nearby_coverage already uses for the anchor itself). This is a
        cheap, in-memory sufficiency estimate, not an actual Dijkstra run -
        it decides whether Step 3's expensive wide fetch is worth paying
        for, never whether the GA can actually use a resource (that's
        still decided for real once the route matrix is built)."""
        node_list = list(nodes)
        count = 0
        for resource in resources:
            if not resource.is_assignable:
                continue
            if (
                haversine_distance_km(latitude, longitude, resource.station_latitude, resource.station_longitude)
                > _LOCAL_FETCH_RADIUS_KM
            ):
                continue
            if GlobalPlanningInputBuilder._has_nearby_coverage(
                resource.station_latitude, resource.station_longitude, node_list
            ):
                count += 1
        return count

    @staticmethod
    def _per_anchor_bbox(latitude: float, longitude: float, radius_km: float) -> tuple[float, float, float, float]:
        lat_degrees = radius_km / _KM_PER_DEGREE
        lon_degrees = lat_degrees  # slightly over-covers east-west at high latitude - never under-covers.
        return (
            max(-90.0, latitude - lat_degrees),
            min(90.0, latitude + lat_degrees),
            max(-180.0, longitude - lon_degrees),
            min(180.0, longitude + lon_degrees),
        )

    @staticmethod
    def _has_nearby_coverage(
        latitude: float, longitude: float, nodes, radius_km: float = _PER_ANCHOR_COVERAGE_CHECK_RADIUS_KM
    ) -> bool:
        """Cheap presence check: any node at all within `radius_km`,
        regardless of edge direction. NodeMappingService itself is stricter
        (a target additionally needs a node with an INCOMING edge, a
        resource an OUTGOING one) - deliberately not replicated here, since
        this check only decides whether a fetch is worth attempting, not
        whether routing will succeed. Being occasionally over-lenient (skips
        a fetch for a lone edgeless node within range, a rare case for real
        OSM data) leaves that one anchor exactly as routable as it was
        before this fix, never worse.

        `radius_km` defaults to _PER_ANCHOR_COVERAGE_CHECK_RADIUS_KM (the
        anchor-level "is anything snappable at all" semantics) - callers
        deciding station-level genuine local coverage instead pass the much
        tighter _STATION_COVERAGE_CHECK_RADIUS_KM explicitly; see that
        constant's docstring for why the two must not share one radius."""
        return any(
            haversine_distance_km(latitude, longitude, node.latitude, node.longitude) <= radius_km for node in nodes
        )

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

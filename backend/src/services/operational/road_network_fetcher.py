"""Fetch and convert OpenStreetMap road network data for a bounding box.

RoadNetworkFetcher wraps osmnx: given a geographic bounding box, it
downloads the driving road network within it and converts it into our
domain models (GraphNode, GraphEdge). It is pure fetch-and-convert - no
persistence, no bounding-box calculation from an incident location (that
lives in OperationalContextService), no orchestration.

Extracted from backend/scripts/seed_road_networks.py's per-location OSM
fetching logic so a lazy-loading ("on demand") caller can reuse the exact
same OSMnx fetch/speed/travel-time pipeline the original pre-seeding script
used, just parameterized by bounding box instead of a point + radius.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import math
import threading
import time

import networkx as nx
import osmnx as ox
import requests
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError

from src.models.graph_edge import GraphEdge
from src.models.graph_node import GraphNode

logger = logging.getLogger(__name__)

NETWORK_TYPE = "drive"

# Graph fidelity fix: verified directly against osmnx's own internal
# default filter for network_type="drive" (osmnx._overpass._get_network_filter,
# confirmed live this session) - it does NOT exclude "secondary" or
# "tertiary" highway tags; those already come through on every fetch,
# local or wide. What IS excluded by default is highway=service entirely.
# A short service-road connector (e.g. a station's own driveway-like link
# to the nearest classified road) is a common, real cause of exactly the
# "Dijkstra takes a needless detour a human/Waze wouldn't" symptom this
# fixes, since the direct link is invisible to it. This filter is
# osmnx's own "drive" filter with "service"/"services" removed from the
# highway exclusion list, while KEEPING its separate, more targeted
# exclusion of service sub-types that are genuinely not through-routes
# (alley/driveway/emergency_access/parking/parking_aisle/private) - so a
# real service road that functions as a real connector is now included,
# but an actual private driveway or parking aisle still is not.
#
# Deliberately NOT the default for every fetch: it costs more data/time to
# process (more edges to fetch and convert) for a benefit that mostly
# matters at the edges of an already-covered area, so it is reserved for
# the wide/tiled fetch path (see GlobalPlanningInputBuilder's Step 3),
# leaving the fast local fetch (Step 1) on the lean, standard filter.
GRAPH_FIDELITY_CUSTOM_FILTER = (
    '["highway"]'
    '["area"!~"yes"]'
    '["access"!~"private"]'
    '["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|escalator|'
    'footway|no|path|pedestrian|planned|platform|proposed|raceway|razed|rest_area|steps|track"]'
    '["motor_vehicle"!~"no"]'
    '["motorcar"!~"no"]'
    '["service"!~"alley|driveway|emergency_access|parking|parking_aisle|private"]'
)

# Task A1.5: osmnx's own per-HTTP-request timeout (ox.settings.requests_timeout,
# 180s by default) does not bound the *overall* call - osmnx can tile a large
# bbox into several sequential Overpass requests, and its rate-limit handling
# (ox.settings.overpass_rate_limit) can sleep for an arbitrary, server-reported
# duration between them. Without an outer bound, a single cache-miss fetch can
# block the calling thread for tens of minutes with near-zero CPU/network
# activity in between, which is indistinguishable from a hang. This bound
# makes the fetch's own documented "degrade gracefully on Overpass timeout"
# contract actually hold for that case too, not just for outright exceptions.
OSM_FETCH_TIMEOUT_SECONDS = 60.0

# Retry (live incident: a station-micro fetch - see
# GlobalPlanningInputBuilder's Step 4 - observed failing every single cycle
# with "OSM fetch/conversion exceeded the 60s bound", i.e. Overpass
# throttling under this session's own repeated load, not a one-off blip).
# Retrying only genuinely transient conditions, matching news_client.py's
# own _is_transient_llm_error precedent:
#   - TimeoutError: our own OSM_FETCH_TIMEOUT_SECONDS bound - Overpass (or
#     the network) was simply too slow THIS attempt, nothing about the bbox
#     itself is wrong.
#   - requests.exceptions.Timeout/ConnectionError: a raw network-level
#     failure before Overpass even responded.
#   - osmnx's own ResponseStatusCodeError: an unhandled HTTP status from
#     Overpass - in practice this is how Overpass reports rate-limiting
#     (429) and server-side overload (504/503) to osmnx, exactly the
#     "external API throttling" failure mode this exists to power through.
# Deliberately NOT retried: osmnx's own InsufficientResponseError (the
# response came back fine and genuinely contains no road data for this
# bbox - a real "nothing here" result, not a failure; retrying 3 times
# would only add latency for a certain, unchanging outcome) or any other
# unexpected exception (a real bug/data problem retries cannot fix).
# Bounded worst case if every attempt times out: MAX_OSM_FETCH_ATTEMPTS *
# OSM_FETCH_TIMEOUT_SECONDS plus backoff (~3 * 60s + 1s + 2s = ~183s) for
# ONE bbox - still finite, and this is exactly the same shape as
# fetch_network_in_bbox's existing never-raises contract: the caller either
# gets real data or a clean ([], []), just after genuinely trying harder
# first.
MAX_OSM_FETCH_ATTEMPTS = 3
_OSM_FETCH_RETRY_BASE_DELAY_SECONDS = 1.0


def _is_transient_osm_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(exc, InsufficientResponseError):
        return False
    return isinstance(exc, ResponseStatusCodeError)

# Used both as OSMnx's own per-edge speed fallback (passed into
# add_edge_speeds) and as a last-resort manual fallback in
# _travel_time_seconds if the speed/travel-time model raises outright for a
# given bounding box.
FALLBACK_AVERAGE_SPEED_KMH = 50.0
_FALLBACK_SPEED_METERS_PER_SECOND = FALLBACK_AVERAGE_SPEED_KMH / 3.6

# Bounding-box tiling (chunking): a single large bbox reliably times out
# against OSM_FETCH_TIMEOUT_SECONDS (measured live: a 30km-side bbox took
# ~50.6s, a 40km-side one ~46.1s - already close to or past the bound, and
# that only gets worse with a wider search radius). Splitting the requested
# area into DEFAULT_TILE_SIZE_KM x DEFAULT_TILE_SIZE_KM tiles and fetching
# each independently keeps every single Overpass request small enough to
# reliably finish, so a genuinely wide search radius (e.g. 30-50km, per the
# real "a closer available station outside 10km was being missed" incident)
# becomes reachable without raising OSM_FETCH_TIMEOUT_SECONDS itself or
# accepting a shorter, less useful radius.
#
# 10km matches the size measured live (same session, same region) to
# reliably complete in ~19s. IMPORTANT, and verified live while tuning this
# value: Overpass is a shared public API with genuinely variable response
# time that no fixed tile size can fully insulate against - later in the
# same session, even a 7km tile against the exact same region timed out
# outright, where a 20km bbox had completed in 19s earlier. Smaller tiles
# reduce the wasted time and blast radius of any single timeout and raise
# the odds more of them land inside the bound, but cannot GUARANTEE every
# tile succeeds. That is precisely why fetch_network_in_bbox_tiled treats a
# failed/timed-out tile as "contributes nothing to the merge" rather than
# aborting the whole fetch (see its own docstring) - resilience to partial
# failure is what actually makes this reliable against a variable external
# service, not tile size alone.
DEFAULT_TILE_SIZE_KM = 10.0

# Adjacent tiles are fetched as fully independent Overpass queries -
# without any shared margin, a real road that crosses a tile boundary can
# be silently dropped by BOTH tiles' own graph_from_bbox call (each only
# returns ways it considers within ITS OWN bbox), fragmenting the merged
# network exactly at every seam - the opposite of the "continuous real road
# network" this tiling exists to produce. Expanding every tile by this
# margin on all four sides means a boundary-crossing road's nodes fall
# safely inside at least one tile's query, so save_network's own
# upsert-by-id/merge-by-(source,target) semantics naturally deduplicate the
# resulting overlap - it costs a little redundant fetching, never a gap.
DEFAULT_TILE_OVERLAP_KM = 2.0

# Bounded (not unbounded) for the same reason as
# GlobalPlanningInputBuilder._MAX_CONCURRENT_ANCHOR_FETCHES: Overpass is a
# shared, rate-limited public API, and tiling one anchor's search radius
# already multiplies request count well beyond the pre-tiling one-request-
# per-anchor world: a 50km-radius (100km-side) bbox at 15km tiles is
# roughly 7x7 = 49 tiles. Capping concurrency keeps that from turning into
# 49 simultaneous requests against one public server.
_MAX_CONCURRENT_TILE_FETCHES = 4

# Local per-degree conversion, matching the same flat-conversion convention
# already used elsewhere in this codebase (see
# GlobalPlanningInputBuilder._KM_PER_DEGREE, news/satellite data
# generators' own _KM_PER_LATITUDE_DEGREE) - slightly over-covers
# east-west at high latitude, never under-covers, and no shared util
# exists for it.
_KM_PER_DEGREE = 111.32


class RoadNetworkFetcher:
    """Fetches an OSM driving network for a bounding box and converts it to domain models."""

    def fetch_network_in_bbox(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        *,
        custom_filter: str | None = None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Return (nodes, edges) for the OSM driving network within a bounding box.

        `custom_filter`: an Overpass QL way-filter string overriding
        NETWORK_TYPE's standard "drive" preset - see
        GRAPH_FIDELITY_CUSTOM_FILTER for the one real caller of this today
        (a deliberately wider filter for the expensive wide/tiled fetch
        path only). `None` (the default) keeps the standard, faster filter.

        Returns ([], []) - rather than raising - if OSMnx finds no network
        in the bbox or if the fetch fails for any other reason (Overpass
        timeouts, connectivity issues, an oversized/slow graph to process,
        etc.) after exhausting MAX_OSM_FETCH_ATTEMPTS retries on genuinely
        transient failures (see _is_transient_osm_error). A lazy-loading
        caller wants "no road data available" to degrade gracefully, not to
        blow up (or hang) the request it's serving.
        """
        self._validate_bbox(min_lat, max_lat, min_lon, max_lon)

        logger.info(
            "Fetching OSM %s network for bbox (min_lat=%.5f, max_lat=%.5f, min_lon=%.5f, max_lon=%.5f)%s...",
            NETWORK_TYPE,
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            " (custom filter)" if custom_filter is not None else "",
        )
        for attempt in range(MAX_OSM_FETCH_ATTEMPTS):
            try:
                nodes, edges = self._fetch_and_convert_bounded(min_lat, max_lat, min_lon, max_lon, custom_filter)
            except Exception as exc:  # noqa: BLE001 - no network found, timed out, or any Overpass/OSMnx failure, degrades to empty.
                is_last_attempt = attempt == MAX_OSM_FETCH_ATTEMPTS - 1
                if not is_last_attempt and _is_transient_osm_error(exc):
                    delay_seconds = _OSM_FETCH_RETRY_BASE_DELAY_SECONDS * (2**attempt)
                    logger.warning(
                        "OSM fetch attempt %d/%d failed transiently for this bbox; retrying in %.1fs.",
                        attempt + 1,
                        MAX_OSM_FETCH_ATTEMPTS,
                        delay_seconds,
                        exc_info=True,
                    )
                    time.sleep(delay_seconds)
                    continue
                logger.warning(
                    "No OSM road network could be fetched for the requested bounding box (attempt %d/%d).",
                    attempt + 1,
                    MAX_OSM_FETCH_ATTEMPTS,
                    exc_info=True,
                )
                return [], []
            else:
                logger.info(
                    "Fetched %d node(s) and %d edge(s) for the requested bounding box.", len(nodes), len(edges)
                )
                return nodes, edges
        return [], []  # unreachable - every loop iteration either returns or continues/returns above.

    def fetch_network_in_bbox_tiled(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        *,
        tile_size_km: float = DEFAULT_TILE_SIZE_KM,
        tile_overlap_km: float = DEFAULT_TILE_OVERLAP_KM,
        max_concurrent_tiles: int = _MAX_CONCURRENT_TILE_FETCHES,
        custom_filter: str | None = None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Fetch a (possibly large) bounding box reliably by subdividing it
        into `tile_size_km` x `tile_size_km` tiles - each one small enough
        to fetch within OSM_FETCH_TIMEOUT_SECONDS on its own - fetching
        them CONCURRENTLY on a small thread pool, and merging the results.

        `custom_filter`: forwarded to every tile's own fetch_network_in_bbox
        call unchanged - see that method's own docstring and
        GRAPH_FIDELITY_CUSTOM_FILTER.

        Same never-raises, degrade-to-partial-or-empty contract as
        `fetch_network_in_bbox`: every tile goes through that exact method
        (so each tile's own timeout/Overpass-failure handling is unchanged),
        and one tile failing/returning nothing simply contributes nothing to
        the merge rather than losing the other tiles' results. Nodes are
        deduplicated by id, edges by (source_node_id, target_node_id) - the
        same merge semantics `RoadNetworkRepository.save_network`'s own
        upsert already relies on for overlapping fetches, so a real road
        that happens to be covered by more than one tile (see
        DEFAULT_TILE_OVERLAP_KM) is simply represented once.

        A bbox that only needs one tile is fetched directly via
        `fetch_network_in_bbox`, with no pool/merge overhead.
        """
        self._validate_bbox(min_lat, max_lat, min_lon, max_lon)
        tiles = _tile_bbox(min_lat, max_lat, min_lon, max_lon, tile_size_km, tile_overlap_km)
        if len(tiles) == 1:
            return self.fetch_network_in_bbox(*tiles[0], custom_filter=custom_filter)

        logger.info(
            "Tiling bbox (min_lat=%.5f, max_lat=%.5f, min_lon=%.5f, max_lon=%.5f) into %d tile(s) of ~%.0fkm each.",
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            len(tiles),
            tile_size_km,
        )

        nodes_by_id: dict[int, GraphNode] = {}
        edges_by_key: dict[tuple[int, int], GraphEdge] = {}
        worker_count = min(max_concurrent_tiles, len(tiles))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="osm-tile-fetch") as executor:
            futures = [
                executor.submit(self.fetch_network_in_bbox, *tile, custom_filter=custom_filter) for tile in tiles
            ]
            for future in as_completed(futures):
                try:
                    tile_nodes, tile_edges = future.result()
                except Exception:  # noqa: BLE001 - fetch_network_in_bbox itself never raises; defense in depth only.
                    logger.warning("One OSM tile fetch failed unexpectedly; continuing with the other tiles.", exc_info=True)
                    continue
                for node in tile_nodes:
                    nodes_by_id[node.id] = node
                for edge in tile_edges:
                    edges_by_key[(edge.source_node_id, edge.target_node_id)] = edge

        logger.info(
            "Tiled fetch merged to %d node(s) and %d edge(s) across %d tile(s).",
            len(nodes_by_id),
            len(edges_by_key),
            len(tiles),
        )
        return list(nodes_by_id.values()), list(edges_by_key.values())

    def _fetch_and_convert_bounded(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        custom_filter: str | None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Run the full fetch+convert pipeline on a daemon thread, bounded to OSM_FETCH_TIMEOUT_SECONDS.

        Deliberately wraps the *entire* pipeline (Overpass download, OSMnx's
        speed/travel-time model, and node/edge conversion) in one bound, not
        just the network call: none of these steps has its own call-level
        timeout, and OSMnx's speed/travel-time computation over a large
        graph can itself take as long as - or longer than - the download.
        Bounding only the download would leave the rest of the pipeline
        able to reproduce the exact same class of stall.

        Running the pipeline on a `daemon=True` thread and only *waiting* up
        to OSM_FETCH_TIMEOUT_SECONDS lets this method return (and the caller
        degrade gracefully) even if the underlying work is still running;
        the abandoned thread cannot block process exit because it is a
        daemon thread, and it is never joined.
        """
        outcome: dict[str, object] = {}

        def _run() -> None:
            try:
                # osmnx expects bbox as (left, bottom, right, top) =
                # (min_lon, min_lat, max_lon, max_lat).
                # custom_filter (when given) overrides network_type entirely
                # per osmnx's own API - see GRAPH_FIDELITY_CUSTOM_FILTER.
                if custom_filter is not None:
                    graph = ox.graph_from_bbox(
                        bbox=(min_lon, min_lat, max_lon, max_lat),
                        custom_filter=custom_filter,
                    )
                else:
                    graph = ox.graph_from_bbox(
                        bbox=(min_lon, min_lat, max_lon, max_lat),
                        network_type=NETWORK_TYPE,
                    )
                graph = self._add_travel_times(graph)
                outcome["nodes"] = self._build_nodes(graph)
                outcome["edges"] = self._build_edges(graph)
            except Exception as exc:  # noqa: BLE001 - reported to the joining thread, not raised here.
                outcome["error"] = exc

        worker = threading.Thread(target=_run, name="osm-fetch", daemon=True)
        worker.start()
        worker.join(timeout=OSM_FETCH_TIMEOUT_SECONDS)

        if worker.is_alive():
            raise TimeoutError(
                f"OSM fetch/conversion exceeded the {OSM_FETCH_TIMEOUT_SECONDS:.0f}s bound for this bounding box."
            )
        if "error" in outcome:
            raise outcome["error"]  # type: ignore[misc]
        return outcome["nodes"], outcome["edges"]  # type: ignore[return-value]

    def _add_travel_times(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Annotate `graph` edges with a `travel_time` (seconds) attribute.

        Uses OSMnx's own speed model (road-type/tag-based speeds via
        add_edge_speeds, then add_edge_travel_times), passing
        FALLBACK_AVERAGE_SPEED_KMH as OSMnx's per-edge fallback so every
        edge gets a speed even when its highway type has no known/tagged
        speed. If the speed/travel-time computation raises outright (e.g.
        malformed edge data), a flat FALLBACK_AVERAGE_SPEED_KMH estimate is
        applied later per-edge in _travel_time_seconds instead.
        """
        try:
            graph = ox.add_edge_speeds(graph, fallback=FALLBACK_AVERAGE_SPEED_KMH)
            graph = ox.add_edge_travel_times(graph)
            logger.info("Computed edge travel times using OSMnx's speed model.")
        except Exception:  # noqa: BLE001 - any speed-model failure falls back to a flat estimate.
            logger.warning(
                "OSMnx speed-based travel time computation failed; falling back to a flat %.0f km/h estimate.",
                FALLBACK_AVERAGE_SPEED_KMH,
                exc_info=True,
            )
        return graph

    @staticmethod
    def _build_nodes(graph: nx.MultiDiGraph) -> list[GraphNode]:
        """Convert OSMnx graph nodes (`y`=latitude, `x`=longitude) into GraphNode models."""
        return [
            GraphNode(id=int(node_id), latitude=float(data["y"]), longitude=float(data["x"]))
            for node_id, data in graph.nodes(data=True)
        ]

    @classmethod
    def _build_edges(cls, graph: nx.MultiDiGraph) -> list[GraphEdge]:
        """Convert OSMnx graph edges into GraphEdge models (one per parallel edge, if any)."""
        edges: list[GraphEdge] = []
        for source_id, target_id, data in graph.edges(data=True):
            distance_meters = float(data["length"])
            edges.append(
                GraphEdge(
                    source_node_id=int(source_id),
                    target_node_id=int(target_id),
                    distance_meters=distance_meters,
                    travel_time_seconds=cls._travel_time_seconds(data, distance_meters),
                )
            )
        return edges

    @staticmethod
    def _travel_time_seconds(edge_data: dict, distance_meters: float) -> float:
        """Return the edge's OSMnx-computed travel time, or a flat-speed estimate as fallback."""
        travel_time = edge_data.get("travel_time")
        if travel_time is not None:
            return float(travel_time)
        return distance_meters / _FALLBACK_SPEED_METERS_PER_SECOND

    @staticmethod
    def _validate_bbox(min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> None:
        for field_name, value in (
            ("min_lat", min_lat),
            ("max_lat", max_lat),
            ("min_lon", min_lon),
            ("max_lon", max_lon),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a number, got {value!r}.")
        if not -90 <= min_lat <= 90 or not -90 <= max_lat <= 90:
            raise ValueError(f"latitude bounds must be within [-90, 90], got min_lat={min_lat!r}, max_lat={max_lat!r}.")
        if not -180 <= min_lon <= 180 or not -180 <= max_lon <= 180:
            raise ValueError(
                f"longitude bounds must be within [-180, 180], got min_lon={min_lon!r}, max_lon={max_lon!r}."
            )
        if min_lat >= max_lat:
            raise ValueError(f"min_lat must be less than max_lat, got min_lat={min_lat!r}, max_lat={max_lat!r}.")
        if min_lon >= max_lon:
            raise ValueError(f"min_lon must be less than max_lon, got min_lon={min_lon!r}, max_lon={max_lon!r}.")


def _tile_bbox(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    tile_size_km: float,
    overlap_km: float,
) -> list[tuple[float, float, float, float]]:
    """Subdivide one bbox into a grid of (min_lat, max_lat, min_lon, max_lon)
    tiles, each `tile_size_km` on a side before `overlap_km` is added to
    every side (see DEFAULT_TILE_OVERLAP_KM for why the overlap exists).
    Pure/deterministic - no I/O, easy to unit test independent of any live
    OSM fetch. Always returns at least one tile (the original bbox,
    unmodified) when it's already tile_size_km or smaller on both axes.
    """
    if tile_size_km <= 0:
        raise ValueError(f"tile_size_km must be positive, got {tile_size_km!r}")
    if overlap_km < 0:
        raise ValueError(f"overlap_km must be non-negative, got {overlap_km!r}")

    # A tiny relative epsilon guards math.ceil() below against floating-point
    # noise from the degrees<->km round-trip: a span that is mathematically
    # EXACTLY N tile_size_km's can come out as e.g. 45.00000000000026 rather
    # than 45.0 (observed live), which would otherwise ceil() to N+1 tiles
    # on that axis alone - silently doubling total tile count when both axes
    # hit this independently (each axis's rounding direction differs).
    _EPSILON = 1e-9
    lat_span_km = (max_lat - min_lat) * _KM_PER_DEGREE
    lon_span_km = (max_lon - min_lon) * _KM_PER_DEGREE
    if lat_span_km <= tile_size_km and lon_span_km <= tile_size_km:
        return [(min_lat, max_lat, min_lon, max_lon)]

    tile_size_deg = tile_size_km / _KM_PER_DEGREE
    overlap_deg = overlap_km / _KM_PER_DEGREE
    lat_tile_count = max(1, math.ceil(lat_span_km / tile_size_km - _EPSILON))
    lon_tile_count = max(1, math.ceil(lon_span_km / tile_size_km - _EPSILON))

    tiles: list[tuple[float, float, float, float]] = []
    for lat_index in range(lat_tile_count):
        tile_min_lat = min_lat + lat_index * tile_size_deg
        tile_max_lat = min(max_lat, tile_min_lat + tile_size_deg)
        for lon_index in range(lon_tile_count):
            tile_min_lon = min_lon + lon_index * tile_size_deg
            tile_max_lon = min(max_lon, tile_min_lon + tile_size_deg)
            tiles.append(
                (
                    max(-90.0, tile_min_lat - overlap_deg),
                    min(90.0, tile_max_lat + overlap_deg),
                    max(-180.0, tile_min_lon - overlap_deg),
                    min(180.0, tile_max_lon + overlap_deg),
                )
            )
    return tiles

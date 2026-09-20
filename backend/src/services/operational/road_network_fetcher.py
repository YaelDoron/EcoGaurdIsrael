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

import logging
import threading

import networkx as nx
import osmnx as ox

from src.models.graph_edge import GraphEdge
from src.models.graph_node import GraphNode

logger = logging.getLogger(__name__)

NETWORK_TYPE = "drive"

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

# Used both as OSMnx's own per-edge speed fallback (passed into
# add_edge_speeds) and as a last-resort manual fallback in
# _travel_time_seconds if the speed/travel-time model raises outright for a
# given bounding box.
FALLBACK_AVERAGE_SPEED_KMH = 50.0
_FALLBACK_SPEED_METERS_PER_SECOND = FALLBACK_AVERAGE_SPEED_KMH / 3.6


class RoadNetworkFetcher:
    """Fetches an OSM driving network for a bounding box and converts it to domain models."""

    def fetch_network_in_bbox(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Return (nodes, edges) for the OSM driving network within a bounding box.

        Returns ([], []) - rather than raising - if OSMnx finds no network
        in the bbox or if the fetch fails for any other reason (Overpass
        timeouts, connectivity issues, an oversized/slow graph to process,
        etc.). A lazy-loading caller wants "no road data available" to
        degrade gracefully, not to blow up (or hang) the request it's
        serving.
        """
        self._validate_bbox(min_lat, max_lat, min_lon, max_lon)

        logger.info(
            "Fetching OSM %s network for bbox (min_lat=%.5f, max_lat=%.5f, min_lon=%.5f, max_lon=%.5f)...",
            NETWORK_TYPE,
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        )
        try:
            nodes, edges = self._fetch_and_convert_bounded(min_lat, max_lat, min_lon, max_lon)
        except Exception:  # noqa: BLE001 - no network found, timed out, or any Overpass/OSMnx failure, degrades to empty.
            logger.warning(
                "No OSM road network could be fetched for the requested bounding box.",
                exc_info=True,
            )
            return [], []

        logger.info("Fetched %d node(s) and %d edge(s) for the requested bounding box.", len(nodes), len(edges))
        return nodes, edges

    def _fetch_and_convert_bounded(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
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

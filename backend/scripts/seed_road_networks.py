"""Seed the road network graph from OpenStreetMap for each simulation location.

Uses osmnx to download the driving road network within a 20km radius of
each predefined simulation location (see src.simulation.simulation_locations)
and persists it via RoadNetworkRepository, so OperationalContextService has
real road data to query bounding boxes against instead of an empty
graph_nodes/graph_edges table.

Usage:
    python backend/scripts/seed_road_networks.py

Requires network access to the OpenStreetMap Overpass API (via osmnx) and a
configured DATABASE_URL. Fetching + saving can take anywhere from tens of
seconds to a few minutes per location, depending on the area's road density.
"""
from __future__ import annotations

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import osmnx as ox

from src.config.settings import settings
from src.database.connection import DatabaseConfigurationError, get_session, init_db
from src.models.graph_edge import GraphEdge
from src.models.graph_node import GraphNode
from src.repositories.road_network_repository import RoadNetworkRepository
from src.simulation.simulation_location import SimulationLocation
from src.simulation.simulation_locations import SIMULATION_LOCATIONS

logger = logging.getLogger(__name__)

SEARCH_RADIUS_METERS = 20_000
NETWORK_TYPE = "drive"

# Used only when OSMnx's own speed/travel-time model is unavailable or fails
# for a given location (e.g. missing regional speed data).
FALLBACK_AVERAGE_SPEED_KMH = 50.0
_FALLBACK_SPEED_METERS_PER_SECOND = FALLBACK_AVERAGE_SPEED_KMH / 3.6


def main() -> int:
    """Fetch and persist the road network for every predefined simulation location."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not settings.DATABASE_URL:
        raise DatabaseConfigurationError("DATABASE_URL is not configured; cannot seed road networks.")
    init_db()

    repository = RoadNetworkRepository()
    locations = list(SIMULATION_LOCATIONS.values())
    logger.info("Seeding road networks for %d simulation location(s).", len(locations))

    succeeded = 0
    failed = 0
    for location in locations:
        try:
            seed_location(location, repository)
            succeeded += 1
        except Exception:  # noqa: BLE001 - one location's OSM/DB failure must not abort the batch.
            failed += 1
            logger.exception("Failed to seed road network for %s.", location.name)

    logger.info("Road network seeding complete: %d succeeded, %d failed.", succeeded, failed)
    return 1 if failed and not succeeded else 0


def seed_location(location: SimulationLocation, repository: RoadNetworkRepository) -> None:
    """Fetch the driving road network around one simulation location and persist it."""
    logger.info(
        "Fetching OSM road network for %s (%.4f, %.4f), radius=%dm...",
        location.name,
        location.latitude,
        location.longitude,
        SEARCH_RADIUS_METERS,
    )
    graph = ox.graph_from_point(
        (location.latitude, location.longitude),
        dist=SEARCH_RADIUS_METERS,
        network_type=NETWORK_TYPE,
    )
    logger.info(
        "Fetched %d node(s) and %d edge(s) for %s.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        location.name,
    )

    graph = _add_travel_times(graph)
    nodes = _build_nodes(graph)
    edges = _build_edges(graph)

    logger.info("Saving %d node(s) and %d edge(s) for %s...", len(nodes), len(edges), location.name)
    with get_session() as db:
        repository.save_network(db, nodes, edges)
    logger.info("Saved road network for %s.", location.name)


def _add_travel_times(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Annotate `graph` edges with a `travel_time` (seconds) attribute.

    Prefers OSMnx's own speed model (road-type/tag-based speeds via
    add_edge_speeds, then add_edge_travel_times). Falls back to a flat
    FALLBACK_AVERAGE_SPEED_KMH assumption - applied later in
    _travel_time_seconds - if that model is unavailable or raises for this
    region's data.
    """
    try:
        graph = ox.add_edge_speeds(graph)
        graph = ox.add_edge_travel_times(graph)
        logger.info("Computed edge travel times using OSMnx's speed model.")
    except Exception:  # noqa: BLE001 - any speed-model failure falls back to a flat estimate.
        logger.warning(
            "OSMnx speed-based travel time computation failed; falling back to a flat %.0f km/h estimate.",
            FALLBACK_AVERAGE_SPEED_KMH,
            exc_info=True,
        )
    return graph


def _build_nodes(graph: nx.MultiDiGraph) -> list[GraphNode]:
    """Convert OSMnx graph nodes (`y`=latitude, `x`=longitude) into GraphNode models."""
    return [
        GraphNode(id=int(node_id), latitude=float(data["y"]), longitude=float(data["x"]))
        for node_id, data in graph.nodes(data=True)
    ]


def _build_edges(graph: nx.MultiDiGraph) -> list[GraphEdge]:
    """Convert OSMnx graph edges into GraphEdge models (one per parallel edge, if any)."""
    edges: list[GraphEdge] = []
    for source_id, target_id, data in graph.edges(data=True):
        distance_meters = float(data["length"])
        edges.append(
            GraphEdge(
                source_node_id=int(source_id),
                target_node_id=int(target_id),
                distance_meters=distance_meters,
                travel_time_seconds=_travel_time_seconds(data, distance_meters),
            )
        )
    return edges


def _travel_time_seconds(edge_data: dict, distance_meters: float) -> float:
    """Return the edge's OSMnx-computed travel time, or a flat-speed estimate as fallback."""
    travel_time = edge_data.get("travel_time")
    if travel_time is not None:
        return float(travel_time)
    return distance_meters / _FALLBACK_SPEED_METERS_PER_SECOND


if __name__ == "__main__":
    raise SystemExit(main())

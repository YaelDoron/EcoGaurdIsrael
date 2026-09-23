"""Regression tests for the Task A1.5 bounded OSM fetch fix.

Root cause: `ox.graph_from_bbox` is a synchronous call with no overall
wall-clock bound (osmnx's own per-HTTP-request timeout does not bound
tiling/rate-limit-wait behavior across an entire fetch), so a slow or
rate-limited Overpass response could block the calling thread indefinitely.
These tests fake `ox.graph_from_bbox` itself (no real network/Neon
involved) to prove the fetch now degrades to an empty result within the
configured bound instead of hanging, and that a fast/successful fetch is
unaffected.
"""
from __future__ import annotations

import time

import pytest
import requests
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError

from src.services.operational import road_network_fetcher as road_network_fetcher_module
from src.services.operational.road_network_fetcher import RoadNetworkFetcher, _is_transient_osm_error

BBOX = (32.7, 32.8, 35.0, 35.1)


def test_fast_successful_fetch_returns_nodes_and_edges(monkeypatch):
    class _FakeGraph:
        def number_of_nodes(self):
            return 1

        def number_of_edges(self):
            return 0

        def nodes(self, data=False):
            return [(1, {"y": 32.75, "x": 35.05})]

        def edges(self, data=False):
            return []

    monkeypatch.setattr(road_network_fetcher_module.ox, "graph_from_bbox", lambda **_: _FakeGraph())
    monkeypatch.setattr(
        road_network_fetcher_module.ox,
        "add_edge_speeds",
        lambda graph, fallback=None: graph,
    )
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_travel_times", lambda graph: graph)

    fetcher = RoadNetworkFetcher()
    nodes, edges = fetcher.fetch_network_in_bbox(*BBOX)

    assert len(nodes) == 1
    assert edges == []


def test_fetch_exceeding_bound_degrades_to_empty_instead_of_hanging(monkeypatch):
    """Reproduces the discovered stall class: a fetch that never returns
    (or takes far longer than the bound) must not block the caller.

    Retry disabled here (MAX_OSM_FETCH_ATTEMPTS=1): this test is about the
    per-attempt timeout bound itself, not the separate retry behavior (see
    the test_osm_fetch_retries_* tests below) - keeping it single-attempt
    keeps the timing assertion tight and this test focused."""
    monkeypatch.setattr(road_network_fetcher_module, "OSM_FETCH_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(road_network_fetcher_module, "MAX_OSM_FETCH_ATTEMPTS", 1)

    def _hangs_forever(**_kwargs):
        time.sleep(5.0)  # far longer than the 0.2s bound this test installs
        raise AssertionError("should never reach this point in the test")

    monkeypatch.setattr(road_network_fetcher_module.ox, "graph_from_bbox", _hangs_forever)

    fetcher = RoadNetworkFetcher()
    started = time.monotonic()
    nodes, edges = fetcher.fetch_network_in_bbox(*BBOX)
    elapsed = time.monotonic() - started

    assert nodes == []
    assert edges == []
    assert elapsed < 2.0, f"fetch_network_in_bbox blocked for {elapsed:.2f}s, expected it to bail out near 0.2s"


def test_fetch_raising_exception_still_degrades_to_empty(monkeypatch):
    """Existing contract (any Overpass/OSMnx exception -> empty) must survive
    the bounded-thread rewrite unchanged."""

    def _raises(**_kwargs):
        raise RuntimeError("simulated Overpass failure")

    monkeypatch.setattr(road_network_fetcher_module.ox, "graph_from_bbox", _raises)

    fetcher = RoadNetworkFetcher()
    nodes, edges = fetcher.fetch_network_in_bbox(*BBOX)

    assert nodes == []
    assert edges == []


def test_abandoned_fetch_thread_is_a_daemon_and_does_not_block_exit(monkeypatch):
    """The thread backing a timed-out fetch must be a daemon thread, so an
    abandoned live Overpass call can never prevent the CLI process from
    exiting once the caller has moved on. Retry disabled (single attempt)
    to keep this test about the daemon flag, not retry count."""
    monkeypatch.setattr(road_network_fetcher_module, "OSM_FETCH_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(road_network_fetcher_module, "MAX_OSM_FETCH_ATTEMPTS", 1)
    seen_daemon_flags = []

    real_thread_cls = road_network_fetcher_module.threading.Thread

    class _RecordingThread(real_thread_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen_daemon_flags.append(self.daemon)

    monkeypatch.setattr(road_network_fetcher_module.threading, "Thread", _RecordingThread)
    monkeypatch.setattr(
        road_network_fetcher_module.ox,
        "graph_from_bbox",
        lambda **_: time.sleep(5.0),
    )

    fetcher = RoadNetworkFetcher()
    fetcher.fetch_network_in_bbox(*BBOX)

    assert seen_daemon_flags == [True]


# ---------------------------------------------------------------------------
# Retry (live incident: a station-micro fetch losing the race against
# Overpass throttling on EVERY cycle - "OSM fetch/conversion exceeded the
# 60s bound"). Only genuinely transient failures are retried; a real "no
# data here" or unexpected error fails fast, same as before.
# ---------------------------------------------------------------------------


class TestIsTransientOsmError:
    def test_timeout_error_is_transient(self):
        assert _is_transient_osm_error(TimeoutError("bound exceeded")) is True

    def test_requests_timeout_is_transient(self):
        assert _is_transient_osm_error(requests.exceptions.Timeout()) is True

    def test_requests_connection_error_is_transient(self):
        assert _is_transient_osm_error(requests.exceptions.ConnectionError()) is True

    def test_response_status_code_error_is_transient(self):
        assert _is_transient_osm_error(ResponseStatusCodeError("HTTP 429")) is True

    def test_insufficient_response_error_is_not_transient(self):
        """A genuinely empty/too-few-results Overpass response is a real
        answer, not a failure - retrying it 3 times would only add latency
        for a certain, unchanging outcome."""
        assert _is_transient_osm_error(InsufficientResponseError("no data")) is False

    def test_unexpected_error_is_not_transient(self):
        assert _is_transient_osm_error(ValueError("some unrelated bug")) is False


def test_osm_fetch_retries_a_transient_error_and_succeeds(monkeypatch):
    monkeypatch.setattr(road_network_fetcher_module, "_OSM_FETCH_RETRY_BASE_DELAY_SECONDS", 0.01)
    call_count = 0

    class _FakeGraph:
        def nodes(self, data=False):
            return [(1, {"y": 32.75, "x": 35.05})]

        def edges(self, data=False):
            return []

    def _fails_twice_then_succeeds(**_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise requests.exceptions.ConnectionError("simulated transient network failure")
        return _FakeGraph()

    monkeypatch.setattr(road_network_fetcher_module.ox, "graph_from_bbox", _fails_twice_then_succeeds)
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_speeds", lambda graph, fallback=None: graph)
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_travel_times", lambda graph: graph)

    fetcher = RoadNetworkFetcher()
    nodes, edges = fetcher.fetch_network_in_bbox(*BBOX)

    assert call_count == 3
    assert len(nodes) == 1
    assert edges == []


def test_osm_fetch_exhausts_retries_on_persistent_transient_failure_and_degrades_to_empty(monkeypatch):
    monkeypatch.setattr(road_network_fetcher_module, "_OSM_FETCH_RETRY_BASE_DELAY_SECONDS", 0.01)
    call_count = 0

    def _always_throttled(**_kwargs):
        nonlocal call_count
        call_count += 1
        raise ResponseStatusCodeError("HTTP 429 Too Many Requests")

    monkeypatch.setattr(road_network_fetcher_module.ox, "graph_from_bbox", _always_throttled)

    fetcher = RoadNetworkFetcher()
    nodes, edges = fetcher.fetch_network_in_bbox(*BBOX)

    assert call_count == road_network_fetcher_module.MAX_OSM_FETCH_ATTEMPTS
    assert nodes == []
    assert edges == []


def test_osm_fetch_does_not_retry_a_non_transient_error(monkeypatch):
    """A real bug/data problem (not throttling) must fail fast on the first
    attempt - retrying it 3 times would only mask/delay the diagnostic
    signal for an outcome retries cannot change."""
    call_count = 0

    def _raises_unexpectedly(**_kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("simulated unexpected non-transient failure")

    monkeypatch.setattr(road_network_fetcher_module.ox, "graph_from_bbox", _raises_unexpectedly)

    fetcher = RoadNetworkFetcher()
    nodes, edges = fetcher.fetch_network_in_bbox(*BBOX)

    assert call_count == 1
    assert nodes == []
    assert edges == []


def test_osm_fetch_does_not_retry_a_genuinely_empty_result(monkeypatch):
    """InsufficientResponseError means Overpass answered and there is
    genuinely no road data here - not a failure to power through."""
    call_count = 0

    def _genuinely_empty(**_kwargs):
        nonlocal call_count
        call_count += 1
        raise InsufficientResponseError("found no graph nodes within the requested bbox")

    monkeypatch.setattr(road_network_fetcher_module.ox, "graph_from_bbox", _genuinely_empty)

    fetcher = RoadNetworkFetcher()
    nodes, edges = fetcher.fetch_network_in_bbox(*BBOX)

    assert call_count == 1
    assert nodes == []
    assert edges == []


# ---------------------------------------------------------------------------
# Bounding-box tiling (chunking): expands the safely-fetchable search radius
# by splitting one large bbox into several small, independently-reliable
# Overpass requests instead of one large one that reliably times out.
# ---------------------------------------------------------------------------


class TestTileBbox:
    """_tile_bbox is a pure function - no fetch, no threading - tested directly."""

    def test_a_bbox_no_larger_than_one_tile_is_returned_unchanged(self):
        from src.services.operational.road_network_fetcher import _tile_bbox

        tiles = _tile_bbox(32.70, 32.75, 35.00, 35.05, tile_size_km=15.0, overlap_km=2.0)

        assert tiles == [(32.70, 32.75, 35.00, 35.05)]

    def test_a_larger_bbox_is_split_into_a_grid_covering_the_whole_area(self):
        from src.services.operational.road_network_fetcher import _tile_bbox, _KM_PER_DEGREE

        # ~45km x ~45km at tile_size_km=15 -> 3x3 = 9 tiles.
        span_deg = 45.0 / _KM_PER_DEGREE
        tiles = _tile_bbox(30.0, 30.0 + span_deg, 34.0, 34.0 + span_deg, tile_size_km=15.0, overlap_km=0.0)

        assert len(tiles) == 9
        # Every corner of the original bbox must be covered by some tile.
        min_lat = min(tile[0] for tile in tiles)
        max_lat = max(tile[1] for tile in tiles)
        min_lon = min(tile[2] for tile in tiles)
        max_lon = max(tile[3] for tile in tiles)
        assert min_lat <= 30.0
        assert max_lat >= 30.0 + span_deg
        assert min_lon <= 34.0
        assert max_lon >= 34.0 + span_deg

    def test_overlap_expands_every_tile_on_all_sides(self):
        from src.services.operational.road_network_fetcher import _tile_bbox, _KM_PER_DEGREE

        span_deg = 45.0 / _KM_PER_DEGREE
        no_overlap = _tile_bbox(30.0, 30.0 + span_deg, 34.0, 34.0 + span_deg, tile_size_km=15.0, overlap_km=0.0)
        with_overlap = _tile_bbox(30.0, 30.0 + span_deg, 34.0, 34.0 + span_deg, tile_size_km=15.0, overlap_km=2.0)

        assert len(no_overlap) == len(with_overlap)
        overlap_deg = 2.0 / _KM_PER_DEGREE
        for plain, overlapped in zip(sorted(no_overlap), sorted(with_overlap)):
            assert overlapped[0] == pytest.approx(plain[0] - overlap_deg)
            assert overlapped[1] == pytest.approx(plain[1] + overlap_deg)
            assert overlapped[2] == pytest.approx(plain[2] - overlap_deg)
            assert overlapped[3] == pytest.approx(plain[3] + overlap_deg)

    def test_adjacent_tiles_overlap_enough_to_share_a_boundary_strip(self):
        """The actual point of overlap: two horizontally adjacent tiles'
        longitude ranges must genuinely intersect, so a road exactly on
        their shared seam falls inside both tiles' queries, not neither."""
        from src.services.operational.road_network_fetcher import _tile_bbox, _KM_PER_DEGREE

        span_deg = 30.0 / _KM_PER_DEGREE  # 2x1 grid at 15km tiles
        tiles = sorted(_tile_bbox(30.0, 30.0 + (15.0 / _KM_PER_DEGREE), 34.0, 34.0 + span_deg, tile_size_km=15.0, overlap_km=2.0))

        assert len(tiles) == 2
        west_tile, east_tile = tiles
        assert west_tile[3] > east_tile[2]  # west tile's max_lon overlaps east tile's min_lon

    def test_rejects_non_positive_tile_size(self):
        from src.services.operational.road_network_fetcher import _tile_bbox

        with pytest.raises(ValueError):
            _tile_bbox(32.0, 32.1, 35.0, 35.1, tile_size_km=0.0, overlap_km=1.0)

    def test_rejects_negative_overlap(self):
        from src.services.operational.road_network_fetcher import _tile_bbox

        with pytest.raises(ValueError):
            _tile_bbox(32.0, 32.1, 35.0, 35.1, tile_size_km=15.0, overlap_km=-1.0)


def test_fetch_network_in_bbox_tiled_merges_distinct_nodes_from_each_tile(monkeypatch):
    """Each tile's fake graph returns a DIFFERENT node/edge, keyed off which
    bbox it was called with - proves the tiled fetch actually issues one
    real per-tile fetch and merges all of their results, not just the
    first or last one."""
    from src.services.operational.road_network_fetcher import _KM_PER_DEGREE

    call_bboxes = []

    class _FakeGraph:
        def __init__(self, node_id, lat, lon):
            self._node_id = node_id
            self._lat = lat
            self._lon = lon

        def nodes(self, data=False):
            return [(self._node_id, {"y": self._lat, "x": self._lon})]

        def edges(self, data=False):
            return []

    def _fake_graph_from_bbox(bbox, network_type):
        min_lon, min_lat, max_lon, max_lat = bbox
        call_bboxes.append((min_lat, max_lat, min_lon, max_lon))
        # One distinct node per tile, placed at its own center, so each
        # tile's contribution is individually identifiable in the merge.
        mid_lat, mid_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
        node_id = int((mid_lat * 1000) * 100000 + (mid_lon * 1000))
        return _FakeGraph(node_id, mid_lat, mid_lon)

    monkeypatch.setattr(road_network_fetcher_module.ox, "graph_from_bbox", _fake_graph_from_bbox)
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_speeds", lambda graph, fallback=None: graph)
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_travel_times", lambda graph: graph)

    fetcher = RoadNetworkFetcher()
    span_deg = 45.0 / _KM_PER_DEGREE  # 3x3 = 9 tiles at 15km
    nodes, edges = fetcher.fetch_network_in_bbox_tiled(
        30.0, 30.0 + span_deg, 34.0, 34.0 + span_deg, tile_size_km=15.0, tile_overlap_km=0.0
    )

    assert len(call_bboxes) == 9  # one real fetch per tile
    assert len(nodes) == 9  # every tile's distinct node survived the merge
    assert len({node.id for node in nodes}) == 9  # no accidental collision/drop


def test_fetch_network_in_bbox_tiled_deduplicates_a_node_seen_in_two_overlapping_tiles(monkeypatch):
    """The overlap margin means the SAME real intersection can legitimately
    come back from more than one tile's query - the merge must keep exactly
    one copy (by node id), matching save_network's own upsert-by-id
    semantics, not double-count it."""
    from src.services.operational.road_network_fetcher import _KM_PER_DEGREE

    SHARED_NODE_ID = 777

    class _FakeGraph:
        def nodes(self, data=False):
            return [(SHARED_NODE_ID, {"y": 30.0, "x": 34.0})]

        def edges(self, data=False):
            return []

    monkeypatch.setattr(road_network_fetcher_module.ox, "graph_from_bbox", lambda bbox, network_type: _FakeGraph())
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_speeds", lambda graph, fallback=None: graph)
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_travel_times", lambda graph: graph)

    fetcher = RoadNetworkFetcher()
    span_deg = 30.0 / _KM_PER_DEGREE  # 2 tiles at 15km, both "finding" the same shared node
    nodes, _edges = fetcher.fetch_network_in_bbox_tiled(
        30.0, 30.0 + (15.0 / _KM_PER_DEGREE), 34.0, 34.0 + span_deg, tile_size_km=15.0, tile_overlap_km=2.0
    )

    assert [node.id for node in nodes] == [SHARED_NODE_ID]


def test_fetch_network_in_bbox_tiled_fetches_tiles_concurrently(monkeypatch):
    """The central performance claim for tiling itself: N tiles must not
    take N x (one tile's fetch time) - their fetch windows must overlap."""
    import time
    from src.services.operational.road_network_fetcher import _KM_PER_DEGREE

    call_windows = []

    class _FakeGraph:
        def nodes(self, data=False):
            return []

        def edges(self, data=False):
            return []

    def _slow_fetch(bbox, network_type):
        start = time.perf_counter()
        time.sleep(0.3)
        call_windows.append((start, time.perf_counter()))
        return _FakeGraph()

    monkeypatch.setattr(road_network_fetcher_module.ox, "graph_from_bbox", _slow_fetch)
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_speeds", lambda graph, fallback=None: graph)
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_travel_times", lambda graph: graph)

    fetcher = RoadNetworkFetcher()
    span_deg = 30.0 / _KM_PER_DEGREE  # 2x2 = 4 tiles at 15km
    start = time.perf_counter()
    fetcher.fetch_network_in_bbox_tiled(
        30.0, 30.0 + span_deg, 34.0, 34.0 + span_deg, tile_size_km=15.0, tile_overlap_km=0.0, max_concurrent_tiles=4
    )
    elapsed = time.perf_counter() - start

    assert len(call_windows) == 4
    # 4 tiles at 0.3s each would take >=1.2s strictly sequential; concurrently, close to 0.3s.
    assert elapsed < 1.0, f"tiled fetch took {elapsed:.2f}s - looks sequential, not concurrent"


def test_fetch_network_in_bbox_tiled_survives_one_failing_tile(monkeypatch):
    from src.services.operational.road_network_fetcher import _KM_PER_DEGREE

    class _FakeGraph:
        def nodes(self, data=False):
            return [(1, {"y": 30.0, "x": 34.0})]

        def edges(self, data=False):
            return []

    call_count = 0

    def _fails_on_first_call(bbox, network_type):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated Overpass failure for this tile")
        return _FakeGraph()

    monkeypatch.setattr(road_network_fetcher_module.ox, "graph_from_bbox", _fails_on_first_call)
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_speeds", lambda graph, fallback=None: graph)
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_travel_times", lambda graph: graph)

    fetcher = RoadNetworkFetcher()
    span_deg = 30.0 / _KM_PER_DEGREE
    nodes, _edges = fetcher.fetch_network_in_bbox_tiled(
        30.0, 30.0 + span_deg, 34.0, 34.0 + span_deg, tile_size_km=15.0, tile_overlap_km=0.0
    )

    # The failing tile contributes nothing (fetch_network_in_bbox's own
    # per-tile try/except already degrades it to ([], [])); the other
    # (identical, by construction of this fake) tiles still merge in.
    assert len(nodes) == 1


# ---------------------------------------------------------------------------
# Graph fidelity: an optional custom_filter overrides NETWORK_TYPE's
# standard "drive" preset - used only for the expensive wide/tiled fetch
# path (see GlobalPlanningInputBuilder's Step 3), not the fast local one.
# ---------------------------------------------------------------------------


def test_default_fetch_uses_the_standard_network_type_not_a_custom_filter(monkeypatch):
    calls = []

    class _FakeGraph:
        def nodes(self, data=False):
            return []

        def edges(self, data=False):
            return []

    def _capture(**kwargs):
        calls.append(kwargs)
        return _FakeGraph()

    monkeypatch.setattr(road_network_fetcher_module.ox, "graph_from_bbox", _capture)
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_speeds", lambda graph, fallback=None: graph)
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_travel_times", lambda graph: graph)

    fetcher = RoadNetworkFetcher()
    fetcher.fetch_network_in_bbox(*BBOX)

    assert len(calls) == 1
    assert calls[0].get("network_type") == road_network_fetcher_module.NETWORK_TYPE
    assert "custom_filter" not in calls[0]


def test_custom_filter_overrides_network_type_when_given(monkeypatch):
    from src.services.operational.road_network_fetcher import GRAPH_FIDELITY_CUSTOM_FILTER

    calls = []

    class _FakeGraph:
        def nodes(self, data=False):
            return []

        def edges(self, data=False):
            return []

    def _capture(**kwargs):
        calls.append(kwargs)
        return _FakeGraph()

    monkeypatch.setattr(road_network_fetcher_module.ox, "graph_from_bbox", _capture)
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_speeds", lambda graph, fallback=None: graph)
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_travel_times", lambda graph: graph)

    fetcher = RoadNetworkFetcher()
    fetcher.fetch_network_in_bbox(*BBOX, custom_filter=GRAPH_FIDELITY_CUSTOM_FILTER)

    assert len(calls) == 1
    assert calls[0].get("custom_filter") == GRAPH_FIDELITY_CUSTOM_FILTER
    assert "network_type" not in calls[0]


def test_graph_fidelity_filter_does_not_exclude_secondary_or_tertiary_roads():
    """Regression guard for the premise: secondary/tertiary were never
    excluded by the standard filter either, but this locks in that both the
    standard and relaxed filters keep them, and that the relaxed filter
    specifically ALSO stops excluding plain highway=service (while still
    excluding actual driveway/parking/alley/private sub-types)."""
    from src.services.operational.road_network_fetcher import GRAPH_FIDELITY_CUSTOM_FILTER

    assert "secondary" not in GRAPH_FIDELITY_CUSTOM_FILTER
    assert "tertiary" not in GRAPH_FIDELITY_CUSTOM_FILTER
    # "service" as a bare highway-exclusion token must be gone...
    highway_exclusions = GRAPH_FIDELITY_CUSTOM_FILTER.split('"highway"!~"')[1].split('"]')[0].split("|")
    assert "service" not in highway_exclusions
    assert "services" not in highway_exclusions
    # ...but the separate, more targeted service-subtype exclusion remains.
    assert "driveway" in GRAPH_FIDELITY_CUSTOM_FILTER
    assert "parking_aisle" in GRAPH_FIDELITY_CUSTOM_FILTER


def test_tiled_fetch_forwards_custom_filter_to_every_tile(monkeypatch):
    from src.services.operational.road_network_fetcher import _KM_PER_DEGREE, GRAPH_FIDELITY_CUSTOM_FILTER

    calls = []

    class _FakeGraph:
        def nodes(self, data=False):
            return []

        def edges(self, data=False):
            return []

    def _capture(**kwargs):
        calls.append(kwargs)
        return _FakeGraph()

    monkeypatch.setattr(road_network_fetcher_module.ox, "graph_from_bbox", _capture)
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_speeds", lambda graph, fallback=None: graph)
    monkeypatch.setattr(road_network_fetcher_module.ox, "add_edge_travel_times", lambda graph: graph)

    fetcher = RoadNetworkFetcher()
    span_deg = 30.0 / _KM_PER_DEGREE  # 2x2 = 4 tiles at 15km
    fetcher.fetch_network_in_bbox_tiled(
        30.0, 30.0 + span_deg, 34.0, 34.0 + span_deg,
        tile_size_km=15.0, tile_overlap_km=0.0, custom_filter=GRAPH_FIDELITY_CUSTOM_FILTER,
    )

    assert len(calls) == 4
    assert all(call.get("custom_filter") == GRAPH_FIDELITY_CUSTOM_FILTER for call in calls)

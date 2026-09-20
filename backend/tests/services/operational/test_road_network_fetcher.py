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

from src.services.operational import road_network_fetcher as road_network_fetcher_module
from src.services.operational.road_network_fetcher import RoadNetworkFetcher

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
    (or takes far longer than the bound) must not block the caller."""
    monkeypatch.setattr(road_network_fetcher_module, "OSM_FETCH_TIMEOUT_SECONDS", 0.2)

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
    exiting once the caller has moved on."""
    monkeypatch.setattr(road_network_fetcher_module, "OSM_FETCH_TIMEOUT_SECONDS", 0.1)
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

"""Tests for GlobalRouteOption and GlobalRouteMatrix (Stage 3 of the Global
Multi-Incident Optimizer refactor)."""
from __future__ import annotations

import pytest

from src.models.global_route_matrix import GlobalRouteMatrix
from src.models.global_route_option import GlobalRouteOption


def make_option(**overrides) -> GlobalRouteOption:
    defaults = dict(
        resource_id="R1",
        fire_event_id=1,
        response_target_id=10,
        eta_seconds=120.0,
        route_distance_meters=1500.0,
        node_path=(1, 2, 3),
    )
    defaults.update(overrides)
    return GlobalRouteOption(**defaults)


# ---------------------------------------------------------------------------
# GlobalRouteOption
# ---------------------------------------------------------------------------


def test_valid_route_option():
    option = make_option()
    assert option.eta_seconds == 120.0
    assert option.route_distance_meters == 1500.0


def test_rejects_negative_eta():
    with pytest.raises(ValueError):
        make_option(eta_seconds=-1.0)


def test_rejects_negative_distance():
    with pytest.raises(ValueError):
        make_option(route_distance_meters=-1.0)


def test_rejects_empty_node_path():
    with pytest.raises(ValueError):
        make_option(node_path=())


def test_rejects_invalid_ids():
    with pytest.raises(ValueError):
        make_option(resource_id="")
    with pytest.raises(ValueError):
        make_option(fire_event_id=0)
    with pytest.raises(ValueError):
        make_option(response_target_id=0)


# ---------------------------------------------------------------------------
# GlobalRouteMatrix
# ---------------------------------------------------------------------------


def test_matrix_lookup_returns_the_matching_option():
    option = make_option(resource_id="R1", response_target_id=10)
    matrix = GlobalRouteMatrix(options=(option,))

    assert matrix.get("R1", 10) == option


def test_matrix_lookup_returns_none_for_unreachable_pair():
    matrix = GlobalRouteMatrix(options=(make_option(resource_id="R1", response_target_id=10),))

    assert matrix.get("R1", 999) is None
    assert matrix.get("R2", 10) is None


def test_matrix_rejects_duplicate_pair():
    with pytest.raises(ValueError):
        GlobalRouteMatrix(
            options=(
                make_option(resource_id="R1", response_target_id=10, eta_seconds=100.0),
                make_option(resource_id="R1", response_target_id=10, eta_seconds=200.0),
            )
        )


def test_matrix_iteration_order_is_deterministic_regardless_of_input_order():
    a = make_option(resource_id="R2", response_target_id=10)
    b = make_option(resource_id="R1", response_target_id=20)
    c = make_option(resource_id="R1", response_target_id=10)

    forward = GlobalRouteMatrix(options=(a, b, c))
    backward = GlobalRouteMatrix(options=(c, b, a))

    assert list(forward) == list(backward)
    assert [(o.resource_id, o.response_target_id) for o in forward] == [
        ("R1", 10),
        ("R1", 20),
        ("R2", 10),
    ]


def test_matrix_len_and_contains():
    matrix = GlobalRouteMatrix(options=(make_option(resource_id="R1", response_target_id=10),))
    assert len(matrix) == 1
    assert ("R1", 10) in matrix
    assert ("R2", 10) not in matrix


def test_empty_matrix_is_valid():
    matrix = GlobalRouteMatrix(options=())
    assert len(matrix) == 0
    assert matrix.get("R1", 10) is None

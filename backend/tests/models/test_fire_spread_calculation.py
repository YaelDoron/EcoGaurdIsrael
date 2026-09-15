"""Tests for the pure wildfire-spread calculation result model."""
from __future__ import annotations

import pytest

from src.models.fire_spread_calculation import FireSpreadCalculation
from src.models.fire_spread_prediction import FireSpreadPredictionCell


def make_cell(**overrides) -> FireSpreadPredictionCell:
    defaults = dict(
        latitude=32.75,
        longitude=35.0,
        spread_probability=0.6,
        spread_risk_score=60.0,
        reached_step=1,
        reached_minutes=5,
    )
    defaults.update(overrides)
    return FireSpreadPredictionCell(**defaults)


def make_calculation(**overrides) -> FireSpreadCalculation:
    defaults = dict(
        cells=(make_cell(),),
        horizon_minutes=30,
        methodology="ECOGUARD_PROPAGATOR_CA",
        methodology_version="1.0",
    )
    defaults.update(overrides)
    return FireSpreadCalculation(**defaults)


def test_valid_calculation_with_cells():
    calculation = make_calculation()
    assert len(calculation.cells) == 1


def test_valid_calculation_with_no_cells():
    """A legitimate 'no predicted spread' result: not an error."""
    calculation = make_calculation(cells=())
    assert calculation.cells == ()


@pytest.mark.parametrize("horizon_minutes", [0, 15, 45, 90, True])
def test_invalid_horizon_rejected(horizon_minutes):
    with pytest.raises(ValueError):
        make_calculation(horizon_minutes=horizon_minutes, cells=())


@pytest.mark.parametrize("methodology", ["", "   ", None])
def test_empty_methodology_rejected(methodology):
    with pytest.raises(ValueError):
        make_calculation(methodology=methodology)


@pytest.mark.parametrize("methodology_version", ["", "   ", None])
def test_empty_methodology_version_rejected(methodology_version):
    with pytest.raises(ValueError):
        make_calculation(methodology_version=methodology_version)


def test_non_tuple_cells_rejected():
    with pytest.raises(ValueError):
        make_calculation(cells=[make_cell()])


def test_cells_with_wrong_type_rejected():
    with pytest.raises(ValueError):
        make_calculation(cells=(make_cell(), "not-a-cell"))


def test_reached_step_exceeding_horizon_rejected():
    # 30 minutes / 5-minute step = 6 max steps; step 7 cannot belong to a 30-min result.
    with pytest.raises(ValueError):
        make_calculation(
            horizon_minutes=30,
            cells=(make_cell(reached_step=7, reached_minutes=35),),
        )


def test_reached_step_at_horizon_boundary_accepted():
    calculation = make_calculation(
        horizon_minutes=30,
        cells=(make_cell(reached_step=6, reached_minutes=30),),
    )
    assert calculation.cells[0].reached_step == 6


def test_cells_out_of_order_rejected():
    first = make_cell(reached_step=2, reached_minutes=10, latitude=32.75, longitude=35.0)
    second = make_cell(reached_step=1, reached_minutes=5, latitude=32.75, longitude=35.0)
    with pytest.raises(ValueError):
        make_calculation(horizon_minutes=30, cells=(first, second))


def test_cells_in_order_accepted():
    first = make_cell(reached_step=1, reached_minutes=5, latitude=32.74, longitude=35.0)
    second = make_cell(reached_step=1, reached_minutes=5, latitude=32.75, longitude=35.0)
    third = make_cell(reached_step=2, reached_minutes=10, latitude=32.70, longitude=35.0)
    calculation = make_calculation(horizon_minutes=30, cells=(first, second, third))
    assert len(calculation.cells) == 3

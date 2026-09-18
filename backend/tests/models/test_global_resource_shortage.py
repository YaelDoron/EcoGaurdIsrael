"""Tests for GlobalResourceShortage (Stage 5 of the Global Multi-Incident
Optimizer refactor, Tasks 18/20/21)."""
from __future__ import annotations

import pytest

from src.models.global_resource_shortage import GlobalResourceShortage


def make_shortage(**overrides) -> GlobalResourceShortage:
    defaults = dict(
        total_required=3,
        total_desired=5,
        total_assigned=4,
        unmet_required=0,
        unmet_desired=1,
        candidate_assignable_resource_count=4,
        committed_resource_count=1,
        unavailable_resource_count=0,
    )
    defaults.update(overrides)
    return GlobalResourceShortage(**defaults)


def test_valid_shortage_round_trips_and_flags_are_correct():
    shortage = make_shortage()
    assert shortage.has_shortage is True
    assert shortage.insufficient_supply is True


def test_no_shortage_when_nothing_unmet():
    shortage = make_shortage(total_assigned=5, unmet_required=0, unmet_desired=0, candidate_assignable_resource_count=5)
    assert shortage.has_shortage is False
    assert shortage.insufficient_supply is False


def test_rejects_total_desired_below_total_required():
    with pytest.raises(ValueError):
        make_shortage(total_required=6, total_desired=5)


def test_rejects_total_assigned_above_total_desired():
    with pytest.raises(ValueError):
        make_shortage(total_assigned=6, total_desired=5)


def test_rejects_unmet_desired_below_unmet_required():
    with pytest.raises(ValueError):
        make_shortage(unmet_required=2, unmet_desired=1)


def test_rejects_negative_field():
    with pytest.raises(ValueError):
        make_shortage(committed_resource_count=-1)


def test_insufficient_supply_false_when_assignable_meets_desired():
    shortage = make_shortage(candidate_assignable_resource_count=5)
    assert shortage.insufficient_supply is False


# ---------------------------------------------------------------------------
# Stage 6 - locked_resources_preserved relaxes the total_assigned ceiling
# ---------------------------------------------------------------------------


def test_total_assigned_may_exceed_total_desired_by_locked_resources_preserved():
    shortage = make_shortage(
        total_required=1, total_desired=2, total_assigned=4, unmet_required=0, unmet_desired=0,
        candidate_assignable_resource_count=4, locked_resources_preserved=2,
    )
    assert shortage.total_assigned == 4


def test_rejects_total_assigned_beyond_desired_plus_locked_resources_preserved():
    with pytest.raises(ValueError):
        make_shortage(
            total_required=1, total_desired=2, total_assigned=4, unmet_required=0, unmet_desired=0,
            candidate_assignable_resource_count=4, locked_resources_preserved=1,
        )


def test_locked_resources_preserved_defaults_to_zero():
    shortage = make_shortage()
    assert shortage.locked_resources_preserved == 0

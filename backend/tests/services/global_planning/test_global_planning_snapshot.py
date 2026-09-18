"""Tests for GlobalPlanningSnapshot (Stage 2 of the Global Multi-Incident
Optimizer refactor, Tasks 6-7)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.models.resource_commitment import ResourceCommitment
from src.services.global_planning.global_planning_snapshot import GlobalPlanningSnapshot

AS_OF = datetime(2026, 9, 20, 8, 0, tzinfo=timezone.utc)
COMMITTED_AT = AS_OF - timedelta(minutes=5)


def _commitment(resource_id: str, fire_event_id: int, response_plan_id: int) -> ResourceCommitment:
    return ResourceCommitment(
        resource_id=resource_id, fire_event_id=fire_event_id, response_plan_id=response_plan_id,
        committed_at=COMMITTED_AT,
    )


def _make_snapshot(**overrides) -> GlobalPlanningSnapshot:
    defaults = dict(
        as_of=AS_OF,
        active_fire_event_ids=(1, 2),
        resource_commitments=(_commitment("R1", 1, 10), _commitment("R2", 2, 20)),
        per_event_effective_state_fingerprints={1: "a" * 64, 2: "b" * 64},
        methodology="legacy_per_event_orchestration",
        methodology_version="1.0",
    )
    defaults.update(overrides)
    return GlobalPlanningSnapshot(**defaults)


def test_fingerprint_is_stable_for_identical_content():
    first = _make_snapshot()
    second = _make_snapshot()

    assert first.fingerprint() == second.fingerprint()


def test_fingerprint_is_independent_of_active_fire_event_ids_order():
    ordered = _make_snapshot(active_fire_event_ids=(1, 2))
    reversed_order = _make_snapshot(active_fire_event_ids=(2, 1))

    assert ordered.fingerprint() == reversed_order.fingerprint()


def test_fingerprint_is_independent_of_resource_commitments_order():
    forward = _make_snapshot(resource_commitments=(_commitment("R1", 1, 10), _commitment("R2", 2, 20)))
    backward = _make_snapshot(resource_commitments=(_commitment("R2", 2, 20), _commitment("R1", 1, 10)))

    assert forward.fingerprint() == backward.fingerprint()


def test_fingerprint_excludes_as_of():
    earlier = _make_snapshot(as_of=AS_OF)
    later = _make_snapshot(as_of=AS_OF + timedelta(hours=1))

    assert earlier.fingerprint() == later.fingerprint()


def test_fingerprint_changes_when_active_fire_event_ids_change():
    baseline = _make_snapshot()
    changed = _make_snapshot(
        active_fire_event_ids=(1, 2, 3),
        per_event_effective_state_fingerprints={1: "a" * 64, 2: "b" * 64, 3: "c" * 64},
    )

    assert baseline.fingerprint() != changed.fingerprint()


def test_fingerprint_changes_when_a_commitment_moves_to_a_different_plan():
    baseline = _make_snapshot()
    changed = _make_snapshot(resource_commitments=(_commitment("R1", 1, 999), _commitment("R2", 2, 20)))

    assert baseline.fingerprint() != changed.fingerprint()


def test_fingerprint_changes_when_a_local_fingerprint_changes():
    baseline = _make_snapshot()
    changed = _make_snapshot(per_event_effective_state_fingerprints={1: "z" * 64, 2: "b" * 64})

    assert baseline.fingerprint() != changed.fingerprint()


def test_fingerprint_changes_when_methodology_version_changes():
    baseline = _make_snapshot()
    changed = _make_snapshot(methodology_version="2.0")

    assert baseline.fingerprint() != changed.fingerprint()


def test_rejects_duplicate_active_fire_event_ids():
    with pytest.raises(ValueError):
        _make_snapshot(active_fire_event_ids=(1, 1))


def test_rejects_fingerprint_key_not_in_active_ids():
    with pytest.raises(ValueError):
        _make_snapshot(per_event_effective_state_fingerprints={1: "a" * 64, 99: "b" * 64})


def test_rejects_naive_as_of():
    with pytest.raises(ValueError):
        _make_snapshot(as_of=datetime(2026, 1, 1))

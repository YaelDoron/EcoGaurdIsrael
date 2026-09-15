"""Tests for SimulationResponseTargetCoordinator trigger policy."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from src.agents.analysis import ResponseTargetGenerationResult, ResponseTargetGenerationStatus
from src.models import ResponseTarget, ResponseTargetType
from src.simulation.analysis import SimulationResponseTargetCoordinator
from src.simulation.analysis.simulation_response_target_coordinator import NO_AFFECTED_FIRE_EVENTS_REASON
from src.simulation.analysis.simulation_response_target_result import SimulationResponseTargetResult

AS_OF = datetime(2026, 9, 14, 12, 1, 5, tzinfo=timezone.utc)


class FakeResponseTargetAgent:
    def __init__(self, fail_ids=(), inactive_ids=()) -> None:
        self.fail_ids = set(fail_ids)
        self.inactive_ids = set(inactive_ids)
        self.calls = []

    def generate(self, *, fire_event_id, as_of):
        self.calls.append({"fire_event_id": fire_event_id, "as_of": as_of})
        if fire_event_id in self.fail_ids:
            raise RuntimeError(f"response targets failed for {fire_event_id}")
        if fire_event_id in self.inactive_ids:
            return ResponseTargetGenerationResult(
                success=True,
                fire_event_id=fire_event_id,
                status=ResponseTargetGenerationStatus.INACTIVE_EVENT,
                target_set_id=None,
                targets=(),
                target_count=0,
            )
        targets = (make_active_target(fire_event_id),)
        return ResponseTargetGenerationResult(
            success=True,
            fire_event_id=fire_event_id,
            status=ResponseTargetGenerationStatus.GENERATED,
            target_set_id=fire_event_id + 1000,
            targets=targets,
            target_count=len(targets),
        )


def make_active_target(fire_event_id: int) -> ResponseTarget:
    return ResponseTarget(
        fire_event_id=fire_event_id,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        latitude=32.731,
        longitude=35.046,
        priority_score=75.0,
    )


def test_empty_fire_event_ids_do_not_call_agent_and_return_non_triggered_result():
    agent = FakeResponseTargetAgent()
    coordinator = SimulationResponseTargetCoordinator(agent)

    result = coordinator.generate_for_fire_events((), AS_OF)

    assert result.triggered is False
    assert result.reason == NO_AFFECTED_FIRE_EVENTS_REASON
    assert result.generation_results == ()
    assert agent.calls == []


def test_duplicate_fire_event_ids_are_deduplicated_and_processed_in_sorted_order():
    agent = FakeResponseTargetAgent()
    coordinator = SimulationResponseTargetCoordinator(agent)

    result = coordinator.generate_for_fire_events((30, 10, 20, 10), AS_OF)

    assert [call["fire_event_id"] for call in agent.calls] == [10, 20, 30]
    assert all(call["as_of"] == AS_OF for call in agent.calls)
    assert result.fire_event_ids == (10, 20, 30)
    assert [generation.fire_event_id for generation in result.generation_results] == [10, 20, 30]


def test_generated_inactive_and_failed_results_are_aggregated():
    agent = FakeResponseTargetAgent(fail_ids={20}, inactive_ids={30})
    coordinator = SimulationResponseTargetCoordinator(agent)

    result = coordinator.generate_for_fire_events((10, 20, 30), AS_OF)

    assert result.triggered is True
    assert result.fire_events_requested == 3
    assert result.fire_events_processed == 3
    assert result.target_sets_generated == 1
    assert result.inactive_events == 1
    assert result.failed_events == 1
    assert result.target_count == 1
    assert result.target_set_ids == (1010,)
    assert result.error_messages == ("response targets failed for 20",)


def test_one_agent_failure_does_not_stop_later_fire_events():
    agent = FakeResponseTargetAgent(fail_ids={10})
    coordinator = SimulationResponseTargetCoordinator(agent)

    result = coordinator.generate_for_fire_events((10, 20), AS_OF)

    assert [call["fire_event_id"] for call in agent.calls] == [10, 20]
    assert [generation.status for generation in result.generation_results] == [
        ResponseTargetGenerationStatus.FAILED,
        ResponseTargetGenerationStatus.GENERATED,
    ]


def test_naive_timestamp_is_rejected_before_agent_call():
    agent = FakeResponseTargetAgent()
    coordinator = SimulationResponseTargetCoordinator(agent)

    with pytest.raises(ValueError):
        coordinator.generate_for_fire_events((10,), datetime(2026, 9, 14, 12, 0))

    assert agent.calls == []


def test_invalid_fire_event_id_is_rejected_before_agent_call():
    agent = FakeResponseTargetAgent()
    coordinator = SimulationResponseTargetCoordinator(agent)

    with pytest.raises(ValueError):
        coordinator.generate_for_fire_events((10, 0), AS_OF)

    assert agent.calls == []


def test_simulation_response_target_result_is_immutable():
    result = SimulationResponseTargetResult(
        triggered=True,
        generation_results=(
            ResponseTargetGenerationResult(
                success=True,
                fire_event_id=10,
                status=ResponseTargetGenerationStatus.GENERATED,
                target_set_id=1010,
                targets=(make_active_target(10),),
                target_count=1,
            ),
        ),
        fire_event_ids=(10,),
    )

    with pytest.raises(FrozenInstanceError):
        result.triggered = False


def test_triggered_result_rejects_fire_event_ids_that_do_not_match_results():
    with pytest.raises(ValueError):
        SimulationResponseTargetResult(
            triggered=True,
            generation_results=(
                ResponseTargetGenerationResult(
                    success=True,
                    fire_event_id=10,
                    status=ResponseTargetGenerationStatus.GENERATED,
                    target_set_id=1010,
                    targets=(make_active_target(10),),
                    target_count=1,
                ),
            ),
            fire_event_ids=(20,),
        )

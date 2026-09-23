"""Tests for FireSeverityRefreshOrchestrator reuse/invalidation behavior
(performance pass, mirroring FireSpreadRefreshOrchestrator's precedent)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models.fire_severity_assessment import FireSeverityAssessment
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_input import FireSeverityInput
from src.models.fire_severity_input_result import FireSeverityInputResult
from src.models.fire_severity_input_status import FireSeverityInputStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.vegetation_data import VegetationData
from src.repositories.fire_severity_assessment_repository import StoredFireSeverityAssessment
from src.services.operational_refresh.fire_severity_refresh_orchestrator import FireSeverityRefreshOrchestrator

AS_OF = datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc)
FIRE_EVENT_ID = 42


def make_vegetation(**overrides) -> VegetationData:
    defaults = dict(fuel_score=0.6, dominant_land_cover="Shrub cover", source="copernicus", dataset_year=2019, radius_km=2.0)
    defaults.update(overrides)
    return VegetationData(**defaults)


def ready_input_result(
    *,
    weather_observation_ids=(1, 2),
    satellite_hotspot_ids=(10, 11),
    selected_frp_hotspot_id=10,
    vegetation_data=None,
) -> FireSeverityInputResult:
    return FireSeverityInputResult(
        status=FireSeverityInputStatus.READY,
        input_data=FireSeverityInput(frp_mw=50.0, wind_speed_kmh=20.0, relative_humidity_pct=30.0),
        fire_event_id=FIRE_EVENT_ID,
        weather_observation_ids=weather_observation_ids,
        satellite_hotspot_ids=satellite_hotspot_ids,
        selected_frp_hotspot_id=selected_frp_hotspot_id,
        vegetation_data=vegetation_data if vegetation_data is not None else make_vegetation(),
    )


def stored_assessment(
    *,
    assessment_id=900,
    status=FireSeverityAssessmentStatus.VALID,
    weather_observation_ids=(1, 2),
    satellite_hotspot_ids=(10, 11),
    selected_frp_hotspot_id=10,
    vegetation=None,
    methodology="ECOGUARD_ACTIVE_FIRE_SEVERITY",
    methodology_version="1.0",
) -> StoredFireSeverityAssessment:
    vegetation = vegetation if vegetation is not None else make_vegetation()
    return StoredFireSeverityAssessment(
        assessment_id=assessment_id,
        assessment=FireSeverityAssessment(
            fire_event_id=FIRE_EVENT_ID,
            assessed_at=AS_OF,
            status=status,
            score=88.0 if status is FireSeverityAssessmentStatus.VALID else None,
            level=FireSeverityLevel.HIGH if status is FireSeverityAssessmentStatus.VALID else None,
            methodology=methodology,
            methodology_version=methodology_version,
            vegetation_source=vegetation.source,
            vegetation_dataset_year=vegetation.dataset_year,
            vegetation_radius_km=vegetation.radius_km,
            vegetation_dominant_land_cover=vegetation.dominant_land_cover,
            vegetation_fuel_score=vegetation.fuel_score,
        ),
        weather_observation_ids=weather_observation_ids,
        satellite_hotspot_ids=satellite_hotspot_ids,
        selected_frp_hotspot_id=selected_frp_hotspot_id,
    )


class FakeInputService:
    def __init__(self, result: FireSeverityInputResult) -> None:
        self.result = result
        self.calls = []

    def prepare_input(self, fire_event_id, as_of):
        self.calls.append({"fire_event_id": fire_event_id, "as_of": as_of})
        return self.result


class FakeAssessmentAgent:
    def __init__(self, *, fail: bool = False, next_id: int = 999) -> None:
        self.calls = []
        self.input_results = []
        self.fail = fail
        self.next_id = next_id

    def assess(self, fire_event_id, assessed_at):
        self.calls.append({"fire_event_id": fire_event_id, "assessed_at": assessed_at})
        if self.fail:
            raise RuntimeError("simulated assess() failure")
        return stored_assessment(assessment_id=self.next_id)

    def assess_from_input_result(self, input_result, assessed_at):
        """Mirrors the production agent's additive method - the orchestrator
        now always calls this (with the input_result it already prepared to
        make its reuse decision) instead of assess()/assess_for_event(),
        which would redundantly re-prepare the identical input."""
        self.calls.append({"fire_event_id": input_result.fire_event_id, "assessed_at": assessed_at})
        self.input_results.append(input_result)
        if self.fail:
            raise RuntimeError("simulated assess() failure")
        return stored_assessment(assessment_id=self.next_id)


class FakeAssessmentRepository:
    def __init__(self, latest: StoredFireSeverityAssessment | None) -> None:
        self.latest = latest
        self.calls = []

    def get_latest_for_event_as_of(self, fire_event_id, as_of):
        self.calls.append({"fire_event_id": fire_event_id, "as_of": as_of})
        return self.latest


def make_orchestrator(input_service, agent, repository) -> FireSeverityRefreshOrchestrator:
    return FireSeverityRefreshOrchestrator(
        input_service=input_service, assessment_agent=agent, assessment_repository=repository
    )


# --- 1/4: identical/irrelevant-unchanged inputs -> cheap reuse, no compute ---


def test_identical_evidence_and_vegetation_reuses_latest_assessment_without_recomputing():
    input_service = FakeInputService(ready_input_result())
    agent = FakeAssessmentAgent()
    repository = FakeAssessmentRepository(stored_assessment())

    result = make_orchestrator(input_service, agent, repository).refresh(FIRE_EVENT_ID, AS_OF)

    assert result.assessment_id == 900
    assert agent.calls == []  # never recomputed/persisted


def test_reuse_returns_the_exact_stored_assessment_object():
    latest = stored_assessment(assessment_id=777)
    input_service = FakeInputService(ready_input_result())
    orchestrator = make_orchestrator(input_service, FakeAssessmentAgent(), FakeAssessmentRepository(latest))

    result = orchestrator.refresh(FIRE_EVENT_ID, AS_OF)

    assert result is latest


# --- 2: relevant weather (evidence-id) change -> recompute -----------------


def test_changed_weather_observation_ids_triggers_recompute():
    input_service = FakeInputService(ready_input_result(weather_observation_ids=(1, 3)))  # was (1, 2)
    agent = FakeAssessmentAgent(next_id=901)
    repository = FakeAssessmentRepository(stored_assessment())

    result = make_orchestrator(input_service, agent, repository).refresh(FIRE_EVENT_ID, AS_OF)

    assert len(agent.calls) == 1
    assert result.assessment_id == 901


# --- 3: relevant satellite evidence change -> recompute ---------------------


def test_changed_satellite_hotspot_ids_triggers_recompute():
    input_service = FakeInputService(ready_input_result(satellite_hotspot_ids=(10, 12)))  # was (10, 11)
    agent = FakeAssessmentAgent(next_id=902)
    repository = FakeAssessmentRepository(stored_assessment())

    result = make_orchestrator(input_service, agent, repository).refresh(FIRE_EVENT_ID, AS_OF)

    assert len(agent.calls) == 1
    assert result.assessment_id == 902


def test_changed_selected_frp_hotspot_triggers_recompute():
    input_service = FakeInputService(ready_input_result(selected_frp_hotspot_id=11))  # was 10
    agent = FakeAssessmentAgent(next_id=903)
    repository = FakeAssessmentRepository(stored_assessment())

    result = make_orchestrator(input_service, agent, repository).refresh(FIRE_EVENT_ID, AS_OF)

    assert len(agent.calls) == 1
    assert result.assessment_id == 903


def test_changed_vegetation_triggers_recompute():
    input_service = FakeInputService(ready_input_result(vegetation_data=make_vegetation(fuel_score=0.9)))
    agent = FakeAssessmentAgent(next_id=904)
    repository = FakeAssessmentRepository(stored_assessment())  # fuel_score=0.6

    result = make_orchestrator(input_service, agent, repository).refresh(FIRE_EVENT_ID, AS_OF)

    assert len(agent.calls) == 1
    assert result.assessment_id == 904


# --- non-VALID latest / no latest / non-READY input -> always recompute ----


def test_no_latest_assessment_recomputes():
    input_service = FakeInputService(ready_input_result())
    agent = FakeAssessmentAgent(next_id=905)
    repository = FakeAssessmentRepository(None)

    result = make_orchestrator(input_service, agent, repository).refresh(FIRE_EVENT_ID, AS_OF)

    assert len(agent.calls) == 1
    assert result.assessment_id == 905


def test_non_valid_latest_assessment_never_reused():
    input_service = FakeInputService(ready_input_result())
    agent = FakeAssessmentAgent(next_id=906)
    repository = FakeAssessmentRepository(stored_assessment(status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA))

    result = make_orchestrator(input_service, agent, repository).refresh(FIRE_EVENT_ID, AS_OF)

    assert len(agent.calls) == 1
    assert result.assessment_id == 906


def test_non_ready_input_always_delegates_to_agent():
    input_service = FakeInputService(
        FireSeverityInputResult(status=FireSeverityInputStatus.INSUFFICIENT_DATA, input_data=None, fire_event_id=FIRE_EVENT_ID)
    )
    agent = FakeAssessmentAgent(next_id=907)
    repository = FakeAssessmentRepository(stored_assessment())

    result = make_orchestrator(input_service, agent, repository).refresh(FIRE_EVENT_ID, AS_OF)

    assert len(agent.calls) == 1
    assert result.assessment_id == 907


def test_methodology_version_bump_forces_recompute_even_with_identical_evidence():
    """Fail-open guarantee: if the currently-configured methodology/version
    differs from what produced the latest row, never reuse it blindly."""
    input_service = FakeInputService(ready_input_result())
    agent = FakeAssessmentAgent(next_id=908)
    repository = FakeAssessmentRepository(stored_assessment(methodology_version="0.9"))

    result = make_orchestrator(input_service, agent, repository).refresh(FIRE_EVENT_ID, AS_OF)

    assert len(agent.calls) == 1
    assert result.assessment_id == 908


# --- performance pass: input prepared once, forwarded directly on recompute -


def test_recompute_prepares_input_exactly_once_not_twice():
    """Regression guard: the reuse check and the fresh-compute path used to
    each independently call the input service (once to decide reuse, once
    more inside the agent to actually build the assessment) - doubling
    weather/satellite lookups and Copernicus vegetation calls on every
    recompute. assess_from_input_result() must consume the SAME input_result
    already prepared for the reuse decision, so prepare_input() runs once."""
    input_service = FakeInputService(ready_input_result(weather_observation_ids=(1, 3)))  # forces recompute
    agent = FakeAssessmentAgent(next_id=909)
    repository = FakeAssessmentRepository(stored_assessment())

    make_orchestrator(input_service, agent, repository).refresh(FIRE_EVENT_ID, AS_OF)

    assert len(input_service.calls) == 1


def test_recompute_forwards_the_exact_input_result_prepared_for_the_reuse_check():
    input_result = ready_input_result(weather_observation_ids=(1, 3))
    input_service = FakeInputService(input_result)
    agent = FakeAssessmentAgent(next_id=910)
    repository = FakeAssessmentRepository(stored_assessment())

    make_orchestrator(input_service, agent, repository).refresh(FIRE_EVENT_ID, AS_OF)

    assert agent.input_results == [input_result]


# --- 5: failed assessment does not establish a reusable baseline -----------


def test_agent_failure_propagates_and_next_call_still_recomputes():
    input_service = FakeInputService(ready_input_result())
    agent = FakeAssessmentAgent(fail=True)
    repository = FakeAssessmentRepository(None)  # nothing reusable yet
    orchestrator = make_orchestrator(input_service, agent, repository)

    with pytest.raises(RuntimeError, match="simulated assess\\(\\) failure"):
        orchestrator.refresh(FIRE_EVENT_ID, AS_OF)

    # A second call (repository still has nothing valid to reuse, exactly as
    # after a real failed persist) must attempt a fresh compute again, not
    # silently treat the failed attempt as an established baseline.
    agent.fail = False
    result = orchestrator.refresh(FIRE_EVENT_ID, AS_OF)
    assert len(agent.calls) == 2
    assert result.assessment_id == 999


# --- 6: persisted-state (not in-memory) reuse - works after "restart" ------


def test_reuse_decision_is_derived_from_persisted_state_not_in_memory_cache():
    """A brand new orchestrator instance (simulating a process restart) must
    still be able to reuse an assessment that was persisted by a totally
    different orchestrator/process, purely from what the repository returns -
    no in-memory state is required."""
    shared_latest = stored_assessment(assessment_id=555)
    repository = FakeAssessmentRepository(shared_latest)

    fresh_orchestrator = make_orchestrator(FakeInputService(ready_input_result()), FakeAssessmentAgent(), repository)
    result = fresh_orchestrator.refresh(FIRE_EVENT_ID, AS_OF)

    assert result.assessment_id == 555


# --- validation --------------------------------------------------------------


def test_invalid_fire_event_id_rejected():
    orchestrator = make_orchestrator(FakeInputService(ready_input_result()), FakeAssessmentAgent(), FakeAssessmentRepository(None))
    with pytest.raises(ValueError):
        orchestrator.refresh(-1, AS_OF)


def test_naive_assessed_at_rejected():
    orchestrator = make_orchestrator(FakeInputService(ready_input_result()), FakeAssessmentAgent(), FakeAssessmentRepository(None))
    with pytest.raises(ValueError):
        orchestrator.refresh(FIRE_EVENT_ID, datetime(2026, 1, 1))

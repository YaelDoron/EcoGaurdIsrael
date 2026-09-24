"""Unit tests for GlobalPlanningRefreshCoordinator using fakes (Stage 6 of
the Global Multi-Incident Optimizer refactor, Tasks 18-19, 33) - the bounded
stale-retry exhaustion path and NO_ACTIVE_EVENTS are exercised precisely
here; the full real-DB ACTIVATED/NO_OP/dynamic-scenario paths are covered by
tests/integration/test_global_planning_refresh_coordinator_integration.py.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.calculators.global_response_optimization.global_hard_dispatch_lock import GlobalHardDispatchLockInfeasible
from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.services.global_planning.global_planning_input_unstable import GlobalPlanningInputUnstable
from src.services.global_planning.global_planning_refresh_coordinator import (
    GlobalPlanningRefreshCoordinator,
    GlobalPlanningRefreshStatus,
    MAX_GLOBAL_STALE_RETRIES,
)
from src.services.global_planning.global_response_plan_activation_service import GlobalPlanningStaleInput

AS_OF = datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc)


class _FakeFireEventRepository:
    def __init__(self, active_ids_sequence):
        self._sequence = list(active_ids_sequence)
        self.calls = 0

    def get_response_eligible_fire_event_ids(self):
        index = min(self.calls, len(self._sequence) - 1)
        self.calls += 1
        return self._sequence[index]


class _FakeGlobalPlanningRunRepository:
    def __init__(self):
        self._next_id = 1
        self.created_runs = []
        self.completed = {}
        self.fingerprints = {}
        self.member_results = []

    def create_run(self, *, started_at, trigger, methodology, methodology_version, input_fingerprint, fire_event_ids):
        run_id = self._next_id
        self._next_id += 1
        self.created_runs.append(run_id)

        class _Stored:
            id = run_id

        return _Stored()

    def set_input_fingerprint(self, run_id, fingerprint):
        self.fingerprints[run_id] = fingerprint

    def record_member_result(self, run_id, fire_event_id, *, result_status, response_plan_id, local_state_fingerprint, error_code):
        self.member_results.append((run_id, fire_event_id, result_status))

    def complete_run(self, run_id, *, status, completed_at):
        self.completed[run_id] = status

    def get_latest_activated(self, *, exclude_run_id=None):
        return None


class _FakeBundle:
    """Duck-typed stand-in for GlobalPlanningPreRoutingBundle - the
    coordinator only reads `.pre_routing_signature` off it directly and
    otherwise passes it straight through to build(precomputed=...)."""

    def __init__(self, pre_routing_signature="sig"):
        self.pre_routing_signature = pre_routing_signature


class _FakeInputBuilder:
    def __init__(self, pre_routing_signature="sig"):
        self.calls = 0
        self.precheck_calls = 0
        self._pre_routing_signature = pre_routing_signature

    def compute_pre_routing_bundle(self, *, global_planning_run_id, as_of):
        self.precheck_calls += 1
        return _FakeBundle(self._pre_routing_signature)

    def build(self, *, global_planning_run_id, as_of, precomputed=None):
        self.calls += 1

        class _FakeInput:
            input_fingerprint = "f" * 64
            current_assignments = ()

        return _FakeInput()


class _AlwaysStaleActivationService:
    """Every activate() call raises GlobalPlanningStaleInput."""

    def __init__(self):
        self.calls = 0

    def activate(self, *, global_planning_input, global_optimization_result, as_of, run_history=None):
        self.calls += 1
        raise GlobalPlanningStaleInput("simulated staleness")


class _FakeOptimizationService:
    def __init__(self):
        self.calls = 0

    def optimize(self, global_planning_input, config, demand_scoring_policy, severity_demand_policy, stability_policy):
        self.calls += 1

        class _FakeResult:
            actions = ()
            assignment_changes = ()
            shortage = None
            event_results = ()

        return _FakeResult()


class _AlwaysHardLockInfeasibleOptimizationService:
    """Every optimize() call raises GlobalHardDispatchLockInfeasible - the
    stale-road-network-coverage scenario reported live: a hard-dispatched
    resource's only feasible route to its locked FireEvent disappeared in
    this cycle's freshly-fetched road network."""

    def __init__(self):
        self.calls = 0

    def optimize(self, global_planning_input, config, demand_scoring_policy, severity_demand_policy, stability_policy):
        self.calls += 1
        raise GlobalHardDispatchLockInfeasible("simulated stale hard lock")


class _AlwaysSucceedsActivationService:
    """Every activate() call succeeds trivially - used to prove the
    coordinator reaches and completes a normal cycle once optimize() stops
    raising, not just that it fails gracefully."""

    def __init__(self):
        self.calls = 0

    def activate(self, *, global_planning_input, global_optimization_result, as_of, run_history=None):
        self.calls += 1

        class _FakeActivation:
            response_plan_ids_by_event: dict[int, int] = {}

        return _FakeActivation()


def _make_coordinator(
    fire_event_ids_sequence, activation_service=None, run_repository=None, optimization_service=None
):
    return GlobalPlanningRefreshCoordinator(
        fire_event_repository=_FakeFireEventRepository(fire_event_ids_sequence),
        global_planning_run_repository=run_repository or _FakeGlobalPlanningRunRepository(),
        input_builder=_FakeInputBuilder(),
        optimization_service=optimization_service or _FakeOptimizationService(),
        activation_service=activation_service or _AlwaysStaleActivationService(),
    )


def test_no_active_events_returns_immediately_without_creating_a_run():
    run_repository = _FakeGlobalPlanningRunRepository()
    coordinator = _make_coordinator(((),), run_repository=run_repository)

    result = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert result.status is GlobalPlanningRefreshStatus.NO_ACTIVE_EVENTS
    assert run_repository.created_runs == []


def test_stale_retries_are_bounded_and_exhausted(monkeypatch):
    run_repository = _FakeGlobalPlanningRunRepository()
    activation_service = _AlwaysStaleActivationService()
    coordinator = _make_coordinator(((1,), (1,), (1,)), activation_service=activation_service, run_repository=run_repository)

    result = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert result.status is GlobalPlanningRefreshStatus.STALE_RETRY_EXHAUSTED
    assert result.retry_count == MAX_GLOBAL_STALE_RETRIES + 1
    # Exactly one full cycle attempt per retry - never partial/per-event patching.
    assert activation_service.calls == MAX_GLOBAL_STALE_RETRIES + 1
    assert len(run_repository.created_runs) == MAX_GLOBAL_STALE_RETRIES + 1
    for run_id in run_repository.created_runs:
        assert run_repository.completed[run_id].value == "failed"


def test_stale_retry_recaptures_active_set_between_attempts():
    """If the active set shrinks to empty between retry attempts (e.g. the
    only FireEvent resolved), the coordinator must report NO_ACTIVE_EVENTS
    rather than attempting a doomed cycle with an empty set."""
    run_repository = _FakeGlobalPlanningRunRepository()
    activation_service = _AlwaysStaleActivationService()
    fire_event_repository = _FakeFireEventRepository(((1,), ()))
    coordinator = GlobalPlanningRefreshCoordinator(
        fire_event_repository=fire_event_repository,
        global_planning_run_repository=run_repository,
        input_builder=_FakeInputBuilder(),
        optimization_service=_FakeOptimizationService(),
        activation_service=activation_service,
    )

    result = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert result.status is GlobalPlanningRefreshStatus.NO_ACTIVE_EVENTS
    # Only the first attempt actually ran a cycle before the set emptied out.
    assert activation_service.calls == 1


def test_hard_dispatch_lock_infeasible_retries_and_exhausts_without_crashing():
    """A hard-dispatched resource losing its only feasible route to its
    locked FireEvent (e.g. a transient road-network coverage gap on a
    later cycle) must never propagate out of refresh() and crash the
    caller - it is exactly the same bounded-retry shape as a stale
    activation input: retried up to MAX_GLOBAL_STALE_RETRIES times, then
    reported as STALE_RETRY_EXHAUSTED with every attempted run marked
    failed, never an uncaught exception."""
    run_repository = _FakeGlobalPlanningRunRepository()
    optimization_service = _AlwaysHardLockInfeasibleOptimizationService()
    coordinator = _make_coordinator(
        ((1,), (1,), (1,)), run_repository=run_repository, optimization_service=optimization_service
    )

    result = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert result.status is GlobalPlanningRefreshStatus.STALE_RETRY_EXHAUSTED
    assert result.retry_count == MAX_GLOBAL_STALE_RETRIES + 1
    assert optimization_service.calls == MAX_GLOBAL_STALE_RETRIES + 1
    assert len(run_repository.created_runs) == MAX_GLOBAL_STALE_RETRIES + 1
    for run_id in run_repository.created_runs:
        assert run_repository.completed[run_id].value == "failed"


def test_hard_dispatch_lock_infeasible_recovers_once_a_later_cycle_finds_a_feasible_route():
    """If the FIRST cycle's optimize() raises GlobalHardDispatchLockInfeasible
    but a retried cycle's (freshly-rebuilt GlobalPlanningInput) optimize()
    succeeds, refresh() must complete normally as ACTIVATED - proving this
    is a genuine retry-and-recover path, not merely "fails without
    crashing"."""

    class _FailsOnceThenSucceeds:
        def __init__(self):
            self.calls = 0

        def optimize(self, global_planning_input, config, demand_scoring_policy, severity_demand_policy, stability_policy):
            self.calls += 1
            if self.calls == 1:
                raise GlobalHardDispatchLockInfeasible("simulated stale hard lock")

            class _FakeResult:
                actions = ()
                assignment_changes = ()
                shortage = None
                event_results = ()

            return _FakeResult()

    run_repository = _FakeGlobalPlanningRunRepository()
    optimization_service = _FailsOnceThenSucceeds()
    activation_service = _AlwaysSucceedsActivationService()
    coordinator = _make_coordinator(
        ((1,), (1,)),
        run_repository=run_repository,
        optimization_service=optimization_service,
        activation_service=activation_service,
    )

    result = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert result.status is GlobalPlanningRefreshStatus.ACTIVATED
    assert optimization_service.calls == 2
    assert activation_service.calls == 1
    # First (failed) run stays FAILED; only the second cycle's run is left active/untouched-by-failure.
    assert run_repository.completed[run_repository.created_runs[0]].value == "failed"
    assert run_repository.created_runs[1] not in run_repository.completed


@pytest.mark.parametrize("active_fire_event_ids", [(1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
def test_refresh_never_assumes_exactly_two_active_fires(active_fire_event_ids):
    """The global optimization is over whatever the currently-active set
    is - one, two, three, or four FireEvents - never a hardcoded pair.
    Uses the always-stale activation fake (same as
    test_stale_retries_are_bounded_and_exhausted) purely so the cycle
    completes deterministically without needing a real optimizer; what
    matters here is that the coordinator runs a full cycle (creates a run,
    calls activation) regardless of the active set's size."""
    run_repository = _FakeGlobalPlanningRunRepository()
    activation_service = _AlwaysStaleActivationService()
    coordinator = _make_coordinator(
        (active_fire_event_ids,) * (MAX_GLOBAL_STALE_RETRIES + 1),
        activation_service=activation_service,
        run_repository=run_repository,
    )

    result = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert result.status is GlobalPlanningRefreshStatus.STALE_RETRY_EXHAUSTED
    assert len(run_repository.created_runs) == MAX_GLOBAL_STALE_RETRIES + 1
    assert activation_service.calls == MAX_GLOBAL_STALE_RETRIES + 1


def test_invalid_trigger_rejected():
    coordinator = _make_coordinator(((1,),))
    with pytest.raises(ValueError):
        coordinator.refresh(trigger="", as_of=AS_OF)


def test_invalid_as_of_rejected():
    coordinator = _make_coordinator(((1,),))
    with pytest.raises(ValueError):
        coordinator.refresh(trigger="manual", as_of=datetime(2026, 1, 1))


# --- run-status regression tests: a run must never stay stuck RUNNING ----
#
# Root cause this section guards against: previously, only
# GlobalPlanningInputUnstable raised from input_builder.build() was caught
# and turned into a FAILED run. Any other exception from build() (e.g. the
# road-network bbox query's PostgreSQL parameter-limit error) propagated
# straight out of refresh(), leaving the just-created GlobalPlanningRun row
# stuck at RUNNING forever, since complete_run() was never reached.


class _AlwaysSucceedsActivationService:
    """Every activate() call succeeds with an empty result."""

    def __init__(self):
        self.calls = 0

    def activate(self, *, global_planning_input, global_optimization_result, as_of, run_history=None):
        self.calls += 1

        class _FakeActivation:
            response_plan_ids_by_event = {}

        return _FakeActivation()


class _FailingInputBuilder:
    """Simulates an unexpected, non-retryable failure inside input
    building - e.g. the road-network bbox query exceeding PostgreSQL's
    parameter limit. Not a GlobalPlanningInputUnstable - a genuine defect
    or infrastructure error that must still fail the run, not retry it."""

    def __init__(self, exception: Exception):
        self._exception = exception
        self.calls = 0

    def compute_pre_routing_bundle(self, *, global_planning_run_id, as_of):
        return _FakeBundle("sig-failing")

    def build(self, *, global_planning_run_id, as_of, precomputed=None):
        self.calls += 1
        raise self._exception


def test_successful_planning_activates_and_does_not_mark_run_failed():
    run_repository = _FakeGlobalPlanningRunRepository()
    activation_service = _AlwaysSucceedsActivationService()
    coordinator = _make_coordinator(((1,),), activation_service=activation_service, run_repository=run_repository)

    result = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert result.status is GlobalPlanningRefreshStatus.ACTIVATED
    assert activation_service.calls == 1
    # complete_run(FAILED) must never have been called for a successful cycle;
    # the ACTIVATED path's own COMPLETED/PARTIAL status is activate()'s
    # responsibility (see run_history docstring), not the coordinator's.
    assert run_repository.completed == {}


def test_unexpected_planning_exception_marks_run_failed_and_reraises_original():
    """The core regression test: an unexpected exception raised out of
    input building (standing in for the road-network parameter-limit
    error) must mark the run FAILED before propagating, and must not be
    swallowed - the caller still sees the original exception."""
    run_repository = _FakeGlobalPlanningRunRepository()
    original_exception = RuntimeError("simulated road-network parameter-limit failure")
    coordinator = GlobalPlanningRefreshCoordinator(
        fire_event_repository=_FakeFireEventRepository(((1,),)),
        global_planning_run_repository=run_repository,
        input_builder=_FailingInputBuilder(original_exception),
        optimization_service=_FakeOptimizationService(),
        activation_service=_AlwaysStaleActivationService(),
    )

    with pytest.raises(RuntimeError, match="simulated road-network parameter-limit failure"):
        coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert len(run_repository.created_runs) == 1
    run_id = run_repository.created_runs[0]
    # The run must have been finalized (not left absent from `completed`,
    # which stands in here for "still RUNNING") and specifically FAILED.
    assert run_id in run_repository.completed
    assert run_repository.completed[run_id].value == "failed"


def test_unexpected_exception_from_optimize_marks_run_failed_and_reraises():
    """Same guarantee, but the unexpected exception comes from the
    optimization step rather than input building - the safety net covers
    the whole cycle, not just build()."""

    class _FailingOptimizationService:
        def optimize(self, *args, **kwargs):
            raise RuntimeError("simulated GA failure")

    run_repository = _FakeGlobalPlanningRunRepository()
    coordinator = GlobalPlanningRefreshCoordinator(
        fire_event_repository=_FakeFireEventRepository(((1,),)),
        global_planning_run_repository=run_repository,
        input_builder=_FakeInputBuilder(),
        optimization_service=_FailingOptimizationService(),
        activation_service=_AlwaysStaleActivationService(),
    )

    with pytest.raises(RuntimeError, match="simulated GA failure"):
        coordinator.refresh(trigger="manual", as_of=AS_OF)

    run_id = run_repository.created_runs[0]
    assert run_repository.completed[run_id].value == "failed"


def test_input_unstable_exception_still_returns_none_and_retries_as_before():
    """The refactor that added a broad safety-net except clause must not
    change the existing, specifically-handled GlobalPlanningInputUnstable
    retry behavior: it still returns None (triggering the bounded-retry
    loop) rather than propagating."""
    run_repository = _FakeGlobalPlanningRunRepository()

    class _AlwaysUnstableInputBuilder:
        def __init__(self):
            self.calls = 0

        def compute_pre_routing_bundle(self, *, global_planning_run_id, as_of):
            return _FakeBundle("sig-unstable")

        def build(self, *, global_planning_run_id, as_of, precomputed=None):
            self.calls += 1
            raise GlobalPlanningInputUnstable(global_planning_run_id=global_planning_run_id, attempts=3)

    input_builder = _AlwaysUnstableInputBuilder()
    coordinator = GlobalPlanningRefreshCoordinator(
        fire_event_repository=_FakeFireEventRepository(((1,), (1,), (1,))),
        global_planning_run_repository=run_repository,
        input_builder=input_builder,
        optimization_service=_FakeOptimizationService(),
        activation_service=_AlwaysStaleActivationService(),
    )

    result = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert result.status is GlobalPlanningRefreshStatus.STALE_RETRY_EXHAUSTED
    assert len(run_repository.created_runs) == MAX_GLOBAL_STALE_RETRIES + 1
    for run_id in run_repository.created_runs:
        assert run_repository.completed[run_id].value == "failed"


# --- cheap NO_OP precheck regression tests (Optimization 3) --------------
#
# Root cause this section guards against: profiling showed NO_OP cycles
# paying for the full road-network load + route matrix + Dijkstra before
# the existing fingerprint comparison ever got a chance to say "nothing
# changed." _run_one_cycle now asks the input builder for a cheap
# pre-routing signature FIRST and compares it against this coordinator
# instance's own settled baseline before ever calling build().


def test_identical_semantic_input_produces_cheap_no_op_without_full_build():
    run_repository = _FakeGlobalPlanningRunRepository()
    activation_service = _AlwaysSucceedsActivationService()
    optimization_service = _FakeOptimizationService()
    input_builder = _FakeInputBuilder(pre_routing_signature="same-signature")
    coordinator = GlobalPlanningRefreshCoordinator(
        fire_event_repository=_FakeFireEventRepository(((1,), (1,))),
        global_planning_run_repository=run_repository,
        input_builder=input_builder,
        optimization_service=optimization_service,
        activation_service=activation_service,
    )

    first = coordinator.refresh(trigger="manual", as_of=AS_OF)
    second = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert first.status is GlobalPlanningRefreshStatus.ACTIVATED
    assert second.status is GlobalPlanningRefreshStatus.NO_OP
    # build() only ran for the settling (first) cycle - the second cycle's
    # cheap precheck alone decided NO_OP.
    assert input_builder.calls == 1
    assert input_builder.precheck_calls == 2
    # Neither optimize() nor activate() ran a second time - proof the cheap
    # path never touches road-network/route-matrix/Dijkstra/GA.
    assert activation_service.calls == 1
    assert optimization_service.calls == 1
    run_id_2 = run_repository.created_runs[1]
    assert run_repository.completed[run_id_2].value == "completed"
    assert (run_id_2, 1, GlobalPlanningRunEventStatus.NO_OP) in run_repository.member_results


def test_changed_semantic_input_triggers_full_planning_not_cheap_no_op():
    """Not a heuristic: a genuinely different signature must always fall
    through to a full build, never be classified NO_OP."""
    run_repository = _FakeGlobalPlanningRunRepository()
    activation_service = _AlwaysSucceedsActivationService()
    input_builder = _FakeInputBuilder(pre_routing_signature="sig-a")
    coordinator = GlobalPlanningRefreshCoordinator(
        fire_event_repository=_FakeFireEventRepository(((1,), (1,))),
        global_planning_run_repository=run_repository,
        input_builder=input_builder,
        optimization_service=_FakeOptimizationService(),
        activation_service=activation_service,
    )

    first = coordinator.refresh(trigger="manual", as_of=AS_OF)
    input_builder._pre_routing_signature = "sig-b"  # simulate a real semantic change
    second = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert first.status is GlobalPlanningRefreshStatus.ACTIVATED
    assert second.status is GlobalPlanningRefreshStatus.ACTIVATED
    assert input_builder.calls == 2
    assert activation_service.calls == 2


def test_failed_cycle_does_not_establish_a_cheap_no_op_baseline():
    """A cycle that never reaches a genuine terminal success must not seed
    the cheap precheck - a later cycle with the identical signature must
    still perform a full build."""
    run_repository = _FakeGlobalPlanningRunRepository()

    class _FailsFirstRefreshActivationService:
        def __init__(self):
            self.calls = 0

        def activate(self, *, global_planning_input, global_optimization_result, as_of, run_history=None):
            self.calls += 1
            if self.calls <= MAX_GLOBAL_STALE_RETRIES + 1:
                raise GlobalPlanningStaleInput("first refresh always fails")

            class _FakeActivation:
                response_plan_ids_by_event = {}

            return _FakeActivation()

    activation_service = _FailsFirstRefreshActivationService()
    input_builder = _FakeInputBuilder(pre_routing_signature="sig-x")
    coordinator = GlobalPlanningRefreshCoordinator(
        fire_event_repository=_FakeFireEventRepository(((1,),) * 10),
        global_planning_run_repository=run_repository,
        input_builder=input_builder,
        optimization_service=_FakeOptimizationService(),
        activation_service=activation_service,
    )

    first = coordinator.refresh(trigger="manual", as_of=AS_OF)
    assert first.status is GlobalPlanningRefreshStatus.STALE_RETRY_EXHAUSTED

    second = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert second.status is GlobalPlanningRefreshStatus.ACTIVATED
    # Every attempt across both refresh() calls did a real build() - the
    # failed first refresh (despite sharing the identical "sig-x" signature)
    # never short-circuited the later successful one into a cheap NO_OP.
    # First refresh() exhausts MAX_GLOBAL_STALE_RETRIES+1 attempts, each a
    # real build(); the second refresh() succeeds on its first attempt.
    assert input_builder.calls == (MAX_GLOBAL_STALE_RETRIES + 1) + 1


def test_settled_state_does_not_leak_across_coordinator_instances():
    """Instance-scoped settled state (never module/global): a second,
    independently-constructed coordinator must not inherit another
    coordinator's settled baseline, even with the identical signature."""
    run_repository_1 = _FakeGlobalPlanningRunRepository()
    coordinator_1 = GlobalPlanningRefreshCoordinator(
        fire_event_repository=_FakeFireEventRepository(((1,),)),
        global_planning_run_repository=run_repository_1,
        input_builder=_FakeInputBuilder(pre_routing_signature="shared-sig"),
        optimization_service=_FakeOptimizationService(),
        activation_service=_AlwaysSucceedsActivationService(),
    )
    first = coordinator_1.refresh(trigger="manual", as_of=AS_OF)
    assert first.status is GlobalPlanningRefreshStatus.ACTIVATED

    run_repository_2 = _FakeGlobalPlanningRunRepository()
    activation_service_2 = _AlwaysSucceedsActivationService()
    coordinator_2 = GlobalPlanningRefreshCoordinator(
        fire_event_repository=_FakeFireEventRepository(((1,),)),
        global_planning_run_repository=run_repository_2,
        input_builder=_FakeInputBuilder(pre_routing_signature="shared-sig"),
        optimization_service=_FakeOptimizationService(),
        activation_service=activation_service_2,
    )
    second = coordinator_2.refresh(trigger="manual", as_of=AS_OF)

    assert second.status is GlobalPlanningRefreshStatus.ACTIVATED
    assert activation_service_2.calls == 1

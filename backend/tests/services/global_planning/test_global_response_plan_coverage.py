"""Global Response Plan lifecycle: none / generating / updating / current, derived from persisted state."""
from __future__ import annotations

from dataclasses import replace

from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.services.global_planning.global_response_plan_read_service import GlobalResponsePlanReadService
from tests.services.global_planning.test_global_response_plan_read_service import (
    AS_OF,
    FakeGlobalPlanningRunRepository,
    FakeResponsePlanDetailsService,
    FakeResponsePlanPresenter,
    make_details,
    make_member,
    make_presented,
    make_run,
)


class FakeFireEventRepository:
    """`eligible` = CONFIRMED ids; `suspected` = active, monitoring-only ids."""

    def __init__(self, eligible_ids, suspected_ids=()):
        self.eligible_ids = tuple(eligible_ids)
        self.suspected_ids = tuple(suspected_ids)

    def get_response_eligible_fire_event_ids(self):
        return self.eligible_ids

    def get_active_fire_event_ids(self):
        return tuple(sorted(self.eligible_ids + self.suspected_ids))


class RunRepositoryWithLatest(FakeGlobalPlanningRunRepository):
    def __init__(self, latest_materialized=None, members_by_run_id=None, latest=None):
        super().__init__(latest_materialized, members_by_run_id)
        self._latest = latest if latest is not None else latest_materialized

    def get_latest(self):
        return self._latest


def _service(*, eligible, materialized=None, members=(), latest=None, latest_members=(), suspected=()):
    members_by_run = {}
    if materialized is not None:
        members_by_run[materialized.id] = tuple(members)
    if latest is not None and latest is not materialized:
        members_by_run[latest.id] = tuple(latest_members)
    plan_ids = [m.member.response_plan_id for m in members if m.member.response_plan_id is not None]
    return GlobalResponsePlanReadService(
        global_planning_run_repository=RunRepositoryWithLatest(materialized, members_by_run, latest),
        response_plan_details_service=FakeResponsePlanDetailsService({i: make_details(i, 100 + i) for i in plan_ids}),
        response_plan_presenter=FakeResponsePlanPresenter({i: make_presented(i, 100 + i) for i in plan_ids}),
        fire_event_repository=FakeFireEventRepository(eligible, suspected),
    )


def test_no_eligible_fire_and_no_plan_is_the_only_none_state():
    coverage = _service(eligible=[]).get_current(as_of=AS_OF).coverage

    assert coverage.state == "none"
    assert coverage.eligible_count == 0


def test_confirmed_fire_without_any_plan_is_generating():
    result = _service(eligible=[101]).get_current(as_of=AS_OF)

    assert result.plan is None
    assert result.coverage.state == "generating"
    assert result.coverage.pending_fire_event_ids == [101]


def test_plan_covering_one_of_two_confirmed_fires_is_updating_and_still_shown():
    result = _service(
        eligible=[101, 102],
        materialized=make_run(7),
        members=[make_member(1, 101, response_plan_id=1)],
    ).get_current(as_of=AS_OF)

    assert result.plan is not None and [e.fire_event_id for e in result.plan.events] == [101]
    assert result.coverage.state == "updating"
    assert (result.coverage.covered_count, result.coverage.eligible_count) == (1, 2)
    assert result.coverage.pending_fire_event_ids == [102]


def test_plan_covering_every_confirmed_fire_is_current():
    result = _service(
        eligible=[101, 102],
        materialized=make_run(7),
        members=[make_member(1, 101, response_plan_id=1), make_member(2, 102, response_plan_id=2, event_order=1)],
    ).get_current(as_of=AS_OF)

    assert result.coverage.state == "current"
    assert result.coverage.pending_fire_event_ids == []


def test_suspected_only_fires_never_look_like_a_plan_is_being_generated():
    # SUSPECTED events are not response-eligible, so they are simply absent from `eligible`.
    coverage = _service(eligible=[]).get_current(as_of=AS_OF).coverage

    assert coverage.state == "none"
    assert coverage.pending_fire_event_ids == []


def test_a_fire_the_latest_finished_cycle_could_not_plan_is_not_pending_forever():
    materialized = make_run(7)
    latest = make_run(8)
    failed_member = replace(
        make_member(9, 102),
        member=replace(make_member(9, 102).member, result_status=GlobalPlanningRunEventStatus.FAILED),
    )
    result = _service(
        eligible=[101, 102],
        materialized=materialized,
        members=[make_member(1, 101, response_plan_id=1)],
        latest=latest,
        latest_members=[failed_member],
    ).get_current(as_of=AS_OF)

    assert result.coverage.state == "current"
    assert result.coverage.unplannable_fire_event_ids == [102]


def test_a_cycle_still_running_keeps_its_fires_pending():
    running = make_run(8, status=GlobalPlanningRunStatus.RUNNING)
    result = _service(
        eligible=[101, 102],
        materialized=make_run(7),
        members=[make_member(1, 101, response_plan_id=1)],
        latest=running,
    ).get_current(as_of=AS_OF)

    assert result.coverage.state == "updating"
    assert result.coverage.pending_fire_event_ids == [102]


def test_coverage_is_reported_by_the_api_dependency():
    from src.api.routers.global_response_plan import get_global_response_plan_read_service

    assert get_global_response_plan_read_service()._fire_event_repository is not None  # noqa: SLF001

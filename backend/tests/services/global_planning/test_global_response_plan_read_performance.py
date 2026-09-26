"""Global Response Plan read path: fast no-plan read, persisted-plan reuse, parallel loads.

Opening the Global Response Plan page must only READ: no plan -> one
repository lookup and nothing else; a persisted plan -> its immutable
presentation parts are hydrated once and reused, never recomputed. Fakes only.
"""
from __future__ import annotations

import threading

from src.database.connection import is_read_only_database_access, read_only_database_access
from src.services.global_planning.global_response_plan_read_service import (
    GlobalResponsePlanReadService,
    PersistedPlanPartsCache,
)
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


class ExplodingDetailsService:
    def get_plan_details_by_id(self, plan_id):  # pragma: no cover - must never run
        raise AssertionError("the no-plan read must not hydrate any response plan")


class ExplodingPresenter:
    def present(self, details):  # pragma: no cover - must never run
        raise AssertionError("the no-plan read must not present any response plan")


def _service(*, members, cache=None, max_parallel=1, details_service=None, presenter=None, repository=None):
    plan_ids = [stored.member.response_plan_id for stored in members if stored.member.response_plan_id is not None]
    repository = repository or FakeGlobalPlanningRunRepository(make_run(7), {7: tuple(members)})
    details_service = details_service or FakeResponsePlanDetailsService(
        {plan_id: make_details(plan_id, 100 + plan_id) for plan_id in plan_ids}
    )
    presenter = presenter or FakeResponsePlanPresenter(
        {plan_id: make_presented(plan_id, 100 + plan_id) for plan_id in plan_ids}
    )
    service = GlobalResponsePlanReadService(
        global_planning_run_repository=repository,
        response_plan_details_service=details_service,
        response_plan_presenter=presenter,
        plan_parts_cache=cache,
        max_parallel_plan_loads=max_parallel,
    )
    return service, repository, details_service, presenter


def test_no_plan_read_returns_immediately_without_touching_plan_hydration():
    repository = FakeGlobalPlanningRunRepository(latest_materialized=None)
    service = GlobalResponsePlanReadService(
        global_planning_run_repository=repository,
        response_plan_details_service=ExplodingDetailsService(),
        response_plan_presenter=ExplodingPresenter(),
        plan_parts_cache=PersistedPlanPartsCache(),
        max_parallel_plan_loads=4,
    )

    result = service.get_current(as_of=AS_OF)

    assert result.plan is None
    assert repository.get_latest_materialized_generation_calls == 1
    assert repository.get_members_calls == []


def test_persisted_plan_is_hydrated_once_then_served_from_the_cache():
    cache = PersistedPlanPartsCache()
    members = [make_member(1, 101, response_plan_id=1), make_member(2, 102, response_plan_id=2, event_order=1)]
    service, _, details_service, presenter = _service(members=members, cache=cache)

    first = service.get_current(as_of=AS_OF)
    second = service.get_current(as_of=AS_OF)

    assert details_service.calls == [1, 2]  # only the first read hydrated the plans
    assert len(presenter.calls) == 2
    assert [event.fire_event_id for event in second.plan.events] == [101, 102]
    assert second.plan.events == first.plan.events


def test_cached_parts_never_freeze_the_fresh_membership_snapshot():
    cache = PersistedPlanPartsCache()
    before = [make_member(1, 101, response_plan_id=1, assigned_resources=1)]
    service, _, _, _ = _service(members=before, cache=cache)
    service.get_current(as_of=AS_OF)

    after = [make_member(1, 101, response_plan_id=1, assigned_resources=3, severity_score=91.0)]
    refreshed, _, details_service, _ = _service(members=after, cache=cache)
    event = refreshed.get_current(as_of=AS_OF).plan.events[0]

    assert details_service.calls == []  # plan parts reused ...
    assert event.assigned_resources == 3  # ... but the member row is read fresh
    assert event.severity_score == 91.0


def test_a_missing_plan_is_skipped_and_not_cached():
    cache = PersistedPlanPartsCache()
    members = [make_member(1, 101, response_plan_id=1)]
    details_service = FakeResponsePlanDetailsService({})  # plan 1 does not exist (yet)
    service, _, _, _ = _service(members=members, cache=cache, details_service=details_service)

    assert service.get_current(as_of=AS_OF).plan.events == []
    assert cache.get(1) is None


def test_parallel_loads_keep_member_order_and_the_read_only_scope():
    seen_scope: list[bool] = []
    seen_threads: set[int] = set()
    barrier = threading.Barrier(3, timeout=5)

    class ConcurrentDetailsService(FakeResponsePlanDetailsService):
        def get_plan_details_by_id(self, plan_id):
            seen_scope.append(is_read_only_database_access())
            seen_threads.add(threading.get_ident())
            barrier.wait()  # all three loads are in flight at the same time
            return super().get_plan_details_by_id(plan_id)

    members = [make_member(i, 100 + i, response_plan_id=i, event_order=i) for i in (3, 1, 2)]
    details_service = ConcurrentDetailsService({i: make_details(i, 100 + i) for i in (1, 2, 3)})
    service, _, _, _ = _service(members=members, max_parallel=4, details_service=details_service)

    with read_only_database_access():
        events = service.get_current(as_of=AS_OF).plan.events

    assert [event.fire_event_id for event in events] == [103, 101, 102]
    assert seen_scope == [True, True, True]
    assert len(seen_threads) == 3


def test_cache_is_bounded():
    cache = PersistedPlanPartsCache(max_entries=2)
    for plan_id in (1, 2, 3):
        cache.put(plan_id, ((), ()))

    assert cache.get(1) is None
    assert cache.get(2) is not None and cache.get(3) is not None


def test_api_dependency_wires_the_shared_cache_and_parallel_loads():
    from src.api.routers.global_response_plan import get_global_response_plan_read_service

    first = get_global_response_plan_read_service()
    second = get_global_response_plan_read_service()

    assert first._plan_parts_cache is second._plan_parts_cache is not None  # noqa: SLF001
    assert first._max_parallel_plan_loads > 1  # noqa: SLF001

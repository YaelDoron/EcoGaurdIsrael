"""EventDetailsService: independent snapshot sections can be read concurrently (opt-in)."""
from __future__ import annotations

import threading

from src.database.connection import is_read_only_database_access, read_only_database_access
from src.services.fire_event_read.event_details_service import EventDetailsService


class _Unused:
    """Placeholder collaborator: _load_sections never touches repositories itself."""


def _service(max_parallel_loads: int) -> EventDetailsService:
    unused = _Unused()
    return EventDetailsService(
        fire_event_repository=unused,
        fire_severity_assessment_repository=unused,
        fire_spread_prediction_repository=unused,
        response_target_repository=unused,
        fire_station_repository=unused,
        firefighting_resource_repository=unused,
        satellite_hotspot_repository=unused,
        news_repository=unused,
        response_plan_details_service=unused,
        max_parallel_loads=max_parallel_loads,
    )


def test_sections_run_concurrently_and_keep_the_read_only_scope():
    barrier = threading.Barrier(3, timeout=5)
    scopes: list[bool] = []

    def loader(value):
        def run():
            scopes.append(is_read_only_database_access())
            barrier.wait()  # all three are in flight at once
            return value

        return run

    with read_only_database_access():
        result = _service(4)._load_sections({"a": loader(1), "b": loader(2), "c": loader(3)})  # noqa: SLF001

    assert result == {"a": 1, "b": 2, "c": 3}
    assert scopes == [True, True, True]


def test_default_is_sequential_on_the_calling_thread():
    caller = threading.get_ident()
    seen: list[int] = []

    def loader():
        seen.append(threading.get_ident())
        return len(seen)

    result = _service(1)._load_sections({"a": loader, "b": loader})  # noqa: SLF001

    assert result == {"a": 1, "b": 2}
    assert seen == [caller, caller]


def test_api_dependency_enables_parallel_section_reads():
    from src.api.dependencies import get_event_details_service

    assert get_event_details_service()._max_parallel_loads > 1  # noqa: SLF001

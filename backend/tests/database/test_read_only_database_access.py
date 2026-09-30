"""Read-path round-trip reductions in src.database.connection (no database needed).

* Inside `read_only_database_access()` the process session binds to an
  AUTOCOMMIT view of the engine (no BEGIN/COMMIT round trips per repository
  call); outside it, and for sessions bound elsewhere, nothing changes.
* Pooled connections are pinged only after sitting idle (not on every
  checkout); a dead idle connection is reported as a disconnect so the pool
  replaces it.
* The pure GET read routers run their endpoint inside that scope.
"""
from __future__ import annotations

import re
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import DisconnectionError

import src.database.connection as connection
from src.api.schemas.global_response_plan import GlobalResponsePlanResponse


@pytest.fixture()
def engines(monkeypatch):
    base = create_engine("postgresql+psycopg://user:secret@localhost:1/none")  # lazy: never connects
    read_only = base.execution_options(isolation_level="AUTOCOMMIT")
    monkeypatch.setattr(connection, "_engine", base)
    monkeypatch.setattr(connection, "_read_only_engine", read_only)
    return base, read_only


def test_session_binds_to_autocommit_view_only_inside_the_read_only_scope(engines):
    base, read_only = engines
    session = connection._ReadAwareSession(bind=base)  # noqa: SLF001

    assert session.get_bind() is base
    with connection.read_only_database_access():
        assert connection.is_read_only_database_access() is True
        assert session.get_bind() is read_only
    assert session.get_bind() is base
    assert connection.is_read_only_database_access() is False


def test_a_session_bound_to_another_engine_is_never_rerouted(engines):
    other = create_engine("postgresql+psycopg://user:secret@localhost:1/other")
    session = connection._ReadAwareSession(bind=other)  # noqa: SLF001

    with connection.read_only_database_access():
        assert session.get_bind() is other


class _Cursor:
    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.executed: list[str] = []

    def execute(self, sql):
        if self.fail:
            raise OSError("server closed the connection")
        self.executed.append(sql)

    def close(self):
        pass


class _DbapiConnection:
    def __init__(self, fail: bool = False) -> None:
        self.cursor_obj = _Cursor(fail)
        self.autocommit = False
        self.rollbacks = 0

    def cursor(self):
        return self.cursor_obj

    def rollback(self):
        self.rollbacks += 1


def _record(last_used: float | None):
    return SimpleNamespace(info={} if last_used is None else {"last_used": last_used})


def test_recently_used_connection_is_handed_out_without_a_ping():
    dbapi = _DbapiConnection()

    connection._ping_if_idle(dbapi, _record(time.monotonic()), None)  # noqa: SLF001

    assert dbapi.cursor_obj.executed == []


def test_brand_new_connection_is_not_pinged():
    dbapi = _DbapiConnection()

    connection._ping_if_idle(dbapi, _record(None), None)  # noqa: SLF001

    assert dbapi.cursor_obj.executed == []


def test_idle_connection_is_pinged_first():
    dbapi = _DbapiConnection()
    idle_since = time.monotonic() - connection.IDLE_PING_AFTER_SECONDS - 1

    connection._ping_if_idle(dbapi, _record(idle_since), None)  # noqa: SLF001

    assert dbapi.cursor_obj.executed == ["SELECT 1"]
    assert dbapi.rollbacks == 1


def test_dead_idle_connection_is_reported_as_a_disconnect():
    dbapi = _DbapiConnection(fail=True)
    idle_since = time.monotonic() - connection.IDLE_PING_AFTER_SECONDS - 1

    with pytest.raises(DisconnectionError):
        connection._ping_if_idle(dbapi, _record(idle_since), None)  # noqa: SLF001


def test_checkin_records_last_use():
    record = _record(None)

    connection._record_checkin(None, record)  # noqa: SLF001

    assert time.monotonic() - record.info["last_used"] < 5


def test_read_router_endpoint_runs_inside_the_read_only_scope():
    from src.api.app import app
    from src.api.routers.global_response_plan import get_global_response_plan_read_service

    observed: list[bool] = []

    class RecordingService:
        def get_current(self):
            observed.append(connection.is_read_only_database_access())
            return GlobalResponsePlanResponse(as_of="2026-09-20T09:00:00Z", plan=None)

    app.dependency_overrides[get_global_response_plan_read_service] = lambda: RecordingService()
    try:
        response = TestClient(app).get("/api/v1/global-response-plan/current")
    finally:
        app.dependency_overrides.pop(get_global_response_plan_read_service, None)

    assert response.status_code == 200
    assert observed == [True]
    assert connection.is_read_only_database_access() is False


def test_only_the_pure_read_routers_get_the_read_only_scope():
    """Every /api/v1 operation (enumerated from the public OpenAPI schema) is
    sent a real request, with `read_only_request` replaced - via FastAPI's
    public dependency_overrides - by a recorder that notes it ran and then
    short-circuits the request (HTTP 418) so no scoped endpoint body or
    database read ever executes. This observes the scope actually being
    applied at request time, instead of reading route internals that newer
    FastAPI versions wrap in `_IncludedRouter` objects (no `.dependant`/
    `.path`). The unscoped routers' collaborators are faked so nothing real
    (simulation run, Gemini call) can happen."""
    from unittest.mock import MagicMock

    from fastapi import HTTPException

    from src.api.app import create_app
    from src.api.dependencies import get_chatbot_agent, get_simulation_run_manager
    from src.api.read_only_request import read_only_request

    scope_ran_for: list[bool] = []

    # No parameters on purpose: this module uses `from __future__ import
    # annotations`, so a locally-imported annotation could not be resolved by
    # FastAPI. Each request is sent one at a time, so a count suffices.
    async def recording_read_only_request():
        scope_ran_for.append(True)
        raise HTTPException(status_code=418, detail="read-only scope reached")

    class FakeChatbotAgent:
        def ask(self, question, *, history=()):
            return "fake answer"

    app = create_app()
    app.dependency_overrides[read_only_request] = recording_read_only_request
    app.dependency_overrides[get_chatbot_agent] = lambda: FakeChatbotAgent()
    app.dependency_overrides[get_simulation_run_manager] = lambda: MagicMock()
    client = TestClient(app, raise_server_exceptions=False)

    operations = [
        (path, method.upper())
        for path, item in app.openapi()["paths"].items()
        if path.startswith("/api/v1/")
        for method in item
    ]
    assert operations, "no /api/v1 operations found"

    scoped_prefixes = set()
    unscoped_prefixes = set()
    scoped_methods = set()
    for path, method in operations:
        concrete_path = re.sub(r"\{[^}]+\}", "1", path)
        before = len(scope_ran_for)
        response = client.request(method, concrete_path, json={"question": "q"} if method == "POST" else None)
        scoped = len(scope_ran_for) > before
        # Status and recorder agree: the short-circuit happened iff the scope ran.
        assert (response.status_code == 418) is scoped, (method, path, response.status_code)
        (scoped_prefixes if scoped else unscoped_prefixes).add(path.split("/")[3])
        if scoped:
            scoped_methods.add(method)

    assert "simulation" in unscoped_prefixes and "chatbot" in unscoped_prefixes
    assert {"fire-events", "global-response-plan", "response-plans", "operations"} <= scoped_prefixes
    assert {"simulation", "chatbot"}.isdisjoint(scoped_prefixes)
    # Only pure GET reads ever run inside the read-only scope.
    assert scoped_methods == {"GET"}
    assert connection.is_read_only_database_access() is False

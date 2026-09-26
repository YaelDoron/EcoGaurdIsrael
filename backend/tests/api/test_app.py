"""Tests for the FastAPI application foundation (Epic 6 / US 6.1, Task 1).

Covers app creation, the /health endpoint, CORS configuration, and that
importing src.api.app has no side effects (no DB connection, no external
calls) - see test_import_app_without_database_url_configured for how that
last property is verified. Also covers the US 6.3 -> US 6.1 router
integration: that `response_plans_router` is registered on the live app
exactly once alongside the pre-existing `fire_events_router` routes.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from datetime import datetime, timezone

from src.api.app import create_app
from src.api.dependencies import get_active_fire_events_service
from src.api.routers.response_plans import (
    get_response_eligibility_reader,
    get_response_plan_details_service,
    get_response_plan_presenter,
)
from src.models.active_fire_events import ActiveFireEventsResult
from src.models.response_plan_details import ResponsePlanDetails

ALLOWED_ORIGIN = "http://localhost:5173"
DISALLOWED_ORIGIN = "http://evil.example"


def test_create_app_returns_fastapi_instance() -> None:
    app = create_app()

    assert isinstance(app, FastAPI)
    assert app.title == "EcoGuard Israel API"


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_allows_configured_frontend_origin() -> None:
    client = TestClient(create_app())

    response = client.get("/health", headers={"Origin": ALLOWED_ORIGIN})

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN


def test_cors_does_not_grant_unrestricted_access_to_other_origins() -> None:
    client = TestClient(create_app())

    response = client.get("/health", headers={"Origin": DISALLOWED_ORIGIN})

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


class _FakeResponsePlanDetailsService:
    """Always reports "no plan found" - enough to prove the route is wired
    through the real app without needing a full ResponsePlanDetails/
    ResponsePlanPresenter fixture (see test_response_plans_router.py for
    that router's own thorough behavioral coverage)."""

    def get_current_plan_details(self, fire_event_id: int) -> ResponsePlanDetails | None:
        return None

    def get_plan_details_by_id(self, plan_id: int) -> ResponsePlanDetails | None:
        return None


def _override_response_plan_dependencies(app: FastAPI) -> None:
    app.dependency_overrides[get_response_plan_details_service] = _FakeResponsePlanDetailsService
    # `get_response_plan_presenter` builds real DB-backed repositories, but
    # `_FakeResponsePlanDetailsService` always returns None, so the
    # presenter is never actually invoked - it just needs to construct
    # without touching the database.
    app.dependency_overrides[get_response_plan_presenter] = lambda: None
    app.dependency_overrides[get_response_eligibility_reader] = lambda: _NotEligible()


class _NotEligible:
    def is_response_eligible(self, fire_event_id: int) -> bool:
        return False


def test_response_plan_current_plan_endpoint_is_exposed_on_the_app() -> None:
    app = create_app()
    _override_response_plan_dependencies(app)

    response = TestClient(app).get("/api/v1/fire-events/3/response-plan")

    assert response.status_code == 200
    assert response.json() == {"plan": None, "plan_status": "not_applicable"}


def test_response_plan_by_id_endpoint_is_exposed_on_the_app() -> None:
    app = create_app()
    _override_response_plan_dependencies(app)

    response = TestClient(app).get("/api/v1/response-plans/999")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RESPONSE_PLAN_NOT_FOUND"


def test_response_plan_and_fire_event_routers_are_each_included_exactly_once() -> None:
    """`v1_router.include_router(...)` is the single mount point for each
    sub-router (see routers/__init__.py's own docstring) - this checks its
    source directly rather than app.routes, since FastAPI's route tree is an
    internal, version-specific structure that isn't a stable thing to
    introspect for a "not duplicated" check.
    """
    source = (Path(__file__).resolve().parents[2] / "src/api/routers/__init__.py").read_text(encoding="utf-8")

    # Prefix match: the include may carry keyword arguments (e.g. the
    # read-only request dependency), but must still appear exactly once.
    assert source.count("v1_router.include_router(fire_events_router") == 1
    assert source.count("v1_router.include_router(response_plans_router") == 1


def test_existing_active_fire_events_route_still_works_after_integration() -> None:
    app = create_app()
    app.dependency_overrides[get_active_fire_events_service] = lambda: _FakeActiveFireEventsService()

    response = TestClient(app).get("/api/v1/fire-events/active")

    assert response.status_code == 200
    assert response.json() == {"as_of": "2026-09-17T13:20:00Z", "items": []}


class _FakeActiveFireEventsService:
    def get_active_events(self, *, as_of=None) -> ActiveFireEventsResult:
        return ActiveFireEventsResult(as_of=datetime(2026, 9, 17, 13, 20, tzinfo=timezone.utc), items=())


def test_import_app_without_database_url_configured() -> None:
    """Importing src.api.app must not touch the database.

    get_engine() raises DatabaseConfigurationError immediately if
    DATABASE_URL is unset and a connection is attempted (see
    src/database/connection.py), so a clean-process import succeeding with
    DATABASE_URL explicitly cleared proves no DB connection is attempted at
    import time. Run in a subprocess so this process's already-imported
    modules/engine state can't mask the check.
    """
    backend_dir = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["DATABASE_URL"] = ""

    result = subprocess.run(
        [sys.executable, "-c", "import src.api.app"],
        cwd=str(backend_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr

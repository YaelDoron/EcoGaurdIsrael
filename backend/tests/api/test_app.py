"""Tests for the FastAPI application foundation (Epic 6 / US 6.1, Task 1).

Covers app creation, the /health endpoint, CORS configuration, and that
importing src.api.app has no side effects (no DB connection, no external
calls) - see test_import_app_without_database_url_configured for how that
last property is verified.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.app import create_app

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

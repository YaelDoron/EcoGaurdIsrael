"""FastAPI application factory for the EcoGuard Israel backend API.

`create_app()` only wires up middleware and routers. It deliberately does
not initialize the database schema, run migrations, or contact any agent
or external wildfire data provider (IMS/FIRMS/Copernicus) - so importing
this module, and calling `create_app()`, is always safe (e.g. in tests).

Run locally from `backend/`:
    python -m uvicorn src.api.app:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import v1_router
from src.api.routers.health import health_router
from src.config.settings import settings

API_TITLE = "EcoGuard Israel API"


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(title=API_TITLE)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.FRONTEND_ORIGINS),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(v1_router)

    return app


app = create_app()

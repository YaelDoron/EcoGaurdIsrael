"""Liveness endpoint.

Proves the API process is up and responding. Deliberately does not query
the database, call agents, or reach IMS/FIRMS/Copernicus.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

health_router = APIRouter()


class HealthResponse(BaseModel):
    status: str


@health_router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(status="ok")

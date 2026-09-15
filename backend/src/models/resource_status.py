"""Operational status of a firefighting resource."""
from __future__ import annotations

from enum import Enum


class ResourceStatus(Enum):
    """Whether a firefighting resource is available for dispatch."""

    AVAILABLE = "available"
    ASSIGNED = "assigned"
    UNAVAILABLE = "unavailable"

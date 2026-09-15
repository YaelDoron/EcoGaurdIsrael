"""Domain-level causes for operational refresh reevaluation."""
from __future__ import annotations

from enum import Enum


class OperationalRefreshTriggerType(Enum):
    """Business events that may require operational information to be revisited."""

    WEATHER_UPDATE = "weather_update"
    FIRE_EVENT_UPDATE = "fire_event_update"
    SEVERITY_UPDATE = "severity_update"
    RESOURCE_STATUS_UPDATE = "resource_status_update"

"""Aggregate result models for production operational refresh orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.agents.analysis.response_target_generation_result import ResponseTargetGenerationResult
from src.models.firefighting_resource import FirefightingResource
from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType
from src.repositories.fire_severity_assessment_repository import StoredFireSeverityAssessment
from src.services.operational_refresh.fire_spread_refresh_result import FireSpreadRefreshResult
from src.services.operational_refresh.resource_status_update_result import ResourceStatusUpdateResult


class OperationalRefreshStatus(Enum):
    """Top-level outcome of one operational refresh request."""

    REFRESHED = "refreshed"
    NO_OP = "no_op"
    INACTIVE_EVENT = "inactive_event"
    RESOURCE_UPDATED = "resource_updated"
    RESOURCE_NO_OP = "resource_no_op"
    RESOURCE_NOT_FOUND = "resource_not_found"
    FAILED = "failed"


@dataclass(frozen=True)
class OperationalRefreshResult:
    """Structured aggregate result for one environmental/fire or resource refresh."""

    trigger_type: OperationalRefreshTriggerType
    status: OperationalRefreshStatus
    success: bool
    fire_event_id: int | None = None
    as_of: datetime | None = None
    severity_result: StoredFireSeverityAssessment | None = None
    spread_refresh_result: FireSpreadRefreshResult | None = None
    response_target_result: ResponseTargetGenerationResult | None = None
    resource_status_result: ResourceStatusUpdateResult | None = None
    available_resources: tuple[FirefightingResource, ...] = ()
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.trigger_type, OperationalRefreshTriggerType):
            raise ValueError(f"trigger_type must be an OperationalRefreshTriggerType, got {self.trigger_type!r}.")
        if not isinstance(self.status, OperationalRefreshStatus):
            raise ValueError(f"status must be an OperationalRefreshStatus, got {self.status!r}.")
        if not isinstance(self.success, bool):
            raise ValueError(f"success must be a bool, got {self.success!r}.")
        if self.fire_event_id is not None and (
            isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0
        ):
            raise ValueError(f"fire_event_id must be a positive integer or None, got {self.fire_event_id!r}.")
        if self.as_of is not None and (not isinstance(self.as_of, datetime) or self.as_of.tzinfo is None):
            raise ValueError(f"as_of must be a timezone-aware datetime or None, got {self.as_of!r}.")
        available_resources = tuple(self.available_resources)
        for resource in available_resources:
            if not isinstance(resource, FirefightingResource):
                raise ValueError(f"available_resources must contain FirefightingResource items, got {resource!r}.")
        object.__setattr__(self, "available_resources", available_resources)
        if self.success and self.error_message is not None:
            raise ValueError("successful operational refresh results must not include error_message.")
        if not self.success and (not isinstance(self.error_message, str) or not self.error_message.strip()):
            raise ValueError("failed operational refresh results must include error_message.")

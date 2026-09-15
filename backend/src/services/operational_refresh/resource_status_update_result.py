"""Result objects for operational firefighting-resource status updates."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.models.firefighting_resource import FirefightingResource
from src.models.resource_status import ResourceStatus


class ResourceStatusUpdateStatus(Enum):
    """Outcome of a requested firefighting-resource status transition."""

    UPDATED = "updated"
    NO_OP = "no_op"
    NOT_FOUND = "not_found"
    FAILED = "failed"


@dataclass(frozen=True)
class ResourceStatusUpdateResult:
    """Service-layer result for updating one existing firefighting resource."""

    resource_id: int | str
    previous_status: ResourceStatus | None
    current_status: ResourceStatus | None
    status: ResourceStatusUpdateStatus
    resource: FirefightingResource | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ResourceStatusUpdateStatus):
            raise ValueError(f"status must be a ResourceStatusUpdateStatus, got {self.status!r}.")
        for field_name, value in (
            ("previous_status", self.previous_status),
            ("current_status", self.current_status),
        ):
            if value is not None and not isinstance(value, ResourceStatus):
                raise ValueError(f"{field_name} must be a ResourceStatus or None, got {value!r}.")
        if self.resource is not None and not isinstance(self.resource, FirefightingResource):
            raise ValueError(f"resource must be a FirefightingResource or None, got {self.resource!r}.")
        if self.error_message is not None and not isinstance(self.error_message, str):
            raise ValueError(f"error_message must be a string or None, got {self.error_message!r}.")
        if self.status in {
            ResourceStatusUpdateStatus.UPDATED,
            ResourceStatusUpdateStatus.NO_OP,
        }:
            if self.previous_status is None or self.current_status is None or self.resource is None:
                raise ValueError("successful resource status results must include statuses and resource.")
            if self.error_message is not None:
                raise ValueError("successful resource status results must not include error_message.")
        if self.status is ResourceStatusUpdateStatus.NOT_FOUND:
            if self.previous_status is not None or self.current_status is not None or self.resource is not None:
                raise ValueError("NOT_FOUND results must not include statuses or resource.")

    @property
    def changed(self) -> bool:
        """Whether the request persisted a changed resource status."""
        return self.status is ResourceStatusUpdateStatus.UPDATED

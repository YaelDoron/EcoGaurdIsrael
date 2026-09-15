"""Resource-status change payloads for simulation timeline events."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.resource_status import ResourceStatus


@dataclass(frozen=True)
class SimulationResourceStatusChange:
    """A deterministic simulated status transition for one firefighting resource."""

    new_status: ResourceStatus
    resource_id: int | str | None = None
    selection_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.new_status, ResourceStatus):
            raise ValueError(f"new_status must be a ResourceStatus, got {self.new_status!r}.")
        if self.resource_id is not None and (
            not isinstance(self.resource_id, (int, str))
            or isinstance(self.resource_id, bool)
            or (isinstance(self.resource_id, int) and self.resource_id <= 0)
            or (isinstance(self.resource_id, str) and not self.resource_id.strip())
        ):
            raise ValueError(
                f"resource_id must be a positive int, non-empty string, or None, got {self.resource_id!r}."
            )
        if self.selection_key is not None and (
            not isinstance(self.selection_key, str) or not self.selection_key.strip()
        ):
            raise ValueError(f"selection_key must be a non-empty string or None, got {self.selection_key!r}.")

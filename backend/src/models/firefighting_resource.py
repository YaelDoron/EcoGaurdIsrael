"""EcoGuard's internal representation of a firefighting resource (e.g. a crew or vehicle)."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.resource_status import ResourceStatus


@dataclass
class FirefightingResource:
    """A firefighting resource attached to a fire station."""

    id: int | str
    station_id: int | str
    status: ResourceStatus

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, (int, str))
            or isinstance(self.id, bool)
            or (isinstance(self.id, int) and self.id <= 0)
            or (isinstance(self.id, str) and not self.id.strip())
        ):
            raise ValueError(f"id must be a positive int or non-empty string, got {self.id!r}")

        if (
            not isinstance(self.station_id, (int, str))
            or isinstance(self.station_id, bool)
            or (isinstance(self.station_id, int) and self.station_id <= 0)
            or (isinstance(self.station_id, str) and not self.station_id.strip())
        ):
            raise ValueError(
                f"station_id must be a positive int or non-empty string, got {self.station_id!r}"
            )

        if not isinstance(self.status, ResourceStatus):
            raise ValueError(f"status must be a ResourceStatus, got {self.status!r}")

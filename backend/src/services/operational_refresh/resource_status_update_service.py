"""Application service for changing firefighting-resource availability."""
from __future__ import annotations

import logging

from src.models.resource_status import ResourceStatus
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.services.operational_refresh.resource_status_update_result import (
    ResourceStatusUpdateResult,
    ResourceStatusUpdateStatus,
)

logger = logging.getLogger(__name__)


class ResourceStatusUpdateService:
    """Update the status of an existing US 4.1 firefighting resource."""

    def __init__(self, resource_repository: FirefightingResourceRepository) -> None:
        self._resource_repository = resource_repository

    def update_status(
        self,
        *,
        resource_id: int | str,
        new_status: ResourceStatus,
    ) -> ResourceStatusUpdateResult:
        """Persist a resource status change, returning a deterministic operation result."""
        self._validate_resource_id(resource_id)
        self._validate_status(new_status)

        try:
            current_resource = self._resource_repository.get_by_id(resource_id)
            if current_resource is None:
                return ResourceStatusUpdateResult(
                    resource_id=resource_id,
                    previous_status=None,
                    current_status=None,
                    status=ResourceStatusUpdateStatus.NOT_FOUND,
                )

            previous_status = current_resource.status
            if previous_status is new_status:
                return ResourceStatusUpdateResult(
                    resource_id=current_resource.id,
                    previous_status=previous_status,
                    current_status=previous_status,
                    status=ResourceStatusUpdateStatus.NO_OP,
                    resource=current_resource,
                )

            updated_resource = self._resource_repository.update_status(current_resource.id, new_status)
            if updated_resource is None:
                return ResourceStatusUpdateResult(
                    resource_id=current_resource.id,
                    previous_status=previous_status,
                    current_status=None,
                    status=ResourceStatusUpdateStatus.NOT_FOUND,
                )

            return ResourceStatusUpdateResult(
                resource_id=updated_resource.id,
                previous_status=previous_status,
                current_status=updated_resource.status,
                status=ResourceStatusUpdateStatus.UPDATED,
                resource=updated_resource,
            )
        except Exception as exc:
            logger.exception("Failed to update firefighting resource %r status", resource_id)
            return ResourceStatusUpdateResult(
                resource_id=resource_id,
                previous_status=None,
                current_status=None,
                status=ResourceStatusUpdateStatus.FAILED,
                error_message=str(exc),
            )

    @staticmethod
    def _validate_resource_id(resource_id: int | str) -> None:
        if (
            not isinstance(resource_id, (int, str))
            or isinstance(resource_id, bool)
            or (isinstance(resource_id, int) and resource_id <= 0)
            or (isinstance(resource_id, str) and not resource_id.strip())
        ):
            raise ValueError(f"resource_id must be a positive int or non-empty string, got {resource_id!r}.")

    @staticmethod
    def _validate_status(status: ResourceStatus) -> None:
        if not isinstance(status, ResourceStatus):
            raise ValueError(f"new_status must be a ResourceStatus, got {status!r}.")

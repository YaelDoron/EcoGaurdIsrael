"""Tests for firefighting-resource domain validation."""
from __future__ import annotations

import pytest

from src.models.firefighting_resource import FirefightingResource
from src.models.resource_status import ResourceStatus


def test_firefighting_resource_accepts_existing_resource_status_enum():
    resource = FirefightingResource(id="TRUCK-1-1", station_id="1", status=ResourceStatus.AVAILABLE)

    assert resource.id == "TRUCK-1-1"
    assert resource.station_id == "1"
    assert resource.status is ResourceStatus.AVAILABLE


@pytest.mark.parametrize("status", ["available", None, object()])
def test_firefighting_resource_rejects_non_enum_status(status):
    with pytest.raises(ValueError):
        FirefightingResource(id="TRUCK-1-1", station_id="1", status=status)

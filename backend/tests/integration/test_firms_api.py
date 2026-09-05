"""Optional live integration test against the NASA FIRMS Area CSV API.

Run explicitly:

    python -m pytest tests/integration/test_firms_api.py -m integration -v

The test performs one request, skips when FIRMS_MAP_KEY is absent, and never
prints the key or full FIRMS URL.
"""
import pytest

from src.config.settings import settings
from src.external.firms.firms_client import FIRMSClient

pytestmark = pytest.mark.integration


def test_firms_area_api_returns_list_for_configured_area() -> None:
    if not settings.FIRMS_MAP_KEY:
        pytest.skip("FIRMS_MAP_KEY is not configured; skipping live FIRMS integration test.")

    client = FIRMSClient()
    detections = client.get_area_hotspots()

    assert isinstance(detections, list)
    for detection in detections:
        assert "latitude" in detection
        assert "longitude" in detection

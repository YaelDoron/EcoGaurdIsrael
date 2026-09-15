"""Copernicus Data Space external integration."""

from src.external.copernicus.copernicus_land_cover_client import (
    CopernicusAuthenticationError,
    CopernicusClientError,
    CopernicusConfigurationError,
    CopernicusCoverFraction,
    CopernicusInvalidResponseError,
    CopernicusLandCoverClient,
    CopernicusLandCoverStatistics,
    CopernicusServiceUnavailableError,
)

__all__ = [
    "CopernicusLandCoverClient",
    "CopernicusLandCoverStatistics",
    "CopernicusCoverFraction",
    "CopernicusClientError",
    "CopernicusConfigurationError",
    "CopernicusAuthenticationError",
    "CopernicusServiceUnavailableError",
    "CopernicusInvalidResponseError",
]

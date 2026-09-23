"""HTTP client for Copernicus Data Space land-cover statistics.

This client owns only external communication with Copernicus Data Space
Ecosystem and Sentinel Hub Statistical API. It does not assign EcoGuard fuel
weights, calculate severity, or access repositories.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time

import requests

from src.config.settings import settings

logger = logging.getLogger(__name__)

COPERNICUS_LAND_COVER_DATASET_YEAR = 2019
COPERNICUS_LAND_COVER_TIME_FROM = "2019-01-01T00:00:00Z"
COPERNICUS_LAND_COVER_TIME_TO = "2020-01-01T00:00:00Z"
TOKEN_EXPIRY_SAFETY_SECONDS = 60

FRACTIONAL_COVER_BANDS: dict[str, str] = {
    "tree": "Tree_Cover_Fraction",
    "shrub": "Shrub_Cover_Fraction",
    "grass": "Grass_Cover_Fraction",
    "crops": "Crops_Cover_Fraction",
    "bare": "Bare_Cover_Fraction",
    "built_up": "BuiltUp_Cover_Fraction",
    "permanent_water": "PermanentWater_Cover_Fraction",
    "seasonal_water": "SeasonalWater_Cover_fraction",
    "moss_lichen": "MossLichen_Cover_Fraction",
    "snow": "Snow_Cover_Fraction",
}

_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: [
        "Tree_Cover_Fraction",
        "Shrub_Cover_Fraction",
        "Grass_Cover_Fraction",
        "Crops_Cover_Fraction",
        "Bare_Cover_Fraction",
        "BuiltUp_Cover_Fraction",
        "PermanentWater_Cover_Fraction",
        "SeasonalWater_Cover_fraction",
        "MossLichen_Cover_Fraction",
        "Snow_Cover_Fraction",
        "dataMask"
      ],
      units: "DN"
    }],
    output: [
      {
        id: "vegetation",
        bands: 10,
        sampleType: "FLOAT32"
      },
      {
        id: "dataMask",
        bands: 1
      }
    ]
  };
}

function evaluatePixel(sample) {
  return {
    vegetation: [
      sample.Tree_Cover_Fraction / 100.0,
      sample.Shrub_Cover_Fraction / 100.0,
      sample.Grass_Cover_Fraction / 100.0,
      sample.Crops_Cover_Fraction / 100.0,
      sample.Bare_Cover_Fraction / 100.0,
      sample.BuiltUp_Cover_Fraction / 100.0,
      sample.PermanentWater_Cover_Fraction / 100.0,
      sample.SeasonalWater_Cover_fraction / 100.0,
      sample.MossLichen_Cover_Fraction / 100.0,
      sample.Snow_Cover_Fraction / 100.0
    ],
    dataMask: [sample.dataMask]
  };
}
""".strip()

_OUTPUT_BAND_NAMES = tuple(FRACTIONAL_COVER_BANDS.keys())
_EARTH_RADIUS_KM = 6371.0088


class CopernicusClientError(Exception):
    """Base exception for Copernicus Data Space client failures."""


class CopernicusConfigurationError(CopernicusClientError):
    """Raised when required Copernicus credentials are missing."""


class CopernicusAuthenticationError(CopernicusClientError):
    """Raised when OAuth authentication fails."""


class CopernicusServiceUnavailableError(CopernicusClientError):
    """Raised when Copernicus is unreachable or returns a server-side error."""


class CopernicusInvalidResponseError(CopernicusClientError):
    """Raised when Copernicus returns malformed or unusable statistics."""


@dataclass(frozen=True)
class CopernicusCoverFraction:
    """Fractional land-cover category from Sentinel Hub statistics."""

    category: str
    fraction: float

    def __post_init__(self) -> None:
        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError(f"category must be non-empty, got {self.category!r}")
        if isinstance(self.fraction, bool) or not isinstance(self.fraction, (int, float)):
            raise ValueError(f"fraction must be numeric, got {self.fraction!r}")
        if not math.isfinite(self.fraction) or self.fraction < 0:
            raise ValueError(f"fraction must be finite and non-negative, got {self.fraction!r}")


@dataclass(frozen=True)
class CopernicusLandCoverStatistics:
    """Compact raw land-cover statistics for a requested geometry."""

    cover_fractions: tuple[CopernicusCoverFraction, ...]
    source: str
    dataset_year: int
    radius_km: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "cover_fractions", tuple(self.cover_fractions))
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError(f"source must be non-empty, got {self.source!r}")
        if isinstance(self.dataset_year, bool) or not isinstance(self.dataset_year, int) or self.dataset_year <= 0:
            raise ValueError(f"dataset_year must be positive, got {self.dataset_year!r}")
        if isinstance(self.radius_km, bool) or not isinstance(self.radius_km, (int, float)):
            raise ValueError(f"radius_km must be numeric, got {self.radius_km!r}")
        if not math.isfinite(self.radius_km) or self.radius_km <= 0:
            raise ValueError(f"radius_km must be positive and finite, got {self.radius_km!r}")


class CopernicusLandCoverClient:
    """Client for Copernicus Data Space OAuth and Sentinel Hub statistics."""

    def __init__(
        self,
        client_id: str = settings.COPERNICUS_CLIENT_ID,
        client_secret: str = settings.COPERNICUS_CLIENT_SECRET,
        token_url: str = settings.COPERNICUS_TOKEN_URL,
        statistics_url: str = settings.COPERNICUS_STATISTICS_URL,
        collection_id: str = settings.COPERNICUS_LAND_COVER_COLLECTION_ID,
        timeout: int = settings.COPERNICUS_REQUEST_TIMEOUT,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._statistics_url = statistics_url
        self._collection_id = collection_id
        self._timeout = timeout
        self._access_token: str | None = None
        self._token_expires_at = 0.0
        # Performance pass: profiling one full demo run showed 17 Severity
        # refreshes issuing 17 live Copernicus statistics calls for only 4
        # distinct (latitude, longitude) pairs - the same FireEvent location
        # queried repeatedly across refresh cycles. The queried dataset is a
        # fixed historical snapshot (COPERNICUS_LAND_COVER_DATASET_YEAR /
        # _TIME_FROM / _TIME_TO are constants, never `as_of`-dependent), so
        # the exact same request is provably time-invariant for the life of
        # this client instance - a request-shaped in-process memo, not a
        # blind TTL. Keyed on every semantic parameter that could change the
        # result (coordinates, radius, collection, dataset time range), so a
        # FireEvent location change (a new latitude/longitude) is a cache
        # miss, never stale data. Invalidation is simply process restart -
        # this cache is never persisted, matching every other reuse mechanism
        # in this codebase which re-derives correctness from persisted state
        # after a restart.
        self._statistics_cache: dict[tuple, CopernicusLandCoverStatistics | None] = {}

    def get_land_cover_statistics(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
    ) -> CopernicusLandCoverStatistics | None:
        """Return fractional land-cover statistics, or None when no data is available."""
        _validate_coordinate("latitude", latitude, -90, 90)
        _validate_coordinate("longitude", longitude, -180, 180)
        _validate_radius(radius_km)

        cache_key = (
            latitude,
            longitude,
            radius_km,
            self._collection_id,
            COPERNICUS_LAND_COVER_TIME_FROM,
            COPERNICUS_LAND_COVER_TIME_TO,
        )
        if cache_key in self._statistics_cache:
            return self._statistics_cache[cache_key]

        statistics = self._fetch_land_cover_statistics(latitude, longitude, radius_km)
        self._statistics_cache[cache_key] = statistics
        return statistics

    def _fetch_land_cover_statistics(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
    ) -> CopernicusLandCoverStatistics | None:
        token = self._get_access_token()
        request_body = self._build_statistics_request(latitude, longitude, radius_km)
        try:
            response = requests.post(
                self._statistics_url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=request_body,
                timeout=self._timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise CopernicusServiceUnavailableError("Copernicus statistics request timed out.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise CopernicusServiceUnavailableError("Could not connect to Copernicus statistics API.") from exc
        except requests.exceptions.RequestException as exc:
            raise CopernicusClientError("Copernicus statistics request failed.") from exc

        if response.status_code in (401, 403):
            raise CopernicusAuthenticationError("Copernicus statistics authentication failed.")
        if 400 <= response.status_code < 500:
            raise CopernicusClientError(f"Copernicus statistics request failed with HTTP {response.status_code}.")
        if response.status_code >= 500:
            raise CopernicusServiceUnavailableError(
                f"Copernicus statistics service returned HTTP {response.status_code}."
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise CopernicusInvalidResponseError("Copernicus statistics returned invalid JSON.") from exc
        return _parse_statistics_payload(payload, radius_km=radius_km)

    def _get_access_token(self) -> str:
        now = time.time()
        if self._access_token and now < self._token_expires_at - TOKEN_EXPIRY_SAFETY_SECONDS:
            return self._access_token
        if not self._client_id or not self._client_secret:
            raise CopernicusConfigurationError("Copernicus credentials are not configured.")

        try:
            response = requests.post(
                self._token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                timeout=self._timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise CopernicusServiceUnavailableError("Copernicus token request timed out.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise CopernicusServiceUnavailableError("Could not connect to Copernicus token endpoint.") from exc
        except requests.exceptions.RequestException as exc:
            raise CopernicusClientError("Copernicus token request failed.") from exc

        if response.status_code in (401, 403):
            raise CopernicusAuthenticationError("Copernicus authentication failed.")
        if response.status_code >= 500:
            raise CopernicusServiceUnavailableError(
                f"Copernicus token service returned HTTP {response.status_code}."
            )
        if response.status_code >= 400:
            raise CopernicusClientError(f"Copernicus token request failed with HTTP {response.status_code}.")

        try:
            payload = response.json()
        except ValueError as exc:
            raise CopernicusInvalidResponseError("Copernicus token endpoint returned invalid JSON.") from exc

        access_token = payload.get("access_token")
        expires_in = payload.get("expires_in")
        if not isinstance(access_token, str) or not access_token.strip():
            raise CopernicusInvalidResponseError("Copernicus token response did not include an access token.")
        if isinstance(expires_in, bool) or not isinstance(expires_in, (int, float)) or expires_in <= 0:
            raise CopernicusInvalidResponseError("Copernicus token response did not include a valid expiry.")

        self._access_token = access_token
        self._token_expires_at = now + float(expires_in)
        return access_token

    def _build_statistics_request(self, latitude: float, longitude: float, radius_km: float) -> dict:
        return {
            "input": {
                "bounds": {
                    "geometry": _square_geometry(latitude=latitude, longitude=longitude, radius_km=radius_km),
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [
                    {
                        "type": f"byoc-{self._collection_id}",
                        "dataFilter": {
                            "timeRange": {
                                "from": COPERNICUS_LAND_COVER_TIME_FROM,
                                "to": COPERNICUS_LAND_COVER_TIME_TO,
                            }
                        },
                    }
                ],
            },
            "aggregation": {
                "timeRange": {
                    "from": COPERNICUS_LAND_COVER_TIME_FROM,
                    "to": COPERNICUS_LAND_COVER_TIME_TO,
                },
                "aggregationInterval": {"of": "P1Y"},
                "resx": 100,
                "resy": 100,
                "evalscript": _EVALSCRIPT,
            },
        }


def _parse_statistics_payload(payload: dict, radius_km: float) -> CopernicusLandCoverStatistics | None:
    try:
        intervals = payload["data"]
    except (KeyError, TypeError) as exc:
        raise CopernicusInvalidResponseError("Copernicus statistics response is missing data intervals.") from exc
    if not intervals:
        return None
    try:
        bands = intervals[0]["outputs"]["vegetation"]["bands"]
    except (KeyError, TypeError) as exc:
        raise CopernicusInvalidResponseError("Copernicus statistics response is missing vegetation bands.") from exc

    fractions = []
    for index, category in enumerate(_OUTPUT_BAND_NAMES):
        band_stats = bands.get(f"B{index}")
        if not isinstance(band_stats, dict):
            continue
        stats = band_stats.get("stats")
        if not isinstance(stats, dict):
            continue
        mean = stats.get("mean")
        if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean):
            continue
        if mean <= 0:
            continue
        fractions.append(CopernicusCoverFraction(category=category, fraction=max(0.0, float(mean))))

    if not fractions:
        return None
    return CopernicusLandCoverStatistics(
        cover_fractions=tuple(fractions),
        source="COPERNICUS_GLOBAL_LAND_COVER_100M_API",
        dataset_year=COPERNICUS_LAND_COVER_DATASET_YEAR,
        radius_km=radius_km,
    )


def _square_geometry(latitude: float, longitude: float, radius_km: float) -> dict:
    latitude_delta = math.degrees(radius_km / _EARTH_RADIUS_KM)
    longitude_scale = _EARTH_RADIUS_KM * math.cos(math.radians(latitude))
    longitude_delta = 180.0 if abs(longitude_scale) < 1e-9 else math.degrees(radius_km / abs(longitude_scale))
    west = longitude - longitude_delta
    east = longitude + longitude_delta
    south = latitude - latitude_delta
    north = latitude + latitude_delta
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]
        ],
    }


def _validate_coordinate(field_name: str, value: object, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number, got {value!r}")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field_name} must be within [{minimum}, {maximum}], got {value!r}")


def _validate_radius(radius_km: object) -> None:
    if isinstance(radius_km, bool) or not isinstance(radius_km, (int, float)) or not math.isfinite(radius_km):
        raise ValueError(f"radius_km must be a finite number, got {radius_km!r}")
    if radius_km <= 0:
        raise ValueError(f"radius_km must be greater than 0, got {radius_km!r}")

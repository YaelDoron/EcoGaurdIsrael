"""Converts a location name into latitude/longitude via Nominatim (OpenStreetMap)."""
import logging

from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

logger = logging.getLogger(__name__)


class Geocoder:
    def __init__(self, config: dict):
        geolocator = Nominatim(
            user_agent=config["user_agent"],
            timeout=config.get("timeout_seconds", 10),
        )
        # Nominatim's usage policy caps free requests at ~1/sec.
        self._geocode = RateLimiter(
            geolocator.geocode, min_delay_seconds=config.get("min_delay_seconds", 1)
        )
        self.country_codes = config.get("country_bias")

    def geocode(self, location_name: str | None) -> tuple[float | None, float | None]:
        """Best-effort geocoding. Never raises: returns (None, None) on any failure or miss."""
        if not location_name:
            return None, None

        try:
            result = self._geocode(
                location_name, country_codes=self.country_codes, exactly_one=True
            )
        except (GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable) as exc:
            logger.error("Geocoding service error for '%s': %s", location_name, exc)
            return None, None
        except Exception:
            logger.exception("Unexpected geocoding error for '%s'", location_name)
            return None, None

        if result is None:
            logger.warning("Geocoding found no match for location: '%s'", location_name)
            return None, None

        return result.latitude, result.longitude

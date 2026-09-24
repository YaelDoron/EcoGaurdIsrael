"""Normalized evidence item for pure active-wildfire detection."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real

from src.models.fire_evidence_type import FireEvidenceType
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength

_ALLOWED_SATELLITE_CONFIDENCE = {"low", "nominal", "high"}


@dataclass(frozen=True)
class FireDetectionEvidence:
    """One direct evidence item that may support active wildfire detection.

    This model intentionally does not duplicate raw satellite/news domain
    objects. `satellite_frp`/`satellite_brightness`/`satellite_day_night` and
    `news_wildfire_signal_strength` are optional, source-specific enrichment
    (Task 4 / V3): selected existing satellite measurements and the LLM-derived
    news signal, exposed here so the ML feature layer can use them. They are
    NOT used by FireDetectionCalculator's Task 1 confidence methodology, which
    is unchanged and continues to use only `satellite_confidence` and evidence
    presence/correlation.

    `location_name` is optional, trustworthy provenance carried forward from
    `SatelliteHotspot.location_name` ONLY (never from news evidence - see
    FireDetectionEvidenceService._normalize_news's own docstring for why
    real-ingested news location strings are not verified/trustworthy enough
    to become a persisted FireEvent's location). It never affects detection
    confidence/correlation - purely a downstream label for FireEvent creation.
    """

    evidence_id: int
    evidence_type: FireEvidenceType
    latitude: float
    longitude: float
    observed_at: datetime
    satellite_confidence: str | None = None
    location_name: str | None = None
    satellite_frp: float | None = None
    satellite_brightness: float | None = None
    satellite_day_night: str | None = None
    news_wildfire_signal_strength: NewsWildfireSignalStrength | None = None
    # Task 5B: the persisted `satellite` / `instrument` of the hotspot (SatelliteHotspot fields that
    # already exist). Used ONLY to keep observation waves of different platforms apart when grouping
    # satellite passes for event history; never by confidence, correlation or the ML features.
    satellite_name: str | None = None
    satellite_instrument: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.evidence_id, bool) or not isinstance(self.evidence_id, int) or self.evidence_id <= 0:
            raise ValueError(f"evidence_id must be a positive integer, got {self.evidence_id!r}")
        if not isinstance(self.evidence_type, FireEvidenceType):
            raise ValueError(f"evidence_type must be a FireEvidenceType, got {self.evidence_type!r}")
        self._validate_coordinate("latitude", self.latitude, -90, 90)
        self._validate_coordinate("longitude", self.longitude, -180, 180)
        if not isinstance(self.observed_at, datetime) or self.observed_at.tzinfo is None:
            raise ValueError(f"observed_at must be a timezone-aware datetime, got {self.observed_at!r}")

        if self.evidence_type is FireEvidenceType.SATELLITE:
            if self.satellite_confidence not in _ALLOWED_SATELLITE_CONFIDENCE:
                raise ValueError(
                    "satellite_confidence must be one of "
                    f"{sorted(_ALLOWED_SATELLITE_CONFIDENCE)}, got {self.satellite_confidence!r}"
                )
            if self.news_wildfire_signal_strength is not None:
                raise ValueError("SATELLITE evidence must not include news_wildfire_signal_strength.")
            self._validate_non_negative_optional_number("satellite_frp", self.satellite_frp)
            self._validate_non_negative_optional_number("satellite_brightness", self.satellite_brightness)
            if self.satellite_day_night is not None and not isinstance(self.satellite_day_night, str):
                raise ValueError(f"satellite_day_night must be a string or None, got {self.satellite_day_night!r}")
            for field_name in ("satellite_name", "satellite_instrument"):
                value = getattr(self, field_name)
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"{field_name} must be a string or None, got {value!r}")
        else:
            if self.satellite_confidence is not None:
                raise ValueError("NEWS evidence must not include satellite_confidence.")
            if self.satellite_frp is not None:
                raise ValueError("NEWS evidence must not include satellite_frp.")
            if self.satellite_brightness is not None:
                raise ValueError("NEWS evidence must not include satellite_brightness.")
            if self.satellite_day_night is not None:
                raise ValueError("NEWS evidence must not include satellite_day_night.")
            if self.satellite_name is not None:
                raise ValueError("NEWS evidence must not include satellite_name.")
            if self.satellite_instrument is not None:
                raise ValueError("NEWS evidence must not include satellite_instrument.")
            if self.news_wildfire_signal_strength is not None and not isinstance(
                self.news_wildfire_signal_strength, NewsWildfireSignalStrength
            ):
                raise ValueError(
                    "news_wildfire_signal_strength must be a NewsWildfireSignalStrength or None, "
                    f"got {self.news_wildfire_signal_strength!r}"
                )

        if self.location_name is not None and not isinstance(self.location_name, str):
            raise ValueError(f"location_name must be a string or None, got {self.location_name!r}")

    @staticmethod
    def _validate_coordinate(field_name: str, value: object, minimum: float, maximum: float) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        if not minimum <= value <= maximum:
            raise ValueError(f"{field_name} must be within [{minimum}, {maximum}], got {value!r}")

    @staticmethod
    def _validate_non_negative_optional_number(field_name: str, value: object) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative finite number or None, got {value!r}")

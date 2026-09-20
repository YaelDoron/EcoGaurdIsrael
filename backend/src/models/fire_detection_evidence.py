"""Normalized evidence item for pure active-wildfire detection."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real

from src.models.fire_evidence_type import FireEvidenceType

_ALLOWED_SATELLITE_CONFIDENCE = {"low", "nominal", "high"}


@dataclass(frozen=True)
class FireDetectionEvidence:
    """One direct evidence item that may support active wildfire detection.

    This model intentionally does not duplicate raw satellite/news domain
    objects. FRP is excluded from Task 1 confidence calculation and remains
    available on SatelliteHotspot for future severity assessment.

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
        elif self.satellite_confidence is not None:
            raise ValueError("NEWS evidence must not include satellite_confidence.")

        if self.location_name is not None and not isinstance(self.location_name, str):
            raise ValueError(f"location_name must be a string or None, got {self.location_name!r}")

    @staticmethod
    def _validate_coordinate(field_name: str, value: object, minimum: float, maximum: float) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        if not minimum <= value <= maximum:
            raise ValueError(f"{field_name} must be within [{minimum}, {maximum}], got {value!r}")

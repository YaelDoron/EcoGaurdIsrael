"""Pure decision returned by active-wildfire evidence detection."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from src.calculators.fire_detection.fire_detection_config import (
    CONFIRMED_THRESHOLD,
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    SUSPECTED_THRESHOLD,
)
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_evidence_ref import FireEvidenceRef


@dataclass(frozen=True)
class FireDetectionDecision:
    """Confidence and status for one evaluated wildfire evidence candidate."""

    confidence: float
    status: FireDetectionStatus
    latitude: float | None
    longitude: float | None
    supporting_evidence: tuple[FireEvidenceRef, ...]

    def __post_init__(self) -> None:
        self._validate_confidence()
        if not isinstance(self.status, FireDetectionStatus):
            raise ValueError(f"status must be a FireDetectionStatus, got {self.status!r}")
        object.__setattr__(
            self,
            "supporting_evidence",
            tuple(sorted(tuple(self.supporting_evidence), key=self._supporting_evidence_sort_key)),
        )
        self._validate_supporting_evidence()

        if self.status is FireDetectionStatus.NO_EVENT:
            if self.confidence >= SUSPECTED_THRESHOLD:
                raise ValueError("NO_EVENT confidence must be below the suspected threshold.")
            if self.latitude is not None:
                self._validate_coordinate("latitude", self.latitude, -90, 90)
            if self.longitude is not None:
                self._validate_coordinate("longitude", self.longitude, -180, 180)
        elif self.status is FireDetectionStatus.SUSPECTED:
            if not SUSPECTED_THRESHOLD <= self.confidence < CONFIRMED_THRESHOLD:
                raise ValueError("SUSPECTED confidence must be within the suspected range.")
            self._validate_required_location_and_support()
        elif self.status is FireDetectionStatus.CONFIRMED:
            if self.confidence < CONFIRMED_THRESHOLD:
                raise ValueError("CONFIRMED confidence must be at or above the confirmed threshold.")
            self._validate_required_location_and_support()

    def _validate_confidence(self) -> None:
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, Real):
            raise ValueError(f"confidence must be numeric, got {self.confidence!r}")
        if not math.isfinite(self.confidence):
            raise ValueError(f"confidence must be finite, got {self.confidence!r}")
        if not MIN_CONFIDENCE <= self.confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"confidence must be within [{MIN_CONFIDENCE}, {MAX_CONFIDENCE}], got {self.confidence!r}"
            )

    def _validate_required_location_and_support(self) -> None:
        self._validate_coordinate("latitude", self.latitude, -90, 90)
        self._validate_coordinate("longitude", self.longitude, -180, 180)
        if not self.supporting_evidence:
            raise ValueError(f"{self.status.name} decisions must include supporting_evidence.")

    def _validate_supporting_evidence(self) -> None:
        if len(set(self.supporting_evidence)) != len(self.supporting_evidence):
            raise ValueError("supporting_evidence must not contain duplicates.")
        for evidence_ref in self.supporting_evidence:
            if not isinstance(evidence_ref, FireEvidenceRef):
                raise ValueError(f"supporting_evidence must contain FireEvidenceRef items, got {evidence_ref!r}")

    @staticmethod
    def _supporting_evidence_sort_key(evidence_ref: FireEvidenceRef) -> tuple[str, int]:
        if not isinstance(evidence_ref, FireEvidenceRef):
            return ("", 0)
        return (evidence_ref.evidence_type.value, evidence_ref.evidence_id)

    @staticmethod
    def _validate_coordinate(field_name: str, value: object, minimum: float, maximum: float) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        if not minimum <= value <= maximum:
            raise ValueError(f"{field_name} must be within [{minimum}, {maximum}], got {value!r}")

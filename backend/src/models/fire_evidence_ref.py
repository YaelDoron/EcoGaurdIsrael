"""Source-aware reference to persisted active-wildfire evidence."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.fire_evidence_type import FireEvidenceType


@dataclass(frozen=True)
class FireEvidenceRef:
    """Globally unambiguous identity for persisted direct fire evidence."""

    evidence_type: FireEvidenceType
    evidence_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_type, FireEvidenceType):
            raise ValueError(f"evidence_type must be a FireEvidenceType, got {self.evidence_type!r}")
        if isinstance(self.evidence_id, bool) or not isinstance(self.evidence_id, int) or self.evidence_id <= 0:
            raise ValueError(f"evidence_id must be a positive integer, got {self.evidence_id!r}")

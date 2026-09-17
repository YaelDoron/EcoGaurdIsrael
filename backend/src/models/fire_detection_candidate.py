"""Candidate active-wildfire evidence group produced before final scoring."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timezone
import math

from src.calculators.fire_detection.fire_detection_config import (
    MAX_EVIDENCE_DISTANCE_KM,
    MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES,
)
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType

_EARTH_RADIUS_KM = 6371.0
_DISTANCE_TOLERANCE_KM = 1e-9
_EVIDENCE_TYPE_SORT_ORDER = {
    FireEvidenceType.SATELLITE: 0,
    FireEvidenceType.NEWS: 1,
}


@dataclass(frozen=True)
class FireDetectionCandidate:
    """A connected component of direct fire evidence.

    Candidate identity is source-aware: satellite row 5 and news row 5 are
    distinct evidence items even though their integer database IDs overlap.

    Evidence belongs to the same candidate when it forms one connected
    component under the configured geographic and temporal correlation
    relation (see ``is_connected``). Direct correlation is not required
    between every pair inside that component: A-B and B-C correlating is
    sufficient to group A, B, and C even when A and C do not directly
    correlate. Any code that validates a group of evidence as "one
    candidate" (including FireDetectionCalculator, which may be called with
    a raw evidence tuple instead of a constructed FireDetectionCandidate)
    must use this same connectivity definition, not a stricter all-pairs
    check, or the two layers will disagree on the same data.
    """

    evidence: tuple[FireDetectionEvidence, ...]

    def __post_init__(self) -> None:
        evidence = self._normalize_evidence(self.evidence)
        if not evidence:
            raise ValueError("FireDetectionCandidate requires at least one evidence item.")
        if not self.is_connected(evidence):
            raise ValueError("FireDetectionCandidate evidence must form one connected component.")

        object.__setattr__(self, "evidence", evidence)

    @classmethod
    def _normalize_evidence(
        cls,
        evidence: Iterable[FireDetectionEvidence],
    ) -> tuple[FireDetectionEvidence, ...]:
        if isinstance(evidence, (str, bytes)):
            raise ValueError("evidence must be an iterable of FireDetectionEvidence items.")

        try:
            items = tuple(evidence)
        except TypeError as exc:
            raise ValueError("evidence must be an iterable of FireDetectionEvidence items.") from exc

        identities: set[FireEvidenceRef] = set()
        for item in items:
            if not isinstance(item, FireDetectionEvidence):
                raise ValueError(f"evidence must contain FireDetectionEvidence items, got {item!r}.")

            identity = FireEvidenceRef(evidence_type=item.evidence_type, evidence_id=item.evidence_id)
            if identity in identities:
                raise ValueError(f"Duplicate evidence identity: {identity!r}.")
            identities.add(identity)

        return tuple(sorted(items, key=cls._sort_key))

    @staticmethod
    def _sort_key(evidence: FireDetectionEvidence) -> tuple[float, int, int]:
        observed_at = evidence.observed_at.astimezone(timezone.utc)
        return (observed_at.timestamp(), _EVIDENCE_TYPE_SORT_ORDER[evidence.evidence_type], evidence.evidence_id)

    @classmethod
    def is_connected(cls, evidence: tuple[FireDetectionEvidence, ...]) -> bool:
        """Return whether evidence forms one connected component by location and time.

        This is the authoritative "valid candidate" definition: evidence is
        one candidate when every item is reachable from every other item
        through a chain of pairwise correlations, not only when every pair
        directly correlates.
        """
        if len(evidence) == 1:
            return True

        seen = {0}
        stack = [0]
        while stack:
            index = stack.pop()
            for candidate_index, candidate in enumerate(evidence):
                if candidate_index in seen:
                    continue
                if cls._is_correlated(evidence[index], candidate):
                    seen.add(candidate_index)
                    stack.append(candidate_index)

        return len(seen) == len(evidence)

    @classmethod
    def _is_correlated(cls, first: FireDetectionEvidence, second: FireDetectionEvidence) -> bool:
        time_difference_minutes = abs(
            (first.observed_at.astimezone(timezone.utc) - second.observed_at.astimezone(timezone.utc))
            .total_seconds()
            / 60
        )
        if time_difference_minutes > MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES:
            return False

        return cls._haversine_distance_km(first, second) <= MAX_EVIDENCE_DISTANCE_KM + _DISTANCE_TOLERANCE_KM

    @staticmethod
    def _haversine_distance_km(first: FireDetectionEvidence, second: FireDetectionEvidence) -> float:
        first_latitude = math.radians(first.latitude)
        second_latitude = math.radians(second.latitude)
        delta_latitude = math.radians(second.latitude - first.latitude)
        delta_longitude = math.radians(second.longitude - first.longitude)

        a = (
            math.sin(delta_latitude / 2) ** 2
            + math.cos(first_latitude)
            * math.cos(second_latitude)
            * math.sin(delta_longitude / 2) ** 2
        )
        return _EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

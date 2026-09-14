"""Retrieve, normalize, and group direct evidence for active wildfire detection."""
from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from datetime import datetime, timezone

from src.calculators.fire_detection.fire_detection_config import (
    MAX_EVIDENCE_DISTANCE_KM,
    MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES,
)
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.repositories.news_repository import NewsRepository, StoredWildfireReport
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository, StoredSatelliteHotspot
from src.services.fire_detection.fire_detection_evidence_config import EVIDENCE_LOOKBACK_MINUTES

logger = logging.getLogger(__name__)

_EARTH_RADIUS_KM = 6371.0
_DISTANCE_TOLERANCE_KM = 1e-9
_SATELLITE_CONFIDENCE_MAP = {
    "l": "low",
    "low": "low",
    "n": "nominal",
    "nominal": "nominal",
    "h": "high",
    "high": "high",
}
_EVIDENCE_TYPE_SORT_ORDER = {
    FireEvidenceType.SATELLITE: 0,
    FireEvidenceType.NEWS: 1,
}


class FireDetectionEvidenceService:
    """Build active-fire evidence candidates from persisted direct evidence."""

    def __init__(
        self,
        satellite_repository: SatelliteHotspotRepository | None = None,
        news_repository: NewsRepository | None = None,
    ) -> None:
        self._satellite_repository = satellite_repository or SatelliteHotspotRepository()
        self._news_repository = news_repository or NewsRepository()

    def build_candidates(self, as_of: datetime) -> tuple[FireDetectionCandidate, ...]:
        """Return deterministic connected evidence groups within the lookback window."""
        self._validate_as_of(as_of)

        evidence = self._load_evidence(as_of)
        if not evidence:
            return ()

        components = self._connected_components(tuple(sorted(evidence.values(), key=self._evidence_sort_key)))
        candidates = tuple(FireDetectionCandidate(component) for component in components)
        return tuple(sorted(candidates, key=self._candidate_sort_key))

    def resolve_evidence_refs(
        self,
        refs: tuple[FireEvidenceRef, ...],
    ) -> tuple[FireDetectionEvidence, ...]:
        """Reload persisted evidence refs and normalize them for calculation."""
        normalized_refs = self._normalize_refs(refs)
        evidence: list[FireDetectionEvidence] = []
        for ref in normalized_refs:
            if ref.evidence_type is FireEvidenceType.SATELLITE:
                stored_hotspot = self._satellite_repository.get_by_id(ref.evidence_id)
                if stored_hotspot is None:
                    raise ValueError(f"Satellite evidence ref was not found: {ref.evidence_id!r}.")
                normalized = self._normalize_satellite(stored_hotspot)
            elif ref.evidence_type is FireEvidenceType.NEWS:
                stored_report = self._news_repository.get_by_id(ref.evidence_id)
                if stored_report is None:
                    raise ValueError(f"News evidence ref was not found: {ref.evidence_id!r}.")
                normalized = self._normalize_news(stored_report)
            else:
                raise ValueError(f"Unsupported evidence type: {ref.evidence_type!r}.")
            if normalized is None:
                raise ValueError(f"Evidence ref could not be normalized: {ref!r}.")
            evidence.append(normalized)
        return tuple(sorted(evidence, key=self._evidence_sort_key))

    def _load_evidence(self, as_of: datetime) -> dict[tuple[FireEvidenceType, int], FireDetectionEvidence]:
        evidence_by_identity: dict[tuple[FireEvidenceType, int], FireDetectionEvidence] = {}

        for stored_hotspot in self._satellite_repository.get_recent_hotspots(
            as_of=as_of,
            lookback_minutes=EVIDENCE_LOOKBACK_MINUTES,
        ):
            evidence = self._normalize_satellite(stored_hotspot)
            if evidence is not None:
                evidence_by_identity.setdefault((evidence.evidence_type, evidence.evidence_id), evidence)

        for stored_report in self._news_repository.get_recent_reports(
            as_of=as_of,
            lookback_minutes=EVIDENCE_LOOKBACK_MINUTES,
        ):
            evidence = self._normalize_news(stored_report)
            if evidence is not None:
                evidence_by_identity.setdefault((evidence.evidence_type, evidence.evidence_id), evidence)

        return evidence_by_identity

    @staticmethod
    def _normalize_refs(refs: tuple[FireEvidenceRef, ...]) -> tuple[FireEvidenceRef, ...]:
        normalized_refs = tuple(refs)
        for ref in normalized_refs:
            if not isinstance(ref, FireEvidenceRef):
                raise ValueError(f"refs must contain FireEvidenceRef items, got {ref!r}.")
        if len(set(normalized_refs)) != len(normalized_refs):
            raise ValueError("refs must not contain duplicates.")
        return tuple(sorted(normalized_refs, key=lambda ref: (ref.evidence_type.value, ref.evidence_id)))

    def _normalize_satellite(self, stored_hotspot: StoredSatelliteHotspot) -> FireDetectionEvidence | None:
        confidence = self._normalize_satellite_confidence(stored_hotspot.hotspot.confidence)
        if confidence is None:
            logger.info(
                "Skipping satellite hotspot %s with unsupported confidence %r.",
                stored_hotspot.id,
                stored_hotspot.hotspot.confidence,
            )
            return None

        try:
            return FireDetectionEvidence(
                evidence_id=stored_hotspot.id,
                evidence_type=FireEvidenceType.SATELLITE,
                latitude=stored_hotspot.hotspot.latitude,
                longitude=stored_hotspot.hotspot.longitude,
                observed_at=self._ensure_aware_datetime(stored_hotspot.hotspot.detected_at),
                satellite_confidence=confidence,
            )
        except ValueError as exc:
            logger.info("Skipping invalid satellite hotspot %s: %s", stored_hotspot.id, exc)
            return None

    def _normalize_news(self, stored_report: StoredWildfireReport) -> FireDetectionEvidence | None:
        if stored_report.report.latitude is None or stored_report.report.longitude is None:
            logger.info("Skipping wildfire report %s without coordinates.", stored_report.id)
            return None

        try:
            return FireDetectionEvidence(
                evidence_id=stored_report.id,
                evidence_type=FireEvidenceType.NEWS,
                latitude=stored_report.report.latitude,
                longitude=stored_report.report.longitude,
                observed_at=self._ensure_aware_datetime(stored_report.observed_at),
                satellite_confidence=None,
            )
        except ValueError as exc:
            logger.info("Skipping invalid wildfire report %s: %s", stored_report.id, exc)
            return None

    @staticmethod
    def _normalize_satellite_confidence(confidence: str | None) -> str | None:
        if confidence is None:
            return None
        return _SATELLITE_CONFIDENCE_MAP.get(confidence.strip().lower())

    @classmethod
    def _connected_components(
        cls,
        evidence: tuple[FireDetectionEvidence, ...],
    ) -> tuple[tuple[FireDetectionEvidence, ...], ...]:
        adjacency: dict[int, list[int]] = defaultdict(list)
        for first_index, first in enumerate(evidence):
            for second_index in range(first_index + 1, len(evidence)):
                if cls._is_correlated(first, evidence[second_index]):
                    adjacency[first_index].append(second_index)
                    adjacency[second_index].append(first_index)

        components: list[tuple[FireDetectionEvidence, ...]] = []
        seen: set[int] = set()
        for start_index in range(len(evidence)):
            if start_index in seen:
                continue

            component_indices: list[int] = []
            queue: deque[int] = deque([start_index])
            seen.add(start_index)
            while queue:
                index = queue.popleft()
                component_indices.append(index)
                for next_index in sorted(adjacency[index]):
                    if next_index not in seen:
                        seen.add(next_index)
                        queue.append(next_index)

            components.append(tuple(evidence[index] for index in sorted(component_indices)))

        return tuple(components)

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

    @staticmethod
    def _ensure_aware_datetime(value: datetime) -> datetime:
        if not isinstance(value, datetime):
            raise ValueError(f"observed timestamp must be a datetime, got {value!r}.")
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @staticmethod
    def _validate_as_of(as_of: datetime) -> None:
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}.")

    @classmethod
    def _evidence_sort_key(cls, evidence: FireDetectionEvidence) -> tuple[float, int, int]:
        observed_at = evidence.observed_at.astimezone(timezone.utc)
        return (observed_at.timestamp(), _EVIDENCE_TYPE_SORT_ORDER[evidence.evidence_type], evidence.evidence_id)

    @classmethod
    def _candidate_sort_key(cls, candidate: FireDetectionCandidate) -> tuple[float, int, int]:
        return cls._evidence_sort_key(candidate.evidence[0])

"""Pure conversion of FireDetectionEvidence groups into FireDetectionFeatures.

This module deliberately mirrors none of FireDetectionCalculator's
confidence-weighting methodology: it only reduces an already-valid evidence
candidate into numeric counts/geometry, and never produces a label or a
detection decision. It does not access Neon, call external APIs, or use any
repository.
"""
from __future__ import annotations

from datetime import timezone
import math

from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_type import FireEvidenceType
from src.ml.fire_detection.fire_detection_features import FireDetectionFeatures

_EARTH_RADIUS_KM = 6371.0


class FireDetectionFeatureExtractor:
    """Extract a deterministic, order-independent FireDetectionFeatures vector.

    Evidence validity (non-duplicate identity, one connected component by
    location/time) is delegated to FireDetectionCandidate - the same
    authoritative definition FireDetectionEvidenceService and
    FireDetectionCalculator use - rather than re-implemented here.
    """

    def extract(self, evidence: tuple[FireDetectionEvidence, ...]) -> FireDetectionFeatures:
        """Return the feature vector for one valid Fire Detection evidence candidate."""
        candidate = FireDetectionCandidate(evidence)
        items = candidate.evidence  # normalized: deduplicated identity, deterministically sorted

        satellite_items = tuple(item for item in items if item.evidence_type is FireEvidenceType.SATELLITE)
        news_items = tuple(item for item in items if item.evidence_type is FireEvidenceType.NEWS)

        return FireDetectionFeatures(
            satellite_count=len(satellite_items),
            news_count=len(news_items),
            satellite_low_count=self._count_confidence(satellite_items, "low"),
            satellite_nominal_count=self._count_confidence(satellite_items, "nominal"),
            satellite_high_count=self._count_confidence(satellite_items, "high"),
            time_span_minutes=self._time_span_minutes(items),
            max_pairwise_distance_km=self._max_pairwise_distance_km(items),
        )

    @staticmethod
    def _count_confidence(satellite_items: tuple[FireDetectionEvidence, ...], confidence: str) -> int:
        return sum(1 for item in satellite_items if item.satellite_confidence == confidence)

    @staticmethod
    def _time_span_minutes(items: tuple[FireDetectionEvidence, ...]) -> float:
        if len(items) == 1:
            return 0.0
        observed_at_utc = [item.observed_at.astimezone(timezone.utc) for item in items]
        return (max(observed_at_utc) - min(observed_at_utc)).total_seconds() / 60.0

    @classmethod
    def _max_pairwise_distance_km(cls, items: tuple[FireDetectionEvidence, ...]) -> float:
        if len(items) == 1:
            return 0.0
        max_distance = 0.0
        for first_index in range(len(items)):
            for second_index in range(first_index + 1, len(items)):
                distance = cls._haversine_distance_km(items[first_index], items[second_index])
                if distance > max_distance:
                    max_distance = distance
        return max_distance

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

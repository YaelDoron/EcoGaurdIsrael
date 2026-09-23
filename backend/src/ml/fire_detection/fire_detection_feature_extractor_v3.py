"""Pure conversion of FireDetectionEvidence groups into FireDetectionFeaturesV3.

Sibling to FireDetectionFeatureExtractor (V1/V2), which this module leaves
completely untouched. Geometry logic (Haversine distance, time span) is the
same formula used there and in FireDetectionCandidate/
FireDetectionEvidenceService - duplicated here rather than imported, matching
this codebase's existing convention of keeping each pure-calculation module
self-contained (see those three modules).

Does not query Neon, call external APIs, call an LLM, or use any repository.
Does not depend on FireDetectionCalculator and never produces a label.
"""
from __future__ import annotations

from datetime import timezone
import math

from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_type import FireEvidenceType
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength
from src.ml.fire_detection.fire_detection_features_v3 import FireDetectionFeaturesV3

_EARTH_RADIUS_KM = 6371.0

_NEWS_SIGNAL_COUNT_FIELD = {
    NewsWildfireSignalStrength.NONE: "news_none_count",
    NewsWildfireSignalStrength.WEAK: "news_weak_count",
    NewsWildfireSignalStrength.MODERATE: "news_moderate_count",
    NewsWildfireSignalStrength.STRONG: "news_strong_count",
}


class FireDetectionFeatureExtractorV3:
    """Extract a deterministic, order-independent FireDetectionFeaturesV3 vector.

    Evidence validity (non-duplicate identity, one connected component by
    location/time) is delegated to FireDetectionCandidate, the same
    authoritative definition every other Fire Detection component uses.
    """

    def extract(self, evidence: tuple[FireDetectionEvidence, ...]) -> FireDetectionFeaturesV3:
        """Return the V3 feature vector for one valid Fire Detection evidence candidate."""
        candidate = FireDetectionCandidate(evidence)
        items = candidate.evidence  # normalized: deduplicated identity, deterministically sorted

        satellite_items = tuple(item for item in items if item.evidence_type is FireEvidenceType.SATELLITE)
        news_items = tuple(item for item in items if item.evidence_type is FireEvidenceType.NEWS)

        frp_ratio, frp_mean, frp_max = self._availability_stats(
            satellite_items, lambda item: item.satellite_frp
        )
        brightness_ratio, brightness_mean, brightness_max = self._availability_stats(
            satellite_items, lambda item: item.satellite_brightness
        )
        news_counts = self._news_signal_counts(news_items)

        return FireDetectionFeaturesV3(
            satellite_low_count=self._count_confidence(satellite_items, "low"),
            satellite_nominal_count=self._count_confidence(satellite_items, "nominal"),
            satellite_high_count=self._count_confidence(satellite_items, "high"),
            satellite_frp_available_ratio=frp_ratio,
            satellite_frp_mean=frp_mean,
            satellite_frp_max=frp_max,
            satellite_brightness_available_ratio=brightness_ratio,
            satellite_brightness_mean=brightness_mean,
            satellite_brightness_max=brightness_max,
            time_span_minutes=self._time_span_minutes(items),
            max_pairwise_distance_km=self._max_pairwise_distance_km(items),
            **news_counts,
        )

    @staticmethod
    def _count_confidence(satellite_items: tuple[FireDetectionEvidence, ...], confidence: str) -> int:
        return sum(1 for item in satellite_items if item.satellite_confidence == confidence)

    @staticmethod
    def _availability_stats(
        satellite_items: tuple[FireDetectionEvidence, ...],
        accessor,
    ) -> tuple[float, float, float]:
        """ratio = items-with-a-value / total-items; mean/max computed only over available values.

        No satellite items, or no available values among them, both yield
        (0.0, 0.0, 0.0) - "no evidence" and "evidence present but unmeasured"
        are distinguished by the ratio itself, never confused with "measured
        as zero".
        """
        if not satellite_items:
            return 0.0, 0.0, 0.0
        values = [value for value in (accessor(item) for item in satellite_items) if value is not None]
        if not values:
            return 0.0, 0.0, 0.0
        ratio = len(values) / len(satellite_items)
        return ratio, sum(values) / len(values), max(values)

    @staticmethod
    def _news_signal_counts(news_items: tuple[FireDetectionEvidence, ...]) -> dict[str, int]:
        counts = {
            "news_none_count": 0,
            "news_weak_count": 0,
            "news_moderate_count": 0,
            "news_strong_count": 0,
            "news_unknown_count": 0,
        }
        for item in news_items:
            signal = item.news_wildfire_signal_strength
            if signal is None:
                counts["news_unknown_count"] += 1
            else:
                counts[_NEWS_SIGNAL_COUNT_FIELD[signal]] += 1
        return counts

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

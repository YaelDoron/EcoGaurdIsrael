"""The V5 Fire Detection feature extractor: the ONE place V5 features are computed.

    FireDetectionCandidate                 ---+
                                              +--> FireDetectionFeatureExtractorV5 --> 25 ordered features
    FireDetectionEventHistory (optional)   ---+

Both offline training (the V5 dataset) and future runtime inference call this class, so the two cannot
drift apart (no train/serving skew).

CURRENT vs HISTORY
    * CURRENT-evidence features (the 14 retained V3/V4 features, FRP sum/std, brightness std, night
      fraction, satellite cluster radius, news/satellite lag) describe the CURRENT candidate only.
    * HISTORY features (pass count, history span, centroid stability, FRP / brightness trend) describe
      the EFFECTIVE satellite history: the satellite evidence of the current candidate united with the
      satellite evidence of the supplied FireDetectionEventHistory, de-duplicated by source-aware identity
      (an event that already has the current candidate's hotspots attached does not count them twice).
    The candidate itself never has to span more than 60 minutes; hours-old passes come from the history.

Composition, not duplication:
    * the 14 retained features come from the unchanged FireDetectionFeatureExtractorV3 (its two dropped
      geometry features are simply not taken);
    * passes are grouped, and trend / span / centroid stability are computed, by the Task 5B primitives
      (`group_satellite_passes`, `satellite_pass_trend`, `satellite_pass_time_span_minutes`,
      `satellite_pass_centroid_rms_km`) - there is exactly one pass algorithm in the code base.

Deliberately NOT done here (the class is pure):
  * no repository, database or service access; it never builds a history - the caller supplies it;
  * no ML call, no labels, regime / subtype / environment, rule status or confidence, ML probability
    or FireEvent status (it does not even import them);
  * no Fire Danger / FFWI;
  * no imputation or scaling: "not computable" is NaN (see fire_detection_features_v5), the
    missing-value strategy belongs to the training pipeline;
  * the history object is never mutated (it is frozen) and nothing is stored.
"""
from __future__ import annotations

from datetime import timezone
import math

from src.ml.fire_detection.fire_detection_feature_extractor_v3 import FireDetectionFeatureExtractorV3
from src.ml.fire_detection.fire_detection_features_v5 import (
    FIRE_DETECTION_FEATURE_NAMES_V5,
    MISSING_VALUE_V5,
    RETAINED_FEATURE_NAMES_V5,
    FireDetectionFeaturesV5,
)
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_detection_event_history import (
    FireDetectionEventHistory,
    satellite_pass_centroid_rms_km,
    satellite_pass_time_span_minutes,
    satellite_pass_trend,
)
from src.models.fire_detection_satellite_pass import group_satellite_passes
from src.models.fire_evidence_type import FireEvidenceType
from src.services.fire_detection.fire_detection_evidence_config import SATELLITE_PASS_GAP_MINUTES
from src.utils.geo import haversine_distance_km

NIGHT_CODE = "N"
DAY_CODE = "D"


class FireDetectionFeatureExtractorV5:
    """Extract the deterministic 25-feature V5 vector for one candidate and its optional event history."""

    def __init__(
        self,
        evidence_extractor: FireDetectionFeatureExtractorV3 | None = None,
        satellite_pass_gap_minutes: float = SATELLITE_PASS_GAP_MINUTES,
    ) -> None:
        self._evidence_extractor = evidence_extractor or FireDetectionFeatureExtractorV3()
        self._pass_gap_minutes = satellite_pass_gap_minutes

    def extract(
        self,
        candidate: FireDetectionCandidate | tuple[FireDetectionEvidence, ...],
        history: FireDetectionEventHistory | None = None,
    ) -> FireDetectionFeaturesV5:
        """Return the V5 features. `candidate` may also be the raw evidence tuple (validated the same way).

        `history=None` is the "no FireEvent yet" case: the effective history is the candidate alone.
        """
        if history is not None and not isinstance(history, FireDetectionEventHistory):
            raise ValueError(f"history must be a FireDetectionEventHistory or None, got {history!r}")
        # Validity (non-duplicate identity, one connected component of <= 5 km / 60 min) is enforced by
        # FireDetectionCandidate itself - exactly as at runtime.
        if not isinstance(candidate, FireDetectionCandidate):
            candidate = FireDetectionCandidate(tuple(candidate))
        evidence = candidate.evidence
        satellite = tuple(item for item in evidence if item.evidence_type is FireEvidenceType.SATELLITE)
        news = tuple(item for item in evidence if item.evidence_type is FireEvidenceType.NEWS)

        values = {
            **self._retained_features(evidence),
            **self._current_features(satellite, news),
            **self._history_features(satellite, history),
        }
        return FireDetectionFeaturesV5(tuple(values[name] for name in FIRE_DETECTION_FEATURE_NAMES_V5))

    # --- the 14 retained features (V3 logic, unchanged) ---

    def _retained_features(self, evidence: tuple[FireDetectionEvidence, ...]) -> dict[str, float]:
        v3 = self._evidence_extractor.extract(evidence).as_dict()
        return {name: v3[name] for name in RETAINED_FEATURE_NAMES_V5}

    # --- current-candidate features ---

    @classmethod
    def _current_features(
        cls,
        satellite: tuple[FireDetectionEvidence, ...],
        news: tuple[FireDetectionEvidence, ...],
    ) -> dict[str, float]:
        frp = [item.satellite_frp for item in satellite if item.satellite_frp is not None]
        brightness = [item.satellite_brightness for item in satellite if item.satellite_brightness is not None]
        return {
            "satellite_frp_sum": float(sum(frp)) if frp else 0.0,
            "satellite_frp_std": cls._population_std(frp),
            "satellite_brightness_std": cls._population_std(brightness),
            "satellite_night_fraction": cls._night_fraction(satellite),
            "satellite_cluster_radius_km": cls._cluster_radius_km(satellite),
            "news_satellite_lag_minutes": cls._news_satellite_lag_minutes(satellite, news),
        }

    @staticmethod
    def _population_std(values: list[float]) -> float:
        """Population standard deviation (ddof=0); 0.0 with fewer than 2 values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    @staticmethod
    def _night_fraction(satellite: tuple[FireDetectionEvidence, ...]) -> float:
        """Share of hotspots with a KNOWN day/night ('D' or 'N') that are 'N'. NaN when none is known:
        unknown is not "0% night"."""
        known = [
            item.satellite_day_night.strip().upper()
            for item in satellite
            if item.satellite_day_night is not None and item.satellite_day_night.strip().upper() in (DAY_CODE, NIGHT_CODE)
        ]
        if not known:
            return MISSING_VALUE_V5
        return sum(1 for code in known if code == NIGHT_CODE) / len(known)

    @staticmethod
    def _cluster_radius_km(satellite: tuple[FireDetectionEvidence, ...]) -> float:
        """RMS great-circle distance of the SATELLITE pixels from their own centroid (km); 0.0 for one pixel.

        News coordinates (geocoding guesses) are never mixed in. NaN when there is no hotspot.
        """
        if not satellite:
            return MISSING_VALUE_V5
        if len(satellite) == 1:
            return 0.0
        centroid_latitude = sum(item.latitude for item in satellite) / len(satellite)
        centroid_longitude = sum(item.longitude for item in satellite) / len(satellite)
        squares = [
            haversine_distance_km(item.latitude, item.longitude, centroid_latitude, centroid_longitude) ** 2
            for item in satellite
        ]
        return math.sqrt(sum(squares) / len(squares))

    @staticmethod
    def _news_satellite_lag_minutes(
        satellite: tuple[FireDetectionEvidence, ...],
        news: tuple[FireDetectionEvidence, ...],
    ) -> float:
        """first news time - first satellite time, in minutes (positive: news came after the satellite).
        NaN unless BOTH families are present: 0.0 would claim near-simultaneous evidence."""
        if not satellite or not news:
            return MISSING_VALUE_V5
        first_news = min(item.observed_at.astimezone(timezone.utc) for item in news)
        first_satellite = min(item.observed_at.astimezone(timezone.utc) for item in satellite)
        return (first_news - first_satellite).total_seconds() / 60.0

    # --- history features ---

    def _history_features(
        self,
        current_satellite: tuple[FireDetectionEvidence, ...],
        history: FireDetectionEventHistory | None,
    ) -> dict[str, float]:
        pool: dict[tuple, FireDetectionEvidence] = {}
        for item in (*current_satellite, *(history.satellite_evidence if history is not None else ())):
            pool.setdefault((item.evidence_type, item.evidence_id), item)  # the current candidate wins on a duplicate
        passes = group_satellite_passes(pool.values(), self._pass_gap_minutes)

        span = satellite_pass_time_span_minutes(passes)
        stability = satellite_pass_centroid_rms_km(passes)
        frp_trend = satellite_pass_trend(passes, lambda p: p.frp_statistic("max"))
        brightness_trend = satellite_pass_trend(passes, lambda p: p.brightness_statistic("max"))
        return {
            "satellite_pass_count": len(passes),
            "satellite_history_span_minutes": MISSING_VALUE_V5 if span is None else span,
            "satellite_centroid_stability_km": MISSING_VALUE_V5 if stability is None else stability,
            "satellite_frp_trend_per_hour": MISSING_VALUE_V5 if frp_trend is None else frp_trend.slope_per_hour,
            "satellite_brightness_trend_per_hour": (
                MISSING_VALUE_V5 if brightness_trend is None else brightness_trend.slope_per_hour
            ),
        }

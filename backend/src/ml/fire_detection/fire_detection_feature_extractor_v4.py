"""The V4 Fire Detection feature extractor: the ONE place V4 features are computed.

    FireDetectionCandidate  ---+
                               +--> FireDetectionFeatureExtractorV4 --> 19 ordered features
    FireDetectionContext    ---+

Both offline training (the V4 dataset) and future runtime inference call this
class, so the two cannot drift apart (no train/serving skew).

Composition, not duplication: the 16 evidence features come from the unchanged
FireDetectionFeatureExtractorV3 (FRP, brightness, news, time and distance logic
is NOT re-implemented here). This class only appends the 3 Fire Danger context
features taken from an already-built FireDetectionContext.

Deliberately NOT done here:
  * no repository, database or service access - in particular it never calls
    FireDetectionContextService or reads a Fire Danger assessment; the caller
    supplies the context;
  * no labels, scenario/family/archetype, rule status/confidence, ML
    probability or FireEvent status (it does not even import them);
  * no imputation or scaling - unavailable Fire Danger is NaN (never 0.0) and
    the missing-value strategy belongs to Task 4's preprocessing;
  * the context's `fire_danger_level` (a function of the score), assessment id
    and timestamps are ignored: only availability, score and age are features.
"""
from __future__ import annotations

from src.ml.fire_detection.fire_detection_feature_extractor_v3 import FireDetectionFeatureExtractorV3
from src.ml.fire_detection.fire_detection_features_v4 import (
    MISSING_VALUE_V4,
    FireDetectionFeaturesV4,
)
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_context import FireDetectionContext
from src.models.fire_detection_evidence import FireDetectionEvidence


class FireDetectionFeatureExtractorV4:
    """Extract the deterministic 19-feature V4 vector for one candidate and its Fire Danger context."""

    def __init__(self, evidence_extractor: FireDetectionFeatureExtractorV3 | None = None) -> None:
        self._evidence_extractor = evidence_extractor or FireDetectionFeatureExtractorV3()

    def extract(
        self,
        candidate: FireDetectionCandidate | tuple[FireDetectionEvidence, ...],
        context: FireDetectionContext,
    ) -> FireDetectionFeaturesV4:
        """Return the V4 features. `candidate` may also be the raw evidence tuple (e.g. merged evidence).

        Evidence validity (non-duplicate, one connected component) is enforced by
        the V3 extractor through FireDetectionCandidate, exactly as at runtime.
        """
        if not isinstance(context, FireDetectionContext):
            raise ValueError(f"context must be a FireDetectionContext, got {context!r}")
        evidence = candidate.evidence if isinstance(candidate, FireDetectionCandidate) else tuple(candidate)

        evidence_features = self._evidence_extractor.extract(evidence).as_tuple()
        return FireDetectionFeaturesV4(evidence_features + self._context_features(context))

    @staticmethod
    def _context_features(context: FireDetectionContext) -> tuple[float, float, float]:
        if not context.fire_danger_available:
            return 0, MISSING_VALUE_V4, MISSING_VALUE_V4
        return 1, float(context.fire_danger_score), float(context.fire_danger_age_minutes)

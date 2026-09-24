"""AI Hybrid V5 runtime classifier: one candidate (+ its event history) -> P(fire) -> NO_EVENT / SUSPECTED / CONFIRMED.

    FireDetectionCandidate  (+ FireDetectionEventHistory of a matching active FireEvent, or None)
        -> FireDetectionFeatureExtractorV5      25 ordered features (the ONE implementation; also used for training)
        -> FireDetectionModelV5Runtime          HGB P(fire)
        -> fire_detection_ai_hybrid_policy_v5   locked policy v5.0 (thresholds live in fire_detection_policy_config_v5)
        -> FireDetectionAIAssessment

Composition only: no feature or policy logic is reimplemented here, and no rule calculator, Fire Danger context, repository,
database or FireEvent is touched. The classifier is stateless between calls (the only shared state is the immutable, cached
model). The caller supplies the history - this class never builds one, so a history failure can never silently become
"no history" here.

Any failure raises FireDetectionAIError (public, path-free message); nothing falls back to another detector.
"""
from __future__ import annotations

import logging

from src.ml.fire_detection.fire_detection_ai_hybrid_policy_v5 import current_satellite_pixel_count, decide_status
from src.ml.fire_detection.fire_detection_feature_extractor_v5 import FireDetectionFeatureExtractorV5
from src.ml.fire_detection.fire_detection_model_runtime_v5 import (
    FireDetectionAIError,
    FireDetectionModelV5InferenceError,
    FireDetectionModelV5Runtime,
)
from src.ml.fire_detection.fire_detection_policy_config_v5 import (
    POLICY_VERSION,
    STATUS_CONFIRMED,
    STATUS_NO_EVENT,
    STATUS_SUSPECTED,
)
from src.models.fire_detection_ai_assessment import FireDetectionAIAssessment
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_event_history import FireDetectionEventHistory
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_detection_status import FireDetectionStatus

logger = logging.getLogger(__name__)

_STATUS_BY_POLICY_NAME = {
    STATUS_NO_EVENT: FireDetectionStatus.NO_EVENT,
    STATUS_SUSPECTED: FireDetectionStatus.SUSPECTED,
    STATUS_CONFIRMED: FireDetectionStatus.CONFIRMED,
}


class FireDetectionAIHybridClassifierV5:
    """Stateless (apart from the immutable cached model) V5 scorer + locked policy."""

    def __init__(
        self,
        model_runtime: FireDetectionModelV5Runtime,
        feature_extractor: FireDetectionFeatureExtractorV5 | None = None,
    ) -> None:
        self._model_runtime = model_runtime
        self._extractor = feature_extractor or FireDetectionFeatureExtractorV5()

    @property
    def model_runtime(self) -> FireDetectionModelV5Runtime:
        return self._model_runtime

    def assess(
        self,
        candidate: FireDetectionCandidate | tuple[FireDetectionEvidence, ...],
        history: FireDetectionEventHistory | None = None,
    ) -> FireDetectionAIAssessment:
        """Score one candidate. `history=None` means "no FireEvent yet": the effective history is the candidate alone."""
        loaded = self._model_runtime.ensure_loaded()  # explicit artifact errors surface here, before any feature work
        try:
            features = self._extractor.extract(candidate, history)
        except FireDetectionAIError:
            raise
        except Exception as exc:  # noqa: BLE001 - never let a malformed candidate turn into a different decision
            logger.error("V5 feature extraction failed: %s: %s", type(exc).__name__, exc)
            raise FireDetectionModelV5InferenceError("V5 feature extraction failed for the candidate.") from exc

        probability = self._model_runtime.predict_probability(features)
        feature_values = features.as_dict()
        pixel_count = current_satellite_pixel_count(feature_values)
        status = _STATUS_BY_POLICY_NAME[decide_status(probability, pixel_count)]
        return FireDetectionAIAssessment(
            probability=probability,
            policy_status=status,
            model_name=loaded.model_name,
            model_version=loaded.model_version,
            feature_schema_version=loaded.feature_schema_version,
            policy_version=POLICY_VERSION,
            current_satellite_pixel_count=pixel_count,
            satellite_pass_count=int(feature_values["satellite_pass_count"]),
            history_available=history is not None,
        )

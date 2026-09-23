"""Runtime Logistic Regression V3 inference component for Fire Detection (Task 5).

Loads the saved V3 pipeline lazily (on first use, not on import) and caches
it, validates it against the metadata recorded at training time, and turns
one candidate's normalized evidence into a FireDetectionMLAssessment. All
sklearn/joblib/feature-array details are contained here -
FireDetectionAgent never touches them directly; it only ever sees
FireDetectionMLAssessment.

`predict_proba(...)[1]` is a model-estimated probability WITHIN the
synthetic V3 training distribution (see backend/docs/
fire_detection_runtime_ml.md) - it is NOT a calibrated real-world
probability of an active wildfire, and must never be presented as
operational certainty.

ML failure of any kind (missing/corrupt file, metadata mismatch, feature
extraction failure, predict_proba failure) NEVER raises out of `assess()` -
it returns an `available=False` assessment so FireDetectionAgent can fall
back to the deterministic calculator alone.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib

from src.ml.fire_detection.fire_detection_feature_extractor_v3 import FireDetectionFeatureExtractorV3
from src.ml.fire_detection.fire_detection_features_v3 import FIRE_DETECTION_FEATURE_NAMES_V3
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_detection_ml_assessment import FireDetectionMLAssessment

logger = logging.getLogger(__name__)

EXPECTED_MODEL_TYPE = "LogisticRegression"
FEATURE_SCHEMA_VERSION = "v3"
DEFAULT_MODEL_NAME = "fire_detection_logistic_v3"


class MLFireDetectionClassifier:
    """Lazily-loaded, cached runtime wrapper around the saved V3 Logistic Regression pipeline."""

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self._model_path = Path(model_path)
        self._metadata_path = Path(metadata_path)
        self._model_name = model_name
        self._extractor = FireDetectionFeatureExtractorV3()

        self._pipeline = None
        self._model_version: str | None = None
        self._load_attempted = False
        self._load_failure_reason: str | None = None

    def initialize(self) -> bool:
        """Explicitly attempt to load the model now (idempotent, cached). Returns True if available."""
        return self._ensure_loaded()

    @property
    def is_available(self) -> bool:
        return self._ensure_loaded()

    def assess(self, evidence: tuple[FireDetectionEvidence, ...]) -> FireDetectionMLAssessment:
        """Return an ML assessment for one candidate's evidence. Never raises."""
        if not self._ensure_loaded():
            return self._unavailable(self._load_failure_reason or "ML model unavailable.")

        try:
            features = self._extractor.extract(evidence)
        except Exception as exc:
            logger.warning("V3 feature extraction failed for ML inference: %s", exc)
            return self._unavailable(f"feature extraction failed: {exc}")

        try:
            probability = float(self._pipeline.predict_proba([list(features.as_tuple())])[0][1])
        except Exception as exc:
            logger.warning("ML predict_proba failed: %s", exc)
            return self._unavailable(f"predict_proba failed: {exc}")

        return FireDetectionMLAssessment(
            available=True,
            probability=probability,
            model_name=self._model_name,
            model_version=self._model_version,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            failure_reason=None,
        )

    def _ensure_loaded(self) -> bool:
        if self._load_attempted:
            return self._pipeline is not None
        self._load_attempted = True

        try:
            if not self._model_path.exists():
                raise FileNotFoundError(f"ML model file not found: {self._model_path}")
            if not self._metadata_path.exists():
                raise FileNotFoundError(f"ML model metadata file not found: {self._metadata_path}")

            metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            model_version = self._validate_metadata(metadata)

            pipeline = joblib.load(self._model_path)
            self._validate_pipeline(pipeline)

            self._pipeline = pipeline
            self._model_version = model_version
            logger.info("Loaded Fire Detection ML model %r (version %s)", self._model_name, model_version)
            return True
        except Exception as exc:
            self._load_failure_reason = f"ML model load failed: {exc}"
            self._pipeline = None
            logger.warning(self._load_failure_reason)
            return False

    @staticmethod
    def _validate_metadata(metadata: dict) -> str:
        model_type = metadata.get("model_type")
        if model_type != EXPECTED_MODEL_TYPE:
            raise ValueError(f"unexpected model_type in metadata: {model_type!r} (expected {EXPECTED_MODEL_TYPE!r})")

        feature_names = tuple(metadata.get("feature_names") or ())
        if feature_names != FIRE_DETECTION_FEATURE_NAMES_V3:
            raise ValueError(
                "ML metadata feature_names do not match FIRE_DETECTION_FEATURE_NAMES_V3 - "
                "refusing to run inference with a mismatched schema."
            )

        model_version = metadata.get("model_version")
        if not model_version or not isinstance(model_version, str):
            raise ValueError("ML metadata is missing a non-empty model_version.")
        return model_version

    @staticmethod
    def _validate_pipeline(pipeline) -> None:
        if not hasattr(pipeline, "predict_proba"):
            raise ValueError("Loaded ML artifact does not expose predict_proba.")
        n_features_in = getattr(pipeline, "n_features_in_", None)
        if n_features_in is not None and n_features_in != len(FIRE_DETECTION_FEATURE_NAMES_V3):
            raise ValueError(
                f"Loaded ML pipeline expects {n_features_in} features, "
                f"but FIRE_DETECTION_FEATURE_NAMES_V3 has {len(FIRE_DETECTION_FEATURE_NAMES_V3)}."
            )

    def _unavailable(self, reason: str) -> FireDetectionMLAssessment:
        return FireDetectionMLAssessment(
            available=False,
            probability=None,
            model_name=self._model_name,
            model_version=self._model_version,
            feature_schema_version=None,
            failure_reason=reason,
        )

"""Runtime loader for the approved Fire Detection HGB V5 artifact (Task 9B).

Loads `fire_detection_hgb_v5.joblib` + `fire_detection_hgb_v5_metadata.json` once, validates them STRICTLY against the
project's own V5 contract, and caches the immutable result so the file is deserialized once per process - not per candidate
and not per simulation run (the trained model is static; only incident/evidence/history state resets between runs).

What is validated (any mismatch raises FireDetectionModelV5UnavailableError - there is no lenient mode):
    * model identity: metadata model_type == HistGradientBoostingClassifier, model_version and feature_schema_version are
      the V5 ones, and the loaded object really is a pipeline ending in a HistGradientBoostingClassifier
    * feature schema: metadata feature_names == FIRE_DETECTION_FEATURE_NAMES_V5 exactly (names AND order), and the model
      was fitted on that many features
    * policy provenance: the Task 7 binary-primary gate is recorded FAILED, the Task 8 AI-hybrid policy gate PASSED, and
      the policy version / thresholds / corroboration definition equal the ones locked in fire_detection_policy_config_v5.
      Thresholds are compared to that config - never copied into a second policy implementation.

This class only turns a V5 feature vector into P(fire). It has no policy, no feature extraction and no incident state.
There is no fallback to rules: a caller that cannot get a probability must fail, never quietly use another detector.
Errors carry a short public message with no filesystem paths.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import threading
from dataclasses import dataclass

import joblib
import numpy as np

from src.ml.fire_detection.fire_detection_features_v5 import FIRE_DETECTION_FEATURE_NAMES_V5, FireDetectionFeaturesV5
from src.ml.fire_detection.fire_detection_model_config_v5 import FEATURE_SCHEMA_VERSION_V5, MODEL_VERSION_V5
from src.ml.fire_detection.fire_detection_policy_config_v5 import (
    CONFIRM_THRESHOLD,
    MODEL_CLASS_NAME,
    MULTI_PIXEL_DEFINITION,
    POLICY_VERSION,
    SUSPECT_THRESHOLD,
    TASK_7_BINARY_PRIMARY_GATE_VERDICT,
)

logger = logging.getLogger(__name__)

FEATURE_COUNT = len(FIRE_DETECTION_FEATURE_NAMES_V5)
EXPECTED_TASK_8_POLICY_GATE_VERDICT = "PASSED"


class FireDetectionAIError(RuntimeError):
    """Base for every AI Hybrid V5 runtime failure. `public_message` is safe to log / return (no paths, no secrets)."""

    def __init__(self, public_message: str) -> None:
        super().__init__(public_message)
        self.public_message = public_message


class FireDetectionModelV5UnavailableError(FireDetectionAIError):
    """The V5 artifact is missing, corrupt, or not the approved artifact (a configuration/deployment error)."""


class FireDetectionModelV5InferenceError(FireDetectionAIError):
    """The artifact loaded, but scoring one feature vector failed or produced an invalid probability."""


class FireDetectionHistoryUnavailableError(FireDetectionAIError):
    """The matching FireEvent's evidence history could not be loaded.

    Treated as a failure, never as "no history": an empty history would silently change the model's inputs
    (pass count 1, no trend) and therefore its probability.
    """


class FireDetectionAIAssessmentPersistenceError(FireDetectionAIError):
    """The AI audit row could not be stored (typically: the Task 9B columns have not been migrated yet)."""


@dataclass(frozen=True)
class LoadedModelV5:
    """The immutable loaded artifact. Holds no incident/event state."""

    model: object
    model_name: str
    model_version: str
    feature_schema_version: str
    policy_version: str


_CACHE: dict[tuple[str, str], LoadedModelV5] = {}
_CACHE_LOCK = threading.Lock()


def clear_model_cache_v5() -> None:
    """Drop every cached artifact (tests / an explicit reload). Never called in the normal runtime path."""
    with _CACHE_LOCK:
        _CACHE.clear()


class FireDetectionModelV5Runtime:
    """Lazily loaded, process-wide cached, strictly validated HGB V5 scorer."""

    def __init__(self, model_path: str | Path, metadata_path: str | Path) -> None:
        self._key = (str(Path(model_path)), str(Path(metadata_path)))
        self._model_path = Path(model_path)
        self._metadata_path = Path(metadata_path)

    def ensure_loaded(self) -> LoadedModelV5:
        """Return the validated artifact, loading it on first use. Failures are NOT cached (a fixed file is picked up)."""
        loaded = _CACHE.get(self._key)
        if loaded is not None:
            return loaded
        with _CACHE_LOCK:
            loaded = _CACHE.get(self._key)
            if loaded is None:
                loaded = self._load_and_validate()
                _CACHE[self._key] = loaded
            return loaded

    @property
    def model_info(self) -> dict[str, str]:
        """Safe, path-free identity of the loaded model (loads it if needed)."""
        loaded = self.ensure_loaded()
        return {
            "model_name": loaded.model_name,
            "model_version": loaded.model_version,
            "feature_schema_version": loaded.feature_schema_version,
            "policy_version": loaded.policy_version,
        }

    def predict_probability(self, features: FireDetectionFeaturesV5) -> float:
        """P(fire) for one V5 feature vector. Raises FireDetectionModelV5InferenceError on any problem."""
        loaded = self.ensure_loaded()
        if not isinstance(features, FireDetectionFeaturesV5):
            raise FireDetectionModelV5InferenceError("invalid feature vector: not a FireDetectionFeaturesV5.")
        values = features.as_tuple()
        if len(values) != FEATURE_COUNT or features.names != FIRE_DETECTION_FEATURE_NAMES_V5:
            raise FireDetectionModelV5InferenceError("invalid feature vector: it does not match the V5 feature schema.")
        try:
            row = np.asarray([values], dtype=float)
            probabilities = loaded.model.predict_proba(row)
            probability = float(probabilities[0][1])
        except Exception as exc:  # noqa: BLE001 - any classifier failure is an explicit inference error
            logger.error("V5 model predict_proba failed: %s: %s", type(exc).__name__, exc)
            raise FireDetectionModelV5InferenceError("the V5 model failed to score the candidate.") from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise FireDetectionModelV5InferenceError("the V5 model returned an invalid probability.")
        return probability

    # --- loading / validation -------------------------------------------------------------------------------

    def _load_and_validate(self) -> LoadedModelV5:
        if not self._metadata_path.is_file():
            raise FireDetectionModelV5UnavailableError("the V5 model metadata file is missing.")
        if not self._model_path.is_file():
            raise FireDetectionModelV5UnavailableError("the V5 model artifact file is missing.")
        try:
            metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FireDetectionModelV5UnavailableError("the V5 model metadata file cannot be read as JSON.") from exc
        if not isinstance(metadata, dict):
            raise FireDetectionModelV5UnavailableError("the V5 model metadata is not a JSON object.")

        self._validate_metadata(metadata)
        try:
            model = joblib.load(self._model_path)
        except Exception as exc:  # noqa: BLE001 - corrupt / incompatible pickle of any kind
            logger.error("V5 model deserialization failed: %s: %s", type(exc).__name__, exc)
            raise FireDetectionModelV5UnavailableError("the V5 model artifact cannot be deserialized.") from exc
        self._validate_model(model)

        logger.info(
            "Loaded Fire Detection AI Hybrid model %s (version %s, policy %s)",
            metadata["model_name"], metadata["model_version"], metadata["policy_version"],
        )
        return LoadedModelV5(
            model=model,
            model_name=metadata["model_name"],
            model_version=metadata["model_version"],
            feature_schema_version=metadata["feature_schema_version"],
            policy_version=metadata["policy_version"],
        )

    @staticmethod
    def _validate_metadata(metadata: dict) -> None:
        def require(condition: bool, message: str) -> None:
            if not condition:
                raise FireDetectionModelV5UnavailableError(f"the V5 model metadata is not the approved artifact: {message}.")

        # model identity
        require(metadata.get("model_type") == MODEL_CLASS_NAME, "model_type is not HistGradientBoostingClassifier")
        require(metadata.get("model_version") == MODEL_VERSION_V5, "model_version is not the V5 model version")
        require(metadata.get("feature_schema_version") == FEATURE_SCHEMA_VERSION_V5, "feature_schema_version is not v5")
        require(isinstance(metadata.get("model_name"), str) and bool(metadata["model_name"].strip()), "model_name is missing")
        # feature schema: exact names AND order
        names = metadata.get("feature_names")
        require(isinstance(names, list) and tuple(names) == FIRE_DETECTION_FEATURE_NAMES_V5, "feature_names differ from FIRE_DETECTION_FEATURE_NAMES_V5")
        # policy provenance (compared to the locked Task 8 config, never copied)
        require(metadata.get("task_7_binary_primary_model_gate") == TASK_7_BINARY_PRIMARY_GATE_VERDICT, "Task 7 gate is not recorded as FAILED")
        require(metadata.get("task_8_ai_hybrid_policy_gate") == EXPECTED_TASK_8_POLICY_GATE_VERDICT, "Task 8 policy gate is not recorded as PASSED")
        gate = metadata.get("policy_gate")
        require(isinstance(gate, dict) and gate.get("passed") is True, "the recorded policy gate did not pass")
        require(metadata.get("policy_version") == POLICY_VERSION, "policy_version differs from the locked policy")
        require(metadata.get("suspect_threshold") == SUSPECT_THRESHOLD, "suspect_threshold differs from the locked policy")
        require(metadata.get("confirm_threshold") == CONFIRM_THRESHOLD, "confirm_threshold differs from the locked policy")
        require(metadata.get("multi_pixel_corroboration") == MULTI_PIXEL_DEFINITION, "the corroboration condition differs from the locked policy")

    @staticmethod
    def _validate_model(model: object) -> None:
        if not hasattr(model, "predict_proba"):
            raise FireDetectionModelV5UnavailableError("the V5 artifact does not expose predict_proba.")
        steps = getattr(model, "steps", None)
        final_estimator = steps[-1][1] if steps else model
        if type(final_estimator).__name__ != MODEL_CLASS_NAME:
            raise FireDetectionModelV5UnavailableError("the V5 artifact is not a HistGradientBoostingClassifier model.")
        n_features_in = getattr(model, "n_features_in_", None)
        if n_features_in is not None and n_features_in != FEATURE_COUNT:
            raise FireDetectionModelV5UnavailableError("the V5 artifact was fitted on a different number of features.")
        classes = getattr(final_estimator, "classes_", None)
        if classes is not None and list(classes) != [0, 1]:
            raise FireDetectionModelV5UnavailableError("the V5 artifact is not a binary 0/1 classifier.")

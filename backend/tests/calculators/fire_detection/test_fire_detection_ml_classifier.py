"""Tests for MLFireDetectionClassifier. No Neon required."""
from __future__ import annotations

from datetime import datetime, timezone
import json

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.calculators.fire_detection.fire_detection_ml_classifier import MLFireDetectionClassifier
from src.ml.fire_detection.fire_detection_features_v3 import FIRE_DETECTION_FEATURE_NAMES_V3
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_type import FireEvidenceType

OBSERVED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
LATITUDE = 32.731
LONGITUDE = 35.046


def evidence() -> tuple[FireDetectionEvidence, ...]:
    return (
        FireDetectionEvidence(
            evidence_id=1,
            evidence_type=FireEvidenceType.SATELLITE,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            observed_at=OBSERVED_AT,
            satellite_confidence="high",
            satellite_frp=80.0,
            satellite_brightness=340.0,
        ),
    )


def _fit_trivial_pipeline() -> Pipeline:
    pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=200))])
    features = [[float(i % 3)] * len(FIRE_DETECTION_FEATURE_NAMES_V3) for i in range(10)]
    labels = [i % 2 for i in range(10)]
    pipeline.fit(features, labels)
    return pipeline


def _write_valid_artifacts(tmp_path, model_version="3.0"):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump(_fit_trivial_pipeline(), model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model_type": "LogisticRegression",
                "model_version": model_version,
                "feature_names": list(FIRE_DETECTION_FEATURE_NAMES_V3),
            }
        ),
        encoding="utf-8",
    )
    return model_path, metadata_path


# --- successful load + inference ---


def test_model_loads_and_infers_successfully(tmp_path):
    model_path, metadata_path = _write_valid_artifacts(tmp_path)
    classifier = MLFireDetectionClassifier(model_path, metadata_path)

    result = classifier.assess(evidence())

    assert result.available is True
    assert result.model_version == "3.0"
    assert result.feature_schema_version == "v3"
    assert result.failure_reason is None


def test_probability_is_within_zero_one(tmp_path):
    model_path, metadata_path = _write_valid_artifacts(tmp_path)
    classifier = MLFireDetectionClassifier(model_path, metadata_path)

    result = classifier.assess(evidence())

    assert 0.0 <= result.probability <= 1.0


def test_same_candidate_produces_deterministic_result(tmp_path):
    model_path, metadata_path = _write_valid_artifacts(tmp_path)
    classifier = MLFireDetectionClassifier(model_path, metadata_path)

    first = classifier.assess(evidence())
    second = classifier.assess(evidence())

    assert first == second


def test_model_is_loaded_once_and_cached(tmp_path, monkeypatch):
    model_path, metadata_path = _write_valid_artifacts(tmp_path)
    classifier = MLFireDetectionClassifier(model_path, metadata_path)

    load_calls = []
    original_load = joblib.load

    def counting_load(path):
        load_calls.append(path)
        return original_load(path)

    monkeypatch.setattr("src.calculators.fire_detection.fire_detection_ml_classifier.joblib.load", counting_load)

    classifier.assess(evidence())
    classifier.assess(evidence())
    classifier.assess(evidence())

    assert len(load_calls) == 1


def test_initialize_is_idempotent(tmp_path):
    model_path, metadata_path = _write_valid_artifacts(tmp_path)
    classifier = MLFireDetectionClassifier(model_path, metadata_path)

    assert classifier.initialize() is True
    assert classifier.initialize() is True
    assert classifier.is_available is True


# --- graceful failure ---


def test_missing_model_file_returns_unavailable():
    classifier = MLFireDetectionClassifier("does/not/exist.joblib", "does/not/exist.json")

    result = classifier.assess(evidence())

    assert result.available is False
    assert result.probability is None
    assert result.failure_reason is not None


def test_missing_metadata_file_returns_unavailable(tmp_path):
    model_path, _ = _write_valid_artifacts(tmp_path)
    classifier = MLFireDetectionClassifier(model_path, tmp_path / "missing_metadata.json")

    result = classifier.assess(evidence())

    assert result.available is False


def test_corrupt_model_file_returns_unavailable(tmp_path):
    _, metadata_path = _write_valid_artifacts(tmp_path)
    corrupt_model_path = tmp_path / "corrupt.joblib"
    corrupt_model_path.write_text("not a real pickle", encoding="utf-8")
    classifier = MLFireDetectionClassifier(corrupt_model_path, metadata_path)

    result = classifier.assess(evidence())

    assert result.available is False
    assert result.failure_reason is not None


def test_metadata_wrong_model_type_returns_unavailable(tmp_path):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump(_fit_trivial_pipeline(), model_path)
    metadata_path.write_text(
        json.dumps({"model_type": "RandomForestClassifier", "model_version": "3.0", "feature_names": list(FIRE_DETECTION_FEATURE_NAMES_V3)}),
        encoding="utf-8",
    )
    classifier = MLFireDetectionClassifier(model_path, metadata_path)

    result = classifier.assess(evidence())

    assert result.available is False
    assert "model_type" in result.failure_reason


def test_metadata_feature_mismatch_returns_unavailable(tmp_path):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump(_fit_trivial_pipeline(), model_path)
    metadata_path.write_text(
        json.dumps({"model_type": "LogisticRegression", "model_version": "3.0", "feature_names": ["wrong_feature"]}),
        encoding="utf-8",
    )
    classifier = MLFireDetectionClassifier(model_path, metadata_path)

    result = classifier.assess(evidence())

    assert result.available is False
    assert "feature_names" in result.failure_reason
    # Never silently feed features in the wrong order: no prediction attempted.
    assert result.probability is None


def test_metadata_missing_model_version_returns_unavailable(tmp_path):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump(_fit_trivial_pipeline(), model_path)
    metadata_path.write_text(
        json.dumps({"model_type": "LogisticRegression", "feature_names": list(FIRE_DETECTION_FEATURE_NAMES_V3)}),
        encoding="utf-8",
    )
    classifier = MLFireDetectionClassifier(model_path, metadata_path)

    result = classifier.assess(evidence())

    assert result.available is False


def test_load_failure_is_cached_not_retried_every_call(tmp_path, monkeypatch):
    classifier = MLFireDetectionClassifier(tmp_path / "missing.joblib", tmp_path / "missing.json")

    exists_calls = []
    original_exists = type(classifier._model_path).exists

    def counting_exists(self):
        exists_calls.append(self)
        return original_exists(self)

    monkeypatch.setattr("pathlib.Path.exists", counting_exists)

    classifier.assess(evidence())
    classifier.assess(evidence())

    # model_path.exists() should only be probed once (first load attempt), not once per assess() call.
    assert exists_calls.count(classifier._model_path) == 1


def test_predict_proba_failure_returns_unavailable(tmp_path):
    model_path, metadata_path = _write_valid_artifacts(tmp_path)
    classifier = MLFireDetectionClassifier(model_path, metadata_path)
    classifier.initialize()

    class ExplodingPipeline:
        def predict_proba(self, x):
            raise RuntimeError("boom")

        n_features_in_ = len(FIRE_DETECTION_FEATURE_NAMES_V3)

    classifier._pipeline = ExplodingPipeline()

    result = classifier.assess(evidence())

    assert result.available is False
    assert "predict_proba" in result.failure_reason


def test_extractor_failure_returns_unavailable(tmp_path):
    model_path, metadata_path = _write_valid_artifacts(tmp_path)
    classifier = MLFireDetectionClassifier(model_path, metadata_path)

    result = classifier.assess(())  # empty evidence -> extractor raises

    assert result.available is False
    assert "feature extraction" in result.failure_reason


def test_assess_never_raises_on_any_failure(tmp_path):
    classifier = MLFireDetectionClassifier(tmp_path / "missing.joblib", tmp_path / "missing.json")

    result = classifier.assess(())  # both load failure and extractor failure possible - must not raise

    assert result.available is False

"""Task 9B: strict validation, lazy loading and caching of the approved HGB V5 artifact (no fallback, no incident state)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from src.config.settings import settings
from src.ml.fire_detection import fire_detection_model_runtime_v5 as runtime_module
from src.ml.fire_detection.fire_detection_features_v5 import FIRE_DETECTION_FEATURE_NAMES_V5, FireDetectionFeaturesV5
from src.ml.fire_detection.fire_detection_model_runtime_v5 import (
    FireDetectionModelV5InferenceError,
    FireDetectionModelV5Runtime,
    FireDetectionModelV5UnavailableError,
    clear_model_cache_v5,
)

REAL_MODEL = Path(settings.FIRE_DETECTION_AI_V5_MODEL_PATH)
REAL_METADATA = Path(settings.FIRE_DETECTION_AI_V5_METADATA_PATH)


@pytest.fixture(autouse=True)
def fresh_cache():
    clear_model_cache_v5()
    yield
    clear_model_cache_v5()


def real_features() -> FireDetectionFeaturesV5:
    values = dict.fromkeys(FIRE_DETECTION_FEATURE_NAMES_V5, 0.0)
    values.update(satellite_low_count=0, satellite_nominal_count=1, satellite_high_count=0, satellite_pass_count=1,
                  satellite_frp_mean=8.0, satellite_frp_max=8.0, satellite_frp_sum=8.0, satellite_frp_available_ratio=1.0,
                  satellite_brightness_mean=330.0, satellite_brightness_max=330.0, satellite_brightness_available_ratio=1.0,
                  news_none_count=0, news_weak_count=0, news_moderate_count=0, news_strong_count=0, news_unknown_count=0,
                  satellite_night_fraction=float("nan"), news_satellite_lag_minutes=float("nan"),
                  satellite_history_span_minutes=0.0, satellite_centroid_stability_km=float("nan"),
                  satellite_frp_trend_per_hour=float("nan"), satellite_brightness_trend_per_hour=float("nan"))
    return FireDetectionFeaturesV5.from_optional_values(values[name] for name in FIRE_DETECTION_FEATURE_NAMES_V5)


def copy_artifact(tmp_path, metadata_edit=None):
    model = tmp_path / "fire_detection_hgb_v5.joblib"
    metadata = tmp_path / "fire_detection_hgb_v5_metadata.json"
    shutil.copy(REAL_MODEL, model)
    data = json.loads(REAL_METADATA.read_text(encoding="utf-8"))
    if metadata_edit:
        metadata_edit(data)
    metadata.write_text(json.dumps(data), encoding="utf-8")
    return model, metadata


# --- valid artifact ---------------------------------------------------------------------------------------------------


def test_the_approved_artifact_loads_and_reports_its_identity_without_paths():
    runtime = FireDetectionModelV5Runtime(REAL_MODEL, REAL_METADATA)

    info = runtime.model_info

    assert info == {
        "model_name": "fire_detection_hgb_v5", "model_version": "5.0", "feature_schema_version": "v5",
        "policy_version": "ai_hybrid_policy_v5.0",
    }
    assert not any("/" in value or "\\" in value for value in info.values())


def test_the_approved_metadata_records_the_two_gate_verdicts_the_loader_relies_on():
    metadata = json.loads(REAL_METADATA.read_text(encoding="utf-8"))
    assert metadata["task_7_binary_primary_model_gate"] == "FAILED"
    assert metadata["task_8_ai_hybrid_policy_gate"] == "PASSED"
    assert tuple(metadata["feature_names"]) == FIRE_DETECTION_FEATURE_NAMES_V5


def test_the_real_model_scores_a_v5_vector_with_a_finite_probability():
    probability = FireDetectionModelV5Runtime(REAL_MODEL, REAL_METADATA).predict_probability(real_features())
    assert 0.0 <= probability <= 1.0 and np.isfinite(probability)


# --- explicit failures (never a silent fallback) ------------------------------------------------------------------------


def test_missing_model_file_fails_explicitly(tmp_path):
    _, metadata = copy_artifact(tmp_path)
    with pytest.raises(FireDetectionModelV5UnavailableError, match="artifact file is missing") as error:
        FireDetectionModelV5Runtime(tmp_path / "absent.joblib", metadata).ensure_loaded()
    assert str(tmp_path) not in error.value.public_message


def test_missing_metadata_file_fails_explicitly(tmp_path):
    model, _ = copy_artifact(tmp_path)
    with pytest.raises(FireDetectionModelV5UnavailableError, match="metadata file is missing"):
        FireDetectionModelV5Runtime(model, tmp_path / "absent.json").ensure_loaded()


def test_a_corrupt_model_file_fails_explicitly(tmp_path):
    model, metadata = copy_artifact(tmp_path)
    model.write_bytes(b"this is not a pickle")
    with pytest.raises(FireDetectionModelV5UnavailableError, match="cannot be deserialized"):
        FireDetectionModelV5Runtime(model, metadata).ensure_loaded()


def test_corrupt_metadata_json_fails_explicitly(tmp_path):
    model, metadata = copy_artifact(tmp_path)
    metadata.write_text("{not json", encoding="utf-8")
    with pytest.raises(FireDetectionModelV5UnavailableError, match="cannot be read as JSON"):
        FireDetectionModelV5Runtime(model, metadata).ensure_loaded()


@pytest.mark.parametrize(
    "edit,fragment",
    [
        (lambda d: d.update(model_type="LogisticRegression"), "model_type"),
        (lambda d: d.update(model_version="4.0"), "model_version"),
        (lambda d: d.update(feature_schema_version="v4"), "feature_schema_version"),
        (lambda d: d.update(feature_names=list(reversed(d["feature_names"]))), "feature_names"),  # same names, wrong ORDER
        (lambda d: d.update(feature_names=d["feature_names"][:-1]), "feature_names"),
        (lambda d: d.update(task_7_binary_primary_model_gate="PASSED"), "Task 7 gate"),
        (lambda d: d.update(task_8_ai_hybrid_policy_gate="FAILED"), "Task 8 policy gate"),
        (lambda d: d.update(policy_version="ai_hybrid_policy_v9"), "policy_version"),
        (lambda d: d.update(suspect_threshold=0.30), "suspect_threshold"),
        (lambda d: d.update(confirm_threshold=0.70), "confirm_threshold"),
        (lambda d: d.update(multi_pixel_corroboration="satellite pixels >= 1"), "corroboration"),
        (lambda d: d["policy_gate"].update(passed=False), "policy gate"),
    ],
)
def test_incompatible_metadata_is_refused(tmp_path, edit, fragment):
    model, metadata = copy_artifact(tmp_path, edit)
    with pytest.raises(FireDetectionModelV5UnavailableError, match=fragment):
        FireDetectionModelV5Runtime(model, metadata).ensure_loaded()


def test_a_model_that_is_not_the_hgb_pipeline_is_refused(tmp_path, monkeypatch):
    class NotHgb:
        n_features_in_ = 25

        def predict_proba(self, rows):
            return np.array([[0.5, 0.5]])

    model, metadata = copy_artifact(tmp_path)
    monkeypatch.setattr(runtime_module.joblib, "load", lambda path: NotHgb())
    with pytest.raises(FireDetectionModelV5UnavailableError, match="not a HistGradientBoostingClassifier"):
        FireDetectionModelV5Runtime(model, metadata).ensure_loaded()


def test_a_model_fitted_on_a_different_feature_count_is_refused(tmp_path, monkeypatch):
    class HistGradientBoostingClassifier:  # same class NAME as the approved final estimator
        n_features_in_ = 19
        classes_ = [0, 1]

        def predict_proba(self, rows):
            return np.array([[0.5, 0.5]])

    model, metadata = copy_artifact(tmp_path)
    monkeypatch.setattr(runtime_module.joblib, "load", lambda path: HistGradientBoostingClassifier())
    with pytest.raises(FireDetectionModelV5UnavailableError, match="different number of features"):
        FireDetectionModelV5Runtime(model, metadata).ensure_loaded()


def test_a_failed_load_is_not_cached_so_a_repaired_file_is_picked_up(tmp_path):
    model, metadata = copy_artifact(tmp_path)
    good = model.read_bytes()
    model.write_bytes(b"corrupt")
    runtime = FireDetectionModelV5Runtime(model, metadata)
    with pytest.raises(FireDetectionModelV5UnavailableError):
        runtime.ensure_loaded()

    model.write_bytes(good)

    assert runtime.ensure_loaded().model_version == "5.0"


# --- inference failures ---------------------------------------------------------------------------------------------------


def test_a_classifier_exception_becomes_an_explicit_inference_error(monkeypatch):
    runtime = FireDetectionModelV5Runtime(REAL_MODEL, REAL_METADATA)
    loaded = runtime.ensure_loaded()

    class Exploding:
        def predict_proba(self, rows):
            raise RuntimeError("boom at C:\\secret\\path")

    monkeypatch.setitem(runtime_module._CACHE, runtime._key, runtime_module.LoadedModelV5(
        Exploding(), loaded.model_name, loaded.model_version, loaded.feature_schema_version, loaded.policy_version))

    with pytest.raises(FireDetectionModelV5InferenceError) as error:
        runtime.predict_probability(real_features())
    assert "secret" not in error.value.public_message


@pytest.mark.parametrize("bad", [float("nan"), 1.5, -0.1])
def test_an_invalid_probability_is_an_inference_error(monkeypatch, bad):
    runtime = FireDetectionModelV5Runtime(REAL_MODEL, REAL_METADATA)
    loaded = runtime.ensure_loaded()

    class Bad:
        def predict_proba(self, rows):
            return np.array([[1 - 0.5, bad]])

    monkeypatch.setitem(runtime_module._CACHE, runtime._key, runtime_module.LoadedModelV5(
        Bad(), loaded.model_name, loaded.model_version, loaded.feature_schema_version, loaded.policy_version))
    with pytest.raises(FireDetectionModelV5InferenceError, match="invalid probability"):
        runtime.predict_probability(real_features())


def test_a_non_v5_feature_object_is_refused():
    with pytest.raises(FireDetectionModelV5InferenceError, match="invalid feature vector"):
        FireDetectionModelV5Runtime(REAL_MODEL, REAL_METADATA).predict_probability([0.0] * 25)


# --- caching ----------------------------------------------------------------------------------------------------------------


def test_repeated_predictions_and_new_runtime_instances_deserialize_the_artifact_only_once(monkeypatch):
    loads = []
    real_load = runtime_module.joblib.load
    monkeypatch.setattr(runtime_module.joblib, "load", lambda path: loads.append(path) or real_load(path))

    first = FireDetectionModelV5Runtime(REAL_MODEL, REAL_METADATA)
    probabilities = [first.predict_probability(real_features()) for _ in range(5)]
    second = FireDetectionModelV5Runtime(REAL_MODEL, REAL_METADATA)  # e.g. the next simulation run's agent
    probabilities.append(second.predict_probability(real_features()))

    assert len(loads) == 1
    assert len(set(probabilities)) == 1  # stateless: the same input always scores the same


def test_loading_is_lazy_constructing_a_runtime_reads_nothing(monkeypatch):
    monkeypatch.setattr(runtime_module.joblib, "load", lambda path: pytest.fail("must not load at construction"))
    FireDetectionModelV5Runtime(REAL_MODEL, REAL_METADATA)


def test_the_cache_holds_only_the_immutable_model_no_incident_state():
    FireDetectionModelV5Runtime(REAL_MODEL, REAL_METADATA).ensure_loaded()
    (loaded,) = runtime_module._CACHE.values()
    assert set(vars(loaded)) == {"model", "model_name", "model_version", "feature_schema_version", "policy_version"}
    with pytest.raises(Exception):
        loaded.model_version = "x"  # frozen

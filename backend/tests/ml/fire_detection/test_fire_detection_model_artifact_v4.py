"""Tests for V4 artifact saving/loading (temp directories only) and the V3-artifacts-unchanged guarantee."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np
import pytest

from src.ml.fire_detection.fire_detection_features_v4 import FIRE_DETECTION_FEATURE_NAMES_V4, validate_feature_names_v4
from src.ml.fire_detection.fire_detection_model_artifact_v4 import (
    artifact_stem,
    build_metadata_v4,
    load_model_artifact_v4,
    save_model_artifact_v4,
)
from src.ml.fire_detection.fire_detection_model_v4 import DEFAULT_TRAINING_CSV_V4, build_pipeline_v4, dataset_sha256

BACKEND_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = BACKEND_ROOT / "models" / "fire_detection"
REPORT_PATH = BACKEND_ROOT / "data" / "fire_detection" / "fire_detection_model_comparison_v4.json"
CREATED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

REQUIRED_METADATA_KEYS = {
    "model_version",
    "model_type",
    "feature_names",
    "training_dataset_version",
    "training_dataset_hash",
    "preprocessing",
    "grouped_cv_metrics",
    "archetype_metrics",
    "suspect_threshold_candidate",
    "confirm_threshold_candidate",
    "calibration",
    "created_at",
    "python_version",
    "scikit_learn_version",
    "random_seed",
}


def selected_report(report: dict, model_key: str = "logistic_regression") -> dict:
    """A copy of a comparison report in which `model_key` passed the gate (test fixture only)."""
    chosen = copy.deepcopy(report)
    chosen["selection"] = {"selected_model": model_key, "reason": "test fixture: pretend the gate passed", "passing_models": [model_key]}
    return chosen


@pytest.fixture()
def saved(v4_matrix, small_comparison_report, tmp_path):
    report = selected_report(small_comparison_report)
    paths = save_model_artifact_v4("logistic_regression", v4_matrix, report, tmp_path, DEFAULT_TRAINING_CSV_V4, CREATED_AT)
    return paths, report


# --- gating ---


def test_saving_is_refused_when_no_model_passed_the_gate(v4_matrix, small_comparison_report, tmp_path):
    report = copy.deepcopy(small_comparison_report)
    report["selection"] = {"selected_model": None, "reason": "No model passed the generalisation gate; no runtime-ready artifact is justified."}

    with pytest.raises(ValueError, match="No model passed"):
        save_model_artifact_v4("logistic_regression", v4_matrix, report, tmp_path, DEFAULT_TRAINING_CSV_V4)
    assert list(tmp_path.iterdir()) == []


def test_saving_is_refused_for_a_model_that_was_not_selected(v4_matrix, small_comparison_report, tmp_path):
    with pytest.raises(ValueError, match="not the selected model"):
        save_model_artifact_v4("random_forest", v4_matrix, selected_report(small_comparison_report), tmp_path, DEFAULT_TRAINING_CSV_V4)
    assert list(tmp_path.iterdir()) == []


def test_unknown_models_are_refused(v4_matrix, small_comparison_report, tmp_path):
    with pytest.raises(ValueError):
        save_model_artifact_v4("neural_net", v4_matrix, selected_report(small_comparison_report), tmp_path, DEFAULT_TRAINING_CSV_V4)


def test_the_train_script_refuses_and_writes_nothing_when_no_model_was_selected(small_comparison_report, tmp_path):
    report = copy.deepcopy(small_comparison_report)
    report["selection"] = {"selected_model": None, "reason": "No model passed the generalisation gate."}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    models_dir = tmp_path / "models"

    result = subprocess.run(
        [sys.executable, "-m", "scripts.train_fire_detection_model_v4", "--report", str(report_path), "--models-dir", str(models_dir)],
        cwd=BACKEND_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "REFUSED" in result.stdout and "No artifact written" in result.stdout
    assert not models_dir.exists()


# --- saved artifact ---


def test_artifact_names_carry_the_v4_suffix_and_cannot_collide_with_v3():
    v3_names = {path.stem for path in MODELS_DIR.glob("*_v3*")}
    stems = [artifact_stem(key) for key in ("logistic_regression", "random_forest", "hist_gradient_boosting")]

    for stem in stems:
        assert stem.startswith("fire_detection_") and stem.endswith("_v4")
    assert len(set(stems)) == 3
    assert not set(stems) & v3_names


def test_saved_files_and_metadata_contract(saved):
    (model_path, metadata_path), _ = saved
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert model_path.name == "fire_detection_logistic_v4.joblib"
    assert metadata_path.name == "fire_detection_logistic_v4_metadata.json"
    assert REQUIRED_METADATA_KEYS <= set(metadata)
    assert metadata["model_version"] == "4.0" and metadata["model_type"] == "LogisticRegression"
    assert metadata["training_dataset_version"] == "training_v4"
    assert metadata["training_dataset_hash"] == dataset_sha256(DEFAULT_TRAINING_CSV_V4)
    assert metadata["created_at"] == CREATED_AT.isoformat()
    assert isinstance(metadata["suspect_threshold_candidate"], (float, type(None)))
    assert "grouped_cv_metrics" in metadata and metadata["grouped_cv_metrics"]["roc_auc"]["mean"] is not None
    assert metadata["calibration"]["calibration_applied"] is False


def test_saved_metadata_feature_order_matches_the_v4_contract(saved):
    (_, metadata_path), _ = saved
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert validate_feature_names_v4(metadata["feature_names"]) == FIRE_DETECTION_FEATURE_NAMES_V4
    assert metadata["feature_count"] == 19 and metadata["feature_schema_version"] == "v4"
    assert metadata["preprocessing"]["feature_names"] == list(FIRE_DETECTION_FEATURE_NAMES_V4)


def test_reloaded_model_gives_the_same_probabilities(saved, v4_matrix):
    (model_path, metadata_path), _ = saved
    reloaded, metadata = load_model_artifact_v4(model_path, metadata_path)
    fresh = build_pipeline_v4("logistic_regression").fit(v4_matrix.X, v4_matrix.y)

    assert np.array_equal(reloaded.predict_proba(v4_matrix.X[:500]), fresh.predict_proba(v4_matrix.X[:500]))
    assert metadata["feature_names"] == list(FIRE_DETECTION_FEATURE_NAMES_V4)


def test_reloaded_model_accepts_unavailable_fire_danger(saved, v4_matrix):
    (model_path, metadata_path), _ = saved
    reloaded, _ = load_model_artifact_v4(model_path, metadata_path)
    row = v4_matrix.X[v4_matrix.X[:, 16] == 0][:5]

    assert np.isnan(row[:, 17]).all()
    assert np.isfinite(reloaded.predict_proba(row)).all()


def test_loading_refuses_a_reordered_feature_schema(saved, tmp_path):
    (model_path, metadata_path), _ = saved
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_names"] = metadata["feature_names"][1:] + metadata["feature_names"][:1]
    tampered = tmp_path / "tampered_metadata.json"
    tampered.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="schema mismatch"):
        load_model_artifact_v4(model_path, tampered)


def test_loading_refuses_a_non_v4_schema_version(saved, tmp_path):
    (model_path, metadata_path), _ = saved
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_schema_version"] = "v3"
    tampered = tmp_path / "tampered_metadata.json"
    tampered.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="not a V4"):
        load_model_artifact_v4(model_path, tampered)


def test_loading_refuses_a_model_with_the_wrong_number_of_features(saved, v4_matrix, tmp_path):
    (_, metadata_path), _ = saved
    wrong = build_pipeline_v4("logistic_regression", FIRE_DETECTION_FEATURE_NAMES_V4[:16]).fit(v4_matrix.X[:, :16], v4_matrix.y)
    wrong_path = tmp_path / "wrong.joblib"
    joblib.dump(wrong, wrong_path)

    with pytest.raises(ValueError, match="expects 16 features"):
        load_model_artifact_v4(wrong_path, metadata_path)


def test_metadata_builder_records_the_gate_and_selection_reason(v4_matrix, small_comparison_report):
    report = selected_report(small_comparison_report)

    metadata = build_metadata_v4("logistic_regression", v4_matrix, report, DEFAULT_TRAINING_CSV_V4, CREATED_AT)

    assert metadata["generalization_gate"]["criteria"]
    assert metadata["selection_reason"] == report["selection"]["reason"]
    assert "synthetic" in metadata["synthetic_data_notice"].lower()


# --- V3 artifacts are untouched, and no V4 artifact exists unless the gate passed ---


def test_v3_artifacts_are_byte_identical_to_the_committed_versions():
    if shutil.which("git") is None:
        pytest.skip("git not available")
    inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=BACKEND_ROOT, capture_output=True, text=True)
    if inside.returncode != 0:
        pytest.skip("not a git work tree")

    v3_files = sorted(MODELS_DIR.glob("*_v3*"))
    assert v3_files, "V3 artifacts must still exist"
    for path in v3_files:
        relative = path.relative_to(BACKEND_ROOT.parent).as_posix()
        working = subprocess.run(["git", "hash-object", str(path)], cwd=BACKEND_ROOT, capture_output=True, text=True).stdout.strip()
        committed = subprocess.run(["git", "rev-parse", f"HEAD:{relative}"], cwd=BACKEND_ROOT, capture_output=True, text=True).stdout.strip()
        assert working == committed, f"{relative} differs from HEAD"


def test_the_v3_runtime_classifier_still_points_at_the_v3_artifact():
    from src.config.settings import DEFAULT_FIRE_DETECTION_ML_METADATA_PATH, DEFAULT_FIRE_DETECTION_ML_MODEL_PATH

    assert DEFAULT_FIRE_DETECTION_ML_MODEL_PATH.endswith("fire_detection_logistic_v3.joblib")
    assert DEFAULT_FIRE_DETECTION_ML_METADATA_PATH.endswith("fire_detection_logistic_v3_metadata.json")
    metadata = json.loads(Path(DEFAULT_FIRE_DETECTION_ML_METADATA_PATH).read_text(encoding="utf-8"))
    assert metadata["model_version"] == "3.0" and len(metadata["feature_names"]) == 16


def test_a_v4_artifact_exists_in_models_only_if_the_committed_report_selected_a_model():
    if not REPORT_PATH.exists():
        pytest.skip("comparison report not generated")
    selected = json.loads(REPORT_PATH.read_text(encoding="utf-8"))["selection"]["selected_model"]
    v4_artifacts = sorted(MODELS_DIR.glob("*_v4*"))

    if selected is None:
        assert v4_artifacts == [], "no model passed the gate, so no V4 artifact may exist"
    else:
        assert any(path.name.startswith(artifact_stem(selected)) for path in v4_artifacts)

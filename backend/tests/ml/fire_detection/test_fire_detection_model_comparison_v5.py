"""Task 7: end-to-end comparison on a small environment subset, report shape, determinism, gate-gated artifacts."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.ml.fire_detection.fire_detection_features_v5 import FIRE_DETECTION_FEATURE_NAMES_V5
from src.ml.fire_detection import fire_detection_model_artifact_v5 as artifact_module
from src.ml.fire_detection.fire_detection_model_artifact_v5 import (
    DEFAULT_MODEL_DIR_V5,
    ArtifactRefusedError,
    _assert_v5_name,
    artifact_stem,
    load_model_v5,
    reload_parity,
    save_selected_model_v5,
)
from src.ml.fire_detection.fire_detection_model_comparison_v5 import DATASET_ORACLE_NOTE, run_model_comparison_v5
from src.ml.fire_detection.fire_detection_model_config_v5 import CANDIDATE_GRID_V5, GATE_V5
from src.ml.fire_detection.fire_detection_model_report_v5 import render_report_markdown_v5
from src.ml.fire_detection.fire_detection_model_v5 import DEFAULT_TRAINING_CSV_V5, dataset_sha256
from src.ml.fire_detection.fire_detection_training_data_generator_v5 import FireDetectionTrainingDataGeneratorV5

REPORT_JSON = Path(__file__).resolve().parents[3] / "data" / "fire_detection" / "fire_detection_model_comparison_v5.json"
SINGLE_GRID = {
    "logistic_regression": (CANDIDATE_GRID_V5["logistic_regression"][1],),
    "hist_gradient_boosting": (CANDIDATE_GRID_V5["hist_gradient_boosting"][0],),
}


@pytest.fixture(scope="module")
def subset(v5_matrix):
    indices = np.flatnonzero(v5_matrix.environments <= 60)
    samples = FireDetectionTrainingDataGeneratorV5(42).generate(10000)
    return v5_matrix.select(indices), tuple(samples[i] for i in indices)


def _run(subset, families=("logistic_regression",), **kwargs):
    matrix, samples = subset
    options = dict(grids=SINGLE_GRID, permutation_repeats=1, bootstrap_repeats=20, csv_sha256=None)
    options.update(kwargs)
    return run_model_comparison_v5(matrix, samples, families=families, **options)


@pytest.fixture(scope="module")
def report(subset):
    return _run(subset, ("logistic_regression", "hist_gradient_boosting"))


# --- report content ---


def test_the_report_records_the_predefined_gate_grid_and_synthetic_notice(report):
    assert report["acceptance_gate"] == json.loads(json.dumps(GATE_V5))
    assert set(report["candidate_grid"]) == set(CANDIDATE_GRID_V5)
    assert "SYNTHETIC" in report["synthetic_data_notice"] and "NOT real-world" in report["synthetic_data_notice"]
    assert report["meta"]["feature_names"] == list(FIRE_DETECTION_FEATURE_NAMES_V5)
    assert report["meta"]["seed"] == 42 and report["meta"]["sklearn_version"]
    assert "primary_evaluation" in report and "environment_id" in report["primary_evaluation"]
    assert report["dataset_oracle"] == DATASET_ORACLE_NOTE and "No generator / Bayes oracle" in report["dataset_oracle"]
    assert json.dumps(report)  # JSON-serialisable (NaN already converted)


def test_the_report_contains_every_required_analysis(report):
    assert report["rule_baseline"]["suspected_or_confirmed_is_fire"]["non_sparse_rows"]["rows"] > 0
    for key, section in report["models"].items():
        assert len(section["grouped_environment"]["per_fold"]) == 5
        assert set(section["grouped_environment"]["summary"]) >= {"roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier"}
        assert set(section["grouped_environment"]["summary"]["roc_auc"]) == {"mean", "std", "min", "max"}
        assert "DIAGNOSTIC ONLY" in section["random_split_diagnostic"]["label"]
        assert len([k for k in section["leave_one_regime_out"] if k != "note"]) == 6
        assert len(section["leave_one_no_fire_subtype_out"]) == 4
        assert section["paired_cases"]["overall"]["pairs"] > 0 and section["paired_cases"]["by_pair_type"]
        assert "fraction_uncertain_0_35_to_0_65" in section["sparse_early_evidence"]
        assert set(section["calibration"]["variants"]) == {"none", "sigmoid", "isotonic"}
        assert section["calibration"]["adopted_variant"] in {"none", "sigmoid", "isotonic"}
        assert "confirmed_probability_analysis" in section and "corroboration" in section
        assert section["feature_importance"]["grouped_permutation"]["top_feature"] in FIRE_DETECTION_FEATURE_NAMES_V5
        assert set(section["missingness_audit"]["model_use"]) <= set(section["missingness_audit"]["univariate_association"])
        assert "acceptance_gate_result" in section and "passed" in section["acceptance_gate_result"]


def test_the_ablation_ladder_and_history_comparison_are_reported(report):
    for section in report["models"].values():
        ablations = section["ablations"]
        assert {"full_v5", "without_history", "retained_14_baseline", "without_news", "without_frp_brightness",
                "without_current_spatial_temporal"} == set(ablations["results"])
        assert [ablations["results"][k]["feature_count"] for k in ("retained_14_baseline", "without_history", "full_v5")] == [14, 20, 25]
        for name in ("full_v5", "without_history"):
            persistent = ablations["results"][name]["persistent_thermal"]
            assert persistent["roc_auc"] is not None and set(persistent["at_0_50"]) == {"precision", "recall", "f1", "false_positive_rate"}
            assert len(persistent["per_fold_roc_auc"]) == 5
        comparison = ablations["comparisons"]["history_effect_full_vs_without_history"]
        assert set(comparison) == {"overall_grouped", "persistent_thermal", "non_persistent_rows"}
        assert comparison["persistent_thermal"]["bootstrap_95_ci"][0] <= comparison["persistent_thermal"]["bootstrap_95_ci"][1]


def test_gate_results_use_the_predefined_criteria_and_the_rule_comparison(report):
    for section in report["models"].values():
        criteria = section["acceptance_gate_result"]["criteria"]
        assert {"grouped_environment_mean_roc_auc", "grouped_environment_worst_fold_roc_auc", "hard_negative_fpr",
                "persistent_thermal_roc_auc", "brier_score", "expected_calibration_error", "top_feature_importance_share",
                "beats_rules_hard_negative_fpr", "beats_rules_recall_not_collapsed", "beats_rules_f1_not_worse"} <= set(criteria)
        assert section["acceptance_gate_result"]["passed"] == (not section["acceptance_gate_result"]["failed_criteria"])


def test_threshold_and_corroboration_sections_are_consistent(report):
    for section in report["models"].values():
        operating = section["suspected_operating_point"]
        if operating is not None:
            assert operating["non_sparse_positive_recall"] >= 0.80
            assert operating["hard_negative_fpr"] is None or 0 <= operating["hard_negative_fpr"] <= 1
            assert f"{operating['threshold']:.2f}" in section["corroboration"]["with_probability"]
        assert "probability_only" in next(iter(section["corroboration"]["with_probability"].values()))


def test_model_selection_is_none_or_a_passing_model(report):
    selection = report["selection"]
    if selection["selected_model"] is None:
        assert selection["passing_models"] == [] and "no runtime artifact" in selection["reason"]
    else:
        assert report["models"][selection["selected_model"]]["acceptance_gate_result"]["passed"]


def test_the_comparison_is_deterministic(subset):
    first = _run(subset, run_ablations=False, run_stress_tests=False)
    second = _run(subset, run_ablations=False, run_stress_tests=False)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_the_comparison_refuses_a_bad_schema(subset):
    from dataclasses import replace

    matrix, samples = subset
    bad = replace(matrix, feature_names=tuple(reversed(matrix.feature_names)))
    with pytest.raises(ValueError, match="schema mismatch"):
        run_model_comparison_v5(bad, samples, families=("logistic_regression",), grids=SINGLE_GRID)


def test_without_samples_there_is_no_rule_baseline_and_no_model_can_pass(subset):
    matrix, _ = subset
    result = run_model_comparison_v5(matrix, None, families=("logistic_regression",), grids=SINGLE_GRID, permutation_repeats=1,
                                     run_stress_tests=False, run_ablations=False)
    assert result["rule_baseline"] is None and result["selection"]["selected_model"] is None


# --- markdown ---


def test_the_markdown_report_states_the_verdict_and_the_synthetic_notice(report):
    text = render_report_markdown_v5(report)
    assert "SYNTHETIC" in text and "Selected model:" in text
    for heading in ("Acceptance gate", "Grouped-environment results", "DIAGNOSTIC ONLY", "Leave-one-regime-out",
                    "Leave-one-no-fire-subtype-out", "Paired cases", "Sparse early evidence", "Rule baseline",
                    "History ablation", "Calibration", "Threshold analysis", "Corroboration", "Feature importance",
                    "Missingness audit", "Dataset oracle", "Limitations"):
        assert heading in text, heading
    assert "persistent_thermal" in text and "only regime with history" in text


# --- artifacts: only when the gate passed ---


def test_saving_is_refused_when_no_model_was_selected(report, subset, tmp_path):
    matrix, _ = subset
    refused = deepcopy(report)
    refused["selection"]["selected_model"] = None
    with pytest.raises(ArtifactRefusedError, match="no V5 model passed"):
        save_selected_model_v5(refused, matrix, DEFAULT_TRAINING_CSV_V5, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_saving_is_refused_when_the_selected_model_did_not_pass(report, subset, tmp_path):
    matrix, _ = subset
    inconsistent = deepcopy(report)
    inconsistent["selection"]["selected_model"] = "logistic_regression"
    inconsistent["models"]["logistic_regression"]["acceptance_gate_result"]["passed"] = False
    with pytest.raises(ArtifactRefusedError, match="did not pass"):
        save_selected_model_v5(inconsistent, matrix, DEFAULT_TRAINING_CSV_V5, tmp_path)
    assert list(tmp_path.iterdir()) == []


def _forced_selection(report, key="logistic_regression"):
    forced = deepcopy(report)
    forced["selection"] = {"selected_model": key, "reason": "forced for the artifact test", "passing_models": [key]}
    forced["models"][key]["acceptance_gate_result"]["passed"] = True
    forced["meta"]["training_csv_sha256"] = dataset_sha256(DEFAULT_TRAINING_CSV_V5)
    return forced


def test_saving_is_refused_for_a_non_frozen_dataset_hash(report, subset, tmp_path):
    matrix, _ = subset
    forced = _forced_selection(report)
    forced["meta"]["training_csv_sha256"] = "0" * 64
    with pytest.raises(ArtifactRefusedError, match="frozen benchmark"):
        save_selected_model_v5(forced, matrix, DEFAULT_TRAINING_CSV_V5, tmp_path)


def test_a_passing_model_is_saved_with_schema_hash_metadata_and_reloads_identically(report, subset, v5_matrix, tmp_path, monkeypatch):
    matrix, _ = subset
    v3_files = sorted(DEFAULT_MODEL_DIR_V5.glob("*_v3*")) + sorted(DEFAULT_MODEL_DIR_V5.glob("*_v1*")) + sorted(DEFAULT_MODEL_DIR_V5.glob("*_v2*"))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in v3_files}
    listing_before = sorted(p.name for p in DEFAULT_MODEL_DIR_V5.iterdir())

    forced = _forced_selection(report)
    created = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    captured = {}
    real_dump = artifact_module.joblib.dump
    monkeypatch.setattr(artifact_module.joblib, "dump", lambda obj, path, *a, **k: (captured.setdefault("model", obj), real_dump(obj, path, *a, **k))[1])
    model_path, metadata_path = save_selected_model_v5(forced, matrix, DEFAULT_TRAINING_CSV_V5, tmp_path, created_at=created)
    saved_in_memory = captured["model"]
    assert model_path.name == f"{artifact_stem('logistic_regression')}.joblib" == "fire_detection_logistic_v5.joblib"
    assert metadata_path.name == "fire_detection_logistic_v5_metadata.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "model_version", "model_type", "feature_names", "dataset_version", "training_csv_sha256", "python_version",
        "scikit_learn_version", "random_seed", "preprocessing", "grouped_environment_metrics", "leave_regime_metrics",
        "leave_nonfire_subtype_metrics", "paired_metrics", "rule_baseline_metrics", "history_ablation",
        "current_feature_ablation", "calibration_metrics", "suspected_threshold_candidate", "confirmed_probability_analysis",
        "feature_importance_summary", "known_limitations", "created_at",
    }
    assert required <= set(metadata)
    assert metadata["feature_names"] == list(FIRE_DETECTION_FEATURE_NAMES_V5)
    assert metadata["training_csv_sha256"] == "37e056793fb4dda72f517a20153b21094b284e1579a065140483b8a6e1d78035"
    assert metadata["created_at"] == created.isoformat() and metadata["dataset_version"] == "training_v5"
    assert "synthetic" in " ".join(metadata["known_limitations"]).lower()

    model, loaded_metadata = load_model_v5(model_path, metadata_path)
    assert loaded_metadata["feature_names"] == metadata["feature_names"]

    # reload parity across the evidence situations the runtime will see
    X, names = v5_matrix.X, list(FIRE_DETECTION_FEATURE_NAMES_V5)
    col = lambda n: X[:, names.index(n)]  # noqa: E731
    sat = col("satellite_low_count") + col("satellite_nominal_count") + col("satellite_high_count")
    news = sum(col(n) for n in ("news_none_count", "news_weak_count", "news_moderate_count", "news_strong_count", "news_unknown_count"))
    cases = {
        "single_pass": (sat == 1) & (news == 0) & (col("satellite_pass_count") == 1),
        "multi_pass_history": col("satellite_pass_count") >= 3,
        "news_only": (sat == 0) & (news > 0),
        "satellite_and_news": (sat > 0) & (news > 0),
        "nan_history_features": np.isnan(col("satellite_centroid_stability_km")),
        "sparse_evidence": v5_matrix.regimes == "sparse_early_evidence",
    }
    for name, mask in cases.items():
        rows = X[np.flatnonzero(mask)[:50]]
        assert len(rows) > 0, name
        assert reload_parity(saved_in_memory, model, rows) == 0.0, name  # same vectors -> same probabilities before / after serialization

    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in v3_files} == before  # V1-V3 artifacts untouched
    assert sorted(p.name for p in DEFAULT_MODEL_DIR_V5.iterdir()) == listing_before  # nothing written outside tmp_path

    with pytest.raises(ArtifactRefusedError, match="already exists"):
        save_selected_model_v5(forced, matrix, DEFAULT_TRAINING_CSV_V5, tmp_path)


def test_only_v5_artifact_names_are_allowed(tmp_path):
    for bad in ("fire_detection_logistic_v3.joblib", "fire_detection_logistic_v4.joblib", "model.joblib", "fire_detection_logistic_v3_metadata.json"):
        with pytest.raises(ArtifactRefusedError):
            _assert_v5_name(tmp_path / bad)
    _assert_v5_name(tmp_path / "fire_detection_random_forest_v5.joblib")
    _assert_v5_name(tmp_path / "fire_detection_random_forest_v5_metadata.json")


def test_a_metadata_file_with_a_wrong_feature_schema_is_refused(tmp_path):
    bad = tmp_path / "fire_detection_logistic_v5_metadata.json"
    bad.write_text(json.dumps({"feature_names": list(reversed(FIRE_DETECTION_FEATURE_NAMES_V5)), "feature_schema_version": "v5"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        load_model_v5(tmp_path / "missing.joblib", bad)


@pytest.mark.skipif(not REPORT_JSON.exists(), reason="the full comparison has not been run")
def test_the_committed_comparison_and_the_artifact_directory_agree():
    committed = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    # The Task 8 AI-hybrid-policy artifact (fire_detection_hgb_v5*) is governed by ITS OWN gate and is not a Task 7 artifact.
    v5_files = sorted(p.name for p in DEFAULT_MODEL_DIR_V5.glob("*_v5*") if not p.name.startswith("fire_detection_hgb_v5"))
    selected = committed["selection"]["selected_model"]
    if selected is None:
        assert v5_files == []  # a failed Task 7 gate leaves no Task 7 runtime artifact behind
    else:
        assert committed["models"][selected]["acceptance_gate_result"]["passed"]
        assert f"{artifact_stem(selected)}.joblib" in v5_files
    assert committed["meta"]["training_csv_sha256"] == "37e056793fb4dda72f517a20153b21094b284e1579a065140483b8a6e1d78035"
    assert committed["acceptance_gate"] == json.loads(json.dumps(GATE_V5))
    assert "SYNTHETIC" in committed["synthetic_data_notice"]

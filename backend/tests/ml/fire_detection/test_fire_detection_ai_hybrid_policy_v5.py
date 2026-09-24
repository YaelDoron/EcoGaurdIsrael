"""Task 8: the locked AI Hybrid policy, its confirmatory dataset, gate, metrics, rule comparison and artifact."""
from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.ml.fire_detection import fire_detection_policy_evaluation_v5 as evaluation_module
from src.ml.fire_detection.fire_detection_ai_hybrid_policy_v5 import (
    STATUS_CODES,
    current_satellite_pixel_count,
    decide_status,
    decide_statuses,
    is_multi_pixel,
)
from src.ml.fire_detection.fire_detection_dataset_v5 import load_training_dataset_rows_v5
from src.ml.fire_detection.fire_detection_features_v5 import FIRE_DETECTION_FEATURE_NAMES_V5
from src.ml.fire_detection.fire_detection_model_config_v5 import GATE_V5
from src.ml.fire_detection.fire_detection_model_v5 import DEFAULT_TRAINING_CSV_V5, load_training_matrix_v5, matrix_from_rows
from src.ml.fire_detection.fire_detection_policy_config_v5 import (
    CONFIRM_THRESHOLD,
    CONFIRMATORY_CSV_SHA256,
    CONFIRMATORY_FILE_NAME,
    CONFIRMATORY_ROWS,
    CONFIRMATORY_SEED,
    ENVIRONMENT_ID_OFFSET,
    MODEL_FAMILY,
    MODEL_PARAMS,
    MULTI_PIXEL_COUNT_FEATURES,
    POLICY_GATE_V5,
    POLICY_VERSION,
    SUSPECT_THRESHOLD,
    TASK_7_BINARY_PRIMARY_GATE_VERDICT,
    TRAINING_CSV_SHA256,
)
from src.ml.fire_detection.fire_detection_policy_evaluation_v5 import (
    POLICY_FAILED,
    POLICY_PASSED,
    ArtifactRefusedError,
    evaluate_policy_gate,
    evaluate_policy_v5,
    reload_parity,
    reload_parity_cases,
    save_policy_artifact,
    train_locked_model,
)
from src.ml.fire_detection.fire_detection_policy_validation_data_v5 import (
    DEFAULT_CONFIRMATORY_CSV,
    assert_disjoint,
    confirmatory_sha256,
    cross_dataset_overlap,
    generate_confirmatory_samples,
    load_confirmatory_rows,
)
from src.ml.fire_detection.fire_detection_training_data_generator_v5 import DEFAULT_TRAINING_DATA_SEED_V5, RegimeV5

DATA = Path(__file__).resolve().parents[3] / "data" / "fire_detection"
MODELS = Path(__file__).resolve().parents[3] / "models" / "fire_detection"
REPORT_JSON = DATA / "fire_detection_v5_policy_validation_report.json"
NO_EVENT, SUSPECTED, CONFIRMED = STATUS_CODES["NO_EVENT"], STATUS_CODES["SUSPECTED"], STATUS_CODES["CONFIRMED"]


@pytest.fixture(scope="module")
def confirmatory_rows():
    return load_confirmatory_rows()


@pytest.fixture(scope="module")
def confirmatory_matrix(confirmatory_rows):
    return matrix_from_rows(confirmatory_rows)


@pytest.fixture(scope="module")
def committed_report():
    return json.loads(REPORT_JSON.read_text(encoding="utf-8"))


# --- the policy and the model were locked before evaluation ---


def test_the_policy_and_model_are_locked_to_the_specification():
    assert (SUSPECT_THRESHOLD, CONFIRM_THRESHOLD) == (0.40, 0.80)
    assert MULTI_PIXEL_COUNT_FEATURES == ("satellite_low_count", "satellite_nominal_count", "satellite_high_count")
    assert MODEL_FAMILY == "hist_gradient_boosting" and MODEL_PARAMS == {"max_depth": 3, "learning_rate": 0.1}
    assert POLICY_VERSION == "ai_hybrid_policy_v5.0"
    assert TASK_7_BINARY_PRIMARY_GATE_VERDICT == "FAILED"  # the historical verdict stays documented


def test_the_policy_acceptance_gate_is_the_predefined_one():
    assert POLICY_GATE_V5 == {
        "confirmed_precision_min": 0.90,
        "confirmed_recall_min_non_sparse": 0.10,
        "false_confirmation_rate_max": 0.10,
        "fire_retained_recall_min_non_sparse": 0.80,
        "non_sparse_no_fire_alert_rate_max_ratio_vs_rules": 0.75,
        "sparse_confirmed_rate_max": 0.05,
        "brier_score_max_exclusive": 0.25,
        "expected_calibration_error_max": 0.10,
        "persistent_thermal_roc_auc_min": 0.70,
    }


def test_the_task_7_result_is_preserved_unchanged():
    task7 = json.loads((DATA / "fire_detection_model_comparison_v5.json").read_text(encoding="utf-8"))
    assert task7["selection"]["selected_model"] is None and task7["selection"]["passing_models"] == []
    assert task7["acceptance_gate"] == json.loads(json.dumps(GATE_V5))  # the historical gate is not weakened
    for section in task7["models"].values():
        assert section["acceptance_gate_result"]["passed"] is False
    assert task7["models"]["hist_gradient_boosting"]["selected_params"] == {"learning_rate": 0.1, "max_depth": 3}
    assert "NONE" in (Path(__file__).resolve().parents[3] / "docs" / "fire_detection_model_comparison_v5.md").read_text(encoding="utf-8")


# --- exact NO_EVENT / SUSPECTED / CONFIRMED mapping ---


@pytest.mark.parametrize(
    "probability,pixels,expected",
    [
        (0.0, 5, "NO_EVENT"), (0.399999, 5, "NO_EVENT"), (0.40, 0, "SUSPECTED"), (0.40, 5, "SUSPECTED"),
        (0.799999, 9, "SUSPECTED"), (0.80, 1, "SUSPECTED"), (0.95, 1, "SUSPECTED"), (1.0, 1, "SUSPECTED"),
        (0.80, 2, "CONFIRMED"), (0.80, 7, "CONFIRMED"), (1.0, 2, "CONFIRMED"), (0.85, 0, "SUSPECTED"),
        (0.79, 8, "SUSPECTED"), (0.39, 8, "NO_EVENT"),
    ],
)
def test_exact_state_mapping(probability, pixels, expected):
    assert decide_status(probability, pixels) == expected


def test_the_multi_pixel_definition_is_low_plus_nominal_plus_high_of_the_current_candidate():
    assert current_satellite_pixel_count({"satellite_low_count": 1, "satellite_nominal_count": 0, "satellite_high_count": 1}) == 2
    assert current_satellite_pixel_count({"satellite_low_count": 0, "satellite_nominal_count": 1, "satellite_high_count": 0}) == 1
    assert is_multi_pixel(2) and not is_multi_pixel(1) and not is_multi_pixel(0)
    # pass count / history are NOT part of the guardrail: three passes of ONE pixel each is not "multi-pixel"
    assert decide_status(0.95, current_satellite_pixel_count({"satellite_low_count": 0, "satellite_nominal_count": 1, "satellite_high_count": 0,
                                                              "satellite_pass_count": 3})) == "SUSPECTED"


def test_invalid_inputs_are_rejected():
    for bad in (float("nan"), -0.01, 1.01, True, None, "0.5"):
        with pytest.raises(ValueError):
            decide_status(bad, 3)
    with pytest.raises(ValueError):
        decide_status(0.5, -1)
    with pytest.raises(ValueError):
        decide_statuses([0.5, 2.0], [1, 1])
    with pytest.raises(ValueError):
        decide_statuses([0.5], [1, 2])


def test_the_vectorised_policy_matches_the_scalar_policy():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.random(500), [0.0, 0.39999, 0.4, 0.79999, 0.8, 1.0]])
    pixels = rng.integers(0, 6, len(p))
    codes = decide_statuses(p, pixels)
    assert [STATUS_CODES[decide_status(float(a), int(b))] for a, b in zip(p, pixels)] == codes.tolist()


def test_news_only_evidence_can_never_be_confirmed_by_the_policy(confirmatory_matrix):
    news_only = evaluation_module.satellite_pixel_counts(confirmatory_matrix) == 0
    codes = decide_statuses(np.ones(len(news_only)), evaluation_module.satellite_pixel_counts(confirmatory_matrix))
    assert not (codes[news_only] == CONFIRMED).any() and (codes[~news_only] == CONFIRMED).any()


# --- the confirmatory dataset ---


def test_the_confirmatory_dataset_uses_a_new_seed_and_the_locked_hash(confirmatory_rows):
    assert CONFIRMATORY_SEED != DEFAULT_TRAINING_DATA_SEED_V5 and CONFIRMATORY_SEED == 8202609
    assert DEFAULT_CONFIRMATORY_CSV.name == CONFIRMATORY_FILE_NAME == "fire_detection_v5_policy_validation.csv"
    assert confirmatory_sha256() == CONFIRMATORY_CSV_SHA256
    assert len(confirmatory_rows) == CONFIRMATORY_ROWS and {r.seed for r in confirmatory_rows} == {CONFIRMATORY_SEED}
    with pytest.raises(ValueError, match="different from the V5 training seed"):
        generate_confirmatory_samples(seed=DEFAULT_TRAINING_DATA_SEED_V5, num_samples=60)


def test_no_environment_pair_or_sample_overlap_with_training(confirmatory_rows):
    overlap = cross_dataset_overlap(load_training_dataset_rows_v5(DEFAULT_TRAINING_CSV_V5), confirmatory_rows)
    assert overlap["shared_environment_ids"] == [] and overlap["shared_sample_ids"] == [] and overlap["shared_pair_ids"] == []
    assert not set(overlap["training_seeds"]) & set(overlap["confirmatory_seeds"])
    assert overlap["confirmatory_environments"] == 250 and overlap["confirmatory_pairs"] == 2500
    assert min(r.environment_id for r in confirmatory_rows) > ENVIRONMENT_ID_OFFSET
    assert 0 < overlap["exact_feature_vector_overlap_fraction"] < 0.15  # a diagnostic; nothing is removed
    assert_disjoint(overlap)


def test_the_disjointness_check_catches_an_overlap(confirmatory_rows):
    from dataclasses import replace

    training = load_training_dataset_rows_v5(DEFAULT_TRAINING_CSV_V5)
    leaky = (replace(confirmatory_rows[0], environment_id=training[0].environment_id), *confirmatory_rows[1:])
    with pytest.raises(ValueError, match="shared_environment_ids"):
        assert_disjoint(cross_dataset_overlap(training, leaky))
    same_seed = (replace(confirmatory_rows[0], seed=training[0].seed), *confirmatory_rows[1:])
    with pytest.raises(ValueError, match="shares a seed"):
        assert_disjoint(cross_dataset_overlap(training, same_seed))


def test_the_confirmatory_data_uses_the_frozen_generator_regimes_and_balance(confirmatory_matrix):
    assert set(confirmatory_matrix.regimes.tolist()) == {r.value for r in RegimeV5}
    for regime in RegimeV5:
        mask = confirmatory_matrix.regimes == regime.value
        assert abs(int(confirmatory_matrix.y[mask].sum()) - int((confirmatory_matrix.y[mask] == 0).sum())) <= 1
    assert confirmatory_matrix.y.sum() == 5000


def test_the_confirmatory_file_is_deterministic(tmp_path):
    from src.ml.fire_detection.fire_detection_policy_validation_data_v5 import write_confirmatory_dataset

    assert confirmatory_sha256(write_confirmatory_dataset(tmp_path / "again.csv")) == CONFIRMATORY_CSV_SHA256


def test_confirmatory_rows_never_reach_training():
    tree = ast.parse(inspect.getsource(evaluation_module))
    fit_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "fit"]
    assert len(fit_calls) == 1  # the single training call, inside train_locked_model
    signature = inspect.signature(train_locked_model)
    assert list(signature.parameters) == ["training_matrix"]
    matrix = load_training_matrix_v5()
    confirmatory_envs = {r.environment_id for r in load_confirmatory_rows()}
    assert not confirmatory_envs & set(matrix.environments.tolist())


# --- gate, metrics, rule comparison (stub model on a confirmatory subset) ---


class StubModel:
    def __init__(self, probabilities):
        self._p = np.asarray(probabilities, dtype=float)

    def predict_proba(self, X):
        assert len(X) == len(self._p)
        return np.column_stack([1 - self._p, self._p])


@pytest.fixture(scope="module")
def subset(confirmatory_matrix):
    indices = np.flatnonzero(confirmatory_matrix.environments <= ENVIRONMENT_ID_OFFSET + 60)
    samples = generate_confirmatory_samples()
    return confirmatory_matrix.select(indices), tuple(samples[i] for i in indices)


def _evaluate(subset, probabilities):
    matrix, samples = subset
    return evaluate_policy_v5(StubModel(probabilities), matrix, samples, training_csv_sha256="t", confirmatory_csv_sha256="c",
                              training_rows=10000, overlap={"shared_environment_ids": [], "shared_sample_ids": [], "shared_pair_ids": [],
                                                            "exact_feature_vector_overlap_rows": 0, "exact_feature_vector_overlap_fraction": 0.0,
                                                            "exact_overlap_by_regime": {}, "training_seeds": [42], "confirmatory_seeds": [1]})


def test_an_ideal_probability_source_passes_the_gate_and_the_metrics_are_computed_correctly(subset):
    matrix, _ = subset
    pixels = evaluation_module.satellite_pixel_counts(matrix)
    p = np.where(matrix.y == 1, 0.9, 0.1)
    report = _evaluate(subset, p)
    m = report["metrics"]
    confirmable = (matrix.y == 1) & (pixels >= 2)
    assert m["confirmed_rows"] == int(confirmable.sum()) and m["confirmed_precision"] == 1.0
    assert m["false_confirmation_rate"] == 0.0 and m["false_confirmed_rows"] == 0
    non_sparse_pos = (matrix.y == 1) & (matrix.regimes != "sparse_early_evidence")
    assert m["fire_retained_recall_non_sparse"] == 1.0
    assert m["confirmed_recall_non_sparse"] == pytest.approx(float((confirmable & non_sparse_pos).sum() / non_sparse_pos.sum()))
    assert m["non_sparse_no_fire_alert_rate"] == 0.0 and m["sparse_confirmed_rate"] == 0.0
    assert report["verdict"] == POLICY_PASSED and report["policy_gate"]["passed"]


def test_an_uninformative_model_fails_the_gate(subset):
    matrix, _ = subset
    report = _evaluate(subset, np.full(len(matrix.y), 0.9))  # alerts everything, confirms every multi-pixel row
    assert report["verdict"] == POLICY_FAILED
    assert {"confirmed_precision", "false_confirmation_rate"} <= set(report["policy_gate"]["failed_criteria"])
    assert report["metrics"]["false_confirmation_rate"] > 0.10


def test_the_false_confirmation_metric_is_no_fire_rows_confirmed_over_all_no_fire_rows(subset):
    matrix, _ = subset
    pixels = evaluation_module.satellite_pixel_counts(matrix)
    p = np.where(matrix.y == 0, 0.95, 0.05)  # every no-fire row looks like a certain fire
    report = _evaluate(subset, p)
    no_fire = matrix.y == 0
    expected = float(((pixels >= 2) & no_fire).sum() / no_fire.sum())
    assert report["metrics"]["false_confirmation_rate"] == pytest.approx(expected)
    by_subtype = report["per_latent_subtype"]
    assert set(by_subtype) == set(matrix.subtypes.tolist())
    for name, entry in by_subtype.items():
        mask = matrix.subtypes == name
        if entry["label"] == 0:
            assert entry["confirmed_rate"] == pytest.approx(float(((pixels >= 2) & mask).sum() / mask.sum()))


def test_the_sparse_confirmation_cap_is_enforced_by_the_gate(subset):
    matrix, _ = subset
    sparse = matrix.regimes == "sparse_early_evidence"
    pixels = evaluation_module.satellite_pixel_counts(matrix)
    p = np.where(matrix.y == 1, 0.9, 0.1)
    p = np.where(sparse & (pixels >= 2), 0.95, p)  # the policy would confirm sparse multi-pixel rows if any existed
    # sparse rows are single-item by construction, so the guardrail alone already keeps the sparse CONFIRMED rate at 0
    assert (pixels[sparse] < 2).all()
    assert _evaluate(subset, p)["metrics"]["sparse_confirmed_rate"] == 0.0
    ok = {"confirmed_precision": 0.95, "confirmed_recall_non_sparse": 0.2, "false_confirmation_rate": 0.01,
          "fire_retained_recall_non_sparse": 0.85, "non_sparse_no_fire_alert_rate": 0.4, "rule_non_sparse_no_fire_alert_rate": 0.8,
          "sparse_confirmed_rate": 0.06, "brier_score": 0.2, "expected_calibration_error": 0.02, "persistent_thermal_roc_auc": 0.75}
    result = evaluate_policy_gate(ok)
    assert result["failed_criteria"] == ["sparse_confirmed_rate"] and not result["passed"]


@pytest.mark.parametrize(
    "override,failed",
    [
        ({"confirmed_precision": 0.89}, "confirmed_precision"),
        ({"confirmed_recall_non_sparse": 0.09}, "confirmed_recall_non_sparse"),
        ({"false_confirmation_rate": 0.11}, "false_confirmation_rate"),
        ({"fire_retained_recall_non_sparse": 0.79}, "fire_retained_recall_non_sparse"),
        ({"non_sparse_no_fire_alert_rate": 0.61}, "non_sparse_no_fire_alert_rate_vs_rules"),
        ({"sparse_confirmed_rate": 0.051}, "sparse_confirmed_rate"),
        ({"brier_score": 0.25}, "brier_score"),
        ({"expected_calibration_error": 0.101}, "expected_calibration_error"),
        ({"persistent_thermal_roc_auc": 0.69}, "persistent_thermal_roc_auc"),
        ({"confirmed_precision": None}, "confirmed_precision"),
    ],
)
def test_every_policy_gate_criterion_is_critical(override, failed):
    base = {"confirmed_precision": 0.95, "confirmed_recall_non_sparse": 0.2, "false_confirmation_rate": 0.01,
            "fire_retained_recall_non_sparse": 0.85, "non_sparse_no_fire_alert_rate": 0.4, "rule_non_sparse_no_fire_alert_rate": 0.8,
            "sparse_confirmed_rate": 0.0, "brier_score": 0.2, "expected_calibration_error": 0.02, "persistent_thermal_roc_auc": 0.75}
    assert evaluate_policy_gate(base)["passed"]
    result = evaluate_policy_gate({**base, **override})
    assert not result["passed"] and failed in result["failed_criteria"]


def test_the_rule_detector_is_evaluated_on_exactly_the_same_rows(subset):
    matrix, _ = subset
    report = _evaluate(subset, np.where(matrix.y == 1, 0.9, 0.1))
    comparison = report["rule_comparison"]
    assert comparison["non_sparse_alert"]["rule_detector"]["rows"] == comparison["non_sparse_alert"]["ai_hybrid_policy"]["rows"]
    assert comparison["non_sparse_alert"]["rule_detector"]["rows"] == int((matrix.regimes != "sparse_early_evidence").sum())
    assert {"precision", "recall", "f1", "false_positives", "hard_negative_fpr"} <= set(comparison["non_sparse_alert"]["rule_detector"])
    assert comparison["false_confirmed_rows"]["rule_detector"] >= 0 and "confirmed_precision" in comparison
    assert report["metrics"]["rule_non_sparse_no_fire_alert_rate"] == pytest.approx(
        comparison["non_sparse_alert"]["rule_detector"]["false_positive_rate"])


def test_the_outcome_matrix_partitions_every_row(subset):
    matrix, _ = subset
    report = _evaluate(subset, np.random.default_rng(3).random(len(matrix.y)))
    for key in ("all_rows",):
        total = sum(report["outcome_matrix"][key][t][s] for t in ("fire", "no_fire") for s in ("NO_EVENT", "SUSPECTED", "CONFIRMED"))
        assert total == len(matrix.y)
    per_regime_total = sum(e[t][s] for e in report["per_regime"].values() for t in ("fire", "no_fire") for s in ("NO_EVENT", "SUSPECTED", "CONFIRMED"))
    assert per_regime_total == len(matrix.y)
    subtype_total = sum(sum(e["statuses"][s] for s in ("NO_EVENT", "SUSPECTED", "CONFIRMED")) for e in report["per_latent_subtype"].values())
    assert subtype_total == len(matrix.y)


# --- artifact: gated, complete, reloads identically ---


def test_saving_is_refused_when_the_policy_gate_failed(tmp_path, committed_report):
    failed = deepcopy(committed_report)
    failed["verdict"] = POLICY_FAILED
    failed["policy_gate"]["passed"] = False
    with pytest.raises(ArtifactRefusedError, match="did not pass"):
        save_policy_artifact(failed, object(), DEFAULT_TRAINING_CSV_V5, DEFAULT_CONFIRMATORY_CSV, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_saving_is_refused_for_non_frozen_data(tmp_path, committed_report):
    tampered = tmp_path / "confirmatory.csv"
    tampered.write_bytes(DEFAULT_CONFIRMATORY_CSV.read_bytes() + b"\n")
    with pytest.raises(ArtifactRefusedError, match="locked confirmatory"):
        save_policy_artifact(committed_report, object(), DEFAULT_TRAINING_CSV_V5, tampered, tmp_path / "models")
    bad_training = tmp_path / "training.csv"
    bad_training.write_bytes(DEFAULT_TRAINING_CSV_V5.read_bytes() + b"\n")
    with pytest.raises(ArtifactRefusedError, match="frozen V5 benchmark"):
        save_policy_artifact(committed_report, object(), bad_training, DEFAULT_CONFIRMATORY_CSV, tmp_path / "models")


def test_the_saved_artifact_metadata_distinguishes_the_two_gates(tmp_path, committed_report):
    model = train_locked_model(load_training_matrix_v5())
    model_path, metadata_path = save_policy_artifact(
        committed_report, model, DEFAULT_TRAINING_CSV_V5, DEFAULT_CONFIRMATORY_CSV, tmp_path, created_at=datetime(2026, 9, 24, tzinfo=timezone.utc))
    assert model_path.name == "fire_detection_hgb_v5.joblib" and metadata_path.name == "fire_detection_hgb_v5_metadata.json"
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert meta["task_7_binary_primary_model_gate"] == "FAILED" and meta["task_8_ai_hybrid_policy_gate"] == "PASSED"
    assert meta["feature_names"] == list(FIRE_DETECTION_FEATURE_NAMES_V5)
    assert meta["hyperparameters"]["max_depth"] == 3 and meta["hyperparameters"]["learning_rate"] == 0.1
    assert meta["training_dataset_sha256"] == TRAINING_CSV_SHA256 and meta["confirmatory_dataset_sha256"] == CONFIRMATORY_CSV_SHA256
    assert (meta["policy_version"], meta["suspect_threshold"], meta["confirm_threshold"]) == (POLICY_VERSION, 0.40, 0.80)
    assert "satellite_low_count" in meta["multi_pixel_corroboration"] and ">= 2" in meta["multi_pixel_corroboration"]
    assert meta["confirmatory_metrics"]["confirmed_precision"] == committed_report["metrics"]["confirmed_precision"]
    assert meta["python_version"] and meta["scikit_learn_version"] and "synthetic" in " ".join(meta["known_limitations"]).lower()
    with pytest.raises(ArtifactRefusedError, match="already exists"):
        save_policy_artifact(committed_report, model, DEFAULT_TRAINING_CSV_V5, DEFAULT_CONFIRMATORY_CSV, tmp_path)


def test_reload_parity_across_every_evidence_situation(tmp_path, committed_report, confirmatory_matrix):
    model = train_locked_model(load_training_matrix_v5())
    model_path, _ = save_policy_artifact(committed_report, model, DEFAULT_TRAINING_CSV_V5, DEFAULT_CONFIRMATORY_CSV, tmp_path)
    parity = reload_parity(model, joblib.load(model_path), confirmatory_matrix)
    assert set(parity) == {"news_only", "one_hotspot", "multi_pixel_candidate", "satellite_and_news", "one_pass_event",
                           "three_pass_event", "sparse_ambiguous_evidence", "nullable_history_features_nan"}
    assert all(value == 0.0 for value in parity.values())
    assert all(len(idx) > 0 for idx in reload_parity_cases(confirmatory_matrix).values())


# --- the committed evaluation and artifact ---


def test_the_committed_report_matches_the_locked_configuration(committed_report):
    r = committed_report
    assert r["task_7_binary_primary_gate"] == "FAILED"
    assert r["verdict"] in (POLICY_PASSED, POLICY_FAILED) and (r["verdict"] == POLICY_PASSED) == r["policy_gate"]["passed"]
    assert r["policy_gate_definition"] == POLICY_GATE_V5
    assert (r["locked_policy"]["suspect_threshold"], r["locked_policy"]["confirm_threshold"]) == (SUSPECT_THRESHOLD, CONFIRM_THRESHOLD)
    assert r["locked_model"]["params"] == MODEL_PARAMS and r["locked_model"]["confirmatory_rows_used_in_training"] == 0
    d = r["confirmatory_dataset"]
    assert (d["seed"], d["rows"], d["sha256"]) == (CONFIRMATORY_SEED, CONFIRMATORY_ROWS, CONFIRMATORY_CSV_SHA256)
    assert d["overlap_with_training"]["shared_environment_ids"] == [] and d["overlap_with_training"]["shared_pair_ids"] == []
    assert "SYNTHETIC" in r["synthetic_data_notice"]


def test_the_committed_artifact_reproduces_the_reported_confirmatory_metrics(committed_report, confirmatory_matrix):
    """Load the SAVED model from disk and recompute the headline gate metrics: they must equal the report."""
    if committed_report["verdict"] != POLICY_PASSED:
        assert not (MODELS / "fire_detection_hgb_v5.joblib").exists()
        return
    model = joblib.load(MODELS / "fire_detection_hgb_v5.joblib")
    p = model.predict_proba(confirmatory_matrix.X)[:, 1]
    codes = decide_statuses(p, evaluation_module.satellite_pixel_counts(confirmatory_matrix))
    y = confirmatory_matrix.y
    confirmed = codes == CONFIRMED
    assert float(y[confirmed].mean()) == pytest.approx(committed_report["metrics"]["confirmed_precision"])
    assert float(confirmed[y == 0].mean()) == pytest.approx(committed_report["metrics"]["false_confirmation_rate"])
    assert int(confirmed.sum()) == committed_report["metrics"]["confirmed_rows"]
    meta = json.loads((MODELS / "fire_detection_hgb_v5_metadata.json").read_text(encoding="utf-8"))
    assert meta["task_7_binary_primary_model_gate"] == "FAILED" and meta["task_8_ai_hybrid_policy_gate"] == "PASSED"
    assert meta["feature_names"] == list(FIRE_DETECTION_FEATURE_NAMES_V5)


def test_the_v3_v1_v2_artifacts_are_untouched_and_no_task_7_artifact_exists():
    names = {p.name for p in MODELS.iterdir()}
    assert {"fire_detection_logistic_v3.joblib", "fire_detection_random_forest_v3.joblib"} <= names
    assert {n for n in names if n.endswith("_v5.joblib")} <= {"fire_detection_hgb_v5.joblib"}
    assert not [n for n in names if "_v4" in n]

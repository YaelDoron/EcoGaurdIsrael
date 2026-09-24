"""Tests for the V4 comparison orchestration (run on a reduced, family-complete subset for speed)."""
from __future__ import annotations

import json

import pytest

from src.ml.fire_detection.fire_detection_features_v4 import FIRE_DETECTION_FEATURE_NAMES_V4
from src.ml.fire_detection.fire_detection_model_comparison_v4 import (
    ablation_subsets,
    fire_danger_verdict,
    run_model_comparison_v4,
)
from src.ml.fire_detection.fire_detection_model_v4 import DEFAULT_TRAINING_CSV_V4

REPORT_SECTIONS = (
    "synthetic_data_notice",
    "dataset",
    "environment",
    "evaluation_design",
    "rule_baseline",
    "models",
    "fire_danger_ablation_and_feature_groups",
    "hyperparameter_sensitivity_grouped_family_cv",
    "generalization_gate_definition",
    "selection",
)


def test_report_has_every_required_section(small_comparison_report):
    for section in REPORT_SECTIONS:
        assert section in small_comparison_report, section


def test_report_states_that_the_data_is_synthetic(small_comparison_report):
    notice = small_comparison_report["synthetic_data_notice"]

    assert "synthetic" in notice.lower() and "NOT real-world" in notice


def test_report_records_dataset_and_environment_provenance(small_comparison_report):
    from src.ml.fire_detection.fire_detection_model_v4 import dataset_sha256

    dataset = small_comparison_report["dataset"]
    environment = small_comparison_report["environment"]

    assert dataset["sha256"] == dataset_sha256(DEFAULT_TRAINING_CSV_V4)
    assert dataset["feature_names"] == list(FIRE_DETECTION_FEATURE_NAMES_V4)
    assert dataset["scenario_families"] == 20
    assert environment["random_seed"] == 42
    assert environment["python"].startswith("3.") and environment["scikit_learn"]
    assert environment["training_dataset_sha256"] == dataset["sha256"]


def test_model_section_contains_every_required_analysis(small_comparison_report):
    model = small_comparison_report["models"]["logistic_regression"]

    assert len(model["grouped_family_cv"]["per_fold"]) == 5
    assert model["grouped_family_cv"]["primary_selection_metric"] is True
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert set(model["grouped_family_cv"]["summary"][metric]) == {"mean", "std", "min", "max"}
        assert set(model["leave_one_archetype_out"]["summary"][metric]) == {"mean", "std", "min", "max"}
    assert {entry["split"] for entry in model["leave_one_archetype_out"]["per_archetype"]} == {
        "satellite_only", "news_led", "satellite_news", "multi_satellite"
    }
    assert "held_out_20_percent" in model["random_split_diagnostic"]
    assert "diagnostic only" in model["random_split_diagnostic"]["note"].lower()
    assert [row["threshold"] for row in model["thresholds"]["candidate_grid_out_of_fold_grouped"]] == [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
    assert "hard_negative_false_positive_rate" in model["subgroups_at_threshold_0.5"]
    assert "weak_positive_recall" in model["subgroups_at_threshold_0.5"]
    assert "brier_score" in model["calibration"]["uncalibrated_out_of_fold"]
    assert model["calibration"]["uncalibrated_out_of_fold"]["bins"]
    assert "permutation_importance_grouped" in model["interpretability"]
    assert "standardized_coefficients_full_fit" in model["interpretability"]
    assert model["leave_one_family_out_supplementary"]["pooled_out_of_fold"]["rows"] == 1200
    assert "generalization_gate" in model


def test_preprocessing_is_documented_in_the_report(small_comparison_report):
    steps = {step["step"]: step for step in small_comparison_report["models"]["logistic_regression"]["preprocessing"]["steps"]}

    assert steps["impute_nullable_fire_danger"]["strategy"] == "median"
    assert "fire_danger_available" in steps["passthrough"]["applied_to"]
    assert "scale" in steps


def test_rule_baseline_is_reported_for_both_fire_definitions(small_comparison_report):
    rule = small_comparison_report["rule_baseline"]

    for variant in ("suspected_or_confirmed_is_fire", "confirmed_only_is_fire"):
        overall = rule[variant]["overall"]
        assert {"precision", "recall", "f1", "false_positive", "false_negative"} <= set(overall)
        assert len(rule[variant]["per_fold"]) == 5
    assert rule["subgroups"]["suspected_or_confirmed_is_fire"]["hard_negative_false_positive_rate"] > 0.9


def test_fire_danger_ablation_covers_every_feature_group(small_comparison_report):
    ablation = small_comparison_report["fire_danger_ablation_and_feature_groups"]["logistic_regression"]

    assert set(ablation) == {"full_19", "no_fire_danger", "no_news", "no_satellite_frp_brightness", "no_geometry_time"}
    assert ablation["full_19"]["features"] == 19 and ablation["no_fire_danger"]["features"] == 16
    assert ablation["no_fire_danger"]["removed"] == ["fire_danger_age_minutes", "fire_danger_available", "fire_danger_score"]
    paired = ablation["no_fire_danger"]["paired_difference_full_minus_reduced"]
    assert set(paired) >= {"f1", "roc_auc", "precision", "recall"}
    assert len(paired["roc_auc"]["per_fold"]) == 5
    assert ablation["no_fire_danger"]["verdict"]["verdict"] in {"improved", "no material effect", "worse"}
    assert set(ablation["no_fire_danger"]["per_family_accuracy_delta_full_minus_reduced"]) and "leave_one_archetype_out" in ablation["no_fire_danger"]


def test_ablation_subsets_are_canonical_subsequences():
    subsets = ablation_subsets()

    assert subsets["full_19"] == FIRE_DETECTION_FEATURE_NAMES_V4
    for name, names in subsets.items():
        assert [n for n in FIRE_DETECTION_FEATURE_NAMES_V4 if n in names] == list(names), name
    assert len(subsets["no_news"]) == 14 and len(subsets["no_satellite_frp_brightness"]) == 13 and len(subsets["no_geometry_time"]) == 17


@pytest.mark.parametrize(
    "mean_delta, improved, expected",
    [(0.03, 5, "improved"), (0.03, 3, "no material effect"), (0.005, 5, "no material effect"), (-0.02, 0, "worse"), (0.0, 2, "no material effect")],
)
def test_fire_danger_verdict_is_a_mechanical_rule(mean_delta, improved, expected):
    paired = {"roc_auc": {"mean_delta": mean_delta, "folds_improved": improved}}

    assert fire_danger_verdict(paired, 5)["verdict"] == expected


def test_selection_is_consistent_with_the_gate(small_comparison_report):
    model = small_comparison_report["models"]["logistic_regression"]
    selection = small_comparison_report["selection"]

    if model["generalization_gate"]["passed"]:
        assert selection["selected_model"] == "logistic_regression"
    else:
        assert selection["selected_model"] is None


def test_report_is_json_serialisable_and_reproducible(small_comparison_inputs, small_comparison_report):
    matrix, samples = small_comparison_inputs

    again = run_model_comparison_v4(
        matrix,
        samples,
        csv_path=DEFAULT_TRAINING_CSV_V4,
        model_keys=("logistic_regression",),
        permutation_repeats=1,
        include_sensitivity=False,
        include_calibration_variants=False,
    )

    assert json.dumps(small_comparison_report, sort_keys=True) == json.dumps(again, sort_keys=True)
    json.dumps(small_comparison_report)  # no NaN / numpy types left


def test_comparison_refuses_samples_that_do_not_match_the_matrix(small_comparison_inputs):
    matrix, samples = small_comparison_inputs

    with pytest.raises(ValueError):
        run_model_comparison_v4(matrix, tuple(reversed(samples)), model_keys=("logistic_regression",))


# --- the committed report ---


def test_committed_report_matches_the_frozen_dataset_and_is_internally_consistent():
    from pathlib import Path

    from src.ml.fire_detection.fire_detection_model_v4 import dataset_sha256

    path = Path(DEFAULT_TRAINING_CSV_V4).parent / "fire_detection_model_comparison_v4.json"
    if not path.exists():
        pytest.skip("comparison report not generated")
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["dataset"]["sha256"] == dataset_sha256(DEFAULT_TRAINING_CSV_V4)
    assert report["dataset"]["feature_names"] == list(FIRE_DETECTION_FEATURE_NAMES_V4)
    assert set(report["models"]) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    assert "NOT real-world" in report["synthetic_data_notice"]
    passing = [key for key, model in report["models"].items() if model["generalization_gate"]["passed"]]
    assert report["selection"]["passing_models"] == passing
    if not passing:
        assert report["selection"]["selected_model"] is None
    else:
        assert report["selection"]["selected_model"] in passing
    for model in report["models"].values():
        assert len(model["grouped_family_cv"]["per_fold"]) == 5
        assert len(model["leave_one_archetype_out"]["per_archetype"]) == 4

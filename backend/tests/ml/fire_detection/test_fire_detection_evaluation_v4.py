"""Tests for the V4 evaluation logic: metrics, grouped/archetype folds, rule baseline, thresholds, calibration, gate."""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from src.ml.fire_detection.fire_detection_dataset_validation_v4 import is_hard_negative, is_weak_positive
from src.ml.fire_detection.fire_detection_evaluation_v4 import (
    CANDIDATE_THRESHOLDS,
    GATE_V4,
    NamedSplit,
    archetype_splits,
    assert_splits_are_grouped,
    binary_metrics,
    calibration_report,
    cross_validated_scores,
    evaluate_gate,
    family_splits,
    grouped_permutation_importance,
    hard_negative_mask,
    importance_shortcut_summary,
    leave_one_family_out_splits,
    paired_differences,
    per_split_metrics,
    recommend_thresholds,
    rule_baseline_outputs,
    select_model,
    subgroup_performance,
    summarize_folds,
    threshold_table,
    to_builtin,
    verify_samples_match_matrix,
    weak_positive_mask,
)
from src.ml.fire_detection.fire_detection_model_v4 import build_pipeline_v4
from src.models.fire_detection_status import FireDetectionStatus


# --- metrics ---


def test_binary_metrics_on_a_known_confusion_matrix():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.4, 0.6, 0.1])

    metrics = binary_metrics(y, scores, 0.5)

    assert (metrics["true_positive"], metrics["false_positive"], metrics["true_negative"], metrics["false_negative"]) == (1, 1, 1, 1)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == metrics["accuracy"] == 0.5
    assert metrics["false_positive_rate"] == metrics["false_negative_rate"] == 0.5
    assert metrics["roc_auc"] == pytest.approx(0.75)


def test_binary_metrics_handles_no_predicted_positives_without_dividing_by_zero():
    metrics = binary_metrics(np.array([1, 0, 1]), np.array([0.1, 0.2, 0.3]), 0.5)

    assert metrics["precision"] == 0.0 and metrics["recall"] == 0.0 and metrics["f1"] == 0.0


def test_binary_metrics_threshold_is_inclusive():
    metrics = binary_metrics(np.array([1, 0]), np.array([0.5, 0.49]), 0.5)

    assert metrics["true_positive"] == 1 and metrics["false_positive"] == 0


def test_roc_auc_is_none_when_a_split_has_a_single_class():
    assert binary_metrics(np.array([1, 1]), np.array([0.2, 0.9]))["roc_auc"] is None


def test_summarize_folds_reports_mean_std_min_max():
    folds = [{"f1": 0.2, "roc_auc": 0.6}, {"f1": 0.4, "roc_auc": 0.8}, {"f1": 0.6, "roc_auc": 0.7}]

    summary = summarize_folds(folds, keys=("f1", "roc_auc"))

    assert summary["f1"] == {"mean": pytest.approx(0.4), "std": pytest.approx(np.std([0.2, 0.4, 0.6])), "min": 0.2, "max": 0.6}
    assert summary["roc_auc"]["min"] == 0.6 and summary["roc_auc"]["max"] == 0.8


def test_paired_differences_are_computed_fold_by_fold():
    full = [{"f1": 0.7, "roc_auc": 0.8}, {"f1": 0.5, "roc_auc": 0.6}, {"f1": 0.6, "roc_auc": 0.7}]
    reduced = [{"f1": 0.6, "roc_auc": 0.7}, {"f1": 0.6, "roc_auc": 0.6}, {"f1": 0.5, "roc_auc": 0.6}]

    diff = paired_differences(full, reduced, keys=("f1",))["f1"]

    assert diff["per_fold"] == pytest.approx([0.1, -0.1, 0.1])
    assert diff["folds_improved"] == 2 and diff["folds_worse"] == 1
    assert diff["mean_delta"] == pytest.approx(0.1 / 3)


# --- grouped folds ---


def test_family_folds_never_mix_scenario_families(v4_matrix):
    splits = family_splits(v4_matrix, 5)

    assert len(splits) == 5
    assert_splits_are_grouped(v4_matrix, splits, v4_matrix.families)
    for split in splits:
        assert not set(v4_matrix.families[split.train]) & set(v4_matrix.families[split.validation])
        assert set(v4_matrix.y[split.validation]) == {0, 1}  # both labels are held out in every fold
    covered = np.sort(np.concatenate([split.validation for split in splits]))
    assert np.array_equal(covered, np.arange(len(v4_matrix.y)))


def test_a_split_that_mixes_families_is_detected(v4_matrix):
    family = v4_matrix.families[0]
    leaky = NamedSplit("leaky", np.arange(len(v4_matrix.y)), np.where(v4_matrix.families == family)[0])

    with pytest.raises(ValueError, match="both train and validation"):
        assert_splits_are_grouped(v4_matrix, [leaky], v4_matrix.families)


def test_archetype_holdout_holds_out_one_whole_archetype_each_time(v4_matrix):
    splits = archetype_splits(v4_matrix)

    assert {split.name for split in splits} == {"satellite_only", "news_led", "satellite_news", "multi_satellite"}
    assert_splits_are_grouped(v4_matrix, splits, v4_matrix.archetypes)
    for split in splits:
        assert set(v4_matrix.archetypes[split.validation]) == {split.name}
        assert split.name not in set(v4_matrix.archetypes[split.train])
        assert set(v4_matrix.y[split.validation]) == {0, 1}


def test_leave_one_family_out_holds_out_exactly_one_family(v4_matrix):
    splits = leave_one_family_out_splits(v4_matrix)

    assert len(splits) == 20
    for split in splits:
        assert set(v4_matrix.families[split.validation]) == {split.name}
        assert split.name not in set(v4_matrix.families[split.train])
        assert len(split.train) + len(split.validation) == len(v4_matrix.y)


# --- cross-validated scoring ---


def test_out_of_fold_scores_come_from_models_that_never_saw_the_group(v4_matrix):
    splits = family_splits(v4_matrix, 5)

    oof, _ = cross_validated_scores("logistic_regression", v4_matrix, splits)

    assert np.isfinite(oof).all() and ((oof >= 0) & (oof <= 1)).all()
    split = splits[2]
    manual = build_pipeline_v4("logistic_regression").fit(v4_matrix.X[split.train], v4_matrix.y[split.train])
    assert np.allclose(oof[split.validation], manual.predict_proba(v4_matrix.X[split.validation])[:, 1])


def test_cross_validated_scores_are_deterministic(v4_matrix):
    splits = family_splits(v4_matrix, 5)

    first, _ = cross_validated_scores("hist_gradient_boosting", v4_matrix, splits)
    second, _ = cross_validated_scores("hist_gradient_boosting", v4_matrix, splits)

    assert np.array_equal(first, second)


def test_scores_must_cover_every_row(v4_matrix):
    partial = [NamedSplit("only_one", np.arange(1000, 9000), np.arange(0, 1000))]

    with pytest.raises(ValueError, match="cover every row"):
        cross_validated_scores("logistic_regression", v4_matrix, partial)


def test_per_split_metrics_returns_one_entry_per_split(v4_matrix):
    splits = family_splits(v4_matrix, 5)
    oof, _ = cross_validated_scores("logistic_regression", v4_matrix, splits)

    folds = per_split_metrics(v4_matrix.y, oof, splits)

    assert [fold["split"] for fold in folds] == [split.name for split in splits]
    assert all(fold["rows"] == len(split.validation) for fold, split in zip(folds, splits))


# --- rule baseline (never a label) ---


def test_the_regenerated_samples_match_the_frozen_csv_exactly(v4_matrix, v4_samples):
    verify_samples_match_matrix(v4_samples, v4_matrix)


def test_a_sample_that_differs_from_the_csv_is_rejected(v4_matrix, v4_samples):
    swapped = list(v4_samples)
    swapped[10] = swapped[11]  # a different sample in row 10's place
    other_context = list(v4_samples)
    other_context[20] = replace(other_context[20], context=v4_samples[21].context)  # same evidence, another Fire Danger

    with pytest.raises(ValueError, match="does not match"):
        verify_samples_match_matrix(tuple(swapped), v4_matrix)
    if v4_samples[20].context != v4_samples[21].context:
        with pytest.raises(ValueError, match="does not match"):
            verify_samples_match_matrix(tuple(other_context), v4_matrix)
    with pytest.raises(ValueError, match="count"):
        verify_samples_match_matrix(v4_samples[:-1], v4_matrix)


def test_rule_baseline_outputs_are_the_calculators_decisions(v4_samples):
    from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator

    outputs = rule_baseline_outputs(v4_samples[:200])
    calculator = FireDetectionCalculator()

    for index, sample in enumerate(v4_samples[:200]):
        decision = calculator.evaluate(sample.evidence)
        assert outputs.predicted_fire[index] == (decision.status is not FireDetectionStatus.NO_EVENT)
        assert outputs.predicted_confirmed[index] == (decision.status is FireDetectionStatus.CONFIRMED)
        assert outputs.confidence[index] == decision.confidence


def test_rule_outputs_disagree_with_the_ground_truth_labels(v4_matrix, v4_samples):
    outputs = rule_baseline_outputs(v4_samples)

    assert 0.4 < np.mean(outputs.predicted_fire == v4_matrix.y) < 0.75  # the rule is far from the labels


# --- subgroups ---


def test_hard_negative_and_weak_positive_masks_use_the_validation_definitions(v4_matrix):
    hard, weak = hard_negative_mask(v4_matrix), weak_positive_mask(v4_matrix)

    assert hard.sum() == sum(is_hard_negative(row) for row in v4_matrix.rows)
    assert weak.sum() == sum(is_weak_positive(row) for row in v4_matrix.rows)
    assert not hard[v4_matrix.y == 1].any() and not weak[v4_matrix.y == 0].any()


def test_subgroup_performance_at_an_all_fire_operating_point(v4_matrix):
    report = subgroup_performance(v4_matrix, np.ones(len(v4_matrix.y)))

    assert report["hard_negative_false_positive_rate"] == 1.0
    assert report["weak_positive_recall"] == 1.0
    assert len(report["per_family"]) == 20
    assert all(entry["predicted_fire_rate"] == 1.0 for entry in report["per_family"].values())
    assert {entry["label"] for entry in report["per_family"].values()} == {0, 1}


def test_subgroup_performance_at_a_never_fire_operating_point(v4_matrix):
    report = subgroup_performance(v4_matrix, np.zeros(len(v4_matrix.y)))

    assert report["hard_negative_false_positive_rate"] == 0.0 and report["weak_positive_recall"] == 0.0


# --- thresholds ---


def test_threshold_table_covers_the_requested_thresholds_and_is_monotone():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    scores = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)

    table = threshold_table(y, scores)

    assert [row["threshold"] for row in table] == list(CANDIDATE_THRESHOLDS) == [0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80]
    recalls = [row["recall"] for row in table]
    assert recalls == sorted(recalls, reverse=True)
    assert table == threshold_table(y, scores)  # deterministic
    for row in table:
        assert row["true_positive"] + row["false_negative"] == int(y.sum())
        assert row["false_positive_rate"] == pytest.approx(row["false_positive"] / int((y == 0).sum()))


def test_threshold_recommendation_on_separable_scores():
    y = np.array([0] * 100 + [1] * 100)
    scores = np.concatenate([np.linspace(0.05, 0.45, 100), np.linspace(0.55, 0.95, 100)])

    recommendation = recommend_thresholds(y, scores)

    assert recommendation["suspect_threshold_candidate"] is not None
    assert recommendation["confirm_threshold_candidate"] is not None
    assert recommendation["confirm_threshold_candidate"] > recommendation["suspect_threshold_candidate"]
    assert recommendation["thresholds_are_ordered"] is True
    assert recommendation["suspect_operating_point"]["overall"]["recall"] >= 0.85
    assert recommendation["confirm_operating_point"]["overall"]["precision"] >= 0.90
    assert recommendation == recommend_thresholds(y, scores)  # deterministic


def test_confirm_threshold_is_none_when_the_precision_target_is_unattainable():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    scores = rng.random(2000)  # uninformative

    recommendation = recommend_thresholds(y, scores)

    assert recommendation["confirm_threshold_candidate"] is None
    assert recommendation["confirm_operating_point"] is None
    assert recommendation["thresholds_are_ordered"] is False
    assert recommendation["highest_precision_operating_point"] is not None  # still reported honestly


def test_recommendation_reports_per_fold_stability():
    y = np.array([0, 1] * 100)
    scores = np.where(y == 1, 0.8, 0.2).astype(float)
    splits = [NamedSplit("a", np.arange(100, 200), np.arange(0, 100)), NamedSplit("b", np.arange(0, 100), np.arange(100, 200))]

    operating_point = recommend_thresholds(y, scores, splits)["suspect_operating_point"]

    assert len(operating_point["per_fold_recall"]) == 2 and operating_point["min_fold_recall"] == 1.0


# --- calibration ---


def test_calibration_report_for_perfectly_calibrated_scores():
    rng = np.random.default_rng(3)
    scores = rng.random(20000)
    y = (rng.random(20000) < scores).astype(int)

    report = calibration_report(y, scores)

    assert report["expected_calibration_error"] < 0.02
    assert sum(bin_["count"] for bin_ in report["bins"]) == 20000
    assert len(report["bins"]) == 10


def test_calibration_report_for_a_constant_predictor():
    y = np.array([0, 1] * 50)

    report = calibration_report(y, np.full(100, 0.5))

    assert report["brier_score"] == pytest.approx(0.25)
    assert report["brier_score_constant_prevalence_baseline"] == pytest.approx(0.25)
    assert report["brier_skill_score"] == pytest.approx(0.0)


def test_calibration_report_detects_overconfidence():
    y = np.array([1] * 60 + [0] * 40)
    scores = np.full(100, 0.95)

    assert calibration_report(y, scores)["expected_calibration_error"] == pytest.approx(0.35)


# --- permutation importance ---


def test_permutation_importance_blocks_fire_danger_and_shares_sum_to_one(v4_matrix):
    splits = family_splits(v4_matrix, 5)
    _, fitted = cross_validated_scores("logistic_regression", v4_matrix, splits, keep_models=True)

    result = grouped_permutation_importance(fitted, v4_matrix, splits, n_repeats=1)["features"]

    assert "fire_danger_block" in result
    assert not {"fire_danger_available", "fire_danger_score", "fire_danger_age_minutes"} & set(result)
    assert len(result) == 17  # 16 evidence features + 1 Fire Danger block
    assert sum(entry["share_of_total_positive_drop"] for entry in result.values()) == pytest.approx(1.0)
    summary = importance_shortcut_summary({"features": result})
    assert summary["top_feature"] in result and 0 < summary["top_feature_share"] <= 1


def test_shortcut_summary_flags_geometry_share():
    features = {
        "a": {"share_of_total_positive_drop": 0.5},
        "max_pairwise_distance_km": {"share_of_total_positive_drop": 0.3},
        "time_span_minutes": {"share_of_total_positive_drop": 0.2},
    }

    summary = importance_shortcut_summary({"features": features})

    assert summary["top_feature"] == "a" and summary["geometry_time_share"] == pytest.approx(0.5)


# --- gate and selection ---

RULE = {"grouped_mean_f1": 0.67, "overall_precision": 0.51}


def passing_candidate(**overrides):
    candidate = {
        "grouped_mean_roc_auc": 0.85,
        "grouped_min_fold_roc_auc": 0.75,
        "grouped_mean_f1": 0.75,
        "suspect_operating_point": {"precision": 0.65, "recall": 0.88},
        "hard_negative_fpr_at_suspect": 0.40,
        "weak_positive_recall_at_suspect": 0.70,
        "archetype_mean_roc_auc": 0.80,
        "archetype_worst_roc_auc": 0.70,
        "calibration_ece": 0.05,
        "top_feature_share": 0.30,
        "geometry_time_share": 0.10,
    }
    candidate.update(overrides)
    return candidate


def test_a_strong_candidate_passes_the_gate():
    gate = evaluate_gate(passing_candidate(), RULE)

    assert gate["passed"] is True and gate["failed"] == []
    assert len(gate["criteria"]) == 12


@pytest.mark.parametrize(
    "override, failed_criterion",
    [
        ({"grouped_mean_roc_auc": 0.79}, "grouped_mean_roc_auc"),
        ({"grouped_min_fold_roc_auc": 0.60}, "grouped_min_fold_roc_auc"),
        ({"grouped_mean_f1": 0.68}, "grouped_mean_f1_minus_rule"),
        ({"suspect_operating_point": {"precision": 0.65, "recall": 0.80}}, "suspect_recall"),
        ({"suspect_operating_point": {"precision": 0.55, "recall": 0.88}}, "suspect_precision_minus_rule"),
        ({"suspect_operating_point": None}, "suspect_recall"),
        ({"hard_negative_fpr_at_suspect": 0.70}, "hard_negative_fpr_at_suspect"),
        ({"weak_positive_recall_at_suspect": 0.50}, "weak_positive_recall_at_suspect"),
        ({"archetype_mean_roc_auc": 0.70}, "archetype_mean_roc_auc"),
        ({"archetype_worst_roc_auc": 0.60}, "archetype_worst_roc_auc"),
        ({"calibration_ece": 0.15}, "calibration_ece"),
        ({"top_feature_share": 0.55}, "max_single_feature_importance_share"),
        ({"geometry_time_share": 0.40}, "geometry_time_importance_share"),
    ],
)
def test_each_gate_criterion_can_fail_independently(override, failed_criterion):
    gate = evaluate_gate(passing_candidate(**override), RULE)

    assert gate["passed"] is False
    assert failed_criterion in gate["failed"]


def test_the_gate_thresholds_are_the_a_priori_values():
    assert GATE_V4["grouped_mean_roc_auc_min"] == 0.80
    assert GATE_V4["suspect_recall_min"] == 0.85
    assert GATE_V4["calibration_ece_max"] == 0.10


def candidate_summary(key, auc, f1, passed=True):
    return {"model_key": key, "gate_passed": passed, "grouped_mean_roc_auc": auc, "grouped_mean_f1": f1}


def test_no_model_is_selected_when_none_passes_the_gate():
    selection = select_model([candidate_summary("logistic_regression", 0.74, 0.64, False), candidate_summary("random_forest", 0.70, 0.60, False)])

    assert selection["selected_model"] is None
    assert "No model passed" in selection["reason"] and selection["passing_models"] == []


def test_a_near_tie_goes_to_the_simplest_model():
    selection = select_model([
        candidate_summary("logistic_regression", 0.84, 0.74),
        candidate_summary("random_forest", 0.845, 0.75),
        candidate_summary("hist_gradient_boosting", 0.85, 0.75),
    ])

    assert selection["selected_model"] == "logistic_regression"
    assert set(selection["tied_models"]) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}


def test_a_clearly_better_complex_model_beats_a_simpler_one():
    selection = select_model([candidate_summary("logistic_regression", 0.81, 0.70), candidate_summary("random_forest", 0.88, 0.78)])

    assert selection["selected_model"] == "random_forest"


def test_models_failing_the_gate_are_never_selected_even_if_they_score_best():
    selection = select_model([candidate_summary("logistic_regression", 0.82, 0.71), candidate_summary("random_forest", 0.95, 0.90, passed=False)])

    assert selection["selected_model"] == "logistic_regression"


# --- serialisation ---


def test_to_builtin_converts_numpy_and_nan():
    converted = to_builtin({"a": np.float64(0.5), "b": np.int64(3), "c": np.array([1.0, np.nan]), "d": (np.bool_(True),), "e": math.nan})

    assert converted == {"a": 0.5, "b": 3, "c": [1.0, None], "d": [True], "e": None}
    assert type(converted["a"]) is float and type(converted["b"]) is int

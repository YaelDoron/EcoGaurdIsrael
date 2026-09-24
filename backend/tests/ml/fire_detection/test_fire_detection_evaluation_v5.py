"""Task 7: the V5 evaluation library - grouping guarantees, metrics, operating points, paired / sparse analysis,
rule baseline, ablation statistics, audits, gate and selection."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.ml.fire_detection.fire_detection_evaluation_v5 import (
    NamedSplit,
    assert_folds_represent_regimes_and_labels,
    assert_pairs_never_split,
    assert_splits_are_grouped,
    binary_metrics,
    calibration_report,
    clustered_bootstrap_auc_difference,
    confirmed_probability_analysis,
    corroboration_analysis,
    environment_splits,
    evaluate_gate,
    grouped_permutation_importance,
    hard_negative_mask,
    make_fit_predict,
    missingness_association,
    missingness_permutation_audit,
    no_fire_subtype_splits,
    non_sparse_mask,
    out_of_fold_scores,
    paired_analysis,
    regime_splits,
    rule_baseline_outputs,
    rule_baseline_report,
    select_model,
    slice_metrics_at_matched_recall,
    sparse_evidence_report,
    summarize_folds,
    suspected_threshold,
    to_builtin,
    verify_samples_match_matrix,
)
from src.ml.fire_detection.fire_detection_model_config_v5 import CANDIDATE_GRID_V5, FEATURE_SETS_V5, GATE_V5
from src.ml.fire_detection.fire_detection_model_v5 import build_pipeline_v5
from src.ml.fire_detection.fire_detection_training_data_generator_v5 import FireDetectionTrainingDataGeneratorV5


# --- grouped environment folds (the primary evaluation) ---


def test_environments_never_cross_folds_and_every_row_is_validated_once(v5_matrix):
    splits = environment_splits(v5_matrix)
    assert len(splits) == 5
    assert_splits_are_grouped(splits, v5_matrix.environments)
    for split in splits:
        assert not set(v5_matrix.environments[split.train]) & set(v5_matrix.environments[split.validation])
    assert sum(len(s.validation) for s in splits) == 10000


def test_pair_members_never_cross_folds(v5_matrix):
    assert_pairs_never_split(v5_matrix, environment_splits(v5_matrix))


def test_folds_keep_every_regime_and_both_labels(v5_matrix):
    splits = environment_splits(v5_matrix)
    assert_folds_represent_regimes_and_labels(v5_matrix, splits)
    sizes = [len(s.validation) for s in splits]
    assert max(sizes) / min(sizes) < 1.1


def test_the_grouping_assertions_catch_a_leaky_split(v5_matrix):
    splits = environment_splits(v5_matrix)
    leaky = NamedSplit("leaky", splits[0].train, np.concatenate([splits[0].validation, splits[0].train[:5]]))
    with pytest.raises(ValueError, match="both training and validation"):
        assert_splits_are_grouped([leaky], v5_matrix.environments, partition=False)
    row = int(np.flatnonzero(v5_matrix.pair_ids >= 0)[0])
    partner = int(np.flatnonzero((v5_matrix.pair_ids == v5_matrix.pair_ids[row]) & (np.arange(10000) != row))[0])
    broken = NamedSplit("broken", np.array([i for i in range(10000) if i != row]), np.array([row]))
    assert partner in broken.train
    with pytest.raises(ValueError, match="split across"):
        assert_pairs_never_split(v5_matrix, [broken])
    with pytest.raises(ValueError, match="lacks a label"):
        assert_folds_represent_regimes_and_labels(v5_matrix, [NamedSplit("thin", splits[0].train, splits[0].validation[:5])])


def test_leave_one_regime_out_and_subtype_splits(v5_matrix):
    regimes = regime_splits(v5_matrix)
    assert len(regimes) == 6
    for split in regimes:
        assert set(v5_matrix.regimes[split.validation]) == {split.name}
        assert split.name not in set(v5_matrix.regimes[split.train])
    subtypes = no_fire_subtype_splits(v5_matrix)
    assert [s.name for s in subtypes] == ["controlled_or_agricultural_burn", "false_or_rumour_report", "industrial_heat_source", "sensor_noise"]
    for split in subtypes:
        assert (v5_matrix.y[split.validation] == 0).all() and set(v5_matrix.subtypes[split.validation]) == {split.name}
        assert split.name not in set(v5_matrix.subtypes[split.train])


def test_out_of_fold_scores_come_from_a_model_that_never_saw_the_rows(v5_matrix):
    calls = []
    splits = environment_splits(v5_matrix)

    def fit_predict(train, predict):
        calls.append((set(v5_matrix.environments[train]), set(v5_matrix.environments[predict])))
        return np.full(len(predict), 0.5), None

    scores, fitted = out_of_fold_scores(fit_predict, 10000, splits)
    assert len(calls) == 5 and all(not train & held for train, held in calls)
    assert (scores == 0.5).all() and len(fitted) == 5


def test_calibrated_variants_only_calibrate_on_training_rows(v5_matrix):
    """The predicted rows are never used to fit the classifier or its calibrator."""
    params = CANDIDATE_GRID_V5["logistic_regression"][1]
    columns = v5_matrix.X
    fit_predict = make_fit_predict(lambda: build_pipeline_v5("logistic_regression", params), columns, v5_matrix.y, v5_matrix.environments, "isotonic")
    split = environment_splits(v5_matrix)[0]
    scores, model = fit_predict(split.train, split.validation)
    assert len(scores) == len(split.validation) and np.isfinite(scores).all()
    assert len(model.calibrated_classifiers_) == 3  # inner grouped folds of the TRAINING rows
    with pytest.raises(ValueError):
        make_fit_predict(lambda: None, columns, v5_matrix.y, v5_matrix.environments, "platt")


# --- metrics ---


def test_binary_metrics_known_values():
    y = np.array([1, 1, 0, 0])
    m = binary_metrics(y, np.array([0.9, 0.4, 0.6, 0.1]), 0.5)
    assert (m["true_positive"], m["false_positive"], m["true_negative"], m["false_negative"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5 and m["false_positive_rate"] == 0.5
    assert m["roc_auc"] == pytest.approx(0.75) and m["brier"] == pytest.approx(np.mean([0.01, 0.36, 0.36, 0.01]))
    assert binary_metrics(np.array([1, 1]), np.array([0.2, 0.8]))["roc_auc"] is None  # one class: AUC undefined, not faked


def test_summarize_folds_reports_mean_std_min_max():
    folds = [{"roc_auc": 0.6}, {"roc_auc": 0.8}, {"roc_auc": 0.7}]
    s = summarize_folds(folds, ("roc_auc",))["roc_auc"]
    assert s["mean"] == pytest.approx(0.7) and s["min"] == 0.6 and s["max"] == 0.8 and s["std"] == pytest.approx(np.std([0.6, 0.8, 0.7]))


def test_calibration_report_perfect_and_overconfident():
    y = np.array([0, 0, 1, 1] * 25)
    perfect = calibration_report(y, y.astype(float) * 0 + np.where(y == 1, 1.0, 0.0))
    assert perfect["expected_calibration_error"] == 0.0 and perfect["brier_score"] == 0.0
    honest = calibration_report(y, np.full(len(y), 0.5))
    assert honest["expected_calibration_error"] == pytest.approx(0.0) and honest["brier_score"] == pytest.approx(0.25)
    overconfident = calibration_report(y, np.where(np.arange(len(y)) % 2 == 0, 0.95, 0.05))
    assert overconfident["expected_calibration_error"] > 0.3
    assert sum(b["count"] for b in honest["bins"]) == len(y)


# --- operating points ---


def test_suspected_threshold_is_the_highest_threshold_meeting_the_recall_target():
    y = np.array([1] * 10 + [0] * 10)
    scores = np.concatenate([np.linspace(0.30, 0.95, 10), np.linspace(0.05, 0.60, 10)])
    eligible = np.ones(20, dtype=bool)
    threshold = suspected_threshold(y, scores, eligible, 0.8)
    positives = scores[:10]
    assert np.mean(positives >= threshold) >= 0.8
    assert np.mean(positives >= threshold + 0.01) < 0.8  # no higher grid threshold still reaches the target
    assert suspected_threshold(y, np.zeros(20), eligible, 0.8) is None  # unattainable -> None, never invented


def test_suspected_threshold_ignores_positives_that_are_not_eligible():
    y = np.array([1] * 10)
    scores = np.array([0.9] * 8 + [0.05, 0.05])
    only_first_eight = np.array([True] * 8 + [False, False])
    assert suspected_threshold(y, scores, only_first_eight) == pytest.approx(0.9)
    assert suspected_threshold(y, scores, np.ones(10, dtype=bool)) == pytest.approx(0.9)  # 8/10 = 0.8 still meets it
    assert suspected_threshold(y, scores, np.ones(10, dtype=bool), 0.9) == pytest.approx(0.05)


def test_confirmed_analysis_reports_attainable_only_when_precision_and_recall_allow(v5_matrix):
    perfect = np.where(v5_matrix.y == 1, 0.95, 0.05)
    result = confirmed_probability_analysis(v5_matrix, perfect)
    assert result["precision_target_attainable"] and result["candidate"]["precision"] == 1.0
    uninformative = np.full(len(v5_matrix.y), 0.5)
    hopeless = confirmed_probability_analysis(v5_matrix, uninformative)
    assert not hopeless["precision_target_attainable"] and hopeless["candidate"] is None  # never forced


def test_hard_negatives_are_no_fire_rows_with_fire_looking_evidence(v5_matrix):
    hard = hard_negative_mask(v5_matrix)
    assert (v5_matrix.y[hard] == 0).all() and hard.sum() > 1000
    strong_or_bright = (v5_matrix.feature("satellite_nominal_count") + v5_matrix.feature("satellite_high_count") > 0) | (
        v5_matrix.feature("news_moderate_count") + v5_matrix.feature("news_strong_count") > 0
    )
    assert (strong_or_bright[hard]).all()
    assert non_sparse_mask(v5_matrix).sum() == 9000


# --- paired and sparse analyses ---


def test_paired_ranking_accuracy_and_margin_on_a_toy_case(v5_matrix):
    scores = np.where(v5_matrix.y == 1, 0.7, 0.3)
    result = paired_analysis(v5_matrix, scores)
    assert result["overall"]["pairs"] == 2500 and result["overall"]["pairwise_ranking_accuracy"] == 1.0
    assert result["overall"]["mean_probability_margin"] == pytest.approx(0.4)
    assert len(result["by_pair_type"]) == 6
    assert paired_analysis(v5_matrix, np.full(10000, 0.5))["overall"]["pairwise_ranking_accuracy"] == 0.5  # ties count 1/2
    assert paired_analysis(v5_matrix, np.where(v5_matrix.y == 1, 0.2, 0.8))["overall"]["pairwise_ranking_accuracy"] == 0.0


def test_a_broken_pair_is_rejected(v5_matrix):
    broken = v5_matrix.select(np.array([i for i in range(10000)][:5000]))
    pair_id = int(broken.pair_ids[broken.pair_ids >= 0][0])
    keep = np.array([i for i in range(len(broken.y)) if not (broken.pair_ids[i] == pair_id and broken.y[i] == 0)])
    with pytest.raises(ValueError, match="exactly one fire and one no-fire"):
        paired_analysis(broken.select(keep), np.full(len(keep), 0.5))


def test_sparse_report_measures_uncertainty_not_accuracy(v5_matrix):
    sparse = v5_matrix.regimes == "sparse_early_evidence"
    honest = np.full(10000, 0.5)
    r = sparse_evidence_report(v5_matrix, honest)
    assert r["brier_score"] == pytest.approx(0.25, abs=0.001) and r["fraction_uncertain_0_35_to_0_65"] == 1.0
    assert r["fraction_p_above_0_90"] == 0.0 and r["fraction_p_below_0_10"] == 0.0 and r["observed_fire_fraction_when_p_above_0_90"] is None
    overconfident = np.where(v5_matrix.y == 1, 0.97, 0.03)
    guessing = np.where(np.arange(10000) % 2 == 0, 0.97, 0.03)
    confident = sparse_evidence_report(v5_matrix, guessing)
    assert confident["fraction_p_above_0_90"] == pytest.approx(0.5, abs=0.02)
    assert confident["brier_score"] > 0.4 and confident["expected_calibration_error"] > 0.3  # extreme guesses are punished
    assert confident["observed_fire_fraction_when_p_above_0_90"] == pytest.approx(0.5, abs=0.1)
    assert sum(r["probability_histogram_10_bins"]) == int(sparse.sum())
    assert sparse_evidence_report(v5_matrix, overconfident)["brier_score"] < 0.01  # a real oracle would be rewarded


# --- rule baseline ---


def test_rule_baseline_runs_the_existing_calculator_on_the_current_candidate(v5_matrix):
    samples = FireDetectionTrainingDataGeneratorV5(42).generate(10000)[:600]
    outputs = rule_baseline_outputs(samples)
    assert set(np.unique(outputs.predicted_fire)) <= {0.0, 1.0}
    assert (outputs.predicted_confirmed <= outputs.predicted_fire).all()  # CONFIRMED implies fire
    assert outputs.predicted_fire.mean() > 0.3  # the rules flag a large share of these evidence-rich rows


def test_rule_baseline_rows_must_match_the_frozen_csv(v5_matrix):
    samples = FireDetectionTrainingDataGeneratorV5(42).generate(10000)
    verify_samples_match_matrix(samples, v5_matrix)  # regenerated == frozen
    with pytest.raises(ValueError, match="does not match"):
        verify_samples_match_matrix((samples[1], samples[0], *samples[2:]), v5_matrix)
    with pytest.raises(ValueError, match="count"):
        verify_samples_match_matrix(samples[:-1], v5_matrix)


def test_rule_report_has_overall_non_sparse_and_per_regime_slices(v5_matrix):
    samples = FireDetectionTrainingDataGeneratorV5(42).generate(10000)
    report = rule_baseline_report(v5_matrix, rule_baseline_outputs(samples))
    fire = report["suspected_or_confirmed_is_fire"]
    assert set(fire["per_regime"]) == set(v5_matrix.regimes.tolist())
    assert fire["all_rows"]["rows"] == 10000 and fire["non_sparse_rows"]["rows"] == 9000
    for key in ("precision", "recall", "f1", "false_positives", "false_negatives", "hard_negative_fpr"):
        assert key in fire["non_sparse_rows"]
    # the rules rarely reject a hard negative: that is exactly the behaviour a V5 model must improve on
    assert fire["non_sparse_rows"]["hard_negative_fpr"] > 0.9


# --- ablation statistics ---


def test_history_feature_sets_really_drop_the_history_columns(v5_matrix):
    names = FEATURE_SETS_V5["without_history"]
    columns = v5_matrix.columns(names)
    assert columns.shape == (10000, 20)
    pipeline = build_pipeline_v5("logistic_regression", CANDIDATE_GRID_V5["logistic_regression"][1], names).fit(columns[:3000], v5_matrix.y[:3000])
    altered = v5_matrix.X.copy()
    history_columns = [v5_matrix.feature_names.index(n) for n in v5_matrix.feature_names if n not in names]
    altered[:, history_columns] = 12345.0
    assert np.array_equal(pipeline.predict_proba(v5_matrix.columns(names)[3000:3200]), pipeline.predict_proba(altered[3000:3200][:, [v5_matrix.feature_names.index(n) for n in names]]))


def test_bootstrap_auc_difference_detects_a_real_gain_and_not_a_null_one(v5_matrix):
    rng = np.random.default_rng(0)
    y = v5_matrix.y
    good = y + rng.normal(0, 0.6, len(y))
    weak = y * 0.1 + rng.normal(0, 0.6, len(y))
    better = clustered_bootstrap_auc_difference(y, good, weak, v5_matrix.environments, repeats=60)
    assert better["auc_difference"] > 0.1 and better["ci_excludes_zero"]
    same = clustered_bootstrap_auc_difference(y, good, good, v5_matrix.environments, repeats=60)
    assert same["auc_difference"] == 0.0 and not same["ci_excludes_zero"]


def test_matched_recall_slice_metrics():
    y = np.array([1] * 10 + [0] * 10)
    scores = np.concatenate([np.linspace(0.5, 0.95, 10), np.linspace(0.05, 0.6, 10)])
    result = slice_metrics_at_matched_recall(y, scores, 0.8)
    assert result["recall"] >= 0.8 and 0 <= result["false_positive_rate"] <= 1
    assert slice_metrics_at_matched_recall(y, np.zeros(20), 0.8)["threshold"] is None


# --- corroboration guardrail candidates ---


def test_corroboration_precision_is_computed_from_the_flagged_rows(v5_matrix):
    scores = np.where(v5_matrix.y == 1, 0.9, 0.1)
    result = corroboration_analysis(v5_matrix, scores, [0.5])
    table = result["with_probability"]["0.50"]
    assert table["probability_only"]["precision"] == 1.0 and table["probability_only"]["false_positives"] == 0
    assert table["and_multiple_passes"]["flagged_rows"] == int(((v5_matrix.feature("satellite_pass_count") >= 2) & (v5_matrix.y == 1)).sum())
    assert set(result["guardrail_alone"]) >= {"and_multiple_passes", "and_satellite_and_news", "and_strong_news", "and_multi_pixel"}
    # a flat probability adds nothing beyond the guardrail itself
    flat = corroboration_analysis(v5_matrix, np.full(10000, 0.9), [0.5])["with_probability"]["0.50"]["and_strong_news"]
    assert flat["precision"] == pytest.approx(result["guardrail_alone"]["and_strong_news"]["precision"])


# --- importance and audits ---


def _single_feature_model(matrix, feature):
    column = matrix.feature_names.index(feature)
    model = Pipeline([("classifier", HistGradientBoostingClassifier(max_iter=50, random_state=0))])
    model.fit(matrix.X[:, [column]], matrix.y)
    return model, column


def test_permutation_importance_concentrates_on_the_only_informative_feature():
    rng = np.random.default_rng(1)
    n = 3000
    signal = rng.normal(size=n)
    y = (signal + rng.normal(0, 0.3, n) > 0).astype(int)
    noise = rng.normal(size=(n, 2))
    X = np.column_stack([signal, noise])
    names = ("satellite_frp_max", "news_strong_count", "satellite_night_fraction")
    model = Pipeline([("classifier", HistGradientBoostingClassifier(max_iter=50, random_state=0))]).fit(X, y)
    splits = [NamedSplit("all", np.arange(n), np.arange(n))]
    result = grouped_permutation_importance([model], X, y, splits, names, n_repeats=3, blocks={"block": ("news_strong_count",)})
    assert result["top_feature"] == "satellite_frp_max" and result["top_feature_share"] > 0.9
    assert result["blocks"]["block"]["mean_auc_drop"] == pytest.approx(0.0, abs=0.02)


def test_missingness_audit_separates_value_use_from_missingness_use():
    rng = np.random.default_rng(2)
    n = 4000
    missing = rng.random(n) < 0.5
    y = missing.astype(int)  # the LABEL is exactly whether the feature is missing
    values = rng.normal(size=n)
    column = np.where(missing, np.nan, values)
    X = np.column_stack([column])
    names = ("satellite_centroid_stability_km",)
    model = Pipeline([("classifier", HistGradientBoostingClassifier(max_iter=30, random_state=0))]).fit(X, y)
    audit = missingness_permutation_audit([model], X, y, [NamedSplit("all", np.arange(n), np.arange(n))], names)
    entry = audit["satellite_centroid_stability_km"]
    assert entry["mean_auc_drop_shuffling_missingness"] > 0.3  # the model lives off missingness ...
    assert abs(entry["mean_auc_drop_shuffling_values"]) < 0.05  # ... and ignores the observed values


def test_missingness_association_on_the_frozen_dataset_finds_no_label_shortcut(v5_matrix):
    association = missingness_association(v5_matrix)
    for name, entry in association.items():
        assert entry["missingness_indicator_auc"] is None or abs(entry["missingness_indicator_auc"] - 0.5) < 0.05, name
        for regime, within in entry["within_regime"].items():
            assert abs(within["missingness_indicator_auc"] - 0.5) < 0.1, (name, regime)
    assert association["satellite_centroid_stability_km"]["missing_rate"] == pytest.approx(0.74, abs=0.01)


# --- gate and selection ---


def _passing_candidate(**overrides):
    candidate = {
        "model_key": "logistic_regression",
        "grouped_mean_roc_auc": 0.80, "grouped_worst_fold_roc_auc": 0.72, "suspected_threshold": 0.4,
        "non_sparse_positive_recall": 0.82, "hard_negative_fpr": 0.30, "persistent_roc_auc": 0.75,
        "brier_score": 0.20, "expected_calibration_error": 0.03, "top_feature_share": 0.25,
        "non_sparse_slice_at_suspected": {"hard_negative_fpr": 0.30, "recall": 0.82, "f1": 0.70},
    }
    candidate.update(overrides)
    return candidate


RULE = {"non_sparse_rows": {"hard_negative_fpr": 0.95, "recall": 0.90, "f1": 0.65}}


def test_a_candidate_meeting_every_criterion_passes():
    result = evaluate_gate(_passing_candidate(), RULE)
    assert result["passed"] and result["failed_criteria"] == []
    assert len(result["criteria"]) == 11


@pytest.mark.parametrize(
    "override,failed",
    [
        ({"grouped_mean_roc_auc": 0.74}, "grouped_environment_mean_roc_auc"),
        ({"grouped_worst_fold_roc_auc": 0.64}, "grouped_environment_worst_fold_roc_auc"),
        ({"hard_negative_fpr": 0.46, "non_sparse_slice_at_suspected": {"hard_negative_fpr": 0.30, "recall": 0.82, "f1": 0.70}}, "hard_negative_fpr"),
        ({"non_sparse_positive_recall": 0.79}, "non_sparse_positive_recall"),
        ({"persistent_roc_auc": 0.69}, "persistent_thermal_roc_auc"),
        ({"brier_score": 0.25}, "brier_score"),
        ({"expected_calibration_error": 0.11}, "expected_calibration_error"),
        ({"top_feature_share": 0.43}, "top_feature_importance_share"),
        ({"non_sparse_slice_at_suspected": {"hard_negative_fpr": 0.80, "recall": 0.82, "f1": 0.70}}, "beats_rules_hard_negative_fpr"),
        ({"non_sparse_slice_at_suspected": {"hard_negative_fpr": 0.30, "recall": 0.60, "f1": 0.70}}, "beats_rules_recall_not_collapsed"),
        ({"non_sparse_slice_at_suspected": {"hard_negative_fpr": 0.30, "recall": 0.82, "f1": 0.60}}, "beats_rules_f1_not_worse"),
    ],
)
def test_every_criterion_is_critical(override, failed):
    result = evaluate_gate(_passing_candidate(**override), RULE)
    assert not result["passed"] and failed in result["failed_criteria"]


def test_the_top_feature_share_tolerance_is_narrow():
    assert evaluate_gate(_passing_candidate(top_feature_share=0.42), RULE)["passed"]  # 0.40 + 0.02 tolerance
    assert not evaluate_gate(_passing_candidate(top_feature_share=0.4201), RULE)["passed"]


def test_uncomputable_criteria_never_pass_silently():
    result = evaluate_gate(_passing_candidate(suspected_threshold=None, non_sparse_positive_recall=None, hard_negative_fpr=None, non_sparse_slice_at_suspected=None), RULE)
    assert not result["passed"]
    assert {"non_sparse_positive_recall", "hard_negative_fpr", "beats_rules_hard_negative_fpr"} <= set(result["failed_criteria"])


def test_a_high_auc_does_not_pass_a_model_that_is_operationally_worse_than_the_rules():
    worse = _passing_candidate(non_sparse_slice_at_suspected={"hard_negative_fpr": 0.95, "recall": 0.60, "f1": 0.40}, hard_negative_fpr=0.44)
    result = evaluate_gate(worse, RULE)
    assert not result["passed"] and "beats_rules_hard_negative_fpr" in result["failed_criteria"]


def _entry(key, auc, passed=True):
    return {"model_key": key, "grouped_mean_roc_auc": auc, "gate": {"passed": passed}}


def test_selection_prefers_the_simpler_model_within_the_margin():
    chosen = select_model([_entry("logistic_regression", 0.79), _entry("random_forest", 0.80), _entry("hist_gradient_boosting", 0.805)])
    assert chosen["selected_model"] == "logistic_regression"
    clear_winner = select_model([_entry("logistic_regression", 0.76), _entry("hist_gradient_boosting", 0.83)])
    assert clear_winner["selected_model"] == "hist_gradient_boosting"  # a large gain is not sacrificed for simplicity


def test_selection_ignores_failing_models_and_returns_none_when_nothing_passes():
    assert select_model([_entry("logistic_regression", 0.9, passed=False), _entry("random_forest", 0.7)])["selected_model"] == "random_forest"
    none = select_model([_entry("logistic_regression", 0.9, passed=False)])
    assert none["selected_model"] is None and none["passing_models"] == [] and "no runtime artifact" in none["reason"]


def test_to_builtin_makes_reports_json_safe():
    import json

    converted = to_builtin({"a": np.float64(1.5), "b": np.array([1, 2]), "c": np.nan, "d": (np.int64(3),), "e": np.bool_(True)})
    assert json.dumps(converted) and converted["c"] is None and converted["d"] == [3]

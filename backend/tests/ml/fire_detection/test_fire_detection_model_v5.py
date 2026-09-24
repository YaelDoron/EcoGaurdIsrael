"""Task 7: frozen-dataset verification, schema validation before training, preprocessing, determinism, config."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer

from src.ml.fire_detection.fire_detection_features_v5 import (
    FIRE_DETECTION_FEATURE_NAMES_V5,
    HISTORY_FEATURE_NAMES_V5,
    NULLABLE_FEATURE_NAMES_V5,
    RETAINED_FEATURE_NAMES_V5,
    TRAINING_DATA_METADATA_COLUMNS_V5,
)
from src.ml.fire_detection.fire_detection_model_config_v5 import (
    ABLATION_LADDER_V5,
    CANDIDATE_GRID_V5,
    EXPECTED_TRAINING_CSV_SHA256_V5,
    FEATURE_SETS_V5,
    GATE_V5,
    MODEL_FAMILIES_V5,
    PERMUTATION_BLOCKS_V5,
)
from src.ml.fire_detection.fire_detection_model_v5 import (
    DEFAULT_TRAINING_CSV_V5,
    EXCLUDED_FROM_MODEL_INPUT_V5,
    assert_feature_schema_v5,
    assert_feature_subset_v5,
    build_pipeline_v5,
    dataset_sha256,
    describe_preprocessing_v5,
    load_training_matrix_v5,
    verify_frozen_dataset_v5,
)


# --- the frozen dataset ---


def test_the_frozen_v5_csv_has_the_expected_sha256():
    assert EXPECTED_TRAINING_CSV_SHA256_V5 == "37e056793fb4dda72f517a20153b21094b284e1579a065140483b8a6e1d78035"
    assert dataset_sha256(DEFAULT_TRAINING_CSV_V5) == EXPECTED_TRAINING_CSV_SHA256_V5
    assert verify_frozen_dataset_v5() == EXPECTED_TRAINING_CSV_SHA256_V5


def test_any_other_file_is_refused_before_training(tmp_path):
    tampered = tmp_path / "training_v5.csv"
    tampered.write_bytes(DEFAULT_TRAINING_CSV_V5.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="not the frozen V5 benchmark"):
        verify_frozen_dataset_v5(tampered)
    with pytest.raises(ValueError, match="not the frozen V5 benchmark"):
        load_training_matrix_v5(tampered)


def test_the_matrix_is_the_canonical_25_features_with_metadata_kept_aside(v5_matrix):
    assert v5_matrix.feature_names == FIRE_DETECTION_FEATURE_NAMES_V5
    assert v5_matrix.X.shape == (10000, 25)
    assert set(np.unique(v5_matrix.y)) == {0, 1} and v5_matrix.y.sum() == 5000
    assert len(set(v5_matrix.environments.tolist())) == 250
    assert v5_matrix.X.dtype == float
    nan_columns = {name for name in FIRE_DETECTION_FEATURE_NAMES_V5 if np.isnan(v5_matrix.feature(name)).any()}
    assert nan_columns == set(NULLABLE_FEATURE_NAMES_V5)  # NaN only where the contract allows it


# --- schema validation happens before every training path ---


def test_the_exact_ordered_schema_is_accepted_and_anything_else_refused():
    assert assert_feature_schema_v5(FIRE_DETECTION_FEATURE_NAMES_V5) == FIRE_DETECTION_FEATURE_NAMES_V5
    with pytest.raises(ValueError, match="schema mismatch"):
        assert_feature_schema_v5(tuple(reversed(FIRE_DETECTION_FEATURE_NAMES_V5)))
    with pytest.raises(ValueError, match="schema mismatch"):
        assert_feature_schema_v5(FIRE_DETECTION_FEATURE_NAMES_V5[:-1])
    for metadata in ("regime", "environment_id", "pair_id", "label", "latent_subtype", "sample_id", "as_of_utc", "seed"):
        with pytest.raises(ValueError):
            assert_feature_schema_v5((*FIRE_DETECTION_FEATURE_NAMES_V5, metadata))


def test_metadata_columns_are_excluded_from_model_input():
    assert set(TRAINING_DATA_METADATA_COLUMNS_V5) | {"label"} == set(EXCLUDED_FROM_MODEL_INPUT_V5)
    assert not set(EXCLUDED_FROM_MODEL_INPUT_V5) & set(FIRE_DETECTION_FEATURE_NAMES_V5)
    for name in ("sample_id", "seed", "environment_id", "regime", "latent_subtype", "pair_id", "label", "as_of_utc"):
        assert name in EXCLUDED_FROM_MODEL_INPUT_V5


def test_feature_subsets_must_be_canonical_and_ordered():
    assert assert_feature_subset_v5(RETAINED_FEATURE_NAMES_V5) == RETAINED_FEATURE_NAMES_V5
    with pytest.raises(ValueError):
        assert_feature_subset_v5(tuple(reversed(RETAINED_FEATURE_NAMES_V5)))
    with pytest.raises(ValueError):
        assert_feature_subset_v5((*RETAINED_FEATURE_NAMES_V5, "regime"))
    with pytest.raises(ValueError):
        assert_feature_subset_v5(())
    with pytest.raises(ValueError):
        assert_feature_subset_v5((RETAINED_FEATURE_NAMES_V5[0],) * 2)


def test_pipelines_refuse_a_bad_feature_list_and_an_off_grid_setting():
    params = CANDIDATE_GRID_V5["logistic_regression"][0]
    with pytest.raises(ValueError):
        build_pipeline_v5("logistic_regression", params, ("satellite_low_count", "regime"))
    with pytest.raises(ValueError, match="predefined"):
        build_pipeline_v5("logistic_regression", {"C": 123.0})
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline_v5("xgboost", {})


# --- the predefined gate and grid (written before evaluation) ---


def test_the_predefined_gate_matches_the_task_specification():
    assert GATE_V5["grouped_environment_mean_roc_auc_min"] == 0.75
    assert GATE_V5["grouped_environment_worst_fold_roc_auc_min"] == 0.65
    assert GATE_V5["hard_negative_fpr_max"] == 0.45
    assert GATE_V5["non_sparse_positive_recall_min"] == 0.80
    assert GATE_V5["persistent_thermal_roc_auc_min"] == 0.70
    assert GATE_V5["brier_score_max_exclusive"] == 0.25
    assert GATE_V5["expected_calibration_error_max"] == 0.10
    assert GATE_V5["top_feature_importance_share_max"] == 0.40
    assert "grouped_environment_mean_roc_auc_min" in GATE_V5 and GATE_V5["grouped_environment_mean_roc_auc_min"] != 0.80  # not V4's gate


def test_the_grid_is_small_predefined_and_uses_only_the_three_allowed_families():
    assert MODEL_FAMILIES_V5 == ("logistic_regression", "random_forest", "hist_gradient_boosting")
    assert {k: len(v) for k, v in CANDIDATE_GRID_V5.items()} == {"logistic_regression": 4, "random_forest": 4, "hist_gradient_boosting": 4}
    assert all("C" in p for p in CANDIDATE_GRID_V5["logistic_regression"])
    assert all(set(p) == {"max_depth", "min_samples_leaf"} for p in CANDIDATE_GRID_V5["random_forest"])
    assert all(set(p) == {"max_depth", "learning_rate"} for p in CANDIDATE_GRID_V5["hist_gradient_boosting"])


def test_feature_sets_are_derived_from_the_canonical_schema():
    assert FEATURE_SETS_V5["full_v5"] == FIRE_DETECTION_FEATURE_NAMES_V5
    assert FEATURE_SETS_V5["retained_14_baseline"] == RETAINED_FEATURE_NAMES_V5 and len(RETAINED_FEATURE_NAMES_V5) == 14
    assert set(FIRE_DETECTION_FEATURE_NAMES_V5) - set(FEATURE_SETS_V5["without_history"]) == set(HISTORY_FEATURE_NAMES_V5)
    assert len(FEATURE_SETS_V5["without_history"]) == 20
    assert ABLATION_LADDER_V5 == ("retained_14_baseline", "without_history", "full_v5")
    for name, names in FEATURE_SETS_V5.items():
        assert_feature_subset_v5(names)
    assert not [n for n in FEATURE_SETS_V5["without_news"] if n.startswith("news_")]
    assert not [n for n in FEATURE_SETS_V5["without_frp_brightness"] if "frp" in n or "brightness" in n]
    assert set(PERMUTATION_BLOCKS_V5["history_block"]) == set(HISTORY_FEATURE_NAMES_V5)


# --- preprocessing ---


def test_logistic_and_forest_impute_only_the_nullable_features_and_hgb_keeps_nan(v5_matrix):
    lr = build_pipeline_v5("logistic_regression", CANDIDATE_GRID_V5["logistic_regression"][1])
    assert isinstance(lr.named_steps["preprocess"], ColumnTransformer)
    assert list(lr.named_steps) == ["preprocess", "scale", "classifier"]
    rf = build_pipeline_v5("random_forest", CANDIDATE_GRID_V5["random_forest"][0])
    assert list(rf.named_steps) == ["preprocess", "classifier"]
    hgb = build_pipeline_v5("hist_gradient_boosting", CANDIDATE_GRID_V5["hist_gradient_boosting"][0])
    assert not isinstance(hgb.named_steps["preprocess"], ColumnTransformer)  # no imputation: native NaN handling
    for family in MODEL_FAMILIES_V5:
        description = describe_preprocessing_v5(family)
        assert description["missing_indicators"] in ("none added", "none")
        assert description["nullable_features"] == list(NULLABLE_FEATURE_NAMES_V5)


@pytest.mark.parametrize("family", MODEL_FAMILIES_V5)
def test_every_pipeline_fits_and_predicts_on_data_with_nan(v5_matrix, family):
    sample = np.arange(0, 2000)
    pipeline = build_pipeline_v5(family, CANDIDATE_GRID_V5[family][0]).fit(v5_matrix.X[sample], v5_matrix.y[sample])
    probabilities = pipeline.predict_proba(v5_matrix.X[2000:2500])[:, 1]
    assert np.isfinite(probabilities).all() and probabilities.min() >= 0 and probabilities.max() <= 1


def test_median_imputation_is_fit_on_the_training_rows_only(v5_matrix):
    pipeline = build_pipeline_v5("logistic_regression", CANDIDATE_GRID_V5["logistic_regression"][1])
    train = np.arange(0, 3000)
    pipeline.fit(v5_matrix.X[train], v5_matrix.y[train])
    imputer = pipeline.named_steps["preprocess"].named_transformers_["impute_nullable"]
    nullable_indices = [FIRE_DETECTION_FEATURE_NAMES_V5.index(n) for n in NULLABLE_FEATURE_NAMES_V5]
    expected = np.nanmedian(v5_matrix.X[train][:, nullable_indices], axis=0)
    assert imputer.statistics_ == pytest.approx(expected)
    held_out = np.nanmedian(v5_matrix.X[3000:][:, nullable_indices], axis=0)
    assert not np.allclose(imputer.statistics_, held_out)  # it did not peek at other rows


def test_a_history_free_row_gets_the_median_not_zero_for_stability(v5_matrix):
    pipeline = build_pipeline_v5("logistic_regression", CANDIDATE_GRID_V5["logistic_regression"][1]).fit(v5_matrix.X[:4000], v5_matrix.y[:4000])
    imputer = pipeline.named_steps["preprocess"].named_transformers_["impute_nullable"]
    stability = NULLABLE_FEATURE_NAMES_V5.index("satellite_centroid_stability_km")
    assert imputer.statistics_[stability] > 0.0  # median of the observed values, never a fake perfect-stability 0


# --- determinism ---


@pytest.mark.parametrize("family", MODEL_FAMILIES_V5)
def test_training_is_deterministic(v5_matrix, family):
    train, test = np.arange(0, 1500), np.arange(1500, 2000)
    first = build_pipeline_v5(family, CANDIDATE_GRID_V5[family][0]).fit(v5_matrix.X[train], v5_matrix.y[train]).predict_proba(v5_matrix.X[test])
    second = build_pipeline_v5(family, CANDIDATE_GRID_V5[family][0]).fit(v5_matrix.X[train], v5_matrix.y[train]).predict_proba(v5_matrix.X[test])
    assert np.array_equal(first, second)

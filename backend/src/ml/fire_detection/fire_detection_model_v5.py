"""V5 training data as arrays, and the sklearn pipelines evaluated in Task 7.

Every training / evaluation path goes through `assert_feature_schema_v5` (exact ordered 25 names from
FIRE_DETECTION_FEATURE_NAMES_V5, no metadata) or `assert_feature_subset_v5` (an ablation subset in canonical
order). The feature list is never re-typed here.

Preprocessing (one sklearn Pipeline = the single artifact; nothing is imputed or scaled outside it):

  logistic_regression      median imputation of the NULLABLE features -> StandardScaler -> LogisticRegression
  random_forest            median imputation of the NULLABLE features -> RandomForestClassifier
  hist_gradient_boosting   NO imputation: sklearn's native NaN handling learns which side missing values take

Median imputation is fit on training folds only. No missing-value indicator columns are added (a missing
`satellite_centroid_stability_km` therefore looks like a median stability to LR / RF - the missingness audit in
the comparison report measures how much that matters).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.ml.fire_detection.fire_detection_dataset_v5 import FireDetectionDatasetRowV5, load_training_dataset_rows_v5
from src.ml.fire_detection.fire_detection_features_v5 import (
    FIRE_DETECTION_FEATURE_NAMES_V5,
    NULLABLE_FEATURE_NAMES_V5,
    TRAINING_DATA_METADATA_COLUMNS_V5,
    forbidden_feature_names,
    validate_feature_names_v5,
)
from src.ml.fire_detection.fire_detection_model_config_v5 import (
    CANDIDATE_GRID_V5,
    EXPECTED_ROWS_V5,
    EXPECTED_TRAINING_CSV_SHA256_V5,
    FIXED_HYPERPARAMETERS_V5,
    MODEL_FAMILIES_V5,
    RANDOM_SEED_V5,
)

DEFAULT_TRAINING_CSV_V5 = Path(__file__).resolve().parents[3] / "data" / "fire_detection" / "training_v5.csv"

# Everything that describes a row rather than its evidence: never model input.
EXCLUDED_FROM_MODEL_INPUT_V5: tuple[str, ...] = TRAINING_DATA_METADATA_COLUMNS_V5 + ("label",)


def dataset_sha256(csv_path: Path) -> str:
    return hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()


def verify_frozen_dataset_v5(csv_path: Path = DEFAULT_TRAINING_CSV_V5, expected_sha256: str = EXPECTED_TRAINING_CSV_SHA256_V5) -> str:
    """Refuse to train / evaluate on anything but the frozen V5 benchmark. Returns the verified hash."""
    actual = dataset_sha256(csv_path)
    if actual != expected_sha256:
        raise ValueError(
            f"{Path(csv_path).name} is not the frozen V5 benchmark: sha256 {actual} != expected {expected_sha256}."
        )
    return actual


def assert_feature_schema_v5(feature_names: Iterable[str]) -> tuple[str, ...]:
    """The exact ordered canonical 25 features, containing no metadata. Raises otherwise."""
    names = validate_feature_names_v5(feature_names)
    leaking = [name for name in names if name in EXCLUDED_FROM_MODEL_INPUT_V5]
    if leaking or forbidden_feature_names(names):
        raise ValueError(f"metadata / leakage names in the model input: {leaking or forbidden_feature_names(names)}")
    return names


def assert_feature_subset_v5(feature_names: Iterable[str]) -> tuple[str, ...]:
    """An ablation subset: canonical names only, canonical relative order, no metadata."""
    names = tuple(feature_names)
    canonical = [name for name in FIRE_DETECTION_FEATURE_NAMES_V5 if name in set(names)]
    if len(set(names)) != len(names) or tuple(canonical) != names:
        raise ValueError(f"feature subset must be canonical V5 names in canonical order, got {names}")
    if not names or [n for n in names if n in EXCLUDED_FROM_MODEL_INPUT_V5] or forbidden_feature_names(names):
        raise ValueError(f"invalid model input features: {names}")
    return names


@dataclass(frozen=True)
class TrainingMatrixV5:
    """The frozen V5 dataset as arrays. `X` is float with NaN for "not computable"; metadata is kept aside."""

    rows: tuple[FireDetectionDatasetRowV5, ...]
    feature_names: tuple[str, ...]
    X: np.ndarray
    y: np.ndarray
    environments: np.ndarray
    regimes: np.ndarray
    subtypes: np.ndarray
    pair_ids: np.ndarray  # -1 for unpaired rows
    pair_types: np.ndarray  # "" for unpaired rows

    def columns(self, names: Iterable[str]) -> np.ndarray:
        names = tuple(names)
        return self.X[:, [self.feature_names.index(name) for name in names]]

    def regime_mask(self, regime: str) -> np.ndarray:
        return self.regimes == regime

    def feature(self, name: str) -> np.ndarray:
        return self.X[:, self.feature_names.index(name)]

    def select(self, indices: np.ndarray) -> TrainingMatrixV5:
        """A row subset (same schema). Used by tests / smoke runs; the real evaluation always uses the full matrix."""
        indices = np.asarray(indices)
        return TrainingMatrixV5(
            rows=tuple(self.rows[i] for i in indices),
            feature_names=self.feature_names,
            X=self.X[indices],
            y=self.y[indices],
            environments=self.environments[indices],
            regimes=self.regimes[indices],
            subtypes=self.subtypes[indices],
            pair_ids=self.pair_ids[indices],
            pair_types=self.pair_types[indices],
        )


def load_training_matrix_v5(csv_path: Path = DEFAULT_TRAINING_CSV_V5, verify_hash: bool = True) -> TrainingMatrixV5:
    """Load training_v5.csv (hash-verified) into arrays. The 25 model features are the canonical schema."""
    if verify_hash:
        verify_frozen_dataset_v5(csv_path)
    rows = load_training_dataset_rows_v5(csv_path)
    if verify_hash and len(rows) != EXPECTED_ROWS_V5:
        raise ValueError(f"expected {EXPECTED_ROWS_V5} rows, got {len(rows)}")
    return matrix_from_rows(rows)


def matrix_from_rows(rows: tuple[FireDetectionDatasetRowV5, ...]) -> TrainingMatrixV5:
    """Arrays for already-loaded dataset rows (the 25 model features are the canonical schema)."""
    names = assert_feature_schema_v5(FIRE_DETECTION_FEATURE_NAMES_V5)
    X = np.array([[np.nan if value is None else value for value in row.features] for row in rows], dtype=float)
    return TrainingMatrixV5(
        rows=rows,
        feature_names=names,
        X=X,
        y=np.array([row.label for row in rows], dtype=int),
        environments=np.array([row.environment_id for row in rows], dtype=int),
        regimes=np.array([row.regime for row in rows]),
        subtypes=np.array([row.latent_subtype for row in rows]),
        pair_ids=np.array([-1 if row.pair_id is None else row.pair_id for row in rows], dtype=int),
        pair_types=np.array(["" if row.pair_type is None else row.pair_type for row in rows]),
    )


# --- pipelines -------------------------------------------------------------------------------------------


def build_preprocessor_v5(feature_names: tuple[str, ...], impute: bool = True):
    """Median-impute the nullable features present in `feature_names`; pass the rest through.

    Output columns: the imputed nullable columns first, then the others in their original order
    (see `transformed_feature_names`). `impute=False` returns an identity (native NaN handling).
    """
    nullable = [index for index, name in enumerate(feature_names) if name in NULLABLE_FEATURE_NAMES_V5]
    if not impute or not nullable:
        return FunctionTransformer(feature_names_out="one-to-one")
    return ColumnTransformer(
        transformers=[("impute_nullable", SimpleImputer(strategy="median", keep_empty_features=True), nullable)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def build_classifier_v5(model_key: str, params: dict):
    fixed = FIXED_HYPERPARAMETERS_V5[model_key]
    if model_key == "logistic_regression":
        return LogisticRegression(C=params["C"], penalty=fixed["penalty"], max_iter=fixed["max_iter"], random_state=RANDOM_SEED_V5)
    if model_key == "random_forest":
        return RandomForestClassifier(
            n_estimators=fixed["n_estimators"], max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
            max_features=fixed["max_features"], random_state=RANDOM_SEED_V5, n_jobs=fixed["n_jobs"],
        )
    if model_key == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_depth=params["max_depth"], learning_rate=params["learning_rate"], max_iter=fixed["max_iter"],
            min_samples_leaf=fixed["min_samples_leaf"], l2_regularization=fixed["l2_regularization"],
            early_stopping=fixed["early_stopping"], random_state=RANDOM_SEED_V5,
        )
    raise ValueError(f"Unknown model {model_key!r}; expected one of {MODEL_FAMILIES_V5}")


def build_pipeline_v5(model_key: str, params: dict, feature_names: tuple[str, ...] = FIRE_DETECTION_FEATURE_NAMES_V5) -> Pipeline:
    """imputation [+ scaling] + classifier for one candidate; the feature list is schema-validated first."""
    if model_key not in MODEL_FAMILIES_V5:
        raise ValueError(f"Unknown model {model_key!r}; expected one of {MODEL_FAMILIES_V5}")
    if tuple(params) and params not in CANDIDATE_GRID_V5[model_key]:
        raise ValueError(f"{params!r} is not in the predefined {model_key} grid.")
    names = assert_feature_schema_v5(feature_names) if tuple(feature_names) == FIRE_DETECTION_FEATURE_NAMES_V5 else assert_feature_subset_v5(feature_names)
    steps = [("preprocess", build_preprocessor_v5(names, impute=model_key != "hist_gradient_boosting"))]
    if model_key == "logistic_regression":
        steps.append(("scale", StandardScaler()))
    steps.append(("classifier", build_classifier_v5(model_key, params)))
    return Pipeline(steps)


def transformed_feature_names(pipeline: Pipeline, feature_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(pipeline.named_steps["preprocess"].get_feature_names_out(np.array(feature_names, dtype=object)))


def describe_preprocessing_v5(model_key: str, feature_names: tuple[str, ...] = FIRE_DETECTION_FEATURE_NAMES_V5) -> dict:
    nullable = [name for name in feature_names if name in NULLABLE_FEATURE_NAMES_V5]
    if model_key == "hist_gradient_boosting":
        return {
            "imputation": "none (HistGradientBoostingClassifier handles NaN natively)",
            "nullable_features": nullable,
            "scaling": "none",
            "missing_indicators": "none",
        }
    return {
        "imputation": "median (fit on training folds only) for the nullable features",
        "nullable_features": nullable,
        "scaling": "StandardScaler after imputation" if model_key == "logistic_regression" else "none (tree model)",
        "missing_indicators": "none added",
    }

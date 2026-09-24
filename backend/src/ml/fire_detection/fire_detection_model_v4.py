"""Fire Detection ML V4: training data loading, schema validation, preprocessing and model pipelines.

OFFLINE TRAINING/EVALUATION ONLY. Nothing here is imported by FireDetectionAgent,
FireDetectionHybridPolicy or any runtime path; V4 does not influence any FireEvent.

Feature schema
    Features come only from the canonical V4 contract (FIRE_DETECTION_FEATURE_NAMES_V4 /
    FireDetectionFeaturesV4). The CSV header is validated against it before any model
    sees the data, and the design matrix is built from `FireDetectionFeaturesV4.as_tuple()`,
    so metadata columns (scenario_family, ground truth, label, timestamps, ...) can never
    become model input.

Preprocessing (identical for every model family)
    1. Missing-value imputation ONLY for the two nullable Fire Danger numeric features
       (`fire_danger_score`, `fire_danger_age_minutes`): SimpleImputer(strategy="median"),
       fitted on training rows only (it sits inside the sklearn Pipeline, so cross-validation
       cannot leak validation statistics). `fire_danger_available` is a real binary feature,
       is NOT imputed, and is passed through so the model can always tell an imputed value
       from an observed one. Median-imputed scores can therefore only ever be read together
       with available == 0.
    2. StandardScaler for Logistic Regression only (tree models are scale-free).
    3. The classifier.
    The Pipeline is the single artifact: nothing is scaled or imputed outside it.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.ml.fire_detection.fire_detection_dataset_v4 import (
    FireDetectionDatasetRowV4,
    load_training_dataset_rows_v4,
)
from src.ml.fire_detection.fire_detection_features_v4 import (
    FIRE_DANGER_CONTEXT_FEATURE_NAMES_V4,
    FIRE_DETECTION_FEATURE_NAMES_V4,
    NULLABLE_FEATURE_NAMES_V4,
    TRAINING_DATA_METADATA_COLUMNS_V4,
    forbidden_feature_names,
    validate_feature_names_v4,
)

DATASET_VERSION_V4 = "training_v4"
MODEL_VERSION_V4 = "4.0"
FEATURE_SCHEMA_VERSION_V4 = "v4"
RANDOM_SEED_V4 = 42

DEFAULT_TRAINING_CSV_V4 = Path(__file__).resolve().parents[3] / "data" / "fire_detection" / "training_v4.csv"


def dataset_sha256(csv_path: Path) -> str:
    """SHA-256 of the training CSV bytes, recorded in reports/metadata to pin the exact dataset."""
    return hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()


# --- feature groups for ablations, derived from the canonical names (never re-typed) ---


def feature_names_without(excluded: Iterable[str]) -> tuple[str, ...]:
    """The canonical V4 names, in canonical order, minus `excluded` (which must all exist)."""
    excluded_set = set(excluded)
    unknown = excluded_set - set(FIRE_DETECTION_FEATURE_NAMES_V4)
    if unknown:
        raise ValueError(f"Unknown V4 feature names: {sorted(unknown)}")
    return tuple(name for name in FIRE_DETECTION_FEATURE_NAMES_V4 if name not in excluded_set)


FEATURE_GROUPS_V4: dict[str, tuple[str, ...]] = {
    "fire_danger": FIRE_DANGER_CONTEXT_FEATURE_NAMES_V4,
    "news": tuple(name for name in FIRE_DETECTION_FEATURE_NAMES_V4 if name.startswith("news_")),
    "satellite_frp_brightness": tuple(
        name for name in FIRE_DETECTION_FEATURE_NAMES_V4 if "_frp_" in name or "_brightness_" in name
    ),
    "geometry_time": ("time_span_minutes", "max_pairwise_distance_km"),
}


# --- training data ---


@dataclass(frozen=True)
class TrainingMatrixV4:
    """The frozen V4 dataset as arrays. `X` columns follow FIRE_DETECTION_FEATURE_NAMES_V4; NaN = Fire Danger unavailable."""

    rows: tuple[FireDetectionDatasetRowV4, ...]
    feature_names: tuple[str, ...]
    X: np.ndarray
    y: np.ndarray
    families: np.ndarray
    archetypes: np.ndarray

    def columns(self, feature_names: Iterable[str]) -> np.ndarray:
        """X restricted to feature_names (in the order given)."""
        indices = [self.feature_names.index(name) for name in feature_names]
        return self.X[:, indices]


def read_csv_header(csv_path: Path) -> tuple[str, ...]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as csv_file:
        return tuple(next(csv.reader(csv_file)))


def load_training_matrix_v4(csv_path: Path = DEFAULT_TRAINING_CSV_V4) -> TrainingMatrixV4:
    """Load the V4 CSV, validating the feature schema/order BEFORE anything is trained on it."""
    header = read_csv_header(csv_path)
    metadata_count = len(TRAINING_DATA_METADATA_COLUMNS_V4)
    if header[:metadata_count] != TRAINING_DATA_METADATA_COLUMNS_V4:
        raise ValueError("V4 CSV metadata columns do not match the schema.")
    validate_feature_names_v4(header[metadata_count:-1])  # the feature columns must be exactly the canonical 19, in order
    if forbidden_feature_names(header[metadata_count:-1]):
        raise ValueError("V4 CSV feature columns contain leakage-looking names.")

    rows = load_training_dataset_rows_v4(csv_path)  # also validates the full header and per-cell rules
    X = np.array([row.to_features_v4().as_tuple() for row in rows], dtype=float)  # validated contract; NaN = missing
    return TrainingMatrixV4(
        rows=rows,
        feature_names=FIRE_DETECTION_FEATURE_NAMES_V4,
        X=X,
        y=np.array([row.label for row in rows], dtype=int),
        families=np.array([row.scenario_family for row in rows]),
        archetypes=np.array([row.scenario_archetype for row in rows]),
    )


# --- preprocessing and models ---


def build_preprocessor_v4(feature_names: tuple[str, ...] = FIRE_DETECTION_FEATURE_NAMES_V4):
    """Median-impute the nullable Fire Danger numerics (only those present in feature_names); pass the rest through.

    Output columns: the imputed nullable columns first, then the remaining columns in their original order
    (use `transformed_feature_names` to map them back).
    """
    nullable_indices = [index for index, name in enumerate(feature_names) if name in NULLABLE_FEATURE_NAMES_V4]
    if not nullable_indices:
        return FunctionTransformer(feature_names_out="one-to-one")  # e.g. the "no Fire Danger" ablation
    return ColumnTransformer(
        transformers=[
            (
                "impute_nullable_fire_danger",
                SimpleImputer(strategy="median", keep_empty_features=True),
                nullable_indices,
            )
        ],
        remainder="passthrough",  # includes fire_danger_available: a real binary feature, never imputed
        verbose_feature_names_out=False,
    )


@dataclass(frozen=True)
class ModelSpecV4:
    """One model family: how to build its full pipeline, and how it ranks for explainability."""

    key: str
    display_name: str
    simplicity_rank: int  # lower = simpler / more explainable (used only to break near-ties)
    scaled: bool
    build_classifier: Callable[[], object]
    hyperparameters: dict


def _logistic_regression(C: float = 1.0):
    return LogisticRegression(C=C, penalty="l2", max_iter=2000, random_state=RANDOM_SEED_V4)


def _random_forest(min_samples_leaf: int = 10):
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=RANDOM_SEED_V4,
        n_jobs=1,  # single-threaded: identical results run to run
    )


def _hist_gradient_boosting(max_depth: int = 3):
    return HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_SEED_V4,
    )


# Hyperparameters are fixed a priori with moderate regularisation (V3's default Random Forest
# overfit badly); they are NOT tuned on the evaluation splits. A small sensitivity check in the
# comparison report shows whether conclusions depend on them.
MODEL_SPECS_V4: dict[str, ModelSpecV4] = {
    "logistic_regression": ModelSpecV4(
        key="logistic_regression",
        display_name="Logistic Regression V4",
        simplicity_rank=0,
        scaled=True,
        build_classifier=_logistic_regression,
        hyperparameters={"C": 1.0, "penalty": "l2", "max_iter": 2000},
    ),
    "random_forest": ModelSpecV4(
        key="random_forest",
        display_name="Random Forest V4",
        simplicity_rank=1,
        scaled=False,
        build_classifier=_random_forest,
        hyperparameters={"n_estimators": 300, "min_samples_leaf": 10, "max_features": "sqrt"},
    ),
    "hist_gradient_boosting": ModelSpecV4(
        key="hist_gradient_boosting",
        display_name="Histogram Gradient Boosting V4",
        simplicity_rank=2,
        scaled=False,
        build_classifier=_hist_gradient_boosting,
        hyperparameters={"max_depth": 3, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 40, "l2_regularization": 1.0},
    ),
}


# Neighbouring hyperparameter settings for the sensitivity check (report only; the a-priori
# settings above are what is evaluated as "the" model).
SENSITIVITY_VARIANTS_V4: dict[str, dict[str, Callable[[], object]]] = {
    "logistic_regression": {
        "C=0.1": lambda: _logistic_regression(0.1),
        "C=1.0 (default)": lambda: _logistic_regression(1.0),
        "C=10.0": lambda: _logistic_regression(10.0),
    },
    "random_forest": {
        "min_samples_leaf=2": lambda: _random_forest(2),
        "min_samples_leaf=10 (default)": lambda: _random_forest(10),
        "min_samples_leaf=30": lambda: _random_forest(30),
    },
    "hist_gradient_boosting": {
        "max_depth=2": lambda: _hist_gradient_boosting(2),
        "max_depth=3 (default)": lambda: _hist_gradient_boosting(3),
        "max_depth=5": lambda: _hist_gradient_boosting(5),
    },
}


def build_pipeline_v4(
    model_key: str,
    feature_names: tuple[str, ...] = FIRE_DETECTION_FEATURE_NAMES_V4,
    classifier=None,
) -> Pipeline:
    """The full sklearn pipeline (imputation [+ scaling] + classifier) for a model family.

    `classifier` overrides the model's default classifier (used by the sensitivity study).
    """
    if model_key not in MODEL_SPECS_V4:
        raise ValueError(f"Unknown model {model_key!r}; expected one of {sorted(MODEL_SPECS_V4)}")
    spec = MODEL_SPECS_V4[model_key]
    steps = [("preprocess", build_preprocessor_v4(feature_names))]
    if spec.scaled:
        steps.append(("scale", StandardScaler()))
    steps.append(("classifier", classifier if classifier is not None else spec.build_classifier()))
    return Pipeline(steps)


def transformed_feature_names(pipeline: Pipeline, feature_names: tuple[str, ...]) -> tuple[str, ...]:
    """Names of the columns the classifier sees (imputed nullable columns come first)."""
    return tuple(pipeline.named_steps["preprocess"].get_feature_names_out(np.array(feature_names, dtype=object)))


def describe_preprocessing_v4(model_key: str, feature_names: tuple[str, ...] = FIRE_DETECTION_FEATURE_NAMES_V4) -> dict:
    """A JSON-serialisable description of the exact preprocessing pipeline, for reports and metadata."""
    spec = MODEL_SPECS_V4[model_key]
    imputed = [name for name in feature_names if name in NULLABLE_FEATURE_NAMES_V4]
    steps = [
        {
            "step": "impute_nullable_fire_danger",
            "transformer": "sklearn.impute.SimpleImputer",
            "strategy": "median",
            "keep_empty_features": True,
            "applied_to": imputed,
            "fitted_on": "training rows only (inside the Pipeline)",
        },
        {
            "step": "passthrough",
            "applied_to": [name for name in feature_names if name not in imputed],
            "note": "includes fire_danger_available, which is a real binary feature and is never imputed",
        },
    ]
    if spec.scaled:
        steps.append({"step": "scale", "transformer": "sklearn.preprocessing.StandardScaler", "applied_to": "all columns"})
    steps.append(
        {
            "step": "classifier",
            "estimator": type(spec.build_classifier()).__name__,
            "hyperparameters": spec.hyperparameters,
            "random_state": RANDOM_SEED_V4,
        }
    )
    return {"feature_names": list(feature_names), "steps": steps}

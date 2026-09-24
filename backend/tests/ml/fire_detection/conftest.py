"""Shared, generated-once fixtures for the Fire Detection ML V4 dataset tests."""
from __future__ import annotations

import pytest

from src.ml.fire_detection.fire_detection_dataset_v4 import row_from_sample
from src.ml.fire_detection.fire_detection_training_data_generator_v4 import (
    DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V4,
    DEFAULT_TRAINING_DATA_SEED_V4,
    FireDetectionTrainingDataGeneratorV4,
)


@pytest.fixture(scope="session")
def v4_samples():
    """The full default V4 dataset (seed 42), generated once per test session."""
    return FireDetectionTrainingDataGeneratorV4(seed=DEFAULT_TRAINING_DATA_SEED_V4).generate(
        DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V4
    )


@pytest.fixture(scope="session")
def v4_rows(v4_samples):
    return tuple(row_from_sample(sample) for sample in v4_samples)


# --- V4 model comparison fixtures (Task 4) ---

from src.ml.fire_detection.fire_detection_model_comparison_v4 import run_model_comparison_v4  # noqa: E402
from src.ml.fire_detection.fire_detection_model_v4 import (  # noqa: E402
    DEFAULT_TRAINING_CSV_V4,
    TrainingMatrixV4,
    load_training_matrix_v4,
)


@pytest.fixture(scope="session")
def v4_matrix():
    """The frozen training_v4.csv as arrays (schema-validated on load)."""
    return load_training_matrix_v4(DEFAULT_TRAINING_CSV_V4)


def subsample_by_family(matrix: TrainingMatrixV4, samples, per_family: int = 60):
    """A small, family-complete subset of the matrix and its aligned samples (first `per_family` rows per family)."""
    kept: list[int] = []
    counts: dict[str, int] = {}
    for index, family in enumerate(matrix.families.tolist()):
        if counts.get(family, 0) < per_family:
            counts[family] = counts.get(family, 0) + 1
            kept.append(index)
    small = TrainingMatrixV4(
        rows=tuple(matrix.rows[i] for i in kept),
        feature_names=matrix.feature_names,
        X=matrix.X[kept],
        y=matrix.y[kept],
        families=matrix.families[kept],
        archetypes=matrix.archetypes[kept],
    )
    return small, tuple(samples[i] for i in kept)


@pytest.fixture(scope="session")
def small_comparison_inputs(v4_matrix, v4_samples):
    return subsample_by_family(v4_matrix, v4_samples)


@pytest.fixture(scope="session")
def small_comparison_report(small_comparison_inputs):
    """A full (reduced-size, Logistic Regression only) comparison report, generated once per session."""
    matrix, samples = small_comparison_inputs
    return run_model_comparison_v4(
        matrix,
        samples,
        csv_path=DEFAULT_TRAINING_CSV_V4,
        model_keys=("logistic_regression",),
        permutation_repeats=1,
        include_sensitivity=False,
        include_calibration_variants=False,
    )


# --- V5 dataset fixtures (Task 6) ---

from src.ml.fire_detection.fire_detection_dataset_v5 import row_from_sample as row_from_sample_v5  # noqa: E402
from src.ml.fire_detection.fire_detection_training_data_generator_v5 import (  # noqa: E402
    DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V5,
    DEFAULT_TRAINING_DATA_SEED_V5,
    FireDetectionTrainingDataGeneratorV5,
)


@pytest.fixture(scope="session")
def v5_samples():
    """The full default V5 dataset (seed 42), generated once per test session."""
    return FireDetectionTrainingDataGeneratorV5(seed=DEFAULT_TRAINING_DATA_SEED_V5).generate(
        DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V5
    )


@pytest.fixture(scope="session")
def v5_rows(v5_samples):
    return tuple(row_from_sample_v5(sample) for sample in v5_samples)


# --- V5 model-comparison fixtures (Task 7) ---

from src.ml.fire_detection.fire_detection_model_v5 import load_training_matrix_v5  # noqa: E402


@pytest.fixture(scope="session")
def v5_matrix():
    """The frozen training_v5.csv as arrays (hash-verified and schema-validated on load)."""
    return load_training_matrix_v5()

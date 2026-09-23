"""Tests for FireDetectionTrainingDataGenerator."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ml.fire_detection.fire_detection_feature_extractor import FireDetectionFeatureExtractor
from src.ml.fire_detection.fire_detection_features import (
    FIRE_DETECTION_FEATURE_NAMES,
    SAMPLE_ID_COLUMN,
    SCENARIO_FAMILY_COLUMN,
    TRAINING_DATA_CSV_COLUMNS,
    TRAINING_DATA_METADATA_COLUMNS,
)
from src.ml.fire_detection.fire_detection_dataset import load_training_dataset_rows
from src.ml.fire_detection.fire_detection_model_comparison import regenerated_samples_match_dataset
from src.ml.fire_detection.fire_detection_training_data_generator import (
    DATASET_VERSION_V1,
    DATASET_VERSION_V2,
    DEFAULT_TRAINING_DATA_SAMPLE_COUNT,
    FireDetectionTrainingDataGenerator,
)
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_evidence_type import FireEvidenceType

SAMPLE_COUNT = 400
V2_SAMPLE_COUNT = 600
V1_CSV_PATH = Path(__file__).resolve().parents[3] / "data" / "fire_detection" / "training_v1.csv"


def generate(seed: int = 42, num_samples: int = SAMPLE_COUNT, dataset_version: str = DATASET_VERSION_V1):
    return FireDetectionTrainingDataGenerator(seed=seed).generate(num_samples=num_samples, dataset_version=dataset_version)


def satellite_confidences(sample) -> set[str]:
    return {item.satellite_confidence for item in sample.evidence if item.evidence_type is FireEvidenceType.SATELLITE}


def evidence_source_types(sample) -> frozenset[FireEvidenceType]:
    return frozenset(item.evidence_type for item in sample.evidence)


def test_same_seed_produces_equivalent_output():
    first_run = generate(seed=42)
    second_run = generate(seed=42)

    assert first_run == second_run


def test_different_seed_can_produce_different_output():
    first_run = generate(seed=1)
    second_run = generate(seed=2)

    assert first_run != second_run


def test_both_labels_exist():
    samples = generate()

    labels = {sample.label for sample in samples}

    assert labels == {0, 1}


def test_label_distribution_is_reasonably_balanced():
    samples = generate()

    positive_count = sum(1 for sample in samples if sample.label == 1)
    negative_count = len(samples) - positive_count

    # Exact 50/50 split is guaranteed by construction (see _balanced_labels).
    assert positive_count == len(samples) // 2
    assert negative_count == len(samples) - len(samples) // 2


def test_positive_rows_contain_evidence():
    samples = generate()

    positive_samples = [sample for sample in samples if sample.label == 1]
    assert positive_samples
    assert all(len(sample.evidence) > 0 for sample in positive_samples)


def test_negative_rows_contain_evidence():
    samples = generate()

    negative_samples = [sample for sample in samples if sample.label == 0]
    assert negative_samples
    assert all(len(sample.evidence) > 0 for sample in negative_samples)


def test_satellite_only_examples_occur_in_both_classes():
    samples = generate()
    satellite_only = {FireEvidenceType.SATELLITE}

    labels_with_satellite_only = {
        sample.label for sample in samples if evidence_source_types(sample) == satellite_only
    }

    assert labels_with_satellite_only == {0, 1}


def test_news_only_examples_occur_in_both_classes():
    samples = generate()
    news_only = {FireEvidenceType.NEWS}

    labels_with_news_only = {sample.label for sample in samples if evidence_source_types(sample) == news_only}

    assert labels_with_news_only == {0, 1}


def test_satellite_and_news_combination_occurs_in_both_classes():
    samples = generate()
    both_sources = {FireEvidenceType.SATELLITE, FireEvidenceType.NEWS}

    labels_with_both_sources = {
        sample.label for sample in samples if evidence_source_types(sample) == both_sources
    }

    assert labels_with_both_sources == {0, 1}


def test_metadata_columns_are_excluded_from_ml_features():
    assert SAMPLE_ID_COLUMN not in FIRE_DETECTION_FEATURE_NAMES
    assert SCENARIO_FAMILY_COLUMN not in FIRE_DETECTION_FEATURE_NAMES
    assert set(TRAINING_DATA_METADATA_COLUMNS) == {SAMPLE_ID_COLUMN, SCENARIO_FAMILY_COLUMN}
    assert TRAINING_DATA_CSV_COLUMNS == TRAINING_DATA_METADATA_COLUMNS + FIRE_DETECTION_FEATURE_NAMES + ("label",)


def test_every_generated_sample_is_a_valid_candidate():
    samples = generate()
    extractor = FireDetectionFeatureExtractor()

    for sample in samples:
        candidate = FireDetectionCandidate(sample.evidence)  # raises if not one connected component
        assert extractor.extract(sample.evidence) is not None
        assert len(candidate.evidence) == len(sample.evidence)


def test_feature_column_ordering_is_stable():
    assert FIRE_DETECTION_FEATURE_NAMES == (
        "satellite_count",
        "news_count",
        "satellite_low_count",
        "satellite_nominal_count",
        "satellite_high_count",
        "time_span_minutes",
        "max_pairwise_distance_km",
    )


def test_sample_ids_are_unique_and_sequential():
    samples = generate()

    sample_ids = [sample.sample_id for sample in samples]

    assert sample_ids == list(range(1, len(samples) + 1))


def test_num_samples_must_be_positive():
    with pytest.raises(ValueError):
        FireDetectionTrainingDataGenerator().generate(num_samples=0)


def test_seed_must_be_an_integer():
    with pytest.raises(ValueError):
        FireDetectionTrainingDataGenerator(seed="42")


# --- V2 (dataset_version="v2") ---


def test_dataset_version_must_be_supported():
    with pytest.raises(ValueError):
        FireDetectionTrainingDataGenerator().generate(num_samples=10, dataset_version="v3")


def test_v1_generation_is_unchanged_by_the_v2_addition():
    """Adding V2 families/logic must not change dataset_version="v1" (the default) output at all."""
    default_call = generate(seed=42, num_samples=DEFAULT_TRAINING_DATA_SAMPLE_COUNT)
    explicit_v1_call = generate(seed=42, num_samples=DEFAULT_TRAINING_DATA_SAMPLE_COUNT, dataset_version=DATASET_VERSION_V1)

    assert default_call == explicit_v1_call


@pytest.mark.skipif(not V1_CSV_PATH.exists(), reason="training_v1.csv artifact not present")
def test_v1_generation_still_reproduces_the_saved_training_v1_csv():
    csv_rows = load_training_dataset_rows(V1_CSV_PATH)
    regenerated = generate(seed=42, num_samples=len(csv_rows), dataset_version=DATASET_VERSION_V1)

    assert regenerated_samples_match_dataset(csv_rows, regenerated)


def test_v2_generation_is_deterministic_under_the_same_seed():
    first_run = generate(seed=42, num_samples=V2_SAMPLE_COUNT, dataset_version=DATASET_VERSION_V2)
    second_run = generate(seed=42, num_samples=V2_SAMPLE_COUNT, dataset_version=DATASET_VERSION_V2)

    assert first_run == second_run


def test_v2_low_confidence_satellite_evidence_occurs_under_both_labels():
    samples = generate(num_samples=V2_SAMPLE_COUNT, dataset_version=DATASET_VERSION_V2)

    labels_with_low = {sample.label for sample in samples if "low" in satellite_confidences(sample)}

    assert labels_with_low == {0, 1}


def test_v2_low_confidence_satellite_evidence_is_not_the_majority_of_positive_rows():
    samples = generate(num_samples=V2_SAMPLE_COUNT, dataset_version=DATASET_VERSION_V2)
    positive_samples = [sample for sample in samples if sample.label == 1]

    positive_with_low = sum(1 for sample in positive_samples if "low" in satellite_confidences(sample))

    # Fixes the V1 artifact (0%) without over-correcting into "low implies fire" (100%).
    assert 0 < positive_with_low < len(positive_samples)


def test_v2_high_confidence_satellite_evidence_occurs_under_both_labels():
    samples = generate(num_samples=V2_SAMPLE_COUNT, dataset_version=DATASET_VERSION_V2)

    labels_with_high = {sample.label for sample in samples if "high" in satellite_confidences(sample)}

    assert labels_with_high == {0, 1}


def test_v2_news_evidence_occurs_under_both_labels():
    samples = generate(num_samples=V2_SAMPLE_COUNT, dataset_version=DATASET_VERSION_V2)
    news_only_or_mixed = {
        sample.label for sample in samples if any(item.evidence_type is FireEvidenceType.NEWS for item in sample.evidence)
    }

    assert news_only_or_mixed == {0, 1}


def test_v2_mixed_confidence_positive_examples_exist():
    """At least one positive sample must carry more than one distinct satellite confidence level."""
    samples = generate(num_samples=V2_SAMPLE_COUNT, dataset_version=DATASET_VERSION_V2)

    mixed_positive = [
        sample for sample in samples if sample.label == 1 and len(satellite_confidences(sample)) >= 2
    ]

    assert mixed_positive


def test_v2_mixed_confidence_negative_examples_exist():
    samples = generate(num_samples=V2_SAMPLE_COUNT, dataset_version=DATASET_VERSION_V2)

    mixed_negative = [
        sample for sample in samples if sample.label == 0 and len(satellite_confidences(sample)) >= 2
    ]

    assert mixed_negative


def test_v2_metadata_columns_are_still_excluded_from_ml_features():
    assert SAMPLE_ID_COLUMN not in FIRE_DETECTION_FEATURE_NAMES
    assert SCENARIO_FAMILY_COLUMN not in FIRE_DETECTION_FEATURE_NAMES


def test_v2_every_generated_sample_is_a_valid_candidate():
    samples = generate(num_samples=V2_SAMPLE_COUNT, dataset_version=DATASET_VERSION_V2)
    extractor = FireDetectionFeatureExtractor()

    for sample in samples:
        candidate = FireDetectionCandidate(sample.evidence)
        assert extractor.extract(sample.evidence) is not None
        assert len(candidate.evidence) == len(sample.evidence)


def test_v2_includes_new_scenario_families_not_present_in_v1():
    v1_families = {sample.scenario_family for sample in generate(num_samples=V2_SAMPLE_COUNT, dataset_version=DATASET_VERSION_V1)}
    v2_families = {sample.scenario_family for sample in generate(num_samples=V2_SAMPLE_COUNT, dataset_version=DATASET_VERSION_V2)}

    assert v1_families < v2_families  # strict subset: V2 adds families, keeps all of V1's

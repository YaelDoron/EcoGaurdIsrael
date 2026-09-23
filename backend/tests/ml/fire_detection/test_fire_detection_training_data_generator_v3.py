"""Tests for FireDetectionTrainingDataGeneratorV3."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ml.fire_detection.fire_detection_dataset import load_training_dataset_rows
from src.ml.fire_detection.fire_detection_feature_extractor_v3 import FireDetectionFeatureExtractorV3
from src.ml.fire_detection.fire_detection_model_comparison import regenerated_samples_match_dataset
from src.ml.fire_detection.fire_detection_training_data_generator import (
    DEFAULT_TRAINING_DATA_SAMPLE_COUNT,
    FireDetectionTrainingDataGenerator,
)
from src.ml.fire_detection.fire_detection_training_data_generator_v3 import (
    FireDetectionTrainingDataGeneratorV3,
    ScenarioArchetype,
)
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_evidence_type import FireEvidenceType

SAMPLE_COUNT = 800
V1_CSV_PATH = Path(__file__).resolve().parents[3] / "data" / "fire_detection" / "training_v1.csv"
V2_CSV_PATH = Path(__file__).resolve().parents[3] / "data" / "fire_detection" / "training_v2.csv"


def generate(seed: int = 42, num_samples: int = SAMPLE_COUNT):
    return FireDetectionTrainingDataGeneratorV3(seed=seed).generate(num_samples=num_samples)


def satellite_confidences(sample) -> set[str]:
    return {item.satellite_confidence for item in sample.evidence if item.evidence_type is FireEvidenceType.SATELLITE}


def news_signals(sample):
    return [item.news_wildfire_signal_strength for item in sample.evidence if item.evidence_type is FireEvidenceType.NEWS]


def satellite_items(sample):
    return tuple(item for item in sample.evidence if item.evidence_type is FireEvidenceType.SATELLITE)


# --- determinism ---


def test_v3_generation_is_deterministic_under_the_same_seed():
    first_run = generate(seed=42)
    second_run = generate(seed=42)

    assert first_run == second_run


def test_num_samples_must_be_positive():
    with pytest.raises(ValueError):
        FireDetectionTrainingDataGeneratorV3().generate(num_samples=0)


def test_seed_must_be_an_integer():
    with pytest.raises(ValueError):
        FireDetectionTrainingDataGeneratorV3(seed="42")


# --- V1/V2 remain untouched by the existence of the V3 generator ---


def test_v1_generator_is_unaffected_by_v3_generator_module():
    samples = FireDetectionTrainingDataGenerator(seed=42).generate(num_samples=DEFAULT_TRAINING_DATA_SAMPLE_COUNT)

    assert len(samples) == DEFAULT_TRAINING_DATA_SAMPLE_COUNT


@pytest.mark.skipif(not V1_CSV_PATH.exists(), reason="training_v1.csv artifact not present")
def test_v1_csv_still_reproducible():
    csv_rows = load_training_dataset_rows(V1_CSV_PATH)
    regenerated = FireDetectionTrainingDataGenerator(seed=42).generate(num_samples=len(csv_rows))

    assert regenerated_samples_match_dataset(csv_rows, regenerated)


@pytest.mark.skipif(not V2_CSV_PATH.exists(), reason="training_v2.csv artifact not present")
def test_v2_csv_still_reproducible():
    csv_rows = load_training_dataset_rows(V2_CSV_PATH)
    regenerated = FireDetectionTrainingDataGenerator(seed=42).generate(num_samples=len(csv_rows), dataset_version="v2")

    assert regenerated_samples_match_dataset(csv_rows, regenerated)


def test_v3_does_not_share_an_output_path_with_v1_or_v2(tmp_path):
    from scripts.generate_fire_detection_training_data import DEFAULT_OUTPUT_PATH as v1_path
    from scripts.generate_fire_detection_training_data_v3 import DEFAULT_OUTPUT_PATH as v3_path

    assert v1_path != v3_path
    assert v3_path.name == "training_v3.csv"


# --- labels ---


def test_both_labels_exist():
    samples = generate()

    assert {sample.label for sample in samples} == {0, 1}


def test_label_distribution_is_exactly_balanced():
    samples = generate()

    positive_count = sum(1 for sample in samples if sample.label == 1)
    assert positive_count == len(samples) // 2


# --- overlapping FRP/brightness/news-signal ---


def test_frp_overlaps_between_labels():
    samples = generate()
    extractor = FireDetectionFeatureExtractorV3()

    frp_by_label = {0: [], 1: []}
    for sample in samples:
        for item in satellite_items(sample):
            if item.satellite_frp is not None:
                frp_by_label[sample.label].append(item.satellite_frp)

    assert frp_by_label[0] and frp_by_label[1]
    # overlap: some negative FRP values exceed some positive FRP values (not perfectly separated)
    assert min(frp_by_label[0]) < max(frp_by_label[1])
    assert min(frp_by_label[1]) < max(frp_by_label[0])


def test_brightness_overlaps_between_labels():
    samples = generate()

    brightness_by_label = {0: [], 1: []}
    for sample in samples:
        for item in satellite_items(sample):
            if item.satellite_brightness is not None:
                brightness_by_label[sample.label].append(item.satellite_brightness)

    assert brightness_by_label[0] and brightness_by_label[1]
    assert min(brightness_by_label[0]) < max(brightness_by_label[1])
    assert min(brightness_by_label[1]) < max(brightness_by_label[0])


def test_news_strength_overlaps_between_labels():
    from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength

    samples = generate()

    strong_labels = {sample.label for sample in samples if NewsWildfireSignalStrength.STRONG in news_signals(sample)}
    weak_labels = {sample.label for sample in samples if NewsWildfireSignalStrength.WEAK in news_signals(sample)}

    assert strong_labels == {0, 1}
    assert weak_labels == {0, 1}


def test_mixed_confidence_positive_examples_exist():
    samples = generate()

    mixed_positive = [sample for sample in samples if sample.label == 1 and len(satellite_confidences(sample)) >= 2]

    assert mixed_positive


def test_mixed_confidence_negative_examples_exist():
    samples = generate()

    mixed_negative = [sample for sample in samples if sample.label == 0 and len(satellite_confidences(sample)) >= 2]

    assert mixed_negative


# --- no obvious one-feature binary separator ---


def test_low_confidence_satellite_evidence_occurs_under_both_labels():
    samples = generate()

    labels_with_low = {sample.label for sample in samples if "low" in satellite_confidences(sample)}

    assert labels_with_low == {0, 1}


def test_missing_frp_occurs_under_both_labels():
    samples = generate()

    labels = {
        sample.label
        for sample in samples
        if any(item.satellite_frp is None for item in satellite_items(sample))
    }

    assert labels == {0, 1}


# --- no metadata leakage ---


def test_metadata_columns_are_excluded_from_v3_feature_names():
    from src.ml.fire_detection.fire_detection_features_v3 import (
        FIRE_DETECTION_FEATURE_NAMES_V3,
        SAMPLE_ID_COLUMN,
        SCENARIO_ARCHETYPE_COLUMN,
        SCENARIO_FAMILY_COLUMN,
    )

    assert SAMPLE_ID_COLUMN not in FIRE_DETECTION_FEATURE_NAMES_V3
    assert SCENARIO_FAMILY_COLUMN not in FIRE_DETECTION_FEATURE_NAMES_V3
    assert SCENARIO_ARCHETYPE_COLUMN not in FIRE_DETECTION_FEATURE_NAMES_V3


# --- archetypes ---


def test_every_archetype_contains_both_labels():
    samples = generate()

    labels_by_archetype: dict[ScenarioArchetype, set[int]] = {}
    for sample in samples:
        labels_by_archetype.setdefault(sample.scenario_archetype, set()).add(sample.label)

    assert set(labels_by_archetype.keys()) == set(ScenarioArchetype)
    for archetype, labels in labels_by_archetype.items():
        assert labels == {0, 1}, f"{archetype} missing a label: {labels}"


# --- candidate validity ---


def test_every_generated_sample_is_a_valid_candidate():
    samples = generate()
    extractor = FireDetectionFeatureExtractorV3()

    for sample in samples:
        candidate = FireDetectionCandidate(sample.evidence)
        assert extractor.extract(sample.evidence) is not None
        assert len(candidate.evidence) == len(sample.evidence)


def test_sample_ids_are_unique_and_sequential():
    samples = generate()

    sample_ids = [sample.sample_id for sample in samples]

    assert sample_ids == list(range(1, len(samples) + 1))

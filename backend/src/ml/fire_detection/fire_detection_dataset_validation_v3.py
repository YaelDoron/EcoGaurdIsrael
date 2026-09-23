"""Automated sanity checks for a generated Fire Detection ML V3 training dataset (Part 27).

Run against raw FireDetectionTrainingSampleV3 objects (not the CSV) so it can
inspect both the extracted V3 features and the underlying evidence (e.g.
individual satellite confidence labels, source-type composition) that the
feature vector alone does not preserve per-item.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.ml.fire_detection.fire_detection_feature_extractor_v3 import FireDetectionFeatureExtractorV3
from src.ml.fire_detection.fire_detection_features_v3 import FIRE_DETECTION_FEATURE_NAMES_V3
from src.ml.fire_detection.fire_detection_training_data_generator_v3 import FireDetectionTrainingSampleV3
from src.models.fire_evidence_type import FireEvidenceType
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength

_MIN_MINORITY_LABEL_FRACTION = 0.3  # "reasonable" balance: neither label under 30%


@dataclass(frozen=True)
class DatasetValidationReportV3:
    """Pass/fail result of validate_training_samples_v3, with human-readable violations."""

    violations: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.violations


def validate_training_samples_v3(samples: tuple[FireDetectionTrainingSampleV3, ...]) -> DatasetValidationReportV3:
    """Run every Part 27 sanity check and return the combined violation list."""
    violations: list[str] = []
    extractor = FireDetectionFeatureExtractorV3()

    if not samples:
        return DatasetValidationReportV3(violations=("no samples were generated.",))

    labels_present = {sample.label for sample in samples}
    if labels_present != {0, 1}:
        violations.append(f"both labels (0 and 1) must be present; found {sorted(labels_present)}.")
        return DatasetValidationReportV3(violations=tuple(violations))

    positive_count = sum(1 for sample in samples if sample.label == 1)
    negative_count = len(samples) - positive_count
    minority_fraction = min(positive_count, negative_count) / len(samples)
    if minority_fraction < _MIN_MINORITY_LABEL_FRACTION:
        violations.append(
            f"class balance is not reasonable: positive={positive_count}, negative={negative_count} "
            f"(minority fraction {minority_fraction:.2%} < {_MIN_MINORITY_LABEL_FRACTION:.0%})."
        )

    def labels_where(predicate) -> set[int]:
        return {sample.label for sample in samples if predicate(sample)}

    def satellite_items(sample):
        return tuple(item for item in sample.evidence if item.evidence_type is FireEvidenceType.SATELLITE)

    def news_items(sample):
        return tuple(item for item in sample.evidence if item.evidence_type is FireEvidenceType.NEWS)

    # 3. LOW/NOMINAL/HIGH satellite confidence under both labels
    for confidence in ("low", "nominal", "high"):
        labels = labels_where(lambda sample, c=confidence: any(item.satellite_confidence == c for item in satellite_items(sample)))
        if labels != {0, 1}:
            violations.append(f"satellite_confidence={confidence!r} must occur under both labels; found under {sorted(labels)} only.")

    # 4-7. FRP / brightness presence and absence under both labels
    for field_name, attr in (("FRP", "satellite_frp"), ("brightness", "satellite_brightness")):
        present_labels = labels_where(lambda sample, a=attr: any(getattr(item, a) is not None for item in satellite_items(sample)))
        if present_labels != {0, 1}:
            violations.append(f"measured {field_name} must occur under both labels; found under {sorted(present_labels)} only.")

        missing_labels = labels_where(
            lambda sample, a=attr: any(getattr(item, a) is None for item in satellite_items(sample))
        )
        if missing_labels != {0, 1}:
            violations.append(f"missing {field_name} must occur under both labels; found under {sorted(missing_labels)} only.")

    # 8-10. WEAK/MODERATE/STRONG news signal under both labels
    for signal in (NewsWildfireSignalStrength.WEAK, NewsWildfireSignalStrength.MODERATE, NewsWildfireSignalStrength.STRONG):
        labels = labels_where(lambda sample, s=signal: any(item.news_wildfire_signal_strength is s for item in news_items(sample)))
        if labels != {0, 1}:
            violations.append(f"news signal {signal.value!r} must occur under both labels; found under {sorted(labels)} only.")

    # 11. news UNKNOWN (analysis unavailable, None) is supported at all
    unknown_labels = labels_where(lambda sample: any(item.news_wildfire_signal_strength is None for item in news_items(sample)))
    if not unknown_labels:
        violations.append("news wildfire_signal_strength=None (unknown/unavailable) never occurs in the dataset.")

    # 12-14. satellite-only / news-only / multi-source under both labels
    def is_satellite_only(sample) -> bool:
        return bool(satellite_items(sample)) and not news_items(sample)

    def is_news_only(sample) -> bool:
        return bool(news_items(sample)) and not satellite_items(sample)

    def is_multi_source(sample) -> bool:
        return bool(satellite_items(sample)) and bool(news_items(sample))

    for description, predicate in (
        ("satellite-only", is_satellite_only),
        ("news-only", is_news_only),
        ("multi-source (satellite+news)", is_multi_source),
    ):
        labels = labels_where(predicate)
        if labels != {0, 1}:
            violations.append(f"{description} rows must exist under both labels; found under {sorted(labels)} only.")

    # 15. mixed-confidence (>=2 distinct satellite confidence levels in one candidate) under both labels
    def is_mixed_confidence(sample) -> bool:
        return len({item.satellite_confidence for item in satellite_items(sample)}) >= 2

    mixed_labels = labels_where(is_mixed_confidence)
    if mixed_labels != {0, 1}:
        violations.append(f"mixed-confidence rows must exist under both labels; found under {sorted(mixed_labels)} only.")

    # 16. no metadata column enters the feature vector (structural check, not data-dependent)
    for metadata_name in ("sample_id", "scenario_family", "scenario_archetype"):
        if metadata_name in FIRE_DETECTION_FEATURE_NAMES_V3:
            violations.append(f"metadata column {metadata_name!r} must never appear in FIRE_DETECTION_FEATURE_NAMES_V3.")

    # Candidate validity: every sample's evidence must extract cleanly.
    for sample in samples:
        try:
            extractor.extract(sample.evidence)
        except ValueError as exc:
            violations.append(f"sample {sample.sample_id} ({sample.scenario_family!r}) is not a valid candidate: {exc}")

    return DatasetValidationReportV3(violations=tuple(violations))

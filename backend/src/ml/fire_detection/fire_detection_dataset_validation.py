"""Automated sanity checks for a generated Fire Detection ML training dataset.

Run against freshly generated FireDetectionTrainingSample objects (not the
CSV) so it can validate raw evidence via FireDetectionFeatureExtractor -
including candidate validity - not just the summarized feature columns.
Used by scripts.generate_fire_detection_training_data to fail loudly on a
malformed dataset instead of silently writing a broken CSV.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.ml.fire_detection.fire_detection_feature_extractor import FireDetectionFeatureExtractor
from src.ml.fire_detection.fire_detection_features import FIRE_DETECTION_FEATURE_NAMES
from src.ml.fire_detection.fire_detection_training_data_generator import FireDetectionTrainingSample

_MIN_MINORITY_LABEL_FRACTION = 0.3  # "reasonable" balance: neither label under 30%


@dataclass(frozen=True)
class DatasetValidationReport:
    """Pass/fail result of validate_training_samples, with human-readable violations."""

    violations: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.violations


def validate_training_samples(samples: tuple[FireDetectionTrainingSample, ...]) -> DatasetValidationReport:
    """Run every Part 9 sanity check and return the combined violation list."""
    violations: list[str] = []
    extractor = FireDetectionFeatureExtractor()

    rows: list[tuple[tuple[float, ...], int]] = []
    for sample in samples:
        try:
            features = extractor.extract(sample.evidence).as_tuple()
        except ValueError as exc:
            violations.append(
                f"sample {sample.sample_id} ({sample.scenario_family!r}) is not a valid Fire "
                f"Detection candidate: {exc}"
            )
            continue
        rows.append((features, sample.label))

    if not rows:
        violations.append("no valid rows were generated - cannot run further sanity checks.")
        return DatasetValidationReport(violations=tuple(violations))

    index = {name: position for position, name in enumerate(FIRE_DETECTION_FEATURE_NAMES)}

    def values_where(label: int, feature: str) -> list[float]:
        return [features[index[feature]] for features, row_label in rows if row_label == label]

    labels_present = {label for _, label in rows}
    if labels_present != {0, 1}:
        violations.append(f"both labels (0 and 1) must be present; found {sorted(labels_present)}.")
        return DatasetValidationReport(violations=tuple(violations))

    positive_count = sum(1 for _, label in rows if label == 1)
    negative_count = len(rows) - positive_count
    minority_fraction = min(positive_count, negative_count) / len(rows)
    if minority_fraction < _MIN_MINORITY_LABEL_FRACTION:
        violations.append(
            f"class balance is not reasonable: positive={positive_count}, negative={negative_count} "
            f"(minority fraction {minority_fraction:.2%} < {_MIN_MINORITY_LABEL_FRACTION:.0%})."
        )

    for label in (0, 1):
        for feature in ("satellite_low_count", "satellite_nominal_count", "satellite_high_count", "news_count"):
            if not any(value > 0 for value in values_where(label, feature)):
                violations.append(f"label={label} rows never have {feature} > 0.")

    for feature in ("satellite_low_count", "satellite_high_count", "news_count"):
        labels_with_positive_feature = {label for features, label in rows if features[index[feature]] > 0}
        if labels_with_positive_feature != {0, 1}:
            violations.append(
                f"{feature} > 0 must occur under both labels; occurs under {sorted(labels_with_positive_feature)} only."
            )

    def is_satellite_only(features: tuple[float, ...]) -> bool:
        return features[index["satellite_count"]] > 0 and features[index["news_count"]] == 0

    def is_multi_source(features: tuple[float, ...]) -> bool:
        return features[index["satellite_count"]] > 0 and features[index["news_count"]] > 0

    satellite_only_labels = {label for features, label in rows if is_satellite_only(features)}
    if satellite_only_labels != {0, 1}:
        violations.append(f"satellite-only rows must exist under both labels; found under {sorted(satellite_only_labels)} only.")

    multi_source_labels = {label for features, label in rows if is_multi_source(features)}
    if multi_source_labels != {0, 1}:
        violations.append(f"multi-source (satellite+news) rows must exist under both labels; found under {sorted(multi_source_labels)} only.")

    for feature_index, feature_name in enumerate(FIRE_DETECTION_FEATURE_NAMES):
        labels_positive = {label for features, label in rows if features[feature_index] > 0}
        labels_zero = {label for features, label in rows if features[feature_index] == 0}
        if len(labels_positive) == 1 and len(labels_zero) == 1 and labels_positive != labels_zero:
            violations.append(
                f"{feature_name} is an exact one-feature label separator "
                f"(feature > 0 <=> label={next(iter(labels_positive))})."
            )

    return DatasetValidationReport(violations=tuple(violations))

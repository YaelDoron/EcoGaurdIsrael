"""Generate the synthetic labeled Fire Detection ML training dataset (CSV).

Offline ML training-data generation only - does not touch Neon, does not
call FireDetectionCalculator, and does not integrate with FireDetectionAgent.

V2 introduces additional scenario families with mixed-confidence satellite
evidence (see fire_detection_training_data_generator.py) to fix a V1
generator artifact where satellite_low_count was 0 for every positive row.
Every generated dataset is validated (see fire_detection_dataset_validation)
before being written; generation fails loudly rather than writing a broken CSV.

Example:
    python -m scripts.generate_fire_detection_training_data
    python -m scripts.generate_fire_detection_training_data --samples 3000 --seed 7
    python -m scripts.generate_fire_detection_training_data --dataset-version v2
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.fire_detection.fire_detection_dataset_validation import validate_training_samples
from src.ml.fire_detection.fire_detection_feature_extractor import FireDetectionFeatureExtractor
from src.ml.fire_detection.fire_detection_features import (
    FIRE_DETECTION_FEATURE_NAMES,
    LABEL_COLUMN,
    TRAINING_DATA_CSV_COLUMNS,
)
from src.ml.fire_detection.fire_detection_training_data_generator import (
    DATASET_VERSION_V1,
    DATASET_VERSION_V2,
    DEFAULT_TRAINING_DATA_SAMPLE_COUNT,
    DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V2,
    DEFAULT_TRAINING_DATA_SEED,
    FireDetectionTrainingDataGenerator,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "fire_detection"
DEFAULT_OUTPUT_PATH = DATA_DIR / "training_v1.csv"

_VERSION_DEFAULTS = {
    DATASET_VERSION_V1: {"samples": DEFAULT_TRAINING_DATA_SAMPLE_COUNT, "output": DATA_DIR / "training_v1.csv"},
    DATASET_VERSION_V2: {"samples": DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V2, "output": DATA_DIR / "training_v2.csv"},
}


def generate_dataset(num_samples: int, seed: int, output_path: Path, dataset_version: str = DATASET_VERSION_V1) -> Path:
    """Generate num_samples labeled rows, validate them, and write output_path as CSV."""
    generator = FireDetectionTrainingDataGenerator(seed=seed)
    extractor = FireDetectionFeatureExtractor()
    samples = generator.generate(num_samples=num_samples, dataset_version=dataset_version)

    # V1 is intentionally preserved with its known generator artifact (see
    # fire_detection_ml.md) and is NOT expected to pass the stricter V2
    # sanity checks - that gap is exactly why V2 exists. Only V2 generation
    # is validated here.
    if dataset_version == DATASET_VERSION_V2:
        report = validate_training_samples(samples)
        if not report.is_valid:
            violations = "\n".join(f"  - {violation}" for violation in report.violations)
            raise ValueError(f"Generated dataset failed validation ({dataset_version}):\n{violations}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(TRAINING_DATA_CSV_COLUMNS)
        for sample in samples:
            features = extractor.extract(sample.evidence).as_dict()
            row = [sample.sample_id, sample.scenario_family]
            row.extend(features[name] for name in FIRE_DETECTION_FEATURE_NAMES)
            row.append(sample.label)
            writer.writerow(row)

    return output_path


def _summarize(output_path: Path) -> None:
    family_counts: dict[str, int] = {}
    positive = 0
    negative = 0
    with output_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            family_counts[row["scenario_family"]] = family_counts.get(row["scenario_family"], 0) + 1
            if int(row[LABEL_COLUMN]) == 1:
                positive += 1
            else:
                negative += 1

    print(f"Wrote {positive + negative} samples to {output_path}")
    print(f"  positive (active wildfire, label=1): {positive}")
    print(f"  negative (no active wildfire, label=0): {negative}")
    print("  scenario families:")
    for family_name in sorted(family_counts):
        print(f"    {family_name:<40} {family_counts[family_name]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Fire Detection ML training dataset (CSV).")
    parser.add_argument("--dataset-version", choices=list(_VERSION_DEFAULTS), default=DATASET_VERSION_V1)
    parser.add_argument("--samples", type=int, default=None, help="Defaults per --dataset-version if omitted.")
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_DATA_SEED)
    parser.add_argument("--output", type=Path, default=None, help="Defaults per --dataset-version if omitted.")
    args = parser.parse_args()

    version_defaults = _VERSION_DEFAULTS[args.dataset_version]
    num_samples = args.samples if args.samples is not None else version_defaults["samples"]
    output_path = args.output if args.output is not None else version_defaults["output"]

    output_path = generate_dataset(
        num_samples=num_samples, seed=args.seed, output_path=output_path, dataset_version=args.dataset_version
    )
    _summarize(output_path)


if __name__ == "__main__":
    main()

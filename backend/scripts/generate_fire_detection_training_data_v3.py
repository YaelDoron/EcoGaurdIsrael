"""Generate the synthetic labeled Fire Detection ML V3 training dataset (CSV).

Sibling to scripts.generate_fire_detection_training_data (V1/V2), which is
unchanged and not touched by this script. Offline ML training-data
generation only - no network calls, no LLM calls, no Neon dependency. Runs
Part 27 sanity validation before writing the CSV and fails loudly on
violation rather than writing a broken dataset.

Example:
    python -m scripts.generate_fire_detection_training_data_v3
    python -m scripts.generate_fire_detection_training_data_v3 --samples 5000 --seed 7
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.fire_detection.fire_detection_dataset_validation_v3 import validate_training_samples_v3
from src.ml.fire_detection.fire_detection_feature_extractor_v3 import FireDetectionFeatureExtractorV3
from src.ml.fire_detection.fire_detection_features_v3 import (
    FIRE_DETECTION_FEATURE_NAMES_V3,
    LABEL_COLUMN,
    TRAINING_DATA_CSV_COLUMNS_V3,
)
from src.ml.fire_detection.fire_detection_training_data_generator_v3 import (
    DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V3,
    DEFAULT_TRAINING_DATA_SEED_V3,
    FireDetectionTrainingDataGeneratorV3,
)

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "fire_detection" / "training_v3.csv"


def generate_dataset(num_samples: int, seed: int, output_path: Path) -> Path:
    """Generate num_samples labeled rows, validate them, and write output_path as CSV."""
    generator = FireDetectionTrainingDataGeneratorV3(seed=seed)
    extractor = FireDetectionFeatureExtractorV3()
    samples = generator.generate(num_samples=num_samples)

    report = validate_training_samples_v3(samples)
    if not report.is_valid:
        violations = "\n".join(f"  - {violation}" for violation in report.violations)
        raise ValueError(f"Generated V3 dataset failed validation:\n{violations}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(TRAINING_DATA_CSV_COLUMNS_V3)
        for sample in samples:
            features = extractor.extract(sample.evidence).as_dict()
            row = [sample.sample_id, sample.scenario_family, sample.scenario_archetype.value]
            row.extend(features[name] for name in FIRE_DETECTION_FEATURE_NAMES_V3)
            row.append(sample.label)
            writer.writerow(row)

    return output_path


def _summarize(output_path: Path) -> None:
    family_counts: dict[str, int] = {}
    archetype_counts: dict[str, int] = {}
    positive = 0
    negative = 0
    with output_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            family_counts[row["scenario_family"]] = family_counts.get(row["scenario_family"], 0) + 1
            archetype_counts[row["scenario_archetype"]] = archetype_counts.get(row["scenario_archetype"], 0) + 1
            if int(row[LABEL_COLUMN]) == 1:
                positive += 1
            else:
                negative += 1

    print(f"Wrote {positive + negative} samples to {output_path}")
    print(f"  positive (active wildfire, label=1): {positive}")
    print(f"  negative (no active wildfire, label=0): {negative}")
    print("  scenario families:")
    for family_name in sorted(family_counts):
        print(f"    {family_name:<45} {family_counts[family_name]}")
    print("  scenario archetypes:")
    for archetype_name in sorted(archetype_counts):
        print(f"    {archetype_name:<20} {archetype_counts[archetype_name]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Fire Detection ML V3 training dataset (CSV).")
    parser.add_argument("--samples", type=int, default=DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V3)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_DATA_SEED_V3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    output_path = generate_dataset(num_samples=args.samples, seed=args.seed, output_path=args.output)
    _summarize(output_path)


if __name__ == "__main__":
    main()

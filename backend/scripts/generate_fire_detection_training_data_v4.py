"""Generate the synthetic labeled Fire Detection ML V4 training dataset (CSV).

Sibling to scripts.generate_fire_detection_training_data_v3, which is untouched.
Offline ML dataset generation only - no network, no LLM, no Neon, no model
training. Labels come from scenario ground truth, never from the rule-based
detector (see backend/docs/fire_detection_dataset_v4.md). The dataset is
validated BEFORE it is written; a failing dataset is never saved.

Example:
    python -m scripts.generate_fire_detection_training_data_v4
    python -m scripts.generate_fire_detection_training_data_v4 --samples 6000 --seed 7 --output /tmp/v4.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.fire_detection.fire_detection_dataset_v4 import row_from_sample, write_training_dataset_v4
from src.ml.fire_detection.fire_detection_dataset_validation_v4 import validate_dataset_rows_v4
from src.ml.fire_detection.fire_detection_training_data_generator_v4 import (
    DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V4,
    DEFAULT_TRAINING_DATA_SEED_V4,
    FireDetectionTrainingDataGeneratorV4,
)

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "fire_detection" / "training_v4.csv"


def generate_dataset(num_samples: int, seed: int, output_path: Path) -> Path:
    """Generate, validate, and write the V4 dataset CSV; raises ValueError if validation fails."""
    samples = FireDetectionTrainingDataGeneratorV4(seed=seed).generate(num_samples=num_samples)
    report = validate_dataset_rows_v4(tuple(row_from_sample(sample) for sample in samples))
    if not report.is_valid:
        violations = "\n".join(f"  - {violation}" for violation in report.violations)
        raise ValueError(f"Generated V4 dataset failed validation:\n{violations}")
    return write_training_dataset_v4(samples, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Fire Detection ML V4 training dataset (CSV).")
    parser.add_argument("--samples", type=int, default=DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V4)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_DATA_SEED_V4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    output_path = generate_dataset(num_samples=args.samples, seed=args.seed, output_path=args.output)
    print(f"Wrote {args.samples} validated V4 samples (seed={args.seed}) to {output_path}")
    print("Next: python -m scripts.validate_fire_detection_dataset_v4  (prints and writes the full report)")


if __name__ == "__main__":
    main()

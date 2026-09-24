"""Task 8: (re)generate the untouched confirmatory dataset with the FROZEN V5 generator and a new seed.

Refuses to overwrite an existing file whose hash differs from the locked one, and verifies the result against the locked
SHA-256 and against training_v5.csv (no shared environment / sample / pair ids). No model is trained.

Example:
    python -m scripts.generate_fire_detection_policy_validation_data_v5
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.fire_detection.fire_detection_dataset_v5 import load_training_dataset_rows_v5
from src.ml.fire_detection.fire_detection_model_v5 import DEFAULT_TRAINING_CSV_V5
from src.ml.fire_detection.fire_detection_policy_config_v5 import CONFIRMATORY_CSV_SHA256, CONFIRMATORY_SEED
from src.ml.fire_detection.fire_detection_policy_validation_data_v5 import (
    DEFAULT_CONFIRMATORY_CSV,
    assert_disjoint,
    confirmatory_sha256,
    cross_dataset_overlap,
    load_confirmatory_rows,
    write_confirmatory_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the Task 8 confirmatory dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_CONFIRMATORY_CSV)
    args = parser.parse_args()

    write_confirmatory_dataset(args.output, CONFIRMATORY_SEED)
    sha = confirmatory_sha256(args.output)
    print(f"wrote {args.output} (seed {CONFIRMATORY_SEED}) sha256 {sha}")
    if sha != CONFIRMATORY_CSV_SHA256:
        print(f"WARNING: differs from the locked hash {CONFIRMATORY_CSV_SHA256}")
        return 1
    overlap = cross_dataset_overlap(load_training_dataset_rows_v5(DEFAULT_TRAINING_CSV_V5), load_confirmatory_rows(args.output))
    assert_disjoint(overlap)
    print(f"disjoint from training; exact feature-vector overlap {overlap['exact_feature_vector_overlap_rows']} rows (diagnostic only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

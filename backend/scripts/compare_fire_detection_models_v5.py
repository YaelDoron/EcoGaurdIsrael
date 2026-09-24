"""Task 7: compare Logistic Regression, Random Forest and Histogram Gradient Boosting on the frozen V5 dataset.

Verifies training_v5.csv against its expected SHA-256 BEFORE anything is trained, regenerates the (frozen, seed-42)
V5 samples so the rule baseline can score the same rows, runs the whole grouped-environment comparison and writes

    data/fire_detection/fire_detection_model_comparison_v5.json
    docs/fire_detection_model_comparison_v5.md

It saves NO model artifact (see scripts.train_fire_detection_model_v5, which refuses unless a model passed the
predefined gate). All data is synthetic. V5 is not integrated into runtime.

Example:
    python -m scripts.compare_fire_detection_models_v5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.fire_detection.fire_detection_model_comparison_v5 import run_model_comparison_v5
from src.ml.fire_detection.fire_detection_model_report_v5 import render_report_markdown_v5
from src.ml.fire_detection.fire_detection_model_v5 import DEFAULT_TRAINING_CSV_V5, load_training_matrix_v5, verify_frozen_dataset_v5
from src.ml.fire_detection.fire_detection_training_data_generator_v5 import (
    DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V5,
    DEFAULT_TRAINING_DATA_SEED_V5,
    FireDetectionTrainingDataGeneratorV5,
)

BACKEND = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_JSON = BACKEND / "data" / "fire_detection" / "fire_detection_model_comparison_v5.json"
DEFAULT_REPORT_MD = BACKEND / "docs" / "fire_detection_model_comparison_v5.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare V5 Fire Detection models (grouped-environment evaluation).")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_TRAINING_CSV_V5)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    started = time.time()
    sha256 = verify_frozen_dataset_v5(args.dataset)  # refuses any other file
    print(f"dataset verified: {args.dataset.name} sha256 {sha256}")
    matrix = load_training_matrix_v5(args.dataset)
    samples = FireDetectionTrainingDataGeneratorV5(DEFAULT_TRAINING_DATA_SEED_V5).generate(DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V5)
    report = run_model_comparison_v5(
        matrix, samples, csv_sha256=sha256, log=lambda message: print(f"[{time.time() - started:6.0f}s] {message}", flush=True)
    )
    args.json_output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_report_markdown_v5(report), encoding="utf-8")
    print(f"wrote {args.json_output} and {args.markdown_output}")
    print(f"SELECTED MODEL: {report['selection']['selected_model'] or 'NONE'} - {report['selection']['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

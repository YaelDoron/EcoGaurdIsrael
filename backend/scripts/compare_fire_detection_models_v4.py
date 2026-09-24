"""Train/evaluate/compare the Fire Detection ML V4 candidate models and write the comparison report.

Offline only: reads the FROZEN training_v4.csv, trains nothing that is saved (see
scripts/train_fire_detection_model_v4.py for artifacts), and never touches runtime
Fire Detection, V3 artifacts or .env. The rule-based FireDetectionCalculator is run
only as a baseline against ground-truth labels.

Example:
    python -m scripts.compare_fire_detection_models_v4
    python -m scripts.compare_fire_detection_models_v4 --output data/fire_detection/fire_detection_model_comparison_v4.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.fire_detection.fire_detection_model_comparison_v4 import run_model_comparison_v4
from src.ml.fire_detection.fire_detection_model_v4 import DEFAULT_TRAINING_CSV_V4, RANDOM_SEED_V4, load_training_matrix_v4
from src.ml.fire_detection.fire_detection_training_data_generator_v4 import (
    DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V4,
    DEFAULT_TRAINING_DATA_SEED_V4,
    FireDetectionTrainingDataGeneratorV4,
)

DEFAULT_REPORT_PATH = Path(__file__).resolve().parents[1] / "data" / "fire_detection" / "fire_detection_model_comparison_v4.json"


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_summary(report: dict) -> None:
    print(report["synthetic_data_notice"])
    rule = report["rule_baseline"]["suspected_or_confirmed_is_fire"]
    print(f"\n{'model':<34}{'grpF1':>7}{'grpAUC':>8}{'minAUC':>8}{'archAUC':>9}{'ECE':>7}  gate")
    print(
        f"{'Rule baseline (SUSPECTED+)':<34}{_fmt(rule['summary']['f1']['mean']):>7}"
        f"{_fmt(rule['summary']['roc_auc']['mean']):>8}{_fmt(rule['summary']['roc_auc']['min']):>8}"
    )
    for key, model in report["models"].items():
        grouped = model["grouped_family_cv"]["summary"]
        archetype = model["leave_one_archetype_out"]["summary"]
        gate = model["generalization_gate"]
        print(
            f"{model['display_name']:<34}{_fmt(grouped['f1']['mean']):>7}{_fmt(grouped['roc_auc']['mean']):>8}"
            f"{_fmt(grouped['roc_auc']['min']):>8}{_fmt(archetype['roc_auc']['mean']):>9}"
            f"{_fmt(model['calibration']['uncalibrated_out_of_fold']['expected_calibration_error']):>7}  "
            f"{'PASS' if gate['passed'] else 'FAIL: ' + ', '.join(gate['failed'])}"
        )
    print("\nSelection:", report["selection"]["selected_model"] or "NONE", "-", report["selection"]["reason"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Fire Detection ML V4 models.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_TRAINING_CSV_V4)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--no-sensitivity", action="store_true")
    args = parser.parse_args()

    started = time.time()
    matrix = load_training_matrix_v4(args.dataset)
    # The rule baseline needs each candidate's evidence, which the CSV does not store: regenerate the
    # dataset's samples deterministically (verified row-by-row against the CSV before use).
    samples = FireDetectionTrainingDataGeneratorV4(seed=DEFAULT_TRAINING_DATA_SEED_V4).generate(
        DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V4
    )
    report = run_model_comparison_v4(
        matrix, samples, csv_path=args.dataset, include_sensitivity=not args.no_sensitivity
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print_summary(report)
    print(f"\nseed={RANDOM_SEED_V4}  elapsed={time.time() - started:.0f}s  report -> {args.output}")


if __name__ == "__main__":
    main()

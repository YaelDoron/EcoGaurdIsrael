"""Validate the Fire Detection ML V4 dataset and print/write a statistics report.

Read-only auditing of training_v4.csv (plus one JSON report file): distribution,
missingness, Fire Danger band x label, feature/target shortcut diagnostics,
duplicates, and the grouped-evaluation layout. Trains nothing.

Example:
    python -m scripts.validate_fire_detection_dataset_v4
    python -m scripts.validate_fire_detection_dataset_v4 --dataset data/fire_detection/training_v4.csv --no-write
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.fire_detection.fire_detection_dataset_v4 import load_training_dataset_rows_v4
from src.ml.fire_detection.fire_detection_dataset_validation_v4 import (
    build_dataset_report_v4,
    validate_dataset_rows_v4,
)
from src.ml.fire_detection.fire_detection_features_v4 import FIRE_DANGER_BANDS

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "fire_detection"
DEFAULT_DATASET_PATH = DATA_DIR / "training_v4.csv"
DEFAULT_REPORT_PATH = DATA_DIR / "training_v4_validation_report.json"
MIN_ROWS = 5000


def _pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def print_report(report: dict, violations: tuple[str, ...]) -> None:
    basic = report["basic"]
    print("=== Basic distribution ===")
    print(
        f"rows={basic['rows']}  fire={basic['positive_rows']}  no-fire={basic['negative_rows']}  "
        f"fire%={basic['positive_percent']}  scenario_families={basic['scenario_families']}"
    )
    print(f"{'family':<50}{'label':<7}{'archetype':<18}rows")
    for name, entry in basic["rows_per_family"].items():
        print(f"{name:<50}{entry['label']:<7}{entry['archetype']:<18}{entry['rows']}")
    print("archetype x label:", basic["archetype_label_counts"])

    missing = report["missingness"]
    print("\n=== Missingness ===")
    print(
        f"Fire Danger unavailable: {_pct(missing['fire_danger_unavailable_rate'])} "
        f"(no-fire {_pct(missing['fire_danger_unavailable_rate_by_label']['no_fire'])}, "
        f"fire {_pct(missing['fire_danger_unavailable_rate_by_label']['fire'])})"
    )
    print(
        f"FRP missing (rows with satellite, n={missing['rows_with_satellite']}): "
        f"{_pct(missing['frp_missing_rate_among_satellite_rows'])}; brightness missing: "
        f"{_pct(missing['brightness_missing_rate_among_satellite_rows'])}"
    )
    print(
        f"news analysis unavailable (rows with news, n={missing['rows_with_news']}): "
        f"{_pct(missing['news_analysis_unavailable_rate_among_news_rows'])}"
    )

    print("\n=== Fire Danger band x label ===")
    print(f"{'band':<11}{'no-fire':>9}{'fire':>8}{'fire share':>12}")
    for band in FIRE_DANGER_BANDS:
        cell = report["fire_danger_band_by_label"][band]
        print(f"{band:<11}{cell['no_fire']:>9}{cell['fire']:>8}{_pct(cell['fire_share']):>12}")

    print("\n=== Feature / target diagnostics (association only, not causation) ===")
    print(f"{'feature':<40}{'mean(no-fire)':>15}{'mean(fire)':>13}{'pearson r':>11}{'AUC':>8}")
    for item in report["univariate_feature_diagnostics"]:
        mean0 = "n/a" if item["mean_no_fire"] is None else f"{item['mean_no_fire']:.3f}"
        mean1 = "n/a" if item["mean_fire"] is None else f"{item['mean_fire']:.3f}"
        pearson = "n/a" if item["pearson_r"] is None else f"{item['pearson_r']:+.3f}"
        flag = "  <-- SHORTCUT" if item["suspicious_shortcut"] else ""
        print(f"{item['feature']:<40}{mean0:>15}{mean1:>13}{pearson:>11}{item['univariate_auc']:>8.3f}{flag}")

    difficulty = report["shape_and_difficulty"]
    print("\n=== Difficulty ===")
    print(f"hard negatives (fire-looking evidence, no fire): {_pct(difficulty['hard_negative_fraction_of_negatives'])} of negatives")
    print(f"weak/incomplete-evidence fires: {_pct(difficulty['weak_positive_fraction_of_positives'])} of fires")
    print(f"single-item candidates that are fires: {_pct(difficulty['single_item_candidate_fire_share'])}")
    print(f"candidates spread over >3 km that are fires: {_pct(difficulty['candidates_over_3km_fire_share'])}")

    print("\n=== Time of day (fire share per 4h bucket) ===")
    print("  ".join(f"{bucket}: {_pct(entry['fire_share'])}" for bucket, entry in report["time_of_day_fire_share"].items()))

    duplicates = report["duplicates"]
    print("\n=== Duplicates ===")
    print(
        f"unique feature rows={duplicates['unique_feature_rows']} of {duplicates['rows']}  "
        f"duplicate rate={_pct(duplicates['duplicate_rate'])}  "
        f"vectors appearing under both labels={duplicates['vectors_with_both_labels']}"
    )

    grouped = report["grouped_evaluation"]
    print("\n=== Grouped evaluation layout ===")
    print("scenario_family folds (5, label-balanced, whole families held out):")
    for fold, entry in grouped["family_folds_5"].items():
        if fold == "error":
            print(f"  ERROR {entry}")
        else:
            print(f"  fold {fold}: fire_rows={entry['fire_rows']} no_fire_rows={entry['no_fire_rows']} families={len(entry['families'])}")
    print("leave-one-archetype-out:")
    for archetype, entry in grouped["leave_one_archetype_out"].items():
        print(f"  hold out {archetype}: validation_rows={entry['validation_rows']} (fire {entry['validation_fire_rows']})")

    print("\n=== Validation ===")
    if violations:
        for violation in violations:
            print(f"  VIOLATION: {violation}")
    else:
        print("  all checks passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Fire Detection ML V4 dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--no-write", action="store_true", help="Print only; do not write the JSON report.")
    args = parser.parse_args()

    rows = load_training_dataset_rows_v4(args.dataset)
    report = build_dataset_report_v4(rows)
    violations = validate_dataset_rows_v4(rows, min_rows=MIN_ROWS).violations
    print_report(report, violations)

    if not args.no_write:
        args.report_output.write_text(
            json.dumps({"dataset": args.dataset.name, "violations": list(violations), **report}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote report to {args.report_output}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())

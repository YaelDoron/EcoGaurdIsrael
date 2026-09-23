"""Analyze the Fire Detection ML V1 dataset (training_v1.csv) before model comparison.

Read-only dataset auditing: per-feature statistics by class, per-scenario-family
statistics, and a feature correlation matrix. Does not train or evaluate any
model, and does not modify training_v1.csv.

Example:
    python -m scripts.analyze_fire_detection_dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.fire_detection.fire_detection_dataset import load_training_dataset_rows
from src.ml.fire_detection.fire_detection_dataset_statistics import (
    compute_feature_correlation_matrix,
    compute_feature_statistics_by_label,
    compute_scenario_family_statistics,
    satellite_count_identity_holds,
    scenario_family_perfectly_predicts_label,
)
from src.ml.fire_detection.fire_detection_features import (
    FIRE_DETECTION_FEATURE_NAMES,
    LABEL_COLUMN,
    SAMPLE_ID_COLUMN,
    SCENARIO_FAMILY_COLUMN,
)

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "fire_detection" / "training_v1.csv"

_SUSPICIOUS_FEATURES = ("max_pairwise_distance_km", "time_span_minutes", "satellite_count")


def _print_dataset_overview(rows) -> None:
    positive = sum(1 for row in rows if row.label == 1)
    negative = len(rows) - positive
    print("=== Dataset overview ===")
    print(f"Total rows: {len(rows)}  positive(label=1): {positive}  negative(label=0): {negative}")
    print(
        f"Metadata columns excluded from ML features: {SAMPLE_ID_COLUMN!r}, {SCENARIO_FAMILY_COLUMN!r} "
        f"(never present in FIRE_DETECTION_FEATURE_NAMES: {SAMPLE_ID_COLUMN in FIRE_DETECTION_FEATURE_NAMES or SCENARIO_FAMILY_COLUMN in FIRE_DETECTION_FEATURE_NAMES})"
    )


def _print_feature_statistics(rows) -> None:
    print("\n=== Feature statistics by class (Part 2) ===")
    stats = compute_feature_statistics_by_label(rows)
    by_feature: dict[str, list] = {}
    for stat in stats:
        by_feature.setdefault(stat.feature_name, []).append(stat)

    for feature_name in FIRE_DETECTION_FEATURE_NAMES:
        flag = "  <-- inspected in Part 3 (unintuitive LR coefficient)" if feature_name in _SUSPICIOUS_FEATURES else ""
        print(f"\nFeature: {feature_name}{flag}")
        for stat in sorted(by_feature[feature_name], key=lambda item: item.label):
            print(
                f"  label={stat.label}  count={stat.count:<5} "
                f"mean={stat.mean:.4f}  median={stat.median:.4f}  std={stat.std:.4f}  "
                f"min={stat.minimum:.4f}  max={stat.maximum:.4f}"
            )


def _print_scenario_family_statistics(rows) -> None:
    print("\n=== Scenario family statistics (Part 4, metadata-only auditing) ===")
    family_stats = compute_scenario_family_statistics(rows)
    for stat in family_stats:
        feature_summary = ", ".join(f"{name}={stat.mean_features[name]:.2f}" for name in FIRE_DETECTION_FEATURE_NAMES)
        print(f"{stat.scenario_family:<40} label={stat.label}  count={stat.count:<5}  mean[{feature_summary}]")

    leakage_safe = scenario_family_perfectly_predicts_label(rows)
    print(
        f"\nscenario_family perfectly determines label per-family: {leakage_safe} "
        "(expected - this is why scenario_family/sample_id must stay metadata-only and are "
        f"never in FIRE_DETECTION_FEATURE_NAMES: {FIRE_DETECTION_FEATURE_NAMES})"
    )


def _print_correlation_matrix(rows) -> None:
    print("\n=== Feature correlation matrix (Part 5) ===")
    matrix = compute_feature_correlation_matrix(rows)
    header = "".join(f"{name[:14]:>16}" for name in FIRE_DETECTION_FEATURE_NAMES)
    print(f"{'':22}{header}")
    for first_name in FIRE_DETECTION_FEATURE_NAMES:
        row_values = "".join(f"{matrix[first_name][second_name]:>16.3f}" for second_name in FIRE_DETECTION_FEATURE_NAMES)
        print(f"{first_name:<22}{row_values}")

    identity_holds = satellite_count_identity_holds(rows)
    print(
        f"\nsatellite_count == satellite_low_count + satellite_nominal_count + satellite_high_count "
        f"for every row: {identity_holds} (structural/exact dependency, not just statistical correlation - "
        "this is the primary source of Logistic Regression multicollinearity between these four features; "
        "see backend/docs/fire_detection_ml.md)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Fire Detection ML V1 dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    args = parser.parse_args()

    rows = load_training_dataset_rows(args.dataset)

    _print_dataset_overview(rows)
    _print_feature_statistics(rows)
    _print_scenario_family_statistics(rows)
    _print_correlation_matrix(rows)


if __name__ == "__main__":
    main()

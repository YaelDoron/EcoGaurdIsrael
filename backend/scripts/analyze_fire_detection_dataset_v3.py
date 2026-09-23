"""Univariate shortcut analysis (Part 28) and V2-vs-V3 feature-vector collision
analysis (Part 29), including a targeted look at the V2 "twin family" pairs.

Read-only dataset auditing - does not train or evaluate any model, and does
not modify training_v2.csv/training_v3.csv.

Example:
    python -m scripts.analyze_fire_detection_dataset_v3
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.fire_detection.fire_detection_dataset import load_training_dataset_rows
from src.ml.fire_detection.fire_detection_dataset_v3 import load_training_dataset_rows_v3
from src.ml.fire_detection.fire_detection_features import FIRE_DETECTION_FEATURE_NAMES
from src.ml.fire_detection.fire_detection_features_v3 import FIRE_DETECTION_FEATURE_NAMES_V3
from src.ml.fire_detection.fire_detection_shortcut_analysis import (
    compute_feature_vector_collisions,
    compute_univariate_shortcut_analysis,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "fire_detection"
DEFAULT_V2_PATH = DATA_DIR / "training_v2.csv"
DEFAULT_V3_PATH = DATA_DIR / "training_v3.csv"

_TWIN_PAIRS = (
    ("news_only_before_satellite", "news_rumor_no_fire"),
    ("satellite_only_before_news", "high_confidence_satellite_false_positive"),
)


def _print_univariate_analysis(rows_v3) -> None:
    features = [list(row.features) for row in rows_v3]
    labels = [row.label for row in rows_v3]
    results = compute_univariate_shortcut_analysis(features, labels, FIRE_DETECTION_FEATURE_NAMES_V3)

    print("=== Univariate shortcut analysis (V3 features, Part 28) ===")
    print(f"{'feature':<38}{'mean(0)':<10}{'mean(1)':<10}{'AUC':<8}{'flag'}")
    for stat in sorted(results, key=lambda item: item.auc, reverse=True):
        flag = "SUSPICIOUS (>=0.95)" if stat.is_suspicious_shortcut else ""
        print(f"{stat.feature_name:<38}{stat.mean_label_0:<10.3f}{stat.mean_label_1:<10.3f}{stat.auc:<8.3f}{flag}")


def _print_collision_analysis(rows_v2, rows_v3) -> None:
    print("\n=== Feature-vector collision analysis (Part 29) ===")
    for version_name, rows, feature_names in (("V2", rows_v2, FIRE_DETECTION_FEATURE_NAMES), ("V3", rows_v3, FIRE_DETECTION_FEATURE_NAMES_V3)):
        features = [list(row.features) for row in rows]
        labels = [row.label for row in rows]
        result = compute_feature_vector_collisions(features, labels)
        print(
            f"{version_name}: total_rows={result.total_rows}  distinct_vectors={result.distinct_feature_vectors}  "
            f"colliding_vectors={result.colliding_vectors}  colliding_rows={result.colliding_rows}  "
            f"collision_rate={result.collision_rate:.2%}"
        )


def _print_twin_family_analysis(rows_v2, rows_v3) -> None:
    print("\n=== Targeted twin-family analysis (Part 20/29) ===")
    for version_name, rows, feature_names in (("V2", rows_v2, FIRE_DETECTION_FEATURE_NAMES), ("V3", rows_v3, FIRE_DETECTION_FEATURE_NAMES_V3)):
        print(f"\n{version_name}:")
        for family_a, family_b in _TWIN_PAIRS:
            vectors_a = {tuple(row.features) for row in rows if row.scenario_family == family_a}
            vectors_b = {tuple(row.features) for row in rows if row.scenario_family == family_b}
            shared = vectors_a & vectors_b
            print(
                f"  {family_a} vs {family_b}: "
                f"distinct_vectors_a={len(vectors_a)}  distinct_vectors_b={len(vectors_b)}  "
                f"shared_exact_vectors={len(shared)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 vs V3 shortcut/collision analysis.")
    parser.add_argument("--v2-dataset", type=Path, default=DEFAULT_V2_PATH)
    parser.add_argument("--v3-dataset", type=Path, default=DEFAULT_V3_PATH)
    args = parser.parse_args()

    rows_v2 = load_training_dataset_rows(args.v2_dataset)
    rows_v3 = load_training_dataset_rows_v3(args.v3_dataset)

    _print_univariate_analysis(rows_v3)
    _print_collision_analysis(rows_v2, rows_v3)
    _print_twin_family_analysis(rows_v2, rows_v3)


if __name__ == "__main__":
    main()

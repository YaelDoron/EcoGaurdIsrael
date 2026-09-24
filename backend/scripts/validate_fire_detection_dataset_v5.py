"""Validate the Fire Detection ML V5 dataset and print/write a statistics report.

Read-only auditing of training_v5.csv (plus one JSON report file): regime x label balance, latent subtypes,
environments and pairs, univariate feature AUCs (shortcut flags), history / day-night / news distributions,
duplicates, and the grouped-evaluation layout for Task 7. Trains nothing and never edits the generator.

Example:
    python -m scripts.validate_fire_detection_dataset_v5
    python -m scripts.validate_fire_detection_dataset_v5 --dataset data/fire_detection/training_v5.csv --no-write
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.fire_detection.fire_detection_dataset_v5 import dataset_sha256, load_training_dataset_rows_v5
from src.ml.fire_detection.fire_detection_dataset_validation_v5 import (
    SUSPICIOUS_UNIVARIATE_AUC,
    WARNING_UNIVARIATE_AUC,
    build_dataset_report_v5,
    validate_dataset_rows_v5,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "fire_detection"
DEFAULT_DATASET_PATH = DATA_DIR / "training_v5.csv"
DEFAULT_REPORT_PATH = DATA_DIR / "training_v5_validation_report.json"
MIN_ROWS = 5000


def _pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _table(title: str, table: dict, columns=("fire", "no_fire", "total", "fire_share")) -> None:
    print(f"\n=== {title} ===")
    width = max(len(str(name)) for name in table) + 2
    print(f"{'':<{width}}" + "".join(f"{column:>11}" for column in columns))
    for name, cell in table.items():
        print(
            f"{name:<{width}}"
            + "".join(f"{_pct(cell[c]) if c == 'fire_share' else cell[c]:>11}" for c in columns)
        )


def print_report(report: dict, violations: tuple[str, ...], sha256: str | None) -> None:
    basic = report["basic"]
    print("=== Basic distribution ===")
    print(
        f"rows={basic['rows']}  fire={basic['positive_rows']}  no-fire={basic['negative_rows']}  "
        f"fire%={basic['positive_percent']}  seeds={basic['seeds']}"
    )
    if sha256:
        print(f"dataset sha256: {sha256}")

    _table("Regime x label (each regime must hold both labels, ~50/50)", report["regimes"])
    print(f"regime-only AUC (predict label from regime alone): {report['regime_only_auc']:.3f}  (0.5 = regime carries no label)")

    subtypes = report["latent_subtypes"]
    print("\n=== Latent subtypes (analysis metadata, never a feature) ===")
    for name, entry in subtypes["rows_per_subtype"].items():
        print(f"  {name:<34}{'fire' if entry['label'] else 'no-fire':<9}{entry['rows']}")
    print("subtypes per regime:")
    for regime, counts in subtypes["subtypes_per_regime"].items():
        print(f"  {regime:<30}" + ", ".join(f"{name}={count}" for name, count in counts.items()))

    env = report["environments"]
    print("\n=== Environments ===")
    print(
        f"environments={env['environments']}  rows/environment min={env['rows_per_environment_min']} "
        f"median={env['rows_per_environment_median']} max={env['rows_per_environment_max']}  "
        f"with both labels={_pct(env['both_label_fraction'])}  "
        f"fire share per environment {_pct(env['environment_fire_share_min'])}..{_pct(env['environment_fire_share_max'])}"
    )

    pairs = report["pairs"]
    print("\n=== Pairs ===")
    print(f"pairs={pairs['pairs']}  same environment={pairs['same_environment']}  same regime={pairs['same_regime']}  "
          f"identical feature vectors={_pct(pairs['identical_feature_vector_fraction'])}")
    for name, count in pairs["pairs_per_type"].items():
        print(f"  {name:<66}{count}")
    print("pair-level overlap (fire member vs no-fire member):")
    print(f"  {'feature':<36}{'pairs':>7}{'fire>nofire':>13}{'ties':>8}{'within 0.5sd':>14}")
    for name, entry in pairs["key_feature_overlap"].items():
        rate = "n/a" if entry["fire_greater_win_rate"] is None else f"{entry['fire_greater_win_rate']:.2f}"
        print(f"  {name:<36}{entry['comparable_pairs']:>7}{rate:>13}{_pct(entry['tie_fraction']):>8}{_pct(entry['within_half_std_fraction']):>14}")

    print("\n=== Univariate feature AUC (best orientation; association only) ===")
    print(f"{'feature':<40}{'present':>9}{'mean(no-fire)':>15}{'mean(fire)':>13}{'AUC':>8}")
    for item in report["univariate_feature_diagnostics"]:
        mean0 = "n/a" if item["mean_no_fire"] is None else f"{item['mean_no_fire']:.3f}"
        mean1 = "n/a" if item["mean_fire"] is None else f"{item['mean_fire']:.3f}"
        flag = "  <-- SHORTCUT" if item["suspicious_shortcut"] else "  <-- warning" if item["warning"] else ""
        print(f"{item['feature']:<40}{item['rows_present']:>9}{mean0:>15}{mean1:>13}{item['univariate_auc']:>8.3f}{flag}")
    print(f"(shortcut threshold {SUSPICIOUS_UNIVARIATE_AUC}, warning threshold {WARNING_UNIVARIATE_AUC})")

    print("\n=== Strongest per-regime univariate AUCs ===")
    for regime, table in report["per_regime_univariate_auc"].items():
        top = list(table.items())[:4]
        print(f"  {regime:<30}" + ", ".join(f"{name}={auc:.2f}" for name, auc in top))

    history = report["history"]
    _table("Satellite pass count x label", history["pass_count_by_label"])
    _table("History span x label", history["history_span_by_label"])
    _table("History feature availability x label", history["history_feature_availability_by_label"])
    _table("Day/night x label (satellite_night_fraction)", report["day_night"])
    _table("News strongest signal x label", report["news"])
    print("\n=== Time of day (fire share per 4h bucket) ===")
    print("  ".join(f"{bucket}: {_pct(entry['fire_share'])}" for bucket, entry in report["time_of_day_fire_share"].items()))
    _table("Anti-shortcut coverage slices (both labels must appear)", report["coverage_slices"])

    duplicates = report["duplicates"]
    print("\n=== Duplicates ===")
    for name in ("overall", "rows_with_satellite", "news_only_rows"):
        entry = duplicates[name]
        print(
            f"{name:<20}rows={entry['rows']:<6} unique={entry['unique_feature_rows']:<6} duplicate rate={_pct(entry['duplicate_rate']):<7} "
            f"vectors under both labels={entry['vectors_with_both_labels']}"
        )
    near = duplicates["near_duplicates"]
    print(
        f"near-duplicates among satellite rows (NN distance < {near.get('near_duplicate_distance_std')} sd): "
        f"{_pct(near['near_duplicate_rate'])}, of which cross-label {_pct(near['cross_label_near_duplicate_rate'])}"
    )

    grouped = report["grouped_evaluation"]
    print("\n=== Grouped evaluation layout (Task 7) ===")
    print("environment folds (5; whole environments held out):")
    for fold, entry in grouped["environment_folds_5"].items():
        print(f"  fold {fold}: rows={entry['rows']} environments={entry['environments']} fire_rows={entry['fire_rows']}")
    print("leave-one-regime-out:")
    for regime, entry in grouped["leave_one_regime_out"].items():
        print(f"  hold out {regime}: validation_rows={entry['validation_rows']} (fire {entry['validation_fire_rows']})")
    print("leave-one-no-fire-subtype-out (validation = the held-out no-fire rows):")
    for subtype, entry in grouped["leave_one_no_fire_subtype_out"].items():
        print(f"  hold out {subtype}: validation_rows={entry['validation_rows']}")

    print("\n=== Validation ===")
    if violations:
        for violation in violations:
            print(f"  VIOLATION: {violation}")
    else:
        print("  all checks passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Fire Detection ML V5 dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--no-write", action="store_true", help="Print only; do not write the JSON report.")
    args = parser.parse_args()

    rows = load_training_dataset_rows_v5(args.dataset)
    report = build_dataset_report_v5(rows)
    violations = validate_dataset_rows_v5(rows, min_rows=MIN_ROWS).violations
    sha256 = dataset_sha256(args.dataset)
    print_report(report, violations, sha256)

    if not args.no_write:
        args.report_output.write_text(
            json.dumps({"dataset": args.dataset.name, "sha256": sha256, "violations": list(violations), **report}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote report to {args.report_output}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())

"""Task 8: confirmatory evaluation of the locked AI Hybrid policy (offline; not wired to runtime).

Steps: verify BOTH datasets against their locked SHA-256 -> check training/confirmatory disjointness -> train the locked HGB
ONCE on training_v5.csv -> score the untouched confirmatory rows -> apply the locked policy and gate -> write the JSON report
and the Markdown report -> save a model artifact ONLY if the policy gate passed.

Example:
    python -m scripts.evaluate_fire_detection_ai_hybrid_policy_v5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.fire_detection.fire_detection_dataset_v5 import load_training_dataset_rows_v5
from src.ml.fire_detection.fire_detection_model_artifact_v5 import DEFAULT_MODEL_DIR_V5
from src.ml.fire_detection.fire_detection_model_v5 import (
    DEFAULT_TRAINING_CSV_V5,
    load_training_matrix_v5,
    matrix_from_rows,
    verify_frozen_dataset_v5,
)
from src.ml.fire_detection.fire_detection_policy_config_v5 import CONFIRMATORY_CSV_SHA256, CONFIRMATORY_SEED, TRAINING_CSV_SHA256
from src.ml.fire_detection.fire_detection_policy_evaluation_v5 import (
    ArtifactRefusedError,
    POLICY_PASSED,
    evaluate_policy_v5,
    reload_parity,
    save_policy_artifact,
    train_locked_model,
)
from src.ml.fire_detection.fire_detection_policy_report_v5 import render_policy_report_markdown
from src.ml.fire_detection.fire_detection_policy_validation_data_v5 import (
    DEFAULT_CONFIRMATORY_CSV,
    assert_disjoint,
    cross_dataset_overlap,
    generate_confirmatory_samples,
    load_confirmatory_rows,
)

BACKEND = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_JSON = BACKEND / "data" / "fire_detection" / "fire_detection_v5_policy_validation_report.json"
DEFAULT_REPORT_MD = BACKEND / "docs" / "fire_detection_ai_hybrid_policy_v5.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Confirmatory evaluation of the Task 8 AI Hybrid policy.")
    parser.add_argument("--confirmatory", type=Path, default=DEFAULT_CONFIRMATORY_CSV)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR_V5)
    args = parser.parse_args()

    training_sha = verify_frozen_dataset_v5(DEFAULT_TRAINING_CSV_V5, TRAINING_CSV_SHA256)
    confirmatory_sha = verify_frozen_dataset_v5(args.confirmatory, CONFIRMATORY_CSV_SHA256)
    print(f"training  {DEFAULT_TRAINING_CSV_V5.name}: {training_sha}\nconfirmatory {args.confirmatory.name}: {confirmatory_sha}")

    training_matrix = load_training_matrix_v5()
    confirmatory_rows = load_confirmatory_rows(args.confirmatory)
    overlap = cross_dataset_overlap(load_training_dataset_rows_v5(DEFAULT_TRAINING_CSV_V5), confirmatory_rows)
    assert_disjoint(overlap)  # before ANY evaluation
    print(f"disjoint: 0 shared environment / sample / pair ids; exact feature-vector overlap {overlap['exact_feature_vector_overlap_rows']} rows")

    model = train_locked_model(training_matrix)  # the ONE training run
    report = evaluate_policy_v5(
        model, matrix_from_rows(confirmatory_rows), generate_confirmatory_samples(CONFIRMATORY_SEED),
        training_csv_sha256=training_sha, confirmatory_csv_sha256=confirmatory_sha, training_rows=len(training_matrix.y), overlap=overlap,
    )
    args.json_output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_policy_report_markdown(report), encoding="utf-8")
    print(f"wrote {args.json_output} and {args.markdown_output}")
    for name, c in report["policy_gate"]["criteria"].items():
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {name}: {c['value']}  ({c['requirement']})")
    print(report["verdict"])

    if report["verdict"] == POLICY_PASSED:
        try:
            model_path, metadata_path = save_policy_artifact(report, model, DEFAULT_TRAINING_CSV_V5, args.confirmatory, args.model_dir)
        except ArtifactRefusedError as exc:
            print(f"NO ARTIFACT SAVED: {exc}")
            return 1
        import joblib

        parity = reload_parity(model, joblib.load(model_path), confirmatory_matrix := matrix_from_rows(confirmatory_rows))
        print(f"saved {model_path} and {metadata_path}; reload parity max |dP| per case: {parity}")
        return 0 if all(v == 0.0 for v in parity.values()) else 1
    print("NO ARTIFACT SAVED: the AI hybrid policy did not pass its gate.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

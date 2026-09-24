"""Task 7: save the V5 model artifact - only if the comparison selected one.

Reads data/fire_detection/fire_detection_model_comparison_v5.json (produced by scripts.compare_fire_detection_models_v5).
If that report selected NONE (no model passed the predefined acceptance gate) this script saves nothing and exits 1.
It never touches V3 / V4 files and never overwrites an existing V5 artifact. V5 is not wired into runtime.

Example:
    python -m scripts.train_fire_detection_model_v5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.fire_detection.fire_detection_model_artifact_v5 import (
    DEFAULT_MODEL_DIR_V5,
    ArtifactRefusedError,
    save_selected_model_v5,
)
from src.ml.fire_detection.fire_detection_model_v5 import DEFAULT_TRAINING_CSV_V5, load_training_matrix_v5

DEFAULT_REPORT_JSON = Path(__file__).resolve().parents[1] / "data" / "fire_detection" / "fire_detection_model_comparison_v5.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Save the selected V5 Fire Detection model (only if the gate passed).")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_TRAINING_CSV_V5)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR_V5)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(f"selected model: {report['selection']['selected_model'] or 'NONE'} - {report['selection']['reason']}")
    try:
        model_path, metadata_path = save_selected_model_v5(report, load_training_matrix_v5(args.dataset), args.dataset, args.model_dir)
    except ArtifactRefusedError as exc:
        print(f"NO ARTIFACT SAVED: {exc}")
        return 1
    print(f"saved {model_path} and {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Save the Fire Detection ML V4 model artifact - ONLY for a model that passed the generalisation gate.

Reads the comparison report written by scripts.compare_fire_detection_models_v4. If the report
selected no model, this script refuses (exit code 2) and writes nothing: a runtime-ready artifact is
never created just to complete a task. V3 artifacts are never modified.

Example:
    python -m scripts.train_fire_detection_model_v4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.fire_detection.fire_detection_model_artifact_v4 import save_model_artifact_v4
from src.ml.fire_detection.fire_detection_model_v4 import DEFAULT_TRAINING_CSV_V4, load_training_matrix_v4

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = BACKEND_ROOT / "data" / "fire_detection" / "fire_detection_model_comparison_v4.json"
DEFAULT_MODELS_DIR = BACKEND_ROOT / "models" / "fire_detection"


def main() -> int:
    parser = argparse.ArgumentParser(description="Save the selected Fire Detection ML V4 model artifact.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_TRAINING_CSV_V4)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    selected = report["selection"]["selected_model"]
    if selected is None:
        print(f"REFUSED: {report['selection']['reason']}")
        print("No artifact written.")
        return 2

    matrix = load_training_matrix_v4(args.dataset)
    model_path, metadata_path = save_model_artifact_v4(selected, matrix, report, args.models_dir, args.dataset)
    print(f"Saved {model_path}\nSaved {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

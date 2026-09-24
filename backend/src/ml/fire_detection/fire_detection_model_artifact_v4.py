"""Save / load a Fire Detection ML V4 model artifact (joblib pipeline + metadata JSON).

Deliberately gated: an artifact is written ONLY for the model the comparison report selected
after passing the generalisation gate. If the report selected nothing, saving raises - a
"runtime-ready" file is never produced just to complete a task. V3 artifacts are never touched
(V4 files are named `fire_detection_<model>_v4.*` and V3 names are refused).

Nothing here is imported by FireDetectionAgent or any runtime path. Wiring a V4 model into
runtime is a later task.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import joblib
import sklearn

from src.ml.fire_detection.fire_detection_features_v4 import FEATURE_COUNT_V4, validate_feature_names_v4
from src.ml.fire_detection.fire_detection_model_v4 import (
    DATASET_VERSION_V4,
    FEATURE_SCHEMA_VERSION_V4,
    MODEL_SPECS_V4,
    MODEL_VERSION_V4,
    RANDOM_SEED_V4,
    TrainingMatrixV4,
    build_pipeline_v4,
    dataset_sha256,
    describe_preprocessing_v4,
)

_STEM_NAMES = {
    "logistic_regression": "logistic",
    "random_forest": "random_forest",
    "hist_gradient_boosting": "hist_gradient_boosting",
}


def artifact_stem(model_key: str) -> str:
    """fire_detection_<model>_v4 - always carries the _v4 suffix, so it can never collide with a V3 artifact."""
    return f"fire_detection_{_STEM_NAMES[model_key]}_v4"


def build_metadata_v4(model_key: str, matrix: TrainingMatrixV4, report: dict, csv_path: Path, created_at: datetime) -> dict:
    """Metadata for a selected model, assembled from the comparison report (no metric is recomputed here)."""
    model_report = report["models"][model_key]
    recommendation = model_report["thresholds"]["recommendation"]
    calibration = model_report["calibration"]
    variants = calibration.get("calibrated_variants_nested_grouped", {})
    return {
        "model_version": MODEL_VERSION_V4,
        "model_type": type(build_pipeline_v4(model_key).named_steps["classifier"]).__name__,
        "model_key": model_key,
        "model_name": artifact_stem(model_key),
        "feature_schema_version": FEATURE_SCHEMA_VERSION_V4,
        "feature_names": list(matrix.feature_names),
        "feature_count": FEATURE_COUNT_V4,
        "training_dataset_version": DATASET_VERSION_V4,
        "training_dataset_file": Path(csv_path).name,
        "training_dataset_hash": dataset_sha256(csv_path),
        "training_rows": int(len(matrix.y)),
        "preprocessing": describe_preprocessing_v4(model_key, matrix.feature_names),
        "grouped_cv_metrics": model_report["grouped_family_cv"]["summary"],
        "archetype_metrics": model_report["leave_one_archetype_out"]["summary"],
        "suspect_threshold_candidate": recommendation["suspect_threshold_candidate"],
        "confirm_threshold_candidate": recommendation["confirm_threshold_candidate"],
        "calibration": {
            "uncalibrated_out_of_fold": {
                key: calibration["uncalibrated_out_of_fold"][key]
                for key in ("brier_score", "expected_calibration_error", "brier_skill_score")
            },
            "calibration_applied": False,
            "calibrated_variants_assessment": variants.get("assessment"),
        },
        "generalization_gate": model_report["generalization_gate"],
        "selection_reason": report["selection"]["reason"],
        "synthetic_data_notice": report["synthetic_data_notice"],
        "random_seed": RANDOM_SEED_V4,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "created_at": created_at.isoformat(),
    }


def save_model_artifact_v4(
    model_key: str,
    matrix: TrainingMatrixV4,
    report: dict,
    output_dir: Path,
    csv_path: Path,
    created_at: datetime | None = None,
) -> tuple[Path, Path]:
    """Fit `model_key` on the full frozen dataset and save pipeline + metadata. Raises unless it was selected."""
    if model_key not in MODEL_SPECS_V4:
        raise ValueError(f"Unknown model {model_key!r}.")
    selected = report.get("selection", {}).get("selected_model")
    if selected is None:
        raise ValueError(f"No model passed the generalisation gate ({report['selection']['reason']}); refusing to save an artifact.")
    if selected != model_key:
        raise ValueError(f"{model_key!r} was not the selected model (selected: {selected!r}).")
    validate_feature_names_v4(matrix.feature_names)  # the metadata's feature order is validated against the contract

    stem = artifact_stem(model_key)
    if not stem.endswith("_v4"):
        raise ValueError("V4 artifacts must carry the _v4 suffix.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{stem}.joblib"
    metadata_path = output_dir / f"{stem}_metadata.json"

    pipeline = build_pipeline_v4(model_key, matrix.feature_names).fit(matrix.X, matrix.y)
    metadata = build_metadata_v4(model_key, matrix, report, csv_path, created_at or datetime.now(timezone.utc))
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return model_path, metadata_path


def load_model_artifact_v4(model_path: Path, metadata_path: Path):
    """Load a saved V4 pipeline, refusing any schema mismatch. Returns (pipeline, metadata)."""
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    validate_feature_names_v4(metadata.get("feature_names", ()))
    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION_V4:
        raise ValueError("Artifact metadata is not a V4 feature schema.")
    pipeline = joblib.load(model_path)
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("Loaded artifact does not expose predict_proba.")
    n_features = getattr(pipeline, "n_features_in_", None)
    if n_features is not None and n_features != FEATURE_COUNT_V4:
        raise ValueError(f"Artifact expects {n_features} features, the V4 contract has {FEATURE_COUNT_V4}.")
    return pipeline, metadata

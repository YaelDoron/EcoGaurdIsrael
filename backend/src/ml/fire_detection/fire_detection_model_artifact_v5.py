"""Save / load a Fire Detection ML V5 model artifact (joblib model + metadata JSON).

Deliberately gated: an artifact is written ONLY for the model the comparison report selected after passing the
predefined acceptance gate. If the report selected nothing, saving raises - a "runtime-ready" file is never produced
just to complete a task. V3 / V4 artifacts are never touched: V5 files are named `fire_detection_<model>_v5.*` and
any other name is refused.

Nothing here is imported by FireDetectionAgent or any runtime path. Wiring a V5 model into runtime is a later task.
All training data is synthetic.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold

from src.ml.fire_detection.fire_detection_features_v5 import FEATURE_COUNT_V5, FIRE_DETECTION_FEATURE_NAMES_V5
from src.ml.fire_detection.fire_detection_model_config_v5 import (
    DATASET_VERSION_V5,
    EXPECTED_TRAINING_CSV_SHA256_V5,
    FEATURE_SCHEMA_VERSION_V5,
    MODEL_VERSION_V5,
    RANDOM_SEED_V5,
)
from src.ml.fire_detection.fire_detection_model_v5 import (
    TrainingMatrixV5,
    assert_feature_schema_v5,
    build_pipeline_v5,
    dataset_sha256,
    describe_preprocessing_v5,
)

DEFAULT_MODEL_DIR_V5 = Path(__file__).resolve().parents[3] / "models" / "fire_detection"

_STEM_NAMES = {
    "logistic_regression": "logistic",
    "random_forest": "random_forest",
    "hist_gradient_boosting": "hist_gradient_boosting",
}


class ArtifactRefusedError(RuntimeError):
    """Raised when saving is refused (gate not passed, bad schema/hash, or a non-V5 file name)."""


def artifact_stem(model_key: str) -> str:
    """fire_detection_<model>_v5 - always carries the _v5 suffix, so it can never collide with a V3 / V4 artifact."""
    return f"fire_detection_{_STEM_NAMES[model_key]}_v5"


def _assert_v5_name(path: Path) -> None:
    if not path.name.endswith(("_v5.joblib", "_v5_metadata.json")):
        raise ArtifactRefusedError(f"refusing to write {path.name!r}: V5 artifacts must end with _v5.joblib / _v5_metadata.json")


def fit_final_model_v5(matrix: TrainingMatrixV5, model_key: str, params: dict, calibration: str):
    """Fit the selected candidate on ALL rows (the calibrator, if any, uses inner grouped folds of these rows)."""
    names = assert_feature_schema_v5(matrix.feature_names)
    columns = matrix.columns(names)
    pipeline = build_pipeline_v5(model_key, params, names)
    if calibration == "none":
        return pipeline.fit(columns, matrix.y)
    inner = list(GroupKFold(n_splits=3).split(columns, matrix.y, matrix.environments))
    return CalibratedClassifierCV(pipeline, method=calibration, cv=inner, ensemble=True).fit(columns, matrix.y)


def build_metadata_v5(report: dict, matrix: TrainingMatrixV5, csv_path: Path, created_at: datetime) -> dict:
    """Metadata for the selected model, assembled from the comparison report (no metric is recomputed here)."""
    key = report["selection"]["selected_model"]
    section = report["models"][key]
    return {
        "model_version": MODEL_VERSION_V5,
        "model_type": key,
        "model_name": artifact_stem(key),
        "feature_schema_version": FEATURE_SCHEMA_VERSION_V5,
        "feature_names": list(matrix.feature_names),
        "feature_count": FEATURE_COUNT_V5,
        "dataset_version": DATASET_VERSION_V5,
        "training_csv_file": Path(csv_path).name,
        "training_csv_sha256": dataset_sha256(csv_path),
        "training_rows": int(len(matrix.y)),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "random_seed": RANDOM_SEED_V5,
        "selected_params": section["selected_params"],
        "calibration_variant": section["calibration"]["adopted_variant"],
        "preprocessing": describe_preprocessing_v5(key, matrix.feature_names),
        "acceptance_gate": report["acceptance_gate"],
        "acceptance_gate_result": section["acceptance_gate_result"],
        "grouped_environment_metrics": section["grouped_environment"]["summary"],
        "leave_regime_metrics": section.get("leave_one_regime_out"),
        "leave_nonfire_subtype_metrics": section.get("leave_one_no_fire_subtype_out"),
        "paired_metrics": section["paired_cases"],
        "sparse_early_evidence": section["sparse_early_evidence"],
        "rule_baseline_metrics": report["rule_baseline"],
        "history_ablation": section.get("ablations", {}).get("comparisons"),
        "current_feature_ablation": section.get("ablations", {}).get("results"),
        "calibration_metrics": section["calibration"],
        "suspected_threshold_candidate": (section["suspected_operating_point"] or {}).get("threshold"),
        "confirmed_probability_analysis": section["confirmed_probability_analysis"],
        "corroboration_analysis": section["corroboration"],
        "feature_importance_summary": section["feature_importance"]["grouped_permutation"],
        "missingness_audit": section["missingness_audit"],
        "selection_reason": report["selection"]["reason"],
        "synthetic_data_notice": report["synthetic_data_notice"],
        "known_limitations": [
            "trained and evaluated ONLY on synthetic data; not a measure of real-world wildfire detection accuracy",
            "history features exist only in the persistent_thermal regime of the benchmark",
            "sparse early evidence is intentionally ambiguous; probabilities there should express uncertainty",
            "no site recurrence, multi-platform or news-source-count signal",
            "not integrated into FireDetectionAgent / the decision policy",
        ],
        "created_at": created_at.isoformat(),
    }


def save_selected_model_v5(
    report: dict,
    matrix: TrainingMatrixV5,
    csv_path: Path,
    model_dir: Path = DEFAULT_MODEL_DIR_V5,
    created_at: datetime | None = None,
) -> tuple[Path, Path]:
    """Refit the SELECTED model on the frozen dataset and save it; refuses unless the gate selected a model."""
    selected = report["selection"]["selected_model"]
    if selected is None:
        raise ArtifactRefusedError("no V5 model passed the acceptance gate; refusing to save a runtime artifact.")
    if not report["models"][selected]["acceptance_gate_result"]["passed"]:
        raise ArtifactRefusedError(f"{selected} did not pass the acceptance gate.")
    assert_feature_schema_v5(matrix.feature_names)
    if list(matrix.feature_names) != list(FIRE_DETECTION_FEATURE_NAMES_V5):
        raise ArtifactRefusedError("feature_names differ from FIRE_DETECTION_FEATURE_NAMES_V5.")
    actual_hash = dataset_sha256(csv_path)
    if actual_hash != EXPECTED_TRAINING_CSV_SHA256_V5 or report["meta"]["training_csv_sha256"] != actual_hash:
        raise ArtifactRefusedError("the training CSV is not the frozen benchmark the report was computed on.")

    section = report["models"][selected]
    model = fit_final_model_v5(matrix, selected, section["selected_params"], section["calibration"]["adopted_variant"])
    metadata = build_metadata_v5(report, matrix, csv_path, created_at or datetime.now(timezone.utc))

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{artifact_stem(selected)}.joblib"
    metadata_path = model_dir / f"{artifact_stem(selected)}_metadata.json"
    _assert_v5_name(model_path)
    _assert_v5_name(metadata_path)
    if model_path.exists() or metadata_path.exists():
        raise ArtifactRefusedError(f"{model_path.name} already exists; V5 artifacts are never silently overwritten.")
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model_path, metadata_path


def load_model_v5(model_path: Path, metadata_path: Path):
    """Load (model, metadata); refuses a metadata file whose feature schema is not exactly the canonical V5 one."""
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    assert_feature_schema_v5(metadata["feature_names"])
    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION_V5:
        raise ArtifactRefusedError("metadata is not a V5 feature schema.")
    return joblib.load(model_path), metadata


def reload_parity(model, reloaded, X: np.ndarray) -> float:
    """Largest absolute probability difference between an in-memory model and its reloaded copy (must be 0.0)."""
    return float(np.max(np.abs(model.predict_proba(X)[:, 1] - reloaded.predict_proba(X)[:, 1])))

"""Task 6 guard rails: V5 is additive. V3/V4 stay frozen, runtime stays untouched, nothing is trained."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.ml.fire_detection.fire_detection_features_v3 import FIRE_DETECTION_FEATURE_NAMES_V3
from src.ml.fire_detection.fire_detection_features_v4 import FIRE_DETECTION_FEATURE_NAMES_V4

BACKEND = Path(__file__).resolve().parents[3]
ML_DIR = BACKEND / "src" / "ml" / "fire_detection"
V5_MODULES = sorted(ML_DIR.glob("*_v5.py"))


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    modules |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    return modules


TASK_6_MODULES = {
    "fire_detection_dataset_v5.py",
    "fire_detection_dataset_validation_v5.py",
    "fire_detection_feature_extractor_v5.py",
    "fire_detection_features_v5.py",
    "fire_detection_training_data_generator_v5.py",
}
TASK_7_MODULES = {
    "fire_detection_model_config_v5.py",
    "fire_detection_model_v5.py",
    "fire_detection_evaluation_v5.py",
    "fire_detection_model_comparison_v5.py",
    "fire_detection_model_report_v5.py",
    "fire_detection_model_artifact_v5.py",
}


TASK_8_MODULES = {
    "fire_detection_policy_config_v5.py",
    "fire_detection_ai_hybrid_policy_v5.py",
    "fire_detection_policy_validation_data_v5.py",
    "fire_detection_policy_evaluation_v5.py",
    "fire_detection_policy_report_v5.py",
}


# Task 9B: the ONLY V5 modules that are runtime code (they load the approved artifact / run the locked policy).
TASK_9B_RUNTIME_MODULES = {
    "fire_detection_model_runtime_v5.py",
    "fire_detection_ai_hybrid_runtime_v5.py",
}
# The ONLY runtime modules (outside src/ml) allowed to import a V5 module.
V5_RUNTIME_IMPORTERS = {
    "src/agents/analysis/fire_detection_agent.py",
}


def test_the_expected_v5_modules_exist():
    assert {p.name for p in V5_MODULES} == TASK_6_MODULES | TASK_7_MODULES | TASK_8_MODULES | TASK_9B_RUNTIME_MODULES


def test_task_8_policy_code_is_offline_and_never_regenerates_or_alters_the_frozen_generator():
    """Only the confirmatory-data module calls the frozen generator; the policy / evaluation code never does."""
    for name in TASK_8_MODULES - {"fire_detection_policy_validation_data_v5.py"}:
        assert "FireDetectionTrainingDataGeneratorV5" not in (ML_DIR / name).read_text(encoding="utf-8"), name
    for name in TASK_8_MODULES:
        for module in imported_modules(ML_DIR / name):
            assert "agents" not in module and "repositor" not in module and "fire_detection_decision_policy" not in module, (name, module)


def test_task_6_modules_train_nothing_and_do_not_import_a_model_library():
    """The schema / extractor / generator / dataset code stays model-free (Task 7 modules are the ones that train)."""
    for path in V5_MODULES:
        if path.name not in TASK_6_MODULES:
            continue
        for module in imported_modules(path):
            assert not module.startswith(("joblib", "pickle")), (path.name, module)
            assert not module.startswith("sklearn") or path.name == "fire_detection_dataset_validation_v5.py", (path.name, module)
    validation_source = (ML_DIR / "fire_detection_dataset_validation_v5.py").read_text(encoding="utf-8")
    assert "LogisticRegression" not in validation_source and ".fit(" not in validation_source.replace("NearestNeighbors(n_neighbors=2).fit(", "")


def test_task_7_modules_never_change_the_frozen_generator_or_dataset_code():
    """Model code only READS the frozen V5 pieces: no Task 7 module may write a dataset or call the generator's generate()
    except the rule-baseline sample regeneration in the scripts / tests."""
    for name in TASK_7_MODULES:
        source = (ML_DIR / name).read_text(encoding="utf-8")
        assert "write_training_dataset_v5" not in source, name
        assert "FireDetectionTrainingDataGeneratorV5" not in source, name


def test_v5_artifacts_exist_only_as_allowed_by_the_two_gates():
    """Task 7 (binary-primary gate) selected NONE, so no Task 7 artifact may exist. The ONLY permitted V5 artifact is the
    Task 8 `fire_detection_hgb_v5*`, and only if the Task 8 AI-hybrid policy gate passed."""
    import json

    data = BACKEND / "data" / "fire_detection"
    task7 = json.loads((data / "fire_detection_model_comparison_v5.json").read_text(encoding="utf-8"))
    assert task7["selection"]["selected_model"] is None  # the historical Task 7 verdict is untouched
    artifacts = sorted(p.name for p in (BACKEND / "models" / "fire_detection").glob("*_v5*"))
    task8_path = data / "fire_detection_v5_policy_validation_report.json"
    task8_passed = task8_path.exists() and json.loads(task8_path.read_text(encoding="utf-8"))["verdict"] == "AI HYBRID POLICY PASSED"
    allowed = ["fire_detection_hgb_v5.joblib", "fire_detection_hgb_v5_metadata.json"] if task8_passed else []
    assert artifacts == allowed


def test_v5_has_no_fire_danger_dependency():
    for path in V5_MODULES + [BACKEND / "scripts" / "generate_fire_detection_training_data_v5.py", BACKEND / "scripts" / "validate_fire_detection_dataset_v5.py"]:
        for module in imported_modules(path):
            assert "fire_danger" not in module and "ffwi" not in module and "fire_detection_context" not in module, (path.name, module)


def test_v3_and_v4_feature_schemas_are_unchanged():
    assert len(FIRE_DETECTION_FEATURE_NAMES_V3) == 16 and FIRE_DETECTION_FEATURE_NAMES_V3[-2:] == ("time_span_minutes", "max_pairwise_distance_km")
    assert len(FIRE_DETECTION_FEATURE_NAMES_V4) == 19
    assert FIRE_DETECTION_FEATURE_NAMES_V4[16:] == ("fire_danger_available", "fire_danger_score", "fire_danger_age_minutes")


def test_frozen_v4_modules_do_not_reference_v5():
    for path in ML_DIR.glob("*_v4.py"):
        assert "_v5" not in path.read_text(encoding="utf-8"), path.name
    for path in (BACKEND / "scripts").glob("*_v4.py"):
        assert "_v5" not in path.read_text(encoding="utf-8"), path.name


def test_only_the_fire_detection_agent_uses_v5_at_runtime():
    """Task 9B replaced "runtime never imports V5" with an allow-list: exactly one runtime module (the agent) may."""
    runtime_roots = [
        BACKEND / "src" / "agents",
        BACKEND / "src" / "calculators" / "fire_detection",
        BACKEND / "src" / "repositories",
        BACKEND / "src" / "services",
        BACKEND / "src" / "api",
        BACKEND / "src" / "simulation",
    ]
    importers = set()
    for root in runtime_roots:
        for path in root.rglob("*.py"):
            if any("_v5" in module for module in imported_modules(path)):
                importers.add(path.relative_to(BACKEND).as_posix())
    assert importers == V5_RUNTIME_IMPORTERS


def test_the_v5_runtime_modules_never_touch_the_rule_calculator_fire_danger_repositories_or_agents():
    for name in TASK_9B_RUNTIME_MODULES:
        for module in imported_modules(ML_DIR / name):
            for forbidden in ("calculators", "fire_danger", "fire_detection_context", "repositor", "agents", "services"):
                assert forbidden not in module, (name, module)


def test_the_active_runtime_ml_still_uses_the_v3_schema_only():
    from src.calculators.fire_detection import fire_detection_ml_classifier

    source = Path(fire_detection_ml_classifier.__file__).read_text(encoding="utf-8")
    assert "_v5" not in source and "V5" not in source


def test_the_decision_modes_are_unchanged():
    from src.models.fire_detection_decision_mode import FireDetectionDecisionMode

    # legacy modes unchanged; Task 9B added only ai_hybrid_v5
    assert {mode.value for mode in FireDetectionDecisionMode} == {"rule_only", "shadow", "hybrid", "ai_hybrid_v5"}


def test_v4_benchmark_files_are_not_touched_by_v5_generation(tmp_path):
    """Generating V5 into another directory leaves the frozen V4 dataset (if present) byte-identical."""
    v4_csv = BACKEND / "data" / "fire_detection" / "training_v4.csv"
    if not v4_csv.exists():
        pytest.skip("training_v4.csv not present")
    import hashlib

    from src.ml.fire_detection.fire_detection_dataset_v5 import write_training_dataset_v5
    from src.ml.fire_detection.fire_detection_training_data_generator_v5 import FireDetectionTrainingDataGeneratorV5

    before = hashlib.sha256(v4_csv.read_bytes()).hexdigest()
    write_training_dataset_v5(FireDetectionTrainingDataGeneratorV5(seed=1).generate(60), tmp_path / "v5.csv")
    assert hashlib.sha256(v4_csv.read_bytes()).hexdigest() == before

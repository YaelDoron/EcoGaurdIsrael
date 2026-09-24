"""Read-only readiness check for FIRE_DETECTION_DECISION_MODE=ai_hybrid_v5 (Task 9C).

Verifies, without changing anything and without printing any secret or path:
  * the configured decision mode (and whether it is ai_hybrid_v5)
  * the approved HGB V5 artifact loads and its metadata / feature schema / policy provenance validate
  * the configured database has the AI audit columns on fire_event_ml_assessments (migration applied)
  * the demo reset guard state (informational)

Usage:
    python -m scripts.check_ai_hybrid_v5_readiness
    FIRE_DETECTION_DECISION_MODE=ai_hybrid_v5 python -m scripts.check_ai_hybrid_v5_readiness   # preview a switch

Exit code 0 = ready, 1 = not ready.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import inspect

from scripts.migrate_add_fire_event_ml_assessment_ai_columns import COLUMNS, TABLE_NAME
from src.config.settings import settings
from src.ml.fire_detection.fire_detection_model_runtime_v5 import FireDetectionAIError, FireDetectionModelV5Runtime
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode


def check() -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    try:
        mode = FireDetectionDecisionMode(settings.FIRE_DETECTION_DECISION_MODE)
        results.append(("decision mode is valid", True, mode.value))
    except ValueError:
        results.append(("decision mode is valid", False, "unknown FIRE_DETECTION_DECISION_MODE value"))
        mode = None
    results.append(("decision mode is ai_hybrid_v5", mode is FireDetectionDecisionMode.AI_HYBRID_V5,
                    mode.value if mode else "invalid"))

    try:
        info = FireDetectionModelV5Runtime(
            settings.FIRE_DETECTION_AI_V5_MODEL_PATH, settings.FIRE_DETECTION_AI_V5_METADATA_PATH
        ).model_info
        results.append(("artifact + metadata + policy validate", True, ", ".join(f"{k}={v}" for k, v in info.items())))
    except FireDetectionAIError as exc:
        results.append(("artifact + metadata + policy validate", False, exc.public_message))

    if not settings.DATABASE_URL:
        results.append(("database has the AI audit columns", False, "DATABASE_URL is not configured"))
    else:
        try:
            from src.database.connection import get_engine

            inspector = inspect(get_engine())
            present = {column["name"] for column in inspector.get_columns(TABLE_NAME)} if inspector.has_table(TABLE_NAME) else set()
            missing = [name for name, _ in COLUMNS if name not in present]
            results.append(("database has the AI audit columns", not missing, "all present" if not missing else f"missing: {missing}"))
        except Exception as exc:  # noqa: BLE001 - report, never print connection details
            results.append(("database has the AI audit columns", False, f"database check failed ({type(exc).__name__})"))

    results.append(("demo reset enabled (informational)", bool(settings.ENABLE_DEMO_DATA_RESET), str(bool(settings.ENABLE_DEMO_DATA_RESET))))
    return results


def main() -> int:
    results = check()
    ready = True
    for name, ok, detail in results:
        informational = "informational" in name or name == "decision mode is ai_hybrid_v5"
        print(f"[{'OK' if ok else ('--' if informational else 'FAIL')}] {name}: {detail}")
        ready = ready and (ok or informational)
    print("READY" if ready else "NOT READY")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

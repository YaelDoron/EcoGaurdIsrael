"""Final live acceptance run of the AI Hybrid V5 detector on the configured demo database (Task 9C).

Drives the dedicated multi-hour scenario (src/simulation/ai_acceptance_scenario.py) through the SAME production per-event pipeline the
demo runner uses (`execute_simulation_event` with the production coordinator builders), with the REAL approved HGB artifact. Every
time is a simulated timestamp - nothing waits.

It REFUSES to run unless FIRE_DETECTION_DECISION_MODE resolves to ai_hybrid_v5, the artifact validates and the AI audit columns exist,
and it starts with the mandatory demo reset (Task 9A). Prints a progression table, per-stage timings and a verdict; no secrets, no paths.

    FIRE_DETECTION_DECISION_MODE=ai_hybrid_v5 python -m scripts.run_ai_hybrid_acceptance [--runs 2]

With --runs 2 a second run follows (after another mandatory reset) to prove the new run starts from pass count 1 with no history.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_ai_hybrid_v5_readiness import check
from src.config.settings import settings
from src.database.connection import get_engine
from src.ml.fire_detection import fire_detection_model_runtime_v5 as runtime_module
from src.models.fire_event_response_eligibility import is_response_eligible
from src.repositories.fire_event_repository import FireEventRepository
from src.services.fire_detection import FireDetectionHistoryService
from src.simulation import simulation_event_timestamp
from src.simulation.ai_acceptance_scenario import AIAcceptanceEvidenceExecutor, build_ai_progression_scenario, record_progression
from src.simulation.demo_run_preparation import prepare_clean_demo_state
from src.simulation.demo_simulation_runner import (
    build_fire_detection_coordinator,
    build_operational_coordinator,
    build_simulation_refresh_coordinator,
    execute_simulation_event,
)
from src.simulation.simulation_event import SimulationEventType

STARTED_AT = datetime(2026, 9, 17, 18, 0, tzinfo=timezone.utc)  # 20:00 local: an overnight fire
RESPONSE_TABLES = (
    "fire_severity_assessments", "fire_spread_predictions", "response_targets", "route_planning_runs", "global_planning_runs",
    "response_plans", "resource_commitments",
)


def response_counts() -> dict[str, int]:
    from sqlalchemy import text

    with get_engine().connect() as connection:
        return {name: connection.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar_one() for name in RESPONSE_TABLES}


class Timer:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    def wrap(self, obj, method: str, key: str) -> None:
        original = getattr(obj, method)

        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.calls[key] = self.calls.get(key, 0) + 1
                self.seconds[key] = self.seconds.get(key, 0.0) + time.perf_counter() - started

        setattr(obj, method, timed)


def run_once(label: str, loads: list, steps: int | None = None) -> dict:
    scenario = build_ai_progression_scenario()
    if steps is not None:  # e.g. --steps 3 stops while the event is still SUSPECTED (to inspect the monitoring state)
        scenario = replace(scenario, steps=scenario.steps[:steps])
    prepare_clean_demo_state()  # mandatory reset before ANY event (Task 9A)
    assert response_counts() == dict.fromkeys(RESPONSE_TABLES, 0), "reset left response artefacts behind"
    executor = AIAcceptanceEvidenceExecutor(scenario)
    detection = build_fire_detection_coordinator()
    operational = build_operational_coordinator()
    refresh = build_simulation_refresh_coordinator(operational_coordinator=operational)
    agent = detection._detection_agent
    events = FireEventRepository()
    timer = Timer()
    report_history = FireDetectionHistoryService(agent._evidence_service, events)  # the report's own reads: not in the timings
    timer.wrap(agent, "detect", "fire_detection_cycle (evidence+history+features+HGB+persist)")
    timer.wrap(agent._history_service, "build_history", "history retrieval")
    timer.wrap(agent._ai_classifier.model_runtime, "predict_probability", "HGB inference")
    timer.wrap(refresh, "handle_event", "refresh (severity->spread->targets->routing->global GA->activation)")

    rows, snapshots, activated = [], [], set()
    started = time.perf_counter()
    for index, step in enumerate(scenario.steps):
        for event in scenario.events_for_step(index):
            outcome = execute_simulation_event(
                scenario=scenario.holder_scenario, event=event, scenario_started_at=STARTED_AT, executor=executor,
                fire_detection_coordinator=detection, operational_coordinator=operational, simulation_refresh_coordinator=refresh,
                activated_fire_event_ids=activated)
            if event.event_type in (SimulationEventType.SATELLITE, SimulationEventType.NEWS):
                result = outcome.fire_detection_result.detection_result
                if not result.success:
                    raise SystemExit(f"detection failed: {result.error_message}")
                name = f"{step.label} / {event.event_type.value}"
                for record in record_progression(step, simulation_event_timestamp(STARTED_AT, event), result, events, report_history):
                    rows.append((name, record))
                snapshots.append((name, response_counts(), outcome.depleted_resource_count))
    total = time.perf_counter() - started

    print(f"\n=== {label} ===")
    print(f"{'time / event':<40} {'P':>5} {'policy':<10} {'event':<10} {'pass':>4} {'span_min':>8} {'pix':>3} {'peak':>5} {'latest':>6} {'resp':>5}")
    for name, r in rows:
        span = "-" if r.history_span_minutes is None else f"{r.history_span_minutes:.0f}"
        print(f"{name:<40} {r.probability:5.2f} {r.policy_status:<10} {str(r.event_status):<10} {r.satellite_pass_count:4d} {span:>8} "
              f"{r.current_satellite_pixel_count:3d} {r.peak_confidence or 0:5.2f} {r.latest_assessment_probability or 0:6.2f} {str(r.response_eligible):>5}")
    print("response artefacts after each cycle (severity, spread, targets, route runs, global runs, plans, commitments):")
    for name, counts, depleted in snapshots:
        print(f"  {name:<40} {tuple(counts.values())}  depleted_resources={depleted}")
    print("timings (s):", {key: f"{value:.2f}s x{timer.calls[key]}" for key, value in timer.seconds.items()}, f"| total {total:.1f}s")
    print("artifact loads in this process so far:", len(loads))
    return {"rows": rows, "snapshots": snapshots}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None, help="run only the first N passes (3 = stop in the SUSPECTED phase)")
    args = parser.parse_args(argv)

    loads: list = []
    original_load = runtime_module.joblib.load
    runtime_module.joblib.load = lambda path, *a, **k: loads.append(1) or original_load(path, *a, **k)

    problems = [f"{name}: {detail}" for name, ok, detail in check()
                if not ok and "informational" not in name]
    if problems:
        print("Refused - not ready for ai_hybrid_v5:\n  " + "\n  ".join(problems))
        return 2
    if not settings.ENABLE_DEMO_DATA_RESET:
        print("Refused: ENABLE_DEMO_DATA_RESET is not enabled.")
        return 2

    zero = dict.fromkeys(RESPONSE_TABLES, 0)
    first = run_once("run 1", loads, args.steps)
    statuses = [r.event_status for _, r in first["rows"] if r.event_id is not None]
    partial = args.steps is not None and args.steps < len(build_ai_progression_scenario().steps)
    ok = statuses and statuses[0] == "suspected" and (statuses[-1] == "suspected" if partial else statuses[-1] == "confirmed")
    suspected_clean = all(counts == zero for (name, counts, _), (_, r) in zip(first["snapshots"], first["rows"]) if r.event_status == "suspected")
    confirmed_active = partial or (first["snapshots"][-1][1]["response_plans"] >= 1 and first["snapshots"][-1][1]["resource_commitments"] >= 1)
    print(f"\nSUSPECTED phase: no response artefacts = {suspected_clean}; CONFIRMED phase: plan + commitment persisted = {confirmed_active}")

    isolated = True
    if args.runs >= 2:
        second = run_once("run 2 (after mandatory reset)", loads, args.steps)
        _, first_record = second["rows"][0]
        isolated = first_record.satellite_pass_count == 1 and first_record.history_span_minutes == 0.0
        print(f"\nRun 2 first observation: pass_count={first_record.satellite_pass_count}, span={first_record.history_span_minutes} -> isolated={isolated}")
    print(f"artifact deserialized {len(loads)} time(s) in this process")
    passed = bool(ok and suspected_clean and confirmed_active and isolated and len(loads) == 1)
    print("ACCEPTANCE RUN " + ("PASSED" if passed else "FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

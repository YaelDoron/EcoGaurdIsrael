"""Task 9A: a demo simulation run can NEVER inherit the previous run's FireEvents, evidence or Fire Detection history.

Supersedes the Task 8 "known leak" tests: the reset is now MANDATORY on every supported start path, so those paths
are exercised here end to end on SQLite with the REAL reset service, the REAL detection stack, the REAL history
service and the V5 feature extractor:

    * API   POST /api/v1/simulation/runs  -> SimulationRunManager
    * CLI   scripts/run_demo_simulation.main  (manual and automatic modes)
    * the run manager directly

No wall-clock waiting: every time is a simulated timestamp. Run B's first observation is deliberately placed INSIDE the
6 h event-matching window of Run A's last update, so only the reset - not the passage of time - can isolate it.
"""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.config.settings  # noqa: F401 - ensures the settings submodule is in sys.modules
from src.agents.analysis import FireDetectionAgent
from src.api.app import create_app
from src.api.dependencies import get_simulation_run_manager
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_detection.fire_detection_decision_policy import FireDetectionHybridPolicy
from src.database.base import Base
from src.ml.fire_detection.fire_detection_feature_extractor_v5 import FireDetectionFeatureExtractorV5
from src.models import SatelliteHotspot
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.fire_detection import FireDetectionEvidenceService, FireDetectionHistoryService
from src.services.simulation_control.simulation_run_manager import SimulationResetDisabledError, SimulationRunManager
from src.simulation import demo_state_reset_service as reset_module
from src.simulation.demo_run_preparation import prepare_clean_demo_state
from src.simulation.demo_simulation_runner import DemoSimulationRunResult, DemoSimulationStatus
from src.simulation.demo_state_reset_service import DemoStateResetService, _DELETE_ORDER_MODELS

T0 = datetime(2026, 9, 14, 6, 0, tzinfo=timezone.utc)
LATITUDE, LONGITUDE = 32.731, 35.046
BACKEND = Path(__file__).resolve().parents[2]


class Stack:
    """The real detection stack over one SQLite session factory."""

    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.satellites = SatelliteHotspotRepository(session_factory=session_factory)
        self.news = NewsRepository(session_factory=session_factory)
        self.events = FireEventRepository(session_factory=session_factory)
        self.evidence = FireDetectionEvidenceService(satellite_repository=self.satellites, news_repository=self.news)
        self.history = FireDetectionHistoryService(self.evidence, self.events)
        self.agent = FireDetectionAgent(
            evidence_service=self.evidence, calculator=FireDetectionCalculator(), fire_event_repository=self.events,
            satellite_repository=self.satellites, news_repository=self.news,
            decision_policy=FireDetectionHybridPolicy(mode=FireDetectionDecisionMode.RULE_ONLY),
        )

    def detect_at(self, moment):
        self.satellites.save_hotspot(
            SatelliteHotspot(latitude=LATITUDE, longitude=LONGITUDE, detected_at=moment, confidence="n", frp=8.0,
                             brightness=330.0, satellite="NOAA-20", instrument="VIIRS", day_night="D")
        )
        result = self.agent.detect(moment + timedelta(minutes=5))
        assert result.success, result.error_message
        return result

    def features_of_latest_observation(self, as_of):
        candidates = self.evidence.build_candidates(as_of)
        candidate = max(candidates, key=lambda c: max(e.observed_at for e in c.evidence))
        event = self.events.find_matching_active_event(LATITUDE, LONGITUDE, max(e.observed_at for e in candidate.evidence))
        history = self.history.build_history(event, as_of) if event is not None else None
        return FireDetectionFeatureExtractorV5().extract(FireDetectionCandidate(candidate.evidence), history), event, history

    def run_a(self):
        """Simulation Run A: one event, T+0 / T+3h / T+6h."""
        for hours in (0, 3, 6):
            result = self.detect_at(T0 + timedelta(hours=hours))
        (event_id,) = result.event_ids
        return event_id


@pytest.fixture
def shared_session_factory():
    """One in-memory SQLite database shared across threads (TestClient serves requests on another thread)."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    try:
        yield sessionmaker(bind=engine, expire_on_commit=False)
    finally:
        engine.dispose()


@pytest.fixture
def stack(shared_session_factory, monkeypatch):
    settings = SimpleNamespace(ENABLE_DEMO_DATA_RESET=True, ENABLE_SIMULATION_CONTROL_API=True)
    monkeypatch.setattr(reset_module, "settings", settings)
    monkeypatch.setattr(sys.modules["src.config.settings"], "settings", settings)
    return Stack(shared_session_factory)


def sqlite_reset_service(stack):
    return DemoStateResetService(stack.session_factory)


class RunBObservation:
    """What Run B's first relevant observation produced (filled by whichever start path ran it)."""

    def __init__(self):
        self.reset_state_at_start = None
        self.event_id = None
        self.features = None
        self.event = None
        self.history = None

    def run(self, stack, moment):
        self.reset_state_at_start = (len(stack.events.get_recent(50)), len(stack.satellites.get_recent_hotspots(moment, 60 * 24 * 3)))
        (self.event_id,) = stack.detect_at(moment).event_ids
        self.features, self.event, self.history = stack.features_of_latest_observation(moment + timedelta(minutes=5))


def assert_run_b_is_clean(stack, observation, run_a_event_id):
    assert observation.reset_state_at_start == (0, 0)  # the runner started on an EMPTY runtime state
    f = observation.features
    assert f["satellite_pass_count"] == 1 and f["satellite_history_span_minutes"] == 0.0
    assert all(math.isnan(f[name]) for name in ("satellite_frp_trend_per_hour", "satellite_brightness_trend_per_hour", "satellite_centroid_stability_km"))
    refs = stack.events.get_evidence_refs(observation.event_id)
    assert len(refs) == 1  # only Run B's own hotspot
    assert len(observation.history.evidence) == 1 and observation.history.distinct_satellite_pass_count == 1
    assert observation.history.evidence[0].observed_at >= T0 + timedelta(hours=7) - timedelta(minutes=5)  # nothing from Run A
    active = stack.events.get_active_events()
    assert len(active) == 1 and active[0].event.detected_at >= T0 + timedelta(hours=7)  # Run A's FireEvent is not active state


# --- Run A ---------------------------------------------------------------------------------------------------------


def test_run_a_builds_one_event_with_three_passes_and_a_360_minute_history(stack):
    event_id = stack.run_a()

    assert len(stack.events.get_recent(10)) == 1
    assert len(stack.events.get_evidence_refs(event_id)) == 3
    features, _, history = stack.features_of_latest_observation(T0 + timedelta(hours=6, minutes=5))
    assert features["satellite_pass_count"] == 3 and history.distinct_satellite_pass_count == 3
    assert features["satellite_history_span_minutes"] == pytest.approx(360.0)
    assert features["satellite_frp_trend_per_hour"] == pytest.approx(0.0)  # a real, flat trend from three passes


# --- every supported start path leaves Run B clean ---------------------------------------------------------------------


class RunBRunner:
    """Stands in for DemoSimulationRunner: its first act is Run B's first observation (T+7h, inside A's 6 h window)."""

    def __init__(self, stack, observation):
        self.stack, self.observation = stack, observation

    def run(self, scenario, config, scenario_started_at=None, on_progress=None):
        self.observation.run(self.stack, T0 + timedelta(hours=7))
        now = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
        return DemoSimulationRunResult(
            status=DemoSimulationStatus.COMPLETED, simulation_started_at=now, simulation_completed_at=now,
            simulation_duration_seconds=1, wall_clock_elapsed_seconds=0.1, events_total=0, events_executed=0,
            events_succeeded=0, events_failed=0, incident_ids=())


class SyncExecutor:
    def submit(self, fn, *args):
        fn(*args)


def _manager(stack, observation):
    return SimulationRunManager(
        runner_factory=lambda: RunBRunner(stack, observation),
        reset_service_factory=lambda: sqlite_reset_service(stack),
        executor=SyncExecutor(),
    )


def test_api_start_path_run_b_starts_clean(stack):
    run_a_event = stack.run_a()
    observation = RunBObservation()
    app = create_app()
    app.dependency_overrides[get_simulation_run_manager] = lambda: _manager(stack, observation)

    response = TestClient(app).post("/api/v1/simulation/runs", json={"preset": "operations_demo", "seed": 1})  # NO reset flag sent

    assert response.status_code == 202
    assert_run_b_is_clean(stack, observation, run_a_event)


def test_api_start_path_with_the_dashboards_explicit_reset_flag_also_starts_clean(stack):
    run_a_event = stack.run_a()
    observation = RunBObservation()
    app = create_app()
    app.dependency_overrides[get_simulation_run_manager] = lambda: _manager(stack, observation)

    response = TestClient(app).post("/api/v1/simulation/runs", json={"preset": "operations_demo", "reset_demo_state": True})

    assert response.status_code == 202
    assert_run_b_is_clean(stack, observation, run_a_event)


def test_manager_start_path_run_b_starts_clean(stack):
    run_a_event = stack.run_a()
    observation = RunBObservation()

    _manager(stack, observation).start_run(preset_id="operations_demo", seed=1)

    assert_run_b_is_clean(stack, observation, run_a_event)


@pytest.mark.parametrize("mode", ["manual", "automatic"])
def test_cli_start_path_run_b_starts_clean_in_both_modes(stack, monkeypatch, mode):
    import scripts.run_demo_simulation as cli

    run_a_event = stack.run_a()
    observation = RunBObservation()
    monkeypatch.setattr(cli, "initialize_database", lambda: None)
    monkeypatch.setattr(cli, "prepare_clean_demo_state", lambda: prepare_clean_demo_state(lambda: sqlite_reset_service(stack)))
    monkeypatch.setattr(cli, "run_manual", lambda scenario: observation.run(stack, T0 + timedelta(hours=7)))
    monkeypatch.setattr(cli, "run_automatic", lambda scenario, poll_interval_seconds: observation.run(stack, T0 + timedelta(hours=7)))

    exit_code = cli.main(["--preset", "operations_demo", "--mode", mode])

    assert exit_code == 0
    assert_run_b_is_clean(stack, observation, run_a_event)


# --- fail closed --------------------------------------------------------------------------------------------------------


def test_with_the_reset_disabled_no_start_path_runs_anything_and_run_a_state_is_untouched(stack, monkeypatch):
    import scripts.run_demo_simulation as cli

    stack.run_a()
    disabled = SimpleNamespace(ENABLE_DEMO_DATA_RESET=False, ENABLE_SIMULATION_CONTROL_API=True)
    monkeypatch.setattr(reset_module, "settings", disabled)
    monkeypatch.setattr(sys.modules["src.config.settings"], "settings", disabled)
    observation = RunBObservation()

    with pytest.raises(SimulationResetDisabledError):
        _manager(stack, observation).start_run(preset_id="operations_demo", seed=1)

    app = create_app()
    app.dependency_overrides[get_simulation_run_manager] = lambda: _manager(stack, observation)
    assert TestClient(app).post("/api/v1/simulation/runs", json={"preset": "operations_demo"}).status_code == 403

    ran = []
    monkeypatch.setattr(cli, "initialize_database", lambda: None)
    monkeypatch.setattr(cli, "run_manual", lambda scenario: ran.append("manual"))
    monkeypatch.setattr(cli, "prepare_clean_demo_state", lambda: prepare_clean_demo_state(lambda: sqlite_reset_service(stack)))
    assert cli.main(["--preset", "operations_demo", "--mode", "manual"]) == 2

    assert observation.event_id is None and ran == []  # nothing ran on any path
    assert len(stack.events.get_recent(10)) == 1  # ...and nothing was deleted either


# --- no supported path can skip the reset ---------------------------------------------------------------------------------


def test_only_the_two_supported_starters_construct_the_runner_and_both_prepare_a_clean_state():
    """DemoSimulationRunner is a library primitive: every non-test module that uses it must go through the mandatory reset."""
    users = {}
    for root in (BACKEND / "src", BACKEND / "scripts"):
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "DemoSimulationRunner" not in text or path.name == "demo_simulation_runner.py":
                continue
            tree = ast.parse(text)
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            names |= {alias.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for alias in n.names}
            if "DemoSimulationRunner" in names:  # a real code reference, not merely a docstring mention
                users[path.relative_to(BACKEND).as_posix()] = names
    assert set(users) == {"src/services/simulation_control/simulation_run_manager.py", "scripts/run_demo_simulation.py"}
    for path, names in users.items():
        assert "prepare_clean_demo_state" in names or "require_demo_reset_enabled" in names, path


def test_there_is_no_opt_out_flag_anywhere():
    manager_source = (BACKEND / "src/services/simulation_control/simulation_run_manager.py").read_text(encoding="utf-8")
    assert "if reset_demo_state:" not in manager_source  # the old conditional reset is gone
    assert "reset_demo_state: bool = True" in manager_source
    cli_source = (BACKEND / "scripts/run_demo_simulation.py").read_text(encoding="utf-8")
    assert "--reset-demo-state" not in cli_source and "args.reset_demo_state" not in cli_source


# --- reset scope --------------------------------------------------------------------------------------------------------------


def test_the_reset_clears_every_runtime_table_and_associations_and_preserves_only_declared_static_data(stack):
    from src.database.models.fire_event_db import FireEventDB
    from src.database.models.fire_event_ml_assessment_db import FireEventMLAssessmentDB
    from src.database.models.fire_event_news_evidence_db import FireEventNewsEvidenceDB
    from src.database.models.fire_event_satellite_evidence_db import FireEventSatelliteEvidenceDB
    from src.database.models.satellite_hotspot_db import SatelliteHotspotDB
    from src.database.models.wildfire_report_db import WildfireReportDB

    stack.run_a()
    sqlite_reset_service(stack).reset_demo_state()

    with stack.session_factory() as session:
        for model in (FireEventDB, FireEventSatelliteEvidenceDB, FireEventNewsEvidenceDB, FireEventMLAssessmentDB, SatelliteHotspotDB, WildfireReportDB):
            assert session.query(model).count() == 0, model.__tablename__


def test_a_new_runtime_table_cannot_silently_escape_the_reset():
    all_tables = set(Base.metadata.tables)
    deleted = {model.__tablename__ for model in _DELETE_ORDER_MODELS}
    declared_preserved = {"fire_stations", "firefighting_resources", "graph_nodes", "graph_edges", "weather_stations"}
    assert deleted <= all_tables
    assert (all_tables - deleted) == declared_preserved, sorted((all_tables - deleted) ^ declared_preserved)

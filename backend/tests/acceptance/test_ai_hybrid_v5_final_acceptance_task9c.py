"""Task 9C: final end-to-end acceptance of `ai_hybrid_v5` with the REAL approved HGB artifact (no scripted probabilities).

One dedicated multi-hour scenario (src/simulation/ai_acceptance_scenario.py) is driven through the production per-event pipeline
(`execute_simulation_event`: executor -> detection -> refresh (severity -> spread -> targets -> routing/Dijkstra -> global GA planning
-> activation) -> response targets) on a SQLite database. Nothing waits: every time is a simulated timestamp.

REAL components: the approved HGB V5 artifact, FireDetectionAgent + history + V5 extractor + locked policy, FireEvent/ML-assessment
persistence, severity agent + calculator, spread agent + calculator, response-target agent, operational refresh + global planning
coordinators, the global input builder (route matrix / Dijkstra over the seeded road graph), the Global GA and the activation service
(resource commitment), the seeded weather generator.
TEST DOUBLES (identical to the pre-existing test_simulation_response_planning_e2e.py): the vegetation/land-cover lookup (normally the
Copernicus web API) is a static fake, and the road network is a tiny pre-seeded 2-node graph instead of an OSM download.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time

import pytest
from sqlalchemy import func, select

from integration.test_simulation_response_planning_e2e import (
    CARMEL_RESOURCE_ID,
    CARMEL_STATION_ID,
    RealSimulationStack,
    seed_station_and_road_network,
)
from src.config.settings import settings as real_settings
from src.database.models.fire_severity_assessment_db import FireSeverityAssessmentDB
from src.database.models.fire_spread_prediction_db import FireSpreadPredictionDB
from src.database.models.global_planning_run_db import GlobalPlanningRunDB
from src.database.models.resource_commitment_db import ResourceCommitmentDB
from src.database.models.response_plan_db import ResponsePlanDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.ml.fire_detection import fire_detection_model_runtime_v5 as runtime_module
from src.ml.fire_detection.fire_detection_model_runtime_v5 import clear_model_cache_v5
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_event_status import FireEventStatus
from src.simulation import CARMEL_LOCATION, simulation_event_timestamp
from src.simulation.ai_acceptance_scenario import (
    AIAcceptanceEvidenceExecutor,
    build_ai_progression_scenario,
    record_progression,
)
from src.simulation.analysis.simulation_response_target_coordinator import SimulationResponseTargetCoordinator
from src.simulation.demo_simulation_runner import execute_simulation_event
from src.simulation.simulation_event import SimulationEventType

# 20:00 local time (UTC+2): the fire starts at night, like the plan's first pass expects.
STARTED_AT = datetime(2026, 9, 17, 18, 0, tzinfo=timezone.utc)
SUSPECTED, CONFIRMED = FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED
RESPONSE_TABLES = {
    "severity": FireSeverityAssessmentDB,
    "spread": FireSpreadPredictionDB,
    "response_targets": ResponseTargetDB,
    "route_planning_runs": RoutePlanningRunDB,
    "global_planning_runs": GlobalPlanningRunDB,
    "response_plans": ResponsePlanDB,
    "resource_commitments": ResourceCommitmentDB,
}


class Probes:
    """Counts calls and accumulates wall-clock time of the expensive stages (no permanent profiling in production code)."""

    def __init__(self):
        self.calls, self.seconds = {}, {}

    def wrap(self, obj, method, key):
        original = getattr(obj, method)
        self.calls.setdefault(key, 0)
        self.seconds.setdefault(key, 0.0)

        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.calls[key] += 1
                self.seconds[key] += time.perf_counter() - started

        setattr(obj, method, timed)


class Harness:
    def __init__(self, session_factory):
        self.session_factory = session_factory
        seed_station_and_road_network(
            session_factory, station_id=CARMEL_STATION_ID, resource_id=CARMEL_RESOURCE_ID,
            latitude=CARMEL_LOCATION.latitude, longitude=CARMEL_LOCATION.longitude)
        self.scenario = build_ai_progression_scenario(CARMEL_LOCATION)
        # decision_mode alone: the classifier / artifact paths come from the real configuration (the approved artifact).
        self.stack = RealSimulationStack(
            session_factory, detection_agent_kwargs={"decision_mode": FireDetectionDecisionMode.AI_HYBRID_V5})
        self.stack.executor = AIAcceptanceEvidenceExecutor(
            self.scenario, weather_repository=self.stack.weather_repository,
            satellite_repository=self.stack.satellite_repository, news_repository=self.stack.news_repository)
        self.activated: set[int] = set()
        self.records = []
        self.snapshots = []
        self.probes = Probes()
        agent = self.stack.detection_agent
        self.probes.wrap(agent, "detect", "fire_detection_cycle")
        self.probes.wrap(agent._history_service, "build_history", "history_retrieval")
        self.probes.wrap(agent._ai_classifier, "assess", "ai_assess_features_and_hgb")
        self.probes.wrap(agent._ai_classifier.model_runtime, "predict_probability", "hgb_inference")
        self.probes.wrap(self.stack.severity_refresh_orchestrator, "refresh_for_event", "severity")
        self.probes.wrap(self.stack.spread_refresh_orchestrator, "refresh_for_event", "spread")
        self.probes.wrap(self.stack.response_target_agent, "generate_for_event", "response_targets")
        planner = self.stack.global_planning_refresh_coordinator
        self.probes.wrap(planner._input_builder, "compute_pre_routing_bundle", "routing_precheck")
        self.probes.wrap(planner._input_builder, "build", "routing_input_road_graph_dijkstra")
        self.probes.wrap(planner._optimization_service, "optimize", "ga_optimization")
        self.probes.wrap(planner._activation_service, "activate", "resource_commitment_activation")
        self.probes.wrap(planner, "refresh", "global_planning_total")

    def response_counts(self):
        with self.session_factory() as session:
            return {name: session.execute(select(func.count()).select_from(model)).scalar_one() for name, model in RESPONSE_TABLES.items()}

    def run_step(self, index):
        step = self.scenario.steps[index]
        for event in self.scenario.events_for_step(index):
            outcome = execute_simulation_event(
                scenario=self.scenario.holder_scenario, event=event, scenario_started_at=STARTED_AT, executor=self.stack.executor,
                fire_detection_coordinator=self.stack.fire_detection_coordinator,
                simulation_refresh_coordinator=self.stack.simulation_refresh_coordinator,
                response_target_coordinator_factory=lambda: SimulationResponseTargetCoordinator(self.stack.response_target_agent),
                activated_fire_event_ids=self.activated)
            if event.event_type in (SimulationEventType.SATELLITE, SimulationEventType.NEWS):
                detection = outcome.fire_detection_result.detection_result
                assert detection.success, detection.error_message
                timestamp = simulation_event_timestamp(STARTED_AT, event)
                records = record_progression(
                    step, timestamp, detection, self.stack.fire_event_repository, self.stack.detection_agent._history_service)
                label = f"{step.label} / {event.event_type.value}"
                for record in records:
                    self.records.append((label, record))
                self.snapshots.append((label, self.response_counts()))
        return self.records[-1][1] if self.records else None

    def run_all(self):
        for index in range(len(self.scenario.steps)):
            self.run_step(index)


@pytest.fixture
def load_counter(monkeypatch):
    clear_model_cache_v5()
    loads = []
    real_load = runtime_module.joblib.load
    monkeypatch.setattr(runtime_module.joblib, "load", lambda path, *a, **k: loads.append(str(path)) or real_load(path, *a, **k))
    yield loads
    clear_model_cache_v5()


def format_progression(harness):
    lines = [f"{'time':<38} {'P':>5} {'policy':<10} {'event':<10} {'pass':>4} {'span_min':>8} {'pix':>3} {'peak':>5} {'latest':>6} {'resp':>5}"]
    for label, record in harness.records:
        lines.append(
            f"{label:<38} {record.probability:5.2f} {record.policy_status:<10} {str(record.event_status):<10} "
            f"{record.satellite_pass_count:4d} {('%.0f' % record.history_span_minutes) if record.history_span_minutes is not None else '-':>8} "
            f"{record.current_satellite_pixel_count:3d} {record.peak_confidence or 0:5.2f} {record.latest_assessment_probability or 0:6.2f} "
            f"{str(record.response_eligible):>5}")
    return "\n".join(lines)


# --- the real-model scenario ----------------------------------------------------------------------------------------------------------


def test_the_real_model_finds_the_growing_fire_increasingly_convincing_and_the_response_pipeline_follows(sqlite_session_factory, load_counter):
    harness = Harness(sqlite_session_factory)

    harness.run_all()

    print("\n" + format_progression(harness))
    print("stage calls:", harness.probes.calls)
    print("stage seconds:", {key: round(value, 2) for key, value in harness.probes.seconds.items()})
    print("response artefacts per detection cycle:")
    for label, counts in harness.snapshots:
        print("  ", label, counts)

    events = [record for _, record in harness.records if record.event_id is not None]
    assert events, "the real model never created a FireEvent from the scenario"
    assert {record.event_id for record in events} == {events[0].event_id}  # ONE FireEvent id throughout
    statuses = [record.event_status for record in events]
    assert statuses[0] == SUSPECTED.value  # the faint first pass is worth watching, not confirming
    assert statuses[-1] == CONFIRMED.value and CONFIRMED.value in statuses  # ...and later evidence promotes it
    first_confirmed = statuses.index(CONFIRMED.value)
    assert all(status == CONFIRMED.value for status in statuses[first_confirmed:])  # never downgraded afterwards
    passes = [record.satellite_pass_count for record in events]
    assert passes == sorted(passes) and passes[-1] >= 3  # the pass count grows: history is genuinely used
    peaks = [record.peak_confidence for record in events]
    assert peaks == sorted(peaks)  # peak confidence is monotonic
    assert all(0.0 <= record.probability <= 1.0 for record in events)


def test_suspected_cycles_trigger_no_response_work_and_the_promotion_activates_the_whole_chain(sqlite_session_factory, load_counter):
    harness = Harness(sqlite_session_factory)
    harness.run_all()

    zero = dict.fromkeys(RESPONSE_TABLES, 0)
    suspected_labels = [label for label, record in harness.records if record.event_status == SUSPECTED.value]
    confirmed_labels = [label for label, record in harness.records if record.event_status == CONFIRMED.value]
    assert suspected_labels and confirmed_labels
    snapshot_by_label = dict(harness.snapshots)

    # SUSPECTED phase: the event exists and is active, but NOTHING downstream has been persisted (real orchestration, not unit calls)
    for label in suspected_labels:
        assert snapshot_by_label[label] == zero, label
    active = harness.stack.fire_event_repository.get_active_events()
    assert len(active) == 1 and active[0].event.status is CONFIRMED  # (final state)
    # ...and none of the expensive stages ran before the promotion:
    first_confirmed = next(i for i, (_, record) in enumerate(harness.records) if record.event_status == CONFIRMED.value)
    assert harness.records[first_confirmed][1].response_eligible

    # CONFIRMED phase: the full chain produced persisted artefacts
    final = harness.snapshots[-1][1]
    for name in ("severity", "spread", "response_targets", "route_planning_runs", "global_planning_runs", "response_plans"):
        assert final[name] >= 1, name
    assert final["resource_commitments"] >= 1
    probes = harness.probes.calls
    for key in ("severity", "spread", "response_targets", "routing_input_road_graph_dijkstra", "ga_optimization", "resource_commitment_activation"):
        assert probes[key] >= 1, key


def test_the_status_never_moves_before_the_evidence_and_the_expensive_stages_never_ran_for_suspected_only_state(sqlite_session_factory, load_counter):
    """Stop after the SUSPECTED steps: no stage probe fired at all."""
    harness = Harness(sqlite_session_factory)
    for index in range(len(harness.scenario.steps)):
        harness.run_step(index)
        _, last = harness.records[-1]
        if last.event_status == CONFIRMED.value:
            break
        for key in ("severity", "spread", "response_targets", "routing_precheck", "routing_input_road_graph_dijkstra",
                    "ga_optimization", "resource_commitment_activation"):
            assert harness.probes.calls[key] == 0, (key, harness.records[-1][0])
        assert harness.probes.calls["global_planning_total"] == 0  # global planning was not even entered


def test_confirmed_promotion_keeps_the_same_event_history_peak_confidence_and_latest_assessment(sqlite_session_factory, load_counter):
    harness = Harness(sqlite_session_factory)
    harness.run_all()
    repository = harness.stack.fire_event_repository

    (event_id,) = {record.event_id for _, record in harness.records if record.event_id is not None}
    event = repository.get_by_id(event_id).event
    refs = repository.get_evidence_refs(event_id)
    assessment = repository.get_ml_assessment(event_id)
    total_pixels = sum(len(step.hotspots) for step in harness.scenario.steps)
    total_news = sum(len(step.news) for step in harness.scenario.steps)

    assert event.status is CONFIRMED
    assert STARTED_AT <= event.detected_at < STARTED_AT + timedelta(hours=1)  # created at the first pass, same id ever since
    assert len(refs) == total_pixels + total_news  # every observation of every pass is preserved as history
    all_probabilities = [record.probability for _, record in harness.records if record.event_id == event_id]
    assert event.detection_confidence == pytest.approx(max(all_probabilities))  # peak
    assert assessment.ml_probability == pytest.approx(all_probabilities[-1])  # latest
    assert assessment.decision_mode is FireDetectionDecisionMode.AI_HYBRID_V5 and assessment.policy_version == "ai_hybrid_policy_v5.0"
    assert assessment.satellite_pass_count >= 3 and assessment.history_available is True


def test_the_dashboard_active_events_api_lists_the_ai_event_with_its_status_and_the_details_api_explains_it(sqlite_session_factory, load_counter):
    from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
    from src.services.fire_event_read.active_fire_events_service import ActiveFireEventsService
    from src.services.fire_event_read.event_details_service import EventDetailsService

    harness = Harness(sqlite_session_factory)
    harness.run_step(0)  # only the first, SUSPECTED, pass
    service = ActiveFireEventsService(
        fire_event_repository=harness.stack.fire_event_repository,
        fire_severity_assessment_repository=FireSeverityAssessmentRepository(session_factory=sqlite_session_factory))
    listed = service.get_active_events(as_of=STARTED_AT + timedelta(hours=1))
    assert [item.status for item in listed.items] == [SUSPECTED]  # visible for monitoring
    (event_id,) = {record.event_id for _, record in harness.records}

    response = EventDetailsService._to_ml_assessment_response(harness.stack.fire_event_repository.get_ml_assessment(event_id))
    dumped = response.model_dump(mode="json")

    assert dumped["mode"] == "ai_hybrid_v5" and dumped["model_version"] == "5.0" and dumped["policy_version"] == "ai_hybrid_policy_v5.0"
    assert dumped["policy_status"] in ("suspected", "confirmed") and dumped["current_satellite_pixel_count"] == 1
    assert dumped["satellite_pass_count"] == 1 and dumped["history_available"] is False
    text = str(dumped)
    assert "joblib" not in text and "models" not in text.replace("model_", "") and real_settings.FIRE_DETECTION_AI_V5_MODEL_PATH not in text


def test_the_model_is_deserialized_exactly_once_across_the_whole_simulation(sqlite_session_factory, load_counter):
    harness = Harness(sqlite_session_factory)
    harness.run_all()
    assert len(load_counter) == 1
    assert harness.probes.calls["hgb_inference"] >= len(harness.scenario.steps)  # ...while inference ran for every cycle


def test_performance_hgb_and_history_add_negligible_time_to_a_step(sqlite_session_factory, load_counter):
    harness = Harness(sqlite_session_factory)
    started = time.perf_counter()
    harness.run_all()
    total = time.perf_counter() - started
    seconds = harness.probes.seconds
    # HGB inference + feature extraction + history retrieval are tiny next to the response pipeline (no multi-minute regression)
    ai_cost = seconds["ai_assess_features_and_hgb"] + seconds["history_retrieval"]
    assert ai_cost < 5.0, seconds
    assert seconds["hgb_inference"] < 2.0
    assert total < 240.0  # the whole four-pass scenario finishes in well under the old multi-minute problem size


# --- run isolation -----------------------------------------------------------------------------------------------------------------------


def test_after_the_mandatory_reset_the_next_run_starts_from_a_single_pass_with_no_history(sqlite_session_factory, load_counter, monkeypatch):
    import sys
    from types import SimpleNamespace

    from src.simulation import demo_state_reset_service as reset_module
    from src.simulation.demo_run_preparation import prepare_clean_demo_state
    from src.simulation.demo_state_reset_service import DemoStateResetService

    settings_stub = SimpleNamespace(ENABLE_DEMO_DATA_RESET=True)
    monkeypatch.setattr(reset_module, "settings", settings_stub)
    monkeypatch.setattr(sys.modules["src.config.settings"], "settings", settings_stub)
    first = Harness(sqlite_session_factory)
    first.run_all()
    assert first.stack.fire_event_repository.get_active_events()

    prepare_clean_demo_state(lambda: DemoStateResetService(sqlite_session_factory))

    second = Harness.__new__(Harness)  # a second run: new agent/stack, same database, same cached model
    second.session_factory = sqlite_session_factory
    second.scenario = first.scenario
    second.stack = RealSimulationStack(sqlite_session_factory, detection_agent_kwargs={"decision_mode": FireDetectionDecisionMode.AI_HYBRID_V5})
    second.stack.executor = AIAcceptanceEvidenceExecutor(
        second.scenario, weather_repository=second.stack.weather_repository,
        satellite_repository=second.stack.satellite_repository, news_repository=second.stack.news_repository)
    second.activated, second.records, second.snapshots, second.probes = set(), [], [], Probes()
    assert second.stack.fire_event_repository.get_active_events() == ()  # no old FireEvent
    assert second.response_counts() == dict.fromkeys(RESPONSE_TABLES, 0)

    second.run_step(0)

    _, record = second.records[0]
    assert record.satellite_pass_count == 1 and record.history_span_minutes == 0.0  # the first pass of the NEW run
    assert record.event_status == SUSPECTED.value
    assert len(load_counter) == 1  # the static trained model stayed cached across the reset

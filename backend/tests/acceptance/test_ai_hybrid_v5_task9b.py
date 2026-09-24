"""Task 9B: the approved HGB V5 AI Hybrid detector inside FireDetectionAgent (mode `ai_hybrid_v5`), on a real SQLite stack.

Real repositories, real evidence service, real history service, real V5 feature extractor and the real locked policy. Only the
model's probability is scripted where a test needs a deterministic status progression; separate tests use the REAL approved
artifact. No sleeping - every time is a simulated timestamp.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import sys
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.config.settings  # noqa: F401 - registers the settings submodule
from src.agents.analysis import FireDetectionAgent
from src.agents.analysis import fire_detection_agent as agent_module
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_detection.fire_detection_decision_policy import FireDetectionHybridPolicy
from src.config.settings import settings as real_settings
from src.database.base import Base
from src.ml.fire_detection import fire_detection_model_runtime_v5 as runtime_module
from src.ml.fire_detection.fire_detection_ai_hybrid_runtime_v5 import FireDetectionAIHybridClassifierV5
from src.ml.fire_detection.fire_detection_model_runtime_v5 import (
    FireDetectionModelV5InferenceError,
    FireDetectionModelV5Runtime,
    clear_model_cache_v5,
)
from src.models import SatelliteHotspot
from src.models.fire_detection_decision import FireDetectionDecision
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_ml_rule_agreement import FireDetectionMLRuleAgreement
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_event_response_eligibility import is_response_eligible
from src.models.fire_event_status import FireEventStatus
from src.models.fire_report import WildfireReport
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength
from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.fire_detection import FireDetectionEvidenceService, FireDetectionHistoryService
from src.services.operational_refresh.operational_refresh_orchestrator import OperationalRefreshOrchestrator
from src.services.operational_refresh.operational_refresh_result import OperationalRefreshStatus
from src.simulation import demo_state_reset_service as reset_module
from src.simulation.demo_state_reset_service import DemoStateResetService

T0 = datetime(2026, 9, 14, 6, 0, tzinfo=timezone.utc)
LAT, LON = 32.731, 35.046
SUSPECTED, CONFIRMED = FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED
MODEL_PATH, METADATA_PATH = real_settings.FIRE_DETECTION_AI_V5_MODEL_PATH, real_settings.FIRE_DETECTION_AI_V5_METADATA_PATH


class ScriptedRuntime:
    """Scripted P(fire) per call; records the V5 feature dict of every call (so 'history was used' is observable)."""

    def __init__(self, *probabilities):
        self.probabilities = list(probabilities)
        self.features = []

    def ensure_loaded(self):
        return SimpleNamespace(model_name="fire_detection_hgb_v5", model_version="5.0", feature_schema_version="v5",
                               policy_version="ai_hybrid_policy_v5.0")

    def predict_probability(self, features):
        self.features.append(features.as_dict())
        return self.probabilities.pop(0)


class FailingRuntime(ScriptedRuntime):
    def predict_probability(self, features):
        raise FireDetectionModelV5InferenceError("the V5 model failed to score the candidate.")


class Stack:
    def __init__(self, session_factory, runtime=None, calculator=None, history_service=None, real_model=False):
        self.session_factory = session_factory
        self.satellites = SatelliteHotspotRepository(session_factory=session_factory)
        self.news = NewsRepository(session_factory=session_factory)
        self.events = FireEventRepository(session_factory=session_factory)
        self.evidence = FireDetectionEvidenceService(satellite_repository=self.satellites, news_repository=self.news)
        self.runtime = FireDetectionModelV5Runtime(MODEL_PATH, METADATA_PATH) if real_model else runtime
        self.agent = FireDetectionAgent(
            evidence_service=self.evidence,
            calculator=calculator or FireDetectionCalculator(),
            fire_event_repository=self.events,
            satellite_repository=self.satellites,
            news_repository=self.news,
            decision_mode=FireDetectionDecisionMode.AI_HYBRID_V5,
            ai_classifier=FireDetectionAIHybridClassifierV5(self.runtime),
            history_service=history_service,
        )

    def hotspot(self, moment, latitude=LAT, longitude=LON, confidence="n", frp=8.0):
        self.satellites.save_hotspot(
            SatelliteHotspot(latitude=latitude, longitude=longitude, detected_at=moment, confidence=confidence, frp=frp,
                             brightness=330.0, satellite="NOAA-20", instrument="VIIRS", day_night="D"))

    def two_hotspots(self, moment):
        self.hotspot(moment)
        self.hotspot(moment, latitude=LAT + 0.002)

    def report(self, moment, index=1, strength=NewsWildfireSignalStrength.STRONG):
        self.news.save_report(
            WildfireReport(source_url=f"https://example.com/{index}", source_feed="Example", title="Wildfire", summary="Smoke.",
                           location_name=None, latitude=LAT, longitude=LON, published_at=moment, fetched_at=moment,
                           wildfire_signal_strength=strength))

    def detect(self, moment):
        return self.agent.detect(moment + timedelta(minutes=5))

    def detect_ok(self, moment):
        result = self.detect(moment)
        assert result.success, result.error_message
        return result

    def event(self, event_id):
        return self.events.get_by_id(event_id).event


@pytest.fixture
def shared_session_factory():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    try:
        yield sessionmaker(bind=engine, expire_on_commit=False)
    finally:
        engine.dispose()


@pytest.fixture(autouse=True)
def fresh_model_cache():
    clear_model_cache_v5()
    yield
    clear_model_cache_v5()


def scripted_stack(session_factory, *probabilities, **kwargs):
    return Stack(session_factory, runtime=ScriptedRuntime(*probabilities), **kwargs)


# --- mode plumbing ---------------------------------------------------------------------------------------------------------


def test_ai_hybrid_v5_is_an_accepted_explicit_mode_that_is_not_the_default():
    assert FireDetectionDecisionMode("ai_hybrid_v5") is FireDetectionDecisionMode.AI_HYBRID_V5
    assert {m.value for m in FireDetectionDecisionMode} >= {"rule_only", "shadow", "hybrid", "ai_hybrid_v5"}
    # The application's SOURCE default stays shadow (Task 9C: a demo environment may opt in through its own .env, so this
    # deliberately does not look at the environment-derived setting).
    assert sys.modules["src.config.settings"].DEFAULT_FIRE_DETECTION_DECISION_MODE == "shadow"


def test_the_configured_mode_builds_the_ai_path_and_loads_nothing_at_construction(shared_session_factory, monkeypatch):
    monkeypatch.setattr(agent_module, "settings", SimpleNamespace(
        FIRE_DETECTION_DECISION_MODE="ai_hybrid_v5", FIRE_DETECTION_AI_V5_MODEL_PATH=MODEL_PATH,
        FIRE_DETECTION_AI_V5_METADATA_PATH=METADATA_PATH))
    monkeypatch.setattr(runtime_module.joblib, "load", lambda path: pytest.fail("the artifact must load lazily"))
    satellites, news = SatelliteHotspotRepository(session_factory=shared_session_factory), NewsRepository(session_factory=shared_session_factory)

    agent = FireDetectionAgent(
        evidence_service=FireDetectionEvidenceService(satellite_repository=satellites, news_repository=news),
        calculator=FireDetectionCalculator(), fire_event_repository=FireEventRepository(session_factory=shared_session_factory))

    assert agent.decision_mode is FireDetectionDecisionMode.AI_HYBRID_V5


def test_the_legacy_rule_policy_refuses_ai_hybrid_v5_so_rule_decisions_can_never_be_labelled_as_ai():
    with pytest.raises(ValueError, match="does not implement ai_hybrid_v5"):
        FireDetectionHybridPolicy(mode=FireDetectionDecisionMode.AI_HYBRID_V5)


def test_the_v5_agent_never_uses_fire_danger_context():
    source = open(agent_module.__file__, encoding="utf-8").read()
    assert "fire_danger" not in source.lower() and "fire_detection_context" not in source


# --- new candidates ---------------------------------------------------------------------------------------------------------


def test_a_no_event_candidate_creates_no_fire_event(shared_session_factory):
    stack = scripted_stack(shared_session_factory, 0.30)
    stack.hotspot(T0)

    result = stack.detect_ok(T0)

    assert result.event_ids == () and result.events_created == 0 and result.no_event_count == 1
    assert stack.events.get_recent(10) == ()
    (assessment,) = result.candidate_assessments  # diagnostics are returned in memory: no fake FireEvent for a negative
    assert assessment.event_id is None and assessment.final_status is FireDetectionStatus.NO_EVENT
    assert assessment.decision_mode is FireDetectionDecisionMode.AI_HYBRID_V5 and assessment.ml_probability == pytest.approx(0.30)


def test_a_suspected_candidate_creates_an_active_suspected_event_with_the_probability_as_confidence(shared_session_factory):
    stack = scripted_stack(shared_session_factory, 0.61)
    stack.hotspot(T0)

    result = stack.detect_ok(T0)

    (event_id,) = result.event_ids
    event = stack.event(event_id)
    assert event.status is SUSPECTED and event.detection_confidence == pytest.approx(0.61)
    assert event.methodology == "ECOGUARD_AI_HYBRID_DETECTION" and event.methodology_version == "ai_hybrid_policy_v5.0"
    assert [e.id for e in stack.events.get_active_events()] == [event_id]
    assert len(stack.events.get_evidence_refs(event_id)) == 1


def test_a_confirmed_candidate_creates_a_confirmed_event(shared_session_factory):
    stack = scripted_stack(shared_session_factory, 0.90)
    stack.two_hotspots(T0)

    (event_id,) = stack.detect_ok(T0).event_ids

    assert stack.event(event_id).status is CONFIRMED and stack.event(event_id).detection_confidence == pytest.approx(0.90)


def test_a_high_probability_with_a_single_pixel_creates_only_a_suspected_event(shared_session_factory):
    stack = scripted_stack(shared_session_factory, 0.95)
    stack.hotspot(T0)
    (event_id,) = stack.detect_ok(T0).event_ids
    assert stack.event(event_id).status is SUSPECTED


def test_the_ml_assessment_row_records_the_full_ai_audit_trail(shared_session_factory):
    stack = scripted_stack(shared_session_factory, 0.61)
    stack.hotspot(T0)

    (event_id,) = stack.detect_ok(T0).event_ids

    row = stack.events.get_ml_assessment(event_id)
    assert row.decision_mode is FireDetectionDecisionMode.AI_HYBRID_V5
    assert row.ml_available and row.ml_probability == pytest.approx(0.61)
    assert (row.ml_model_name, row.ml_model_version, row.ml_feature_schema_version) == ("fire_detection_hgb_v5", "5.0", "v5")
    assert row.policy_version == "ai_hybrid_policy_v5.0" and row.policy_status is FireDetectionStatus.SUSPECTED
    assert row.history_available is False and row.satellite_pass_count == 1 and row.current_satellite_pixel_count == 1
    assert row.updated_at.tzinfo is not None


# --- existing event: progression, monotonic status and peak confidence ---------------------------------------------------------


def test_a_suspected_event_stays_suspected_on_a_suspected_result_and_is_not_dismissed_by_a_no_event_result(shared_session_factory):
    stack = scripted_stack(shared_session_factory, 0.55, 0.60, 0.20)
    stack.hotspot(T0)
    (event_id,) = stack.detect_ok(T0).event_ids
    stack.hotspot(T0 + timedelta(hours=3))
    stack.detect_ok(T0 + timedelta(hours=3))
    assert stack.event(event_id).status is SUSPECTED

    stack.hotspot(T0 + timedelta(hours=6))
    result = stack.detect_ok(T0 + timedelta(hours=6))  # P = 0.20 -> NO_EVENT for this candidate

    assert result.event_ids == (event_id,) and len(stack.events.get_recent(10)) == 1
    assert stack.event(event_id).status is SUSPECTED  # never dismissed here (expiry is a separate lifecycle task)
    assert len(stack.events.get_evidence_refs(event_id)) == 3  # a weak pass is still a pass in the event history


def test_suspected_is_promoted_to_confirmed_on_the_same_event_id_with_its_history(shared_session_factory):
    stack = scripted_stack(shared_session_factory, 0.61, 0.84)
    stack.hotspot(T0)
    (event_id,) = stack.detect_ok(T0).event_ids
    assert stack.event(event_id).status is SUSPECTED

    stack.two_hotspots(T0 + timedelta(hours=3))
    result = stack.detect_ok(T0 + timedelta(hours=3))

    assert result.event_ids == (event_id,) and result.events_updated == 1 and len(stack.events.get_recent(10)) == 1
    assert stack.event(event_id).status is CONFIRMED
    assert stack.event(event_id).detected_at == T0  # identity and history preserved
    assert len(stack.events.get_evidence_refs(event_id)) == 3


def test_a_confirmed_event_is_never_downgraded_and_confidence_is_the_monotonic_peak(shared_session_factory):
    """T+0 P=0.61 -> 0.61; T+3h P=0.84 -> 0.84 (confirmed); T+6h P=0.55 -> confidence stays 0.84, status stays CONFIRMED."""
    stack = scripted_stack(shared_session_factory, 0.61, 0.84, 0.55)
    stack.hotspot(T0)
    (event_id,) = stack.detect_ok(T0).event_ids
    assert stack.event(event_id).detection_confidence == pytest.approx(0.61)
    stack.two_hotspots(T0 + timedelta(hours=3))
    stack.detect_ok(T0 + timedelta(hours=3))
    assert stack.event(event_id).detection_confidence == pytest.approx(0.84)
    stack.hotspot(T0 + timedelta(hours=6))
    stack.detect_ok(T0 + timedelta(hours=6))

    event = stack.event(event_id)
    assert event.status is CONFIRMED and event.detection_confidence == pytest.approx(0.84)  # PEAK confidence
    row = stack.events.get_ml_assessment(event_id)
    assert row.ml_probability == pytest.approx(0.55)  # LATEST assessment - a different value with a different meaning
    assert row.policy_status is FireDetectionStatus.SUSPECTED  # what the policy said this time; the event stays CONFIRMED


def test_a_no_event_result_never_lowers_the_peak_confidence(shared_session_factory):
    stack = scripted_stack(shared_session_factory, 0.70, 0.10)
    stack.hotspot(T0)
    (event_id,) = stack.detect_ok(T0).event_ids
    stack.hotspot(T0 + timedelta(hours=2, minutes=30))  # outside the 120 min lookback, inside the 6 h matching window

    stack.detect_ok(T0 + timedelta(hours=2, minutes=30))

    assert stack.event(event_id).detection_confidence == pytest.approx(0.70)
    assert stack.events.get_ml_assessment(event_id).ml_probability == pytest.approx(0.10)


# --- sequential T+0 / T+3h / T+6h with real history ----------------------------------------------------------------------------


def test_sequential_observations_build_one_event_with_growing_history(shared_session_factory):
    stack = scripted_stack(shared_session_factory, 0.50, 0.70, 0.90)
    ids, ref_counts = [], []
    for hours in (0, 3, 6):
        moment = T0 + timedelta(hours=hours)
        if hours == 6:
            stack.two_hotspots(moment)  # multi-pixel corroboration on the last pass
        else:
            stack.hotspot(moment)
        result = stack.detect_ok(moment)
        ids.append(result.event_ids)
        ref_counts.append(len(stack.events.get_evidence_refs(result.event_ids[0])))

    assert len({event_ids for event_ids in ids}) == 1 and len(ids[0]) == 1  # ONE FireEvent id throughout
    (event_id,) = ids[0]
    assert ref_counts == [1, 2, 4]  # evidence accumulates
    passes = [features["satellite_pass_count"] for features in stack.runtime.features]
    spans = [features["satellite_history_span_minutes"] for features in stack.runtime.features]
    assert passes == [1, 2, 3] and spans == [0.0, pytest.approx(180.0), pytest.approx(360.0)]  # history IS used
    assert math.isnan(stack.runtime.features[0]["satellite_frp_trend_per_hour"])
    assert not math.isnan(stack.runtime.features[2]["satellite_frp_trend_per_hour"])
    assert stack.event(event_id).status is CONFIRMED  # promoted at T+6h (P 0.90 with 2 current pixels)
    row = stack.events.get_ml_assessment(event_id)
    assert row.history_available is True and row.satellite_pass_count == 3 and row.current_satellite_pixel_count == 2


def test_the_history_service_failure_fails_the_cycle_instead_of_becoming_empty_history(shared_session_factory):
    class BrokenHistory:
        def build_history(self, fire_event, as_of):
            raise RuntimeError("database is down")

    stack = scripted_stack(shared_session_factory, 0.61, 0.70)
    stack.hotspot(T0)
    (event_id,) = stack.detect_ok(T0).event_ids
    stack.agent._history_service = BrokenHistory()
    stack.hotspot(T0 + timedelta(hours=3))

    result = stack.detect(T0 + timedelta(hours=3))

    assert result.success is False and "history could not be loaded" in result.error_message
    assert "ai_hybrid_v5" in result.error_message and "database is down" not in result.error_message
    assert len(stack.runtime.features) == 1  # the model was NOT called with a made-up empty history
    assert len(stack.events.get_evidence_refs(event_id)) == 1  # nothing was attached
    assert stack.event(event_id).status is SUSPECTED


# --- the rule decision can never override the AI decision --------------------------------------------------------------------


class StubCalculator:
    """A rule calculator with a fixed verdict (the real one never says NO_EVENT for real evidence)."""

    def __init__(self, status):
        self.status = status
        self.calls = 0

    def evaluate(self, evidence):
        self.calls += 1
        if self.status is FireDetectionStatus.NO_EVENT:
            return FireDetectionDecision(status=FireDetectionStatus.NO_EVENT, confidence=0.0, latitude=None, longitude=None,
                                         supporting_evidence=())
        return FireDetectionCalculator().evaluate(evidence)


def test_a_rule_suspected_verdict_cannot_create_an_event_when_the_ai_says_no_event(shared_session_factory):
    stack = scripted_stack(shared_session_factory, 0.20)
    stack.hotspot(T0)  # the rule calculator alone would create a SUSPECTED event for a lone hotspot

    result = stack.detect_ok(T0)

    assert stack.events.get_recent(10) == () and result.no_event_count == 1
    (assessment,) = result.candidate_assessments
    assert assessment.rule_status is FireDetectionStatus.SUSPECTED  # diagnostics only
    assert assessment.final_status is FireDetectionStatus.NO_EVENT
    assert assessment.agreement is FireDetectionMLRuleAgreement.RULE_STRONGER


def test_a_rule_confirmed_verdict_cannot_confirm_when_the_ai_says_suspected(shared_session_factory):
    stack = scripted_stack(shared_session_factory, 0.50)
    stack.hotspot(T0)
    stack.report(T0 + timedelta(minutes=5))  # satellite + strong news: the RULE calculator says CONFIRMED

    result = stack.detect_ok(T0)

    (event_id,) = result.event_ids
    assert result.candidate_assessments[0].rule_status is FireDetectionStatus.CONFIRMED
    assert stack.event(event_id).status is SUSPECTED  # the V5 policy decides, not the rules
    assert stack.event(event_id).detection_confidence == pytest.approx(0.50)  # AI probability, not the rule confidence


def test_a_rule_no_event_verdict_cannot_block_an_ai_suspected_event(shared_session_factory):
    calculator = StubCalculator(FireDetectionStatus.NO_EVENT)
    stack = scripted_stack(shared_session_factory, 0.65, calculator=calculator)
    stack.hotspot(T0)

    result = stack.detect_ok(T0)

    (event_id,) = result.event_ids
    assert calculator.calls == 1 and stack.event(event_id).status is SUSPECTED
    assert result.candidate_assessments[0].agreement is FireDetectionMLRuleAgreement.ML_STRONGER


def test_a_rule_suspected_verdict_cannot_cap_an_ai_confirmed_event(shared_session_factory):
    stack = scripted_stack(shared_session_factory, 0.90)
    stack.two_hotspots(T0)  # rules: SUSPECTED (satellite-only is never CONFIRMED by rules)
    (event_id,) = stack.detect_ok(T0).event_ids
    assert stack.events.get_ml_assessment(event_id).rule_status is FireDetectionStatus.SUSPECTED
    assert stack.event(event_id).status is CONFIRMED


# --- explicit failure, no silent fallback ------------------------------------------------------------------------------------------


def rule_confirmable(stack):
    """Evidence the RULE calculator would happily turn into a CONFIRMED FireEvent - so any fallback would be visible."""
    stack.hotspot(T0)
    stack.report(T0 + timedelta(minutes=5))


def failing_stack(session_factory, runtime):
    return Stack(session_factory, runtime=runtime)


def test_a_missing_artifact_fails_the_cycle_explicitly_and_creates_no_event(shared_session_factory, tmp_path):
    stack = failing_stack(shared_session_factory, FireDetectionModelV5Runtime(tmp_path / "gone.joblib", tmp_path / "gone.json"))
    rule_confirmable(stack)

    result = stack.detect(T0)

    assert result.success is False and "ai_hybrid_v5" in result.error_message and "missing" in result.error_message
    assert str(tmp_path) not in result.error_message
    assert stack.events.get_recent(10) == ()  # NOT a rule-based event in disguise


def test_a_corrupt_artifact_fails_the_cycle_explicitly(shared_session_factory, tmp_path):
    import shutil

    model, metadata = tmp_path / "m.joblib", tmp_path / "m.json"
    shutil.copy(METADATA_PATH, metadata)
    model.write_bytes(b"corrupt")
    stack = failing_stack(shared_session_factory, FireDetectionModelV5Runtime(model, metadata))
    rule_confirmable(stack)

    result = stack.detect(T0)

    assert result.success is False and "deserialized" in result.error_message and stack.events.get_recent(10) == ()


def test_a_metadata_mismatch_fails_the_cycle_explicitly(shared_session_factory, tmp_path):
    import json
    import shutil

    metadata = tmp_path / "m.json"
    data = json.loads(open(METADATA_PATH, encoding="utf-8").read())
    data["feature_names"] = data["feature_names"][:-1]
    metadata.write_text(json.dumps(data), encoding="utf-8")
    model = tmp_path / "m.joblib"
    shutil.copy(MODEL_PATH, model)
    stack = failing_stack(shared_session_factory, FireDetectionModelV5Runtime(model, metadata))
    rule_confirmable(stack)

    result = stack.detect(T0)

    assert result.success is False and "not the approved artifact" in result.error_message and stack.events.get_recent(10) == ()


def test_a_classifier_exception_fails_the_cycle_without_a_fallback_event(shared_session_factory):
    stack = failing_stack(shared_session_factory, FailingRuntime())
    rule_confirmable(stack)

    result = stack.detect(T0)

    assert result.success is False and "failed to score" in result.error_message and stack.events.get_recent(10) == ()


def test_the_control_the_same_evidence_does_create_an_event_under_rule_only(shared_session_factory):
    """Proves the failure tests above are meaningful: rules WOULD have produced a CONFIRMED event from this evidence."""
    session_factory = shared_session_factory
    satellites, news = SatelliteHotspotRepository(session_factory=session_factory), NewsRepository(session_factory=session_factory)
    events = FireEventRepository(session_factory=session_factory)
    agent = FireDetectionAgent(
        evidence_service=FireDetectionEvidenceService(satellite_repository=satellites, news_repository=news),
        calculator=FireDetectionCalculator(), fire_event_repository=events,
        decision_policy=FireDetectionHybridPolicy(mode=FireDetectionDecisionMode.RULE_ONLY))
    stack = Stack(session_factory, runtime=ScriptedRuntime())
    rule_confirmable(stack)

    result = agent.detect(T0 + timedelta(minutes=5))

    assert result.success and len(result.event_ids) == 1
    assert events.get_by_id(result.event_ids[0]).event.status is CONFIRMED


# --- real artifact smoke -----------------------------------------------------------------------------------------------------------------


def test_real_artifact_end_to_end_smoke_over_a_multi_pass_incident(shared_session_factory):
    stack = Stack(shared_session_factory, real_model=True)
    for hours in (0, 3, 6):
        stack.two_hotspots(T0 + timedelta(hours=hours))
        stack.report(T0 + timedelta(hours=hours, minutes=5), index=hours + 1)
        result = stack.detect_ok(T0 + timedelta(hours=hours))
        for assessment in result.candidate_assessments:
            assert assessment.ml_available and 0.0 <= assessment.ml_probability <= 1.0
            assert assessment.decision_mode is FireDetectionDecisionMode.AI_HYBRID_V5
    events = stack.events.get_recent(10)
    assert len(events) <= 1  # at most one incident (the real model may also call a pass NO_EVENT)
    for event in events:
        row = stack.events.get_ml_assessment(event.id)
        assert row is not None and row.decision_mode is FireDetectionDecisionMode.AI_HYBRID_V5
        assert row.policy_version == "ai_hybrid_policy_v5.0"
        assert 0.0 <= event.event.detection_confidence <= 1.0


def test_two_agents_in_a_row_share_one_cached_artifact_across_runs(shared_session_factory, monkeypatch):
    loads = []
    real_load = runtime_module.joblib.load
    monkeypatch.setattr(runtime_module.joblib, "load", lambda path: loads.append(path) or real_load(path))
    for run in range(2):  # two "simulation runs", each builds its own agent
        stack = Stack(shared_session_factory, real_model=True)
        stack.hotspot(T0 + timedelta(days=run))
        stack.detect_ok(T0 + timedelta(days=run))
    assert len(loads) == 1


# --- Task 9A regression: response semantics in ai_hybrid_v5 mode -----------------------------------------------------------------------------


class SpyStage:
    def __init__(self, result=None):
        self.calls, self._result = [], result

    def refresh_for_event(self, *a, **k):
        self.calls.append("severity")
        return self._result

    def refresh(self, *a, **k):
        self.calls.append("spread")
        return self._result


class SpyTargets:
    def __init__(self):
        self.calls = []

    def generate_for_event(self, **k):
        self.calls.append("targets")
        return SimpleNamespace(status=__import__("src.agents.analysis.response_target_generation_result", fromlist=["x"]).ResponseTargetGenerationStatus.GENERATED, error_message=None)


def operational_orchestrator(events):
    severity, spread, targets = SpyStage(SimpleNamespace(name="sev")), SpyStage(SimpleNamespace(failed=False, horizon_results=())), SpyTargets()
    orchestrator = OperationalRefreshOrchestrator(
        severity_refresh_orchestrator=severity, spread_refresh_orchestrator=spread, response_target_agent=targets,
        resource_status_service=None, resource_repository=None, fire_event_repository=events)
    return orchestrator, severity, spread, targets


def test_an_ai_suspected_event_is_monitored_but_triggers_no_response_work_while_an_ai_confirmed_one_does(shared_session_factory):
    stack = scripted_stack(shared_session_factory, 0.55, 0.90)
    stack.hotspot(T0, latitude=LAT, longitude=LON)
    (suspected_id,) = stack.detect_ok(T0).event_ids
    far_lat = LAT + 0.5  # a second, distant fire
    stack.hotspot(T0 + timedelta(hours=3), latitude=far_lat, longitude=LON)
    stack.hotspot(T0 + timedelta(hours=3), latitude=far_lat + 0.002, longitude=LON)
    (confirmed_id,) = tuple(set(stack.detect_ok(T0 + timedelta(hours=3)).event_ids) - {suspected_id})

    assert stack.event(suspected_id).status is SUSPECTED and stack.event(confirmed_id).status is CONFIRMED
    # monitoring sees both; response eligibility only the confirmed one
    assert set(stack.events.get_active_fire_event_ids()) == {suspected_id, confirmed_id}
    assert set(stack.events.get_response_eligible_fire_event_ids()) == {confirmed_id}
    assert is_response_eligible(stack.event(confirmed_id).status) and not is_response_eligible(stack.event(suspected_id).status)

    later = T0 + timedelta(hours=4)
    orchestrator, severity, spread, targets = operational_orchestrator(stack.events)
    suspected_result = orchestrator.refresh_fire_event(
        fire_event_id=suspected_id, trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE, as_of=later)
    assert suspected_result.status is OperationalRefreshStatus.NOT_RESPONSE_ELIGIBLE
    assert severity.calls == [] and spread.calls == [] and targets.calls == []  # no severity / spread / targets

    confirmed_result = orchestrator.refresh_fire_event(
        fire_event_id=confirmed_id, trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE, as_of=later)
    assert confirmed_result.status is not OperationalRefreshStatus.NOT_RESPONSE_ELIGIBLE
    assert severity.calls and spread.calls and targets.calls  # the full Task 9A pipeline


def test_the_global_planner_sees_only_the_ai_confirmed_event_in_a_mixed_state(shared_session_factory):
    from src.services.global_planning.global_planning_refresh_coordinator import GlobalPlanningRefreshCoordinator

    stack = scripted_stack(shared_session_factory, 0.55, 0.90)
    stack.hotspot(T0)
    (suspected_id,) = stack.detect_ok(T0).event_ids
    stack.hotspot(T0 + timedelta(hours=3), latitude=LAT + 0.5)
    stack.hotspot(T0 + timedelta(hours=3), latitude=LAT + 0.502)
    (confirmed_id,) = tuple(set(stack.detect_ok(T0 + timedelta(hours=3)).event_ids) - {suspected_id})

    created = []

    class Runs:
        def create_run(self, *, started_at, trigger, methodology, methodology_version, input_fingerprint, fire_event_ids):
            created.append(tuple(fire_event_ids))
            return SimpleNamespace(id=len(created))

        def set_input_fingerprint(self, *a, **k): pass
        def record_member_result(self, *a, **k): pass
        def complete_run(self, *a, **k): pass
        def get_latest_activated(self, **k): return None

    builder = SimpleNamespace(
        compute_pre_routing_bundle=lambda **k: SimpleNamespace(pre_routing_signature="s"),
        build=lambda **k: SimpleNamespace(input_fingerprint="f" * 64, current_assignments=()))
    optimization = SimpleNamespace(optimize=lambda *a, **k: SimpleNamespace(actions=(), assignment_changes=(), shortage=None, event_results=()))
    activation = SimpleNamespace(activate=lambda **k: SimpleNamespace(response_plan_ids_by_event={}))
    coordinator = GlobalPlanningRefreshCoordinator(
        fire_event_repository=stack.events, global_planning_run_repository=Runs(), input_builder=builder,
        optimization_service=optimization, activation_service=activation)

    coordinator.refresh(trigger="fire_event_update", as_of=T0 + timedelta(hours=4))

    assert created == [(confirmed_id,)] and suspected_id not in created[0]


def test_an_ai_suspected_event_is_visible_in_the_active_fire_events_service(shared_session_factory):
    from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
    from src.services.fire_event_read.active_fire_events_service import ActiveFireEventsService

    stack = scripted_stack(shared_session_factory, 0.55)
    stack.hotspot(T0)
    (event_id,) = stack.detect_ok(T0).event_ids

    result = ActiveFireEventsService(
        fire_event_repository=stack.events,
        fire_severity_assessment_repository=FireSeverityAssessmentRepository(session_factory=shared_session_factory),
    ).get_active_events(as_of=T0 + timedelta(hours=1))

    assert [item.fire_event_id for item in result.items] == [event_id]
    assert result.items[0].status is SUSPECTED
    assert result.items[0].ml_summary.mode == "ai_hybrid_v5"  # Task 9C: the dashboard card can label AI scores correctly


# --- Task 9A regression: simulation-run isolation with the V5 history ---------------------------------------------------------------------------


def test_run_b_sees_only_its_own_evidence_after_the_mandatory_reset(shared_session_factory, monkeypatch):
    settings_stub = SimpleNamespace(ENABLE_DEMO_DATA_RESET=True)
    monkeypatch.setattr(reset_module, "settings", settings_stub)
    monkeypatch.setattr(sys.modules["src.config.settings"], "settings", settings_stub)
    loads = []
    real_load = runtime_module.joblib.load
    monkeypatch.setattr(runtime_module.joblib, "load", lambda path: loads.append(path) or real_load(path))

    # Run A: T+0 / T+3h / T+6h with the REAL model (three passes of history).
    run_a = Stack(shared_session_factory, real_model=True)
    for hours in (0, 3, 6):
        run_a.hotspot(T0 + timedelta(hours=hours))
        run_a.detect_ok(T0 + timedelta(hours=hours))
    assert len(loads) == 1

    from src.simulation.demo_run_preparation import prepare_clean_demo_state

    prepare_clean_demo_state(lambda: DemoStateResetService(shared_session_factory))

    # Run B starts INSIDE Run A's 6 h matching window: only the reset can isolate it. A new agent, the same cached model.
    run_b = Stack(shared_session_factory, runtime=ScriptedRuntime(0.60))
    run_b.hotspot(T0 + timedelta(hours=7))
    result = run_b.detect_ok(T0 + timedelta(hours=7))

    (features,) = run_b.runtime.features
    assert features["satellite_pass_count"] == 1 and features["satellite_history_span_minutes"] == 0.0
    assert math.isnan(features["satellite_frp_trend_per_hour"]) and math.isnan(features["satellite_centroid_stability_km"])
    (event_id,) = result.event_ids
    assert result.events_created == 1 and len(run_b.events.get_evidence_refs(event_id)) == 1
    assert run_b.events.get_ml_assessment(event_id).history_available is False
    assert [e.id for e in run_b.events.get_active_events()] == [event_id]
    # the trained model stays cached in memory across the reset: only incident state is reset
    run_b_real = Stack(shared_session_factory, real_model=True)
    run_b_real.hotspot(T0 + timedelta(hours=8), latitude=LAT + 0.5)
    run_b_real.detect_ok(T0 + timedelta(hours=8))
    assert len(loads) == 1


# --- legacy modes are untouched ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [FireDetectionDecisionMode.RULE_ONLY, FireDetectionDecisionMode.SHADOW, FireDetectionDecisionMode.HYBRID])
def test_legacy_modes_behave_as_before_and_never_touch_the_v5_artifact(shared_session_factory, monkeypatch, mode):
    real_load = runtime_module.joblib.load  # the V3 classifier legitimately loads its own artifact through the same joblib

    def guarded_load(path, *args, **kwargs):
        assert "hgb_v5" not in str(path), "legacy modes must not load the V5 artifact"
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(runtime_module.joblib, "load", guarded_load)
    satellites, news = SatelliteHotspotRepository(session_factory=shared_session_factory), NewsRepository(session_factory=shared_session_factory)
    events = FireEventRepository(session_factory=shared_session_factory)
    agent = FireDetectionAgent(
        evidence_service=FireDetectionEvidenceService(satellite_repository=satellites, news_repository=news),
        calculator=FireDetectionCalculator(), fire_event_repository=events, satellite_repository=satellites, news_repository=news,
        decision_policy=FireDetectionHybridPolicy(mode=mode, ml_suspect_threshold=0.70))
    stack = Stack(shared_session_factory, runtime=ScriptedRuntime())
    rule_confirmable(stack)

    result = agent.detect(T0 + timedelta(minutes=5))

    assert agent.decision_mode is mode and result.success
    (event_id,) = result.event_ids
    stored = events.get_by_id(event_id).event
    assert stored.status is CONFIRMED and stored.methodology == "ECOGUARD_MULTI_SOURCE_DETECTION"  # rule decision, rule methodology
    assert stored.detection_confidence == pytest.approx(result.candidate_assessments[0].rule_confidence)
    row = events.get_ml_assessment(event_id)
    if mode is FireDetectionDecisionMode.RULE_ONLY:
        assert row is None
    else:
        assert row.decision_mode is mode and row.policy_version is None and row.policy_status is None
        assert row.history_available is None and row.satellite_pass_count is None and row.current_satellite_pixel_count is None
    assert result.candidate_assessments[0].ai_policy_version is None


# --- schema tolerance: legacy modes on a database without the Task 9B columns -------------------------------------------------------------------


class FakeV3Classifier:
    def assess(self, evidence):
        from src.models.fire_detection_ml_assessment import FireDetectionMLAssessment

        return FireDetectionMLAssessment(available=True, probability=0.9, model_name="m", model_version="1",
                                         feature_schema_version="v3", failure_reason=None)


def _unmigrated_session_factory():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        for column in ("policy_version", "policy_status", "history_available", "satellite_pass_count", "current_satellite_pixel_count"):
            connection.execute(text(f"ALTER TABLE fire_event_ml_assessments DROP COLUMN {column}"))
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def test_legacy_modes_keep_working_on_a_database_that_has_not_been_migrated():
    import sqlite3

    if tuple(int(part) for part in sqlite3.sqlite_version.split(".")[:2]) < (3, 35):
        pytest.skip("SQLite too old for DROP COLUMN")
    engine, session_factory = _unmigrated_session_factory()
    try:
        satellites, news = SatelliteHotspotRepository(session_factory=session_factory), NewsRepository(session_factory=session_factory)
        events = FireEventRepository(session_factory=session_factory)
        agent = FireDetectionAgent(
            evidence_service=FireDetectionEvidenceService(satellite_repository=satellites, news_repository=news),
            calculator=FireDetectionCalculator(), fire_event_repository=events, satellite_repository=satellites, news_repository=news,
            decision_policy=FireDetectionHybridPolicy(mode=FireDetectionDecisionMode.SHADOW),
            ml_classifier=FakeV3Classifier())
        stack = Stack(session_factory, runtime=ScriptedRuntime())
        stack.hotspot(T0)
        (event_id,) = agent.detect(T0 + timedelta(minutes=5)).event_ids
        row = events.get_ml_assessment(event_id)  # upsert + read both worked without the new columns
        assert row.decision_mode is FireDetectionDecisionMode.SHADOW and row.ml_probability == pytest.approx(0.9)
        assert [r.fire_event_id for r in events.get_ml_assessments_for_events([event_id]).values()] == [event_id]
    finally:
        engine.dispose()


def test_ai_hybrid_v5_on_an_unmigrated_database_fails_with_an_actionable_message():
    import sqlite3

    if tuple(int(part) for part in sqlite3.sqlite_version.split(".")[:2]) < (3, 35):
        pytest.skip("SQLite too old for DROP COLUMN")
    engine, session_factory = _unmigrated_session_factory()
    try:
        stack = Stack(session_factory, runtime=ScriptedRuntime(0.61))
        stack.hotspot(T0)

        result = stack.detect(T0)

        assert result.success is False
        assert "migrate_add_fire_event_ml_assessment_ai_columns" in result.error_message
        assert "ai_hybrid_v5" in result.error_message
        assert stack.events.get_recent(10) == ()  # Task 9C: rolled back - no orphan FireEvent without its assessment
    finally:
        engine.dispose()


# --- explainability reaches the (unchanged) event-details API shape ------------------------------------------------------------------------


def test_the_event_details_response_carries_the_ai_explainability_fields_but_no_feature_vector(shared_session_factory):
    from src.services.fire_event_read.event_details_service import EventDetailsService

    stack = scripted_stack(shared_session_factory, 0.61)
    stack.hotspot(T0)
    (event_id,) = stack.detect_ok(T0).event_ids

    response = EventDetailsService._to_ml_assessment_response(stack.events.get_ml_assessment(event_id))

    assert response.mode is FireDetectionDecisionMode.AI_HYBRID_V5
    assert response.model_score == pytest.approx(0.61) and response.model_version == "5.0"
    assert response.policy_version == "ai_hybrid_policy_v5.0" and response.policy_status is FireDetectionStatus.SUSPECTED
    assert response.current_satellite_pixel_count == 1 and response.satellite_pass_count == 1 and response.history_available is False
    assert "features" not in response.model_dump() and "feature_vector" not in response.model_dump()


def test_the_simulation_detection_coordinator_reports_response_eligibility_of_ai_events(shared_session_factory):
    """The simulation path (detect -> which events are response-eligible) works unchanged with AI-decided statuses."""
    from src.simulation.analysis.simulation_fire_detection_coordinator import SimulationFireDetectionCoordinator
    from src.simulation.simulation_event import SimulationEvent, SimulationEventType
    from src.simulation.simulation_event_executor import SimulationEventExecutionResult
    from src.simulation.simulation_scenario import SimulationScenario

    stack = scripted_stack(shared_session_factory, 0.55, 0.90)
    coordinator = SimulationFireDetectionCoordinator(detection_agent=stack.agent, fire_event_repository=stack.events)
    event = SimulationEvent(offset_seconds=0, event_type=SimulationEventType.SATELLITE, incident_id="i1", source_event_index=0)
    executed = SimulationEventExecutionResult(event=event, success=True, generated_count=1, saved_count=1, duplicates_skipped=0, failed_count=0)
    scenario = SimpleNamespace()

    stack.hotspot(T0)  # -> SUSPECTED
    suspected = coordinator.handle_event(scenario, event, executed, T0 + timedelta(minutes=5))
    stack.two_hotspots(T0 + timedelta(hours=3, minutes=0))  # same fire, later -> promoted by the scripted 0.90
    promoted = coordinator.handle_event(scenario, event, executed, T0 + timedelta(hours=3, minutes=5))

    assert suspected.triggered and suspected.detection_result.success
    assert suspected.response_eligible_event_ids == ()  # a SUSPECTED event never becomes response-eligible
    (event_id,) = promoted.detection_result.event_ids
    assert promoted.response_eligible_event_ids == (event_id,)  # ...until the AI policy confirms it

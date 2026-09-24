"""Task 9A: SUSPECTED is active for MONITORING but not eligible for EMERGENCY RESPONSE; CONFIRMED is both.

Full stack on SQLite where the persistence matters (real repositories, evidence service, FireDetectionAgent), and
spies at the expensive boundaries so "SUSPECTED never reaches Dijkstra / the GA / commitment" is proven, not assumed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.agents.analysis import FireDetectionAgent
from src.agents.analysis.response_target_generation_result import ResponseTargetGenerationStatus
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_detection.fire_detection_decision_policy import FireDetectionHybridPolicy
from src.models import SatelliteHotspot
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_event import FireEvent
from src.models.fire_event_response_eligibility import (
    ACTIVE_FOR_MONITORING_STATUSES,
    RESPONSE_ELIGIBLE_STATUSES,
    is_active_for_monitoring,
    is_response_eligible,
)
from src.models.fire_event_status import FireEventStatus
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.models.fire_report import WildfireReport
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength
from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.fire_detection import FireDetectionEvidenceService
from src.services.global_planning.global_planning_refresh_coordinator import (
    GlobalPlanningRefreshCoordinator,
    GlobalPlanningRefreshStatus,
)
from src.services.operational_planning_refresh.operational_planning_refresh_coordinator import (
    OperationalPlanningRefreshCoordinator,
)
from src.services.operational_refresh.operational_refresh_orchestrator import OperationalRefreshOrchestrator
from src.services.operational_refresh.operational_refresh_result import OperationalRefreshResult, OperationalRefreshStatus

T0 = datetime(2026, 9, 14, 6, 0, tzinfo=timezone.utc)
LAT, LON = 32.731, 35.046
SUSPECTED, CONFIRMED = FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED


# --- shared semantics ------------------------------------------------------------------------------------------


def test_active_for_monitoring_is_not_response_eligible():
    assert ACTIVE_FOR_MONITORING_STATUSES == {SUSPECTED, CONFIRMED}  # NOT narrowed: SUSPECTED must keep accumulating evidence
    assert RESPONSE_ELIGIBLE_STATUSES == {CONFIRMED}
    table = {
        SUSPECTED: (True, False),
        CONFIRMED: (True, True),
        FireEventStatus.RESOLVED: (False, False),
        FireEventStatus.DISMISSED: (False, False),
    }
    for status, (active, eligible) in table.items():
        assert is_active_for_monitoring(status) is active and is_response_eligible(status) is eligible, status
        assert is_response_eligible(status.value) is eligible  # accepts the persisted string value too


def test_the_eligibility_definition_is_the_only_place_that_names_confirmed_in_services():
    """No service branches on `== CONFIRMED`: eligibility is centralised (see fire_event_response_eligibility)."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src"
    offenders = []
    for path in list((root / "services").rglob("*.py")) + list((root / "simulation").rglob("*.py")) + list((root / "api").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "CONFIRMED" and getattr(node.value, "id", "") == "FireEventStatus":
                inside_set, ancestor = False, parents.get(node)
                while ancestor is not None:
                    inside_set = inside_set or isinstance(ancestor, ast.Set)
                    ancestor = parents.get(ancestor)
                if not inside_set:
                    offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == [], offenders


# --- a real SQLite stack -----------------------------------------------------------------------------------------


class Stack:
    def __init__(self, session_factory):
        self.satellites = SatelliteHotspotRepository(session_factory=session_factory)
        self.news = NewsRepository(session_factory=session_factory)
        self.events = FireEventRepository(session_factory=session_factory)
        self.evidence = FireDetectionEvidenceService(satellite_repository=self.satellites, news_repository=self.news)
        self.agent = FireDetectionAgent(
            evidence_service=self.evidence,
            calculator=FireDetectionCalculator(),
            fire_event_repository=self.events,
            satellite_repository=self.satellites,
            news_repository=self.news,
            decision_policy=FireDetectionHybridPolicy(mode=FireDetectionDecisionMode.RULE_ONLY),
        )

    def hotspot(self, moment, latitude=LAT, longitude=LON, confidence="n"):
        self.satellites.save_hotspot(
            SatelliteHotspot(latitude=latitude, longitude=longitude, detected_at=moment, confidence=confidence, frp=8.0,
                             satellite="NOAA-20", instrument="VIIRS", day_night="D")
        )

    def report(self, moment, index):
        self.news.save_report(
            WildfireReport(source_url=f"https://example.com/{index}", source_feed="Example", title="Wildfire", summary="Smoke.",
                           location_name=None, latitude=LAT, longitude=LON, published_at=moment, fetched_at=moment,
                           wildfire_signal_strength=NewsWildfireSignalStrength.STRONG)
        )

    def detect(self, moment):
        result = self.agent.detect(moment + timedelta(minutes=5))
        assert result.success, result.error_message
        return result

    def event(self, status, latitude=LAT, longitude=LON, moment=T0):
        """A persisted FireEvent with the given status (backed by one real hotspot)."""
        self.hotspot(moment, latitude, longitude)
        hotspot_id = max(h.id for h in self.satellites.get_recent_hotspots(moment + timedelta(hours=1), 600))
        return self.events.create_event(
            FireEvent(latitude=latitude, longitude=longitude, detected_at=moment, updated_at=moment, status=status,
                      detection_confidence=0.6, methodology="test", methodology_version="1"),
            supporting_evidence=(FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),),
        )


@pytest.fixture
def stack(sqlite_session_factory):
    return Stack(sqlite_session_factory)


# --- repository: active vs response-eligible ----------------------------------------------------------------------


def test_the_repository_keeps_suspected_active_but_only_confirmed_is_response_eligible(stack):
    suspected = stack.event(SUSPECTED, LAT, LON, T0)
    confirmed = stack.event(CONFIRMED, LAT + 0.3, LON, T0 + timedelta(minutes=1))

    assert {e.id for e in stack.events.get_active_events()} == {suspected.id, confirmed.id}
    assert set(stack.events.get_active_fire_event_ids()) == {suspected.id, confirmed.id}
    assert stack.events.get_response_eligible_fire_event_ids() == (confirmed.id,)
    near_a = stack.events.get_active_events_near(LAT, LON, 5.0, T0 + timedelta(hours=1))
    near_eligible = stack.events.get_response_eligible_events_near(LAT, LON, 5.0, T0 + timedelta(hours=1))
    assert [e.id for e in near_a] == [suspected.id] and near_eligible == ()
    assert [e.id for e in stack.events.get_response_eligible_events_near(LAT + 0.3, LON, 5.0, T0 + timedelta(hours=1))] == [confirmed.id]
    # a SUSPECTED event still matches later evidence (that is why ACTIVE is not narrowed)
    assert stack.events.find_matching_active_event(LAT, LON, T0 + timedelta(hours=1)).id == suspected.id


# --- the operational-refresh orchestration boundary -----------------------------------------------------------------


class SpyStage:
    def __init__(self, result=None):
        self.calls = []
        self._result = result

    def _record(self, name, *args, **kwargs):
        self.calls.append(name)
        return self._result

    def refresh_for_event(self, *args, **kwargs):
        return self._record("refresh_for_event")

    def refresh(self, *args, **kwargs):
        return self._record("refresh")


class SpyTargetAgent:
    def __init__(self):
        self.calls = []

    def generate_for_event(self, **kwargs):
        self.calls.append("generate_for_event")
        return SimpleNamespace(status=ResponseTargetGenerationStatus.GENERATED, error_message=None)

    def generate(self, **kwargs):
        self.calls.append("generate")
        return SimpleNamespace(status=ResponseTargetGenerationStatus.GENERATED, error_message=None)


def _orchestrator(events):
    spread_result = SimpleNamespace(failed=False, horizon_results=())
    severity, spread, targets = SpyStage(SimpleNamespace(name="severity")), SpyStage(spread_result), SpyTargetAgent()
    orchestrator = OperationalRefreshOrchestrator(
        severity_refresh_orchestrator=severity, spread_refresh_orchestrator=spread, response_target_agent=targets,
        resource_status_service=None, resource_repository=None, fire_event_repository=events,
    )
    return orchestrator, severity, spread, targets


@pytest.mark.parametrize("trigger", list(OperationalRefreshTriggerType))
def test_a_suspected_event_runs_no_severity_spread_or_target_work_for_any_trigger(stack, trigger):
    if trigger is OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE:
        pytest.skip("resource updates go through refresh_resource, not refresh_fire_event")
    suspected = stack.event(SUSPECTED)
    orchestrator, severity, spread, targets = _orchestrator(stack.events)

    result = orchestrator.refresh_fire_event(fire_event_id=suspected.id, trigger_type=trigger, as_of=T0 + timedelta(hours=1))

    assert result.status is OperationalRefreshStatus.NOT_RESPONSE_ELIGIBLE and result.success is True
    assert severity.calls == [] and spread.calls == [] and targets.calls == []
    assert result.severity_result is None and result.spread_refresh_result is None and result.response_target_result is None


def test_a_confirmed_event_runs_the_full_operational_chain(stack):
    confirmed = stack.event(CONFIRMED)
    orchestrator, severity, spread, targets = _orchestrator(stack.events)

    result = orchestrator.refresh_fire_event(
        fire_event_id=confirmed.id, trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE, as_of=T0 + timedelta(hours=1))

    assert result.status is not OperationalRefreshStatus.NOT_RESPONSE_ELIGIBLE and result.success
    assert severity.calls and spread.calls and targets.calls  # severity -> spread -> response targets all ran


# --- operational planning coordinator: no global planning for SUSPECTED-only updates ---------------------------------


class ScriptedOperationalOrchestrator:
    def __init__(self, status_by_id):
        self.status_by_id = status_by_id
        self.calls = []

    def refresh_fire_event(self, *, fire_event_id, trigger_type, as_of):
        self.calls.append(fire_event_id)
        return OperationalRefreshResult(trigger_type=trigger_type, status=self.status_by_id[fire_event_id], success=True,
                                        fire_event_id=fire_event_id, as_of=as_of)


class SpyGlobalPlanning:
    def __init__(self):
        self.calls = []

    def refresh(self, *, trigger, as_of):
        self.calls.append(trigger)
        return SimpleNamespace(status="refreshed")


TRIGGER = OperationalRefreshTriggerType.FIRE_EVENT_UPDATE


def test_a_suspected_only_update_never_starts_a_global_planning_cycle():
    operational = ScriptedOperationalOrchestrator({1: OperationalRefreshStatus.NOT_RESPONSE_ELIGIBLE})
    planning = SpyGlobalPlanning()
    coordinator = OperationalPlanningRefreshCoordinator(operational_refresh_orchestrator=operational, global_planning_refresh=planning)

    single = coordinator.refresh_fire_event(fire_event_id=1, trigger_type=TRIGGER, as_of=T0)
    batch = coordinator.refresh_fire_events_batch(fire_event_ids=(1,), trigger_type=TRIGGER, as_of=T0)

    assert planning.calls == [] and single.global_planning_result is None and batch.global_planning_result is None


def test_a_batch_with_one_confirmed_event_triggers_exactly_one_global_planning_cycle():
    operational = ScriptedOperationalOrchestrator({1: OperationalRefreshStatus.NOT_RESPONSE_ELIGIBLE, 2: OperationalRefreshStatus.REFRESHED})
    planning = SpyGlobalPlanning()
    coordinator = OperationalPlanningRefreshCoordinator(operational_refresh_orchestrator=operational, global_planning_refresh=planning)

    batch = coordinator.refresh_fire_events_batch(fire_event_ids=(1, 2), trigger_type=TRIGGER, as_of=T0)

    assert planning.calls == [TRIGGER.value] and batch.global_planning_result is not None


# --- global planning: routing / GA / commitment only for CONFIRMED -----------------------------------------------------


class SpyRunRepository:
    def __init__(self):
        self.created = []
        self.completed = {}

    def create_run(self, *, started_at, trigger, methodology, methodology_version, input_fingerprint, fire_event_ids):
        self.created.append(tuple(fire_event_ids))
        return SimpleNamespace(id=len(self.created))

    def set_input_fingerprint(self, run_id, fingerprint):
        pass

    def record_member_result(self, *args, **kwargs):
        pass

    def complete_run(self, run_id, *, status, completed_at):
        self.completed[run_id] = status

    def get_latest_activated(self, *, exclude_run_id=None):
        return None


class SpyInputBuilder:
    """compute_pre_routing_bundle / build are where the road graph, route matrix and Dijkstra work begin."""

    def __init__(self, signature="sig"):
        self.precheck_calls = 0
        self.build_calls = 0
        self.signature = signature

    def compute_pre_routing_bundle(self, *, global_planning_run_id, as_of):
        self.precheck_calls += 1
        return SimpleNamespace(pre_routing_signature=self.signature)

    def build(self, *, global_planning_run_id, as_of, precomputed=None):
        self.build_calls += 1
        return SimpleNamespace(input_fingerprint="f" * 64, current_assignments=())


class SpyOptimization:
    """The genetic algorithm."""

    def __init__(self):
        self.calls = 0

    def optimize(self, *args, **kwargs):
        self.calls += 1
        return SimpleNamespace(actions=(), assignment_changes=(), shortage=None, event_results=())


class SpyActivation:
    """Response-plan activation = resource commitment."""

    def __init__(self):
        self.calls = 0

    def activate(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(response_plan_ids_by_event={})


def _global_coordinator(stack):
    run_repository, builder, optimization, activation = SpyRunRepository(), SpyInputBuilder(), SpyOptimization(), SpyActivation()
    coordinator = GlobalPlanningRefreshCoordinator(
        fire_event_repository=stack.events, global_planning_run_repository=run_repository, input_builder=builder,
        optimization_service=optimization, activation_service=activation)
    return coordinator, run_repository, builder, optimization, activation


def test_a_suspected_event_never_causes_road_graph_dijkstra_ga_or_commitment(stack):
    stack.event(SUSPECTED)
    coordinator, runs, builder, optimization, activation = _global_coordinator(stack)

    result = coordinator.refresh(trigger="fire_event_update", as_of=T0 + timedelta(hours=1))

    assert result.status is GlobalPlanningRefreshStatus.NO_ACTIVE_EVENTS
    assert runs.created == [] and builder.precheck_calls == 0 and builder.build_calls == 0
    assert optimization.calls == 0 and activation.calls == 0


def test_a_confirmed_event_does_trigger_routing_input_ga_and_activation(stack):
    confirmed = stack.event(CONFIRMED)
    coordinator, runs, builder, optimization, activation = _global_coordinator(stack)

    coordinator.refresh(trigger="fire_event_update", as_of=T0 + timedelta(hours=1))

    assert runs.created == [(confirmed.id,)]
    assert builder.precheck_calls == 1 and builder.build_calls == 1 and optimization.calls == 1 and activation.calls == 1


def test_mixed_state_the_planner_sees_only_the_confirmed_event_while_detection_sees_both(stack):
    suspected = stack.event(SUSPECTED, LAT, LON, T0)
    confirmed = stack.event(CONFIRMED, LAT + 0.3, LON, T0 + timedelta(minutes=1))
    coordinator, runs, builder, optimization, activation = _global_coordinator(stack)

    coordinator.refresh(trigger="fire_event_update", as_of=T0 + timedelta(hours=1))

    assert runs.created == [(confirmed.id,)]  # ONLY B receives response planning / resources
    assert suspected.id not in runs.created[0]
    assert set(stack.events.get_active_fire_event_ids()) == {suspected.id, confirmed.id}  # detection / monitoring sees both
    assert stack.events.get_by_id(suspected.id).event.status is SUSPECTED  # A is untouched and stays visible


def test_a_repeat_planning_cycle_on_unchanged_inputs_does_no_duplicate_expensive_work(stack):
    """Characterises the existing NO_OP precheck: a second cycle with an identical pre-routing signature does not
    rebuild the input or rerun the GA/activation (so re-running detect() does not duplicate response work)."""
    stack.event(CONFIRMED)
    coordinator, runs, builder, optimization, activation = _global_coordinator(stack)

    first = coordinator.refresh(trigger="fire_event_update", as_of=T0 + timedelta(hours=1))
    second = coordinator.refresh(trigger="fire_event_update", as_of=T0 + timedelta(hours=1, minutes=5))

    assert first.status is not GlobalPlanningRefreshStatus.NO_OP
    assert second.status is GlobalPlanningRefreshStatus.NO_OP
    assert builder.build_calls == 1 and optimization.calls == 1 and activation.calls == 1


# --- promotion SUSPECTED -> CONFIRMED keeps identity and history ---------------------------------------------------------


def test_promotion_keeps_the_same_event_and_history_and_makes_it_response_eligible(stack):
    # T+0: a lone hotspot -> SUSPECTED (rules never confirm satellite-only evidence)
    stack.hotspot(T0)
    first = stack.detect(T0)
    (event_id,) = first.event_ids
    assert stack.events.get_by_id(event_id).event.status is SUSPECTED
    assert stack.events.get_response_eligible_fire_event_ids() == ()  # visible, but no response yet
    assert event_id in stack.events.get_active_fire_event_ids()
    orchestrator, severity, spread, targets = _orchestrator(stack.events)
    assert orchestrator.refresh_fire_event(fire_event_id=event_id, trigger_type=TRIGGER, as_of=T0 + timedelta(minutes=10)).status is (
        OperationalRefreshStatus.NOT_RESPONSE_ELIGIBLE)
    assert severity.calls == [] and spread.calls == [] and targets.calls == []

    # T+3h: satellite + news corroboration -> the same event is promoted
    stack.hotspot(T0 + timedelta(hours=3), confidence="h")
    stack.report(T0 + timedelta(hours=3, minutes=2), 1)
    second = stack.detect(T0 + timedelta(hours=3))

    assert second.event_ids == (event_id,) and second.events_created == 0  # NO new FireEvent
    promoted = stack.events.get_by_id(event_id)
    assert promoted.event.status is CONFIRMED and promoted.event.detected_at == T0  # identity + first-seen preserved
    assert len(stack.events.get_evidence_refs(event_id)) == 3  # T+0 hotspot, T+3h hotspot, the report: history intact
    assert stack.events.get_response_eligible_fire_event_ids() == (event_id,)
    result = orchestrator.refresh_fire_event(fire_event_id=event_id, trigger_type=TRIGGER, as_of=T0 + timedelta(hours=3, minutes=10))
    assert result.status is not OperationalRefreshStatus.NOT_RESPONSE_ELIGIBLE
    assert severity.calls and spread.calls and targets.calls  # the response chain now runs, once eligible


def test_a_confirmed_event_is_never_downgraded_by_a_weaker_later_observation(stack):
    stack.hotspot(T0, confidence="h")
    stack.report(T0 + timedelta(minutes=2), 1)
    (event_id,) = stack.detect(T0).event_ids
    assert stack.events.get_by_id(event_id).event.status is CONFIRMED

    stack.hotspot(T0 + timedelta(hours=3), confidence="n")  # a lone nominal hotspot: the rules say SUSPECTED
    stack.detect(T0 + timedelta(hours=3))

    assert stack.events.get_by_id(event_id).event.status is CONFIRMED  # monotonic: stays response-eligible


def test_a_suspected_event_can_later_be_dismissed_without_ever_being_confirmed(stack):
    """Nothing assumes every active event becomes CONFIRMED: SUSPECTED -> DISMISSED is a legal end of life."""
    from src.services.fire_event_lifecycle.fire_event_lifecycle_service import FireEventLifecycleService

    suspected = stack.event(SUSPECTED)
    from src.repositories.resource_commitment_repository import ResourceCommitmentRepository

    session_factory = stack.events._session_factory
    lifecycle = FireEventLifecycleService(
        fire_event_repository=stack.events,
        resource_commitment_repository=ResourceCommitmentRepository(session_factory=session_factory),
        session_factory=session_factory,
    )

    dismissed = lifecycle.dismiss_event(suspected.id, as_of=T0 + timedelta(hours=7))

    assert dismissed.event.status is FireEventStatus.DISMISSED
    assert stack.events.get_active_fire_event_ids() == () and stack.events.get_response_eligible_fire_event_ids() == ()


# --- API contract: SUSPECTED stays visible, has no response plan ------------------------------------------------------------


def test_the_response_plan_endpoint_reports_no_plan_for_a_not_response_eligible_event():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routers.response_plans import (
        get_response_plan_details_service,
        get_response_plan_presenter,
        response_plans_router,
    )

    class NoPlanService:
        def get_current_plan_details(self, fire_event_id):
            return None  # a SUSPECTED event never had a plan generated

    app = FastAPI()
    app.include_router(response_plans_router, prefix="/api/v1")
    app.dependency_overrides[get_response_plan_details_service] = lambda: NoPlanService()
    app.dependency_overrides[get_response_plan_presenter] = lambda: SimpleNamespace(present=lambda details: details)

    response = TestClient(app).get("/api/v1/fire-events/7/response-plan")

    assert response.status_code == 200 and response.json() == {"plan": None}  # truthful: no plan yet, none manufactured


def test_suspected_events_remain_in_the_active_fire_events_service(stack):
    from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
    from src.services.fire_event_read.active_fire_events_service import ActiveFireEventsService

    suspected = stack.event(SUSPECTED)
    confirmed = stack.event(CONFIRMED, LAT + 0.3, LON, T0 + timedelta(minutes=1))
    service = ActiveFireEventsService(
        fire_event_repository=stack.events,
        fire_severity_assessment_repository=FireSeverityAssessmentRepository(session_factory=stack.events._session_factory),
    )

    result = service.get_active_events(as_of=T0 + timedelta(hours=1))

    listed = {item.fire_event_id for item in result.items}
    assert listed == {suspected.id, confirmed.id}  # SUSPECTED is NOT hidden from the dashboard / API

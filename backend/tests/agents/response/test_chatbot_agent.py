"""Unit tests for ChatbotAgent (Task 4; compact-context Task 7D-B).

ActiveFireEventsService, EventDetailsService, and GeminiClient are all
replaced with hand-rolled fakes/test doubles - no real database access and
no real Gemini API call happens in these tests.
"""
from __future__ import annotations

import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.agents.response.chatbot_agent import (
    MAX_CHAT_HISTORY_MESSAGES,
    ChatbotAgent,
    ChatMessage,
    _SYSTEM_INSTRUCTION,
)
from src.api.schemas.event_details import (
    CurrentResponsePlanResponse,
    DetectionEvidenceResponse,
    EventDetailsResult,
    FireEventMLAssessmentResponse,
    FireEventSummaryResponse,
    FireStationResponse,
    FirefightingResourceResponse,
    NewsEvidenceResponse,
    ResponseActionResponse,
    ResponseTargetResponse,
    SatelliteEvidenceResponse,
    SeverityAssessmentResponse,
    SpreadPredictionCellResponse,
    SpreadPredictionResponse,
)
from src.api.schemas.global_response_plan import (
    GlobalEventPlan,
    GlobalPlanCoverage,
    GlobalPlanMetrics,
    GlobalPlanResponse,
    GlobalPlanShortage,
    GlobalResponsePlanResponse,
)
from src.api.schemas.response_plans import (
    ResponsePlanActionResponse,
    ResponsePlanResourceResponse,
    ResponsePlanRouteResponse,
    ResponsePlanTargetResponse,
)
from src.external.gemini.exceptions import GeminiServiceUnavailableError
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.active_fire_events import (
    ActiveFireEventMLSummary,
    ActiveFireEventSeveritySummary,
    ActiveFireEventsResult,
    ActiveFireEventSummary,
)
from src.models.fire_danger_areas import (
    FireDangerAreaAssessmentSummary,
    FireDangerAreasResult,
    FireDangerAreaSnapshot,
)
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_ml_rule_agreement import FireDetectionMLRuleAgreement
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_event_status import FireEventStatus
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.fire_spread_prediction_status import FireSpreadPredictionStatus
from src.models.resource_status import ResourceStatus
from src.models.response_target_type import ResponseTargetType
from src.services.weather.weather_conditions_query_service import AreaWeatherConditions

DETECTED_AT = datetime(2026, 9, 20, 10, 0, 0, tzinfo=timezone.utc)
UPDATED_AT = DETECTED_AT + timedelta(minutes=5)
ASSESSED_AT = DETECTED_AT + timedelta(minutes=6)
AS_OF = DETECTED_AT + timedelta(minutes=30)

# Internal DB identifier keys that must never appear anywhere in the
# outgoing Gemini snapshot (Task 7D-B, Part 3).
FORBIDDEN_INTERNAL_ID_KEYS = (
    "fire_event_id",
    "response_plan_id",
    "response_target_id",
    "resource_id",
    "station_id",
    "assessment_id",
    "plan_id",
    "fire_danger_assessment_id",
    "observation_id",
    "observation_ids",
    "station_external_id",
    "area_id",
    "response_plan_id",
    "run_id",
    "resource_id",
    "fire_event_ids",
    "covered_fire_event_ids",
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeActiveFireEventsService:
    def __init__(self, result: ActiveFireEventsResult):
        self._result = result
        self.calls = 0

    def get_active_events(self, *, as_of=None):
        self.calls += 1
        return self._result


class FakeEventDetailsService:
    def __init__(self, details_by_id: dict[int, EventDetailsResult | None]):
        self._details_by_id = details_by_id
        self.requested_ids: list[int] = []

    def get_event_details(self, fire_event_id, *, as_of=None):
        self.requested_ids.append(fire_event_id)
        return self._details_by_id.get(fire_event_id)


class FakeWeatherConditionsQueryService:
    def __init__(self, conditions: tuple[AreaWeatherConditions, ...] = ()):
        self._conditions = tuple(conditions)
        self.calls = 0

    def get_latest_for_all_areas(self):
        self.calls += 1
        return self._conditions


class FakeFireDangerQueryService:
    def __init__(self, areas: tuple[FireDangerAreaSnapshot, ...] = ()):
        self._areas = tuple(areas)
        self.requested_as_of: list = []

    def get_latest_for_all_areas(self, *, as_of=None):
        self.requested_as_of.append(as_of)
        return FireDangerAreasResult(as_of=as_of, areas=self._areas)


class FakeGlobalResponsePlanReadService:
    def __init__(self, response: GlobalResponsePlanResponse | None = None):
        self._response = response
        self.requested_as_of: list = []

    def get_current(self, *, as_of=None):
        self.requested_as_of.append(as_of)
        return self._response if self._response is not None else GlobalResponsePlanResponse(as_of=as_of, plan=None)


class FakeGeminiClient:
    def __init__(self, response_text: str = "The requested EcoGuard information.", *, raise_exc=None):
        self._response_text = response_text
        self._raise_exc = raise_exc
        self.calls: list[dict] = []

    def generate_content(self, contents, *, system_instruction=None):
        self.calls.append({"contents": contents, "system_instruction": system_instruction})
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response_text


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_severity(**overrides) -> ActiveFireEventSeveritySummary:
    values = dict(
        assessment_id=44,
        status=FireSeverityAssessmentStatus.VALID,
        score=81.4,
        level=FireSeverityLevel.CRITICAL,
        assessed_at=ASSESSED_AT,
    )
    values.update(overrides)
    return ActiveFireEventSeveritySummary(**values)


def make_ml_summary(**overrides) -> ActiveFireEventMLSummary:
    values = dict(available=True, model_score=0.87)
    values.update(overrides)
    return ActiveFireEventMLSummary(**values)


def make_active_event(**overrides) -> ActiveFireEventSummary:
    values = dict(
        fire_event_id=1,
        status=FireEventStatus.CONFIRMED,
        latitude=32.794,
        longitude=34.989,
        detection_confidence=0.91,
        detected_at=DETECTED_AT,
        updated_at=UPDATED_AT,
        created_at=UPDATED_AT,
        severity=None,
        location_name="Haifa",
        ml_summary=None,
    )
    values.update(overrides)
    return ActiveFireEventSummary(**values)


def make_active_result(*events: ActiveFireEventSummary) -> ActiveFireEventsResult:
    return ActiveFireEventsResult(as_of=AS_OF, items=tuple(events))


def make_fire_event_summary(**overrides) -> FireEventSummaryResponse:
    values = dict(
        fire_event_id=1,
        status=FireEventStatus.CONFIRMED,
        latitude=32.794,
        longitude=34.989,
        detection_confidence=0.91,
        detected_at=DETECTED_AT,
        updated_at=UPDATED_AT,
        methodology="hybrid",
        methodology_version="v1",
    )
    values.update(overrides)
    return FireEventSummaryResponse(**values)


def make_severity_response(**overrides) -> SeverityAssessmentResponse:
    values = dict(
        assessment_id=44,
        status=FireSeverityAssessmentStatus.VALID,
        score=81.4,
        level=FireSeverityLevel.CRITICAL,
        assessed_at=ASSESSED_AT,
    )
    values.update(overrides)
    return SeverityAssessmentResponse(**values)


def make_ml_assessment_response(**overrides) -> FireEventMLAssessmentResponse:
    values = dict(
        available=True,
        mode=FireDetectionDecisionMode.SHADOW,
        rule_status=FireDetectionStatus.CONFIRMED,
        rule_confidence=0.9,
        model_score=0.87,
        agreement=FireDetectionMLRuleAgreement.AGREE_FIRE,
        model_name="fire_detection_logistic_v3",
        model_version="3.0",
        feature_schema_version="v3",
        failure_reason=None,
        updated_at=UPDATED_AT,
    )
    values.update(overrides)
    return FireEventMLAssessmentResponse(**values)


def make_satellite_evidence(**overrides) -> SatelliteEvidenceResponse:
    values = dict(
        id=901,
        detected_at=DETECTED_AT,
        latitude=32.8,
        longitude=35.0,
        confidence="h",
        frp=48.2,
        brightness=356.1,
        satellite="SIM-NOAA-20",
        instrument="VIIRS",
        day_night="D",
    )
    values.update(overrides)
    return SatelliteEvidenceResponse(**values)


def make_news_evidence(**overrides) -> NewsEvidenceResponse:
    values = dict(
        id=902,
        title="Heavy smoke reported",
        summary="Residents report heavy smoke near the area.",
        source="EcoGuard Simulation News",
        observed_at=DETECTED_AT,
        location_name="Haifa",
        latitude=32.8,
        longitude=35.0,
    )
    values.update(overrides)
    return NewsEvidenceResponse(**values)


def make_spread_cell(**overrides) -> SpreadPredictionCellResponse:
    values = dict(
        latitude=32.81,
        longitude=35.01,
        spread_probability=0.6,
        spread_risk_score=0.7,
        reached_step=1,
        reached_minutes=10,
    )
    values.update(overrides)
    return SpreadPredictionCellResponse(**values)


def make_spread_prediction(**overrides) -> SpreadPredictionResponse:
    values = dict(
        horizon_minutes=30,
        status=FireSpreadPredictionStatus.VALID,
        predicted_at=DETECTED_AT,
        cells=[],
    )
    values.update(overrides)
    return SpreadPredictionResponse(**values)


def make_target(**overrides) -> ResponseTargetResponse:
    values = dict(
        target_order=0,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        latitude=32.8,
        longitude=35.0,
        priority_score=184.3,
        prediction_horizon_minutes=None,
    )
    values.update(overrides)
    return ResponseTargetResponse(**values)


def make_station(**overrides) -> FireStationResponse:
    values = dict(
        station_id="35",
        name="Central Galilee",
        latitude=32.9,
        longitude=35.3,
        station_type="regional",
        address="Industry St 64, Karmiel",
    )
    values.update(overrides)
    return FireStationResponse(**values)


def make_resource(**overrides) -> FirefightingResourceResponse:
    values = dict(resource_id="TRUCK-35-2", station_id="35", status=ResourceStatus.ASSIGNED)
    values.update(overrides)
    return FirefightingResourceResponse(**values)


def make_response_action(**overrides) -> ResponseActionResponse:
    values = dict(
        resource_id="TRUCK-35-2",
        station_id="35",
        response_target_id=6395,
        target_type="active_fire",
        target_priority=184.3,
        eta_seconds=790.4,
        route_distance_meters=12645.5,
        node_path=[101, 102, 103, 104, 105],
    )
    values.update(overrides)
    return ResponseActionResponse(**values)


def make_response_plan(**overrides) -> CurrentResponsePlanResponse:
    values = dict(
        plan_id=1081,
        generated_at=UPDATED_AT,
        methodology="global_genetic_resource_allocation",
        methodology_version="1.0",
        plan_score=6021029.3,
        coverage_score=100.0,
        average_eta_seconds=790.4,
        actions=[make_response_action()],
        uncovered_target_ids=[],
        baseline_comparison=None,
    )
    values.update(overrides)
    return CurrentResponsePlanResponse(**values)


def make_event_details(**overrides) -> EventDetailsResult:
    values = dict(
        as_of=AS_OF,
        fire_event=make_fire_event_summary(),
        severity=None,
        ml_assessment=None,
        danger=None,
        detection_evidence=DetectionEvidenceResponse(satellite=[], news=[]),
        spread_predictions=[],
        targets=[],
        stations=[],
        resources=[],
        station_summaries=[],
        current_response_plan=None,
    )
    values.update(overrides)
    return EventDetailsResult(**values)


def make_agent(
    *,
    active_result: ActiveFireEventsResult,
    details_by_id: dict[int, EventDetailsResult | None],
    gemini_client: FakeGeminiClient | None = None,
    weather_conditions: tuple[AreaWeatherConditions, ...] = (),
    fire_danger_areas: tuple[FireDangerAreaSnapshot, ...] = (),
    global_plan: GlobalResponsePlanResponse | None = None,
) -> tuple[ChatbotAgent, FakeActiveFireEventsService, FakeEventDetailsService, FakeGeminiClient]:
    active_service = FakeActiveFireEventsService(active_result)
    details_service = FakeEventDetailsService(details_by_id)
    gemini = gemini_client or FakeGeminiClient()
    agent = ChatbotAgent(
        active_fire_events_service=active_service,
        event_details_service=details_service,
        weather_conditions_query_service=FakeWeatherConditionsQueryService(weather_conditions),
        fire_danger_query_service=FakeFireDangerQueryService(fire_danger_areas),
        global_response_plan_read_service=FakeGlobalResponsePlanReadService(global_plan),
        gemini_client=gemini,
    )
    return agent, active_service, details_service, gemini


_SNAPSHOT_LABEL_MARKER = "Current EcoGuard data snapshot (authoritative JSON):\n"
_JSON_DECODER = json.JSONDecoder()


def raw_snapshot_text(gemini: FakeGeminiClient) -> str:
    """The exact raw JSON substring sent to Gemini, unparsed - used for
    compact-serialization/absence checks that must look at the real bytes.
    Parses only the JSON value's own natural extent, so an optional
    resolved-focus block (follow-up referent resolution) that may follow it
    in the same turn is never included here."""
    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    after_label = final_text.split(_SNAPSHOT_LABEL_MARKER, 1)[1]
    _, end_index = _JSON_DECODER.raw_decode(after_label)
    return after_label[:end_index]


def sent_snapshot(gemini: FakeGeminiClient) -> dict:
    """Parse the structured EcoGuard snapshot JSON out of the final `contents` turn."""
    return json.loads(raw_snapshot_text(gemini))


def _walk_keys(value):
    """Yield every dict key appearing anywhere in a nested JSON-like structure."""
    if isinstance(value, dict):
        for key, sub_value in value.items():
            yield key
            yield from _walk_keys(sub_value)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


# ---------------------------------------------------------------------------
# 1-2: general question / question by location
# ---------------------------------------------------------------------------


def test_general_question_with_multiple_active_events():
    haifa = make_active_event(fire_event_id=1, location_name="Haifa")
    jerusalem = make_active_event(fire_event_id=2, location_name="Jerusalem")
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(haifa, jerusalem),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    answer = agent.ask("What are the active fires right now?")

    assert answer == "The requested EcoGuard information."
    snapshot = sent_snapshot(gemini)
    assert snapshot["active_fire_event_count"] == 2
    assert {e["event_ref"] for e in snapshot["active_fire_events"]} == {"A", "B"}


def test_question_by_location_includes_location_name_for_gemini_to_match():
    haifa = make_active_event(fire_event_id=1, location_name="Haifa")
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(haifa),
        details_by_id={1: make_event_details()},
    )

    agent.ask("What is the status of the fire in Haifa?")

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "Haifa" in final_text
    assert "What is the status of the fire in Haifa?" in final_text


# ---------------------------------------------------------------------------
# 3-4: language passthrough
# ---------------------------------------------------------------------------


def test_hebrew_question_is_preserved_verbatim():
    agent, _, _, gemini = make_agent(active_result=make_active_result(), details_by_id={})

    agent.ask("מה מצב השריפה בחיפה?")

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "מה מצב השריפה בחיפה?" in final_text
    assert "the language of the user's current question only" in gemini.calls[-1]["system_instruction"].lower()


def test_english_question_is_preserved_verbatim():
    agent, _, _, gemini = make_agent(active_result=make_active_result(), details_by_id={})

    agent.ask("What is the status of the fire in Haifa?")

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "What is the status of the fire in Haifa?" in final_text


# ---------------------------------------------------------------------------
# 5-6: every active event is fetched and included
# ---------------------------------------------------------------------------


def test_all_active_events_included_in_snapshot():
    events = [make_active_event(fire_event_id=i, location_name=f"Area {i}") for i in range(1, 4)]
    details = {i: make_event_details(fire_event=make_fire_event_summary(fire_event_id=i)) for i in range(1, 4)}
    agent, _, _, gemini = make_agent(active_result=make_active_result(*events), details_by_id=details)

    agent.ask("What are the active fires?")

    snapshot = sent_snapshot(gemini)
    assert snapshot["active_fire_event_count"] == 3
    assert {e["event_ref"] for e in snapshot["active_fire_events"]} == {"A", "B", "C"}
    assert {e["location_name"] for e in snapshot["active_fire_events"]} == {"Area 1", "Area 2", "Area 3"}


def test_event_details_fetched_for_every_active_event():
    events = [make_active_event(fire_event_id=i) for i in (5, 6, 7)]
    details = {i: make_event_details(fire_event=make_fire_event_summary(fire_event_id=i)) for i in (5, 6, 7)}
    agent, _, details_service, _ = make_agent(active_result=make_active_result(*events), details_by_id=details)

    agent.ask("What are the active fires?")

    assert details_service.requested_ids == [5, 6, 7]


# ---------------------------------------------------------------------------
# 7-9: location representation
# ---------------------------------------------------------------------------


def test_location_name_is_preserved():
    event = make_active_event(fire_event_id=1, location_name="Haifa")
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(event), details_by_id={1: make_event_details()}
    )

    agent.ask("What are the active fires?")

    snapshot = sent_snapshot(gemini)
    assert snapshot["active_fire_events"][0]["location_name"] == "Haifa"


def test_multiple_events_with_same_location_are_both_included_distinctly_with_different_refs():
    first = make_active_event(fire_event_id=1, location_name="Haifa")
    second = make_active_event(fire_event_id=2, location_name="Haifa")
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(first, second),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    agent.ask("What is the status of the fire in Haifa?")

    snapshot = sent_snapshot(gemini)
    haifa_events = [e for e in snapshot["active_fire_events"] if e["location_name"] == "Haifa"]
    assert len(haifa_events) == 2
    assert {e["event_ref"] for e in haifa_events} == {"A", "B"}


def test_missing_location_name_is_not_fabricated():
    event = make_active_event(fire_event_id=1, location_name=None)
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(event), details_by_id={1: make_event_details()}
    )

    agent.ask("What are the active fires?")

    snapshot = sent_snapshot(gemini)
    assert snapshot["active_fire_events"][0]["location_name"] is None
    # Coordinates must still be present even without a location name.
    assert snapshot["active_fire_events"][0]["latitude"] == event.latitude


# ---------------------------------------------------------------------------
# 10-11: missing downstream data / no active events
# ---------------------------------------------------------------------------


def test_missing_downstream_data_represented_as_unavailable_not_fabricated():
    event = make_active_event(fire_event_id=1, severity=None, ml_summary=None)
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(event),
        details_by_id={1: make_event_details(severity=None, ml_assessment=None, current_response_plan=None)},
    )

    agent.ask("What is the severity of the fire?")

    snapshot = sent_snapshot(gemini)
    event_context = snapshot["active_fire_events"][0]
    assert event_context["severity"] is None
    assert event_context["ml_assessment"] is None
    assert event_context["current_response_plan"] is None
    assert event_context["spread_predictions"] == []
    assert event_context["targets"] == []


def test_event_details_unavailable_is_represented_explicitly_not_dropped():
    event = make_active_event(fire_event_id=1, location_name="Haifa")
    # Simulates the event becoming inactive between the two read calls.
    agent, _, _, gemini = make_agent(active_result=make_active_result(event), details_by_id={})

    agent.ask("What are the active fires?")

    snapshot = sent_snapshot(gemini)
    event_context = snapshot["active_fire_events"][0]
    assert event_context["event_ref"] == "A"
    assert event_context["location_name"] == "Haifa"
    assert event_context["event_details_available"] is False
    assert event_context["detection_evidence"] is None
    assert event_context["current_response_plan"] is None


def test_no_active_events_still_calls_gemini_with_explicit_empty_snapshot():
    agent, active_service, details_service, gemini = make_agent(
        active_result=make_active_result(), details_by_id={}
    )

    answer = agent.ask("Are there any active fires?")

    assert answer == "The requested EcoGuard information."
    assert active_service.calls == 1
    assert details_service.requested_ids == []
    snapshot = sent_snapshot(gemini)
    assert snapshot["active_fire_event_count"] == 0
    assert snapshot["active_fire_events"] == []


# ---------------------------------------------------------------------------
# 12-15: conversation history
# ---------------------------------------------------------------------------


def test_conversation_history_is_included_in_contents():
    history = [
        ChatMessage(role="user", content="What is the status of the fire in Haifa?"),
        ChatMessage(role="assistant", content="The active event in Haifa is currently CONFIRMED."),
    ]
    agent, _, _, gemini = make_agent(active_result=make_active_result(), details_by_id={})

    agent.ask("And what is its spread prediction?", history=history)

    contents = gemini.calls[-1]["contents"]
    assert len(contents) == 3  # 2 history turns + 1 final turn
    assert contents[0] == {"role": "user", "parts": [{"text": "What is the status of the fire in Haifa?"}]}
    assert contents[1] == {
        "role": "model",
        "parts": [{"text": "The active event in Haifa is currently CONFIRMED."}],
    }


def test_history_is_trimmed_to_latest_max_messages():
    assert MAX_CHAT_HISTORY_MESSAGES == 8
    history = [ChatMessage(role="user", content=f"message {i}") for i in range(10)]
    agent, _, _, gemini = make_agent(active_result=make_active_result(), details_by_id={})

    agent.ask("latest question", history=history)

    contents = gemini.calls[-1]["contents"]
    history_turns = contents[:-1]
    assert len(history_turns) == MAX_CHAT_HISTORY_MESSAGES
    assert history_turns[0]["parts"][0]["text"] == "message 2"
    assert history_turns[-1]["parts"][0]["text"] == "message 9"


def test_history_order_is_preserved():
    history = [ChatMessage(role="user", content=f"message {i}") for i in range(3)]
    agent, _, _, gemini = make_agent(active_result=make_active_result(), details_by_id={})

    agent.ask("latest question", history=history)

    history_turns = gemini.calls[-1]["contents"][:-1]
    texts = [turn["parts"][0]["text"] for turn in history_turns]
    assert texts == ["message 0", "message 1", "message 2"]


def test_current_question_is_preserved_exactly():
    question = "Compare the fires in Haifa and Jerusalem."
    agent, _, _, gemini = make_agent(active_result=make_active_result(), details_by_id={})

    agent.ask(question)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert question in final_text


# ---------------------------------------------------------------------------
# 16-18: system instruction content
# ---------------------------------------------------------------------------


def test_system_instruction_marks_current_data_authoritative_over_history():
    assert "authoritative" in _SYSTEM_INSTRUCTION
    assert "Conversation history" in _SYSTEM_INSTRUCTION
    assert "always prefer the current" in _SYSTEM_INSTRUCTION


def test_system_instruction_contains_model_score_rule():
    assert "model_score" in _SYSTEM_INSTRUCTION
    assert "NOT a calibrated real-world probability" in _SYSTEM_INSTRUCTION
    assert "fire-risk score" in _SYSTEM_INSTRUCTION
    assert "never" in _SYSTEM_INSTRUCTION.lower()


def test_system_instruction_contains_all_required_rules():
    required_phrases = (
        "EcoGuard's AI decision-support assistant",
        "Answer ONLY using the EcoGuard information supplied",
        "do not fabricate",
        "explicitly say",
        "Do not perform any new wildfire calculations",
        "replace or override",
        "observed/detected facts",
        "ML model outputs",
        "severity assessments",
        "spread predictions",
        "response targets",
        "response plans",
        "model_score",
        "NOT a calibrated real-world probability",
        "the language of the user's current question only",
        "never the dominant language of the conversation history",
        "authoritative",
        "multiple active events share the same or a similar location",
        "never silently",
        "no active wildfire event",
        "complete list of currently active wildfire events",
        "never invent",
        "internal `event_ref` label",
        "never expose an `event_ref` value",
        "insufficient_data",
        "never infer or guess a cause",
        "does not specify a reason",
        "use the conversation history only to determine which",
        "never to supply its facts",
        "answer about that one event only",
        "cannot be resolved unambiguously from the history",
    )
    lowered = _SYSTEM_INSTRUCTION.lower()
    for phrase in required_phrases:
        assert phrase.lower() in lowered, f"missing required system-instruction phrase: {phrase!r}"


def test_system_instruction_prohibits_inventing_a_missing_data_cause():
    lowered = _SYSTEM_INSTRUCTION.lower()
    assert "insufficient_data" in lowered
    assert "never infer or guess a cause" in lowered
    # The explicit forbidden-inference examples from the real manual-test bug
    # (Gemini inventing "because there are no active cells" from cell_count/
    # has_cells) must be named, not just a vague "don't guess".
    assert "zero count" in lowered
    assert "empty array" in lowered
    assert "another status field" in lowered
    assert "missing evidence" in lowered
    assert "does not specify a reason" in lowered


def test_system_instruction_distinguishes_no_active_event_from_missing_field():
    """Reported regression (Issue 2): a location with no matching active
    event must be described as 'no active event there', never as data being
    'unavailable'/'missing' - that phrasing is reserved for a real event
    whose specific field is null/missing."""
    lowered = _SYSTEM_INSTRUCTION.lower()
    assert "complete list of currently active wildfire events" in lowered
    assert "no active wildfire event there" in lowered
    assert 'never phrase this as the data being "unavailable" or "missing"' in lowered
    assert "wrongly imply such an event exists" in lowered
    assert "say that specific field's data is unavailable for that event" in lowered
    assert "never say no event exists there" in lowered


def test_system_instruction_no_active_event_rule_mentions_current_active_locations_are_optional():
    lowered = _SYSTEM_INSTRUCTION.lower()
    assert "mention which locations are currently active, if useful" in lowered


def test_system_instruction_gives_current_question_language_priority_over_history():
    lowered = _SYSTEM_INSTRUCTION.lower()
    assert "the language of the user's current question only" in lowered
    assert "never the dominant language of the conversation history" in lowered


def test_system_instruction_gives_explicit_english_and_hebrew_examples():
    lowered = _SYSTEM_INSTRUCTION.lower()
    assert "if the current question is in english, answer in english" in lowered
    assert "if it is in hebrew, answer in hebrew" in lowered


def test_system_instruction_explicitly_allows_switching_language_between_turns():
    lowered = _SYSTEM_INSTRUCTION.lower()
    assert "a follow-up question may switch language from the previous turn" in lowered
    assert "your response must switch with it" in lowered


def test_hebrew_history_then_english_question_still_carries_the_language_priority_rule():
    """Reproduces the real manual-test bug: mostly-Hebrew history, then an
    English current question - the system instruction (sent fresh on every
    call) must state current-question priority, and the final turn must
    carry the English question verbatim, never overridden by history."""
    history = [
        ChatMessage(role="user", content="מה מצב השריפות הפעילות?"),
        ChatMessage(role="assistant", content="קיימות שתי שריפות פעילות."),
        ChatMessage(role="user", content="מה לגבי הערכת החומרה?"),
        ChatMessage(role="assistant", content="רמת החומרה קריטית."),
    ]
    agent, _, _, gemini = make_agent(active_result=make_active_result(), details_by_id={})

    agent.ask("What is the current status of the fire in the Judean Hills?", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "What is the current status of the fire in the Judean Hills?" in final_text
    system_instruction = gemini.calls[-1]["system_instruction"].lower()
    assert "the language of the user's current question only" in system_instruction
    assert "never the dominant language of the conversation history" in system_instruction


def test_english_history_then_hebrew_question_still_carries_the_language_priority_rule():
    """The symmetric switch: mostly-English history, then a Hebrew current question."""
    history = [
        ChatMessage(role="user", content="What are the active fires?"),
        ChatMessage(role="assistant", content="There are two active fire events."),
    ]
    agent, _, _, gemini = make_agent(active_result=make_active_result(), details_by_id={})

    agent.ask("מה מצב השריפה ביהודה ושומרון?", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "מה מצב השריפה ביהודה ושומרון?" in final_text
    system_instruction = gemini.calls[-1]["system_instruction"].lower()
    assert "the language of the user's current question only" in system_instruction


# ---------------------------------------------------------------------------
# Follow-up referent resolution (history resolves WHICH event, never its facts)
# ---------------------------------------------------------------------------


def test_system_instruction_explicitly_allows_history_for_referent_resolution():
    lowered = _SYSTEM_INSTRUCTION.lower()
    assert "use the conversation history only to determine which" in lowered
    assert '"the event"' in lowered
    assert '"that event"' in lowered
    assert "האירוע" in _SYSTEM_INSTRUCTION  # Hebrew referring expressions, not lowercased (no case in Hebrew)
    assert "אותו אירוע" in _SYSTEM_INSTRUCTION


def test_system_instruction_keeps_current_snapshot_as_the_factual_source_even_for_resolved_referents():
    lowered = _SYSTEM_INSTRUCTION.lower()
    assert "never to supply its facts" in lowered
    assert "using its current values from the snapshot" in lowered
    # The pre-existing, broader authoritative-snapshot rule must still be present too.
    assert "the current ecoguard data snapshot supplied with this request is always" in lowered
    assert "authoritative" in lowered


def test_system_instruction_requires_scoping_a_resolved_referent_to_one_event_only():
    lowered = _SYSTEM_INSTRUCTION.lower()
    assert "answer about that one event only" in lowered
    assert "do not expand the answer to other active events" in lowered


def test_system_instruction_requires_saying_so_when_referent_is_ambiguous():
    lowered = _SYSTEM_INSTRUCTION.lower()
    assert "cannot be resolved unambiguously from the history" in lowered
    assert "rather than silently guessing which event is meant" in lowered


def test_uniquely_resolved_followup_question_and_history_are_sent_unchanged_for_gemini_to_scope():
    """ChatbotAgent never filters/removes events from the snapshot by
    referent - every active event's data is still sent (Strategy A: every
    active event is always sent, per the existing architecture decision).
    This test proves the follow-up question and its resolving history reach
    Gemini unchanged, and every active event's data is still present -
    scoping to one event is added ON TOP via the deterministic resolved-
    focus block (see the "deterministic follow-up referent resolution"
    tests below), not by ever removing another event from the snapshot."""
    judean_hills = make_active_event(fire_event_id=1, location_name="Judean Hills")
    galilee = make_active_event(fire_event_id=2, location_name="Galilee")
    history = [
        ChatMessage(role="user", content="What is the current status of the fire in the Judean Hills?"),
        ChatMessage(role="assistant", content="The Judean Hills event is currently CONFIRMED."),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(judean_hills, galilee),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    agent.ask("ומה חומרת האירוע?", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "ומה חומרת האירוע?" in final_text
    contents = gemini.calls[-1]["contents"]
    assert contents[0]["parts"][0]["text"] == "What is the current status of the fire in the Judean Hills?"
    assert contents[1]["parts"][0]["text"] == "The Judean Hills event is currently CONFIRMED."
    # Both active events remain in the snapshot (Strategy A) - the
    # instruction, not a backend filter, is what keeps the answer scoped.
    snapshot = sent_snapshot(gemini)
    assert {e["location_name"] for e in snapshot["active_fire_events"]} == {"Judean Hills", "Galilee"}


# ---------------------------------------------------------------------------
# Deterministic follow-up referent resolution (no vague extra system-prompt
# sentence - a small, testable, LLM-free mechanism inside ChatbotAgent)
# ---------------------------------------------------------------------------


def test_exact_reported_bug_hebrew_generic_followup_resolves_to_the_discussed_event():
    """Reproduces the exact manual-test bug: English question about Galilee,
    then a Hebrew generic follow-up ("ומה חומרת האירוע?" = "and what is the
    event's severity?") with THREE active events (Carmel, Galilee, Jerusalem
    Forest) in the snapshot. The resolved focus must name Galilee Demo Area
    only, and must explicitly scope the answer to it."""
    carmel = make_active_event(fire_event_id=1, location_name="Carmel Demo Area")
    galilee = make_active_event(fire_event_id=2, location_name="Galilee Demo Area")
    jerusalem_forest = make_active_event(fire_event_id=3, location_name="Jerusalem Forest Demo Area")
    history = [
        ChatMessage(role="user", content="What is the current status of the fire in the Galilee?"),
        ChatMessage(role="assistant", content="The Galilee event is currently CONFIRMED."),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(carmel, galilee, jerusalem_forest),
        details_by_id={1: make_event_details(), 2: make_event_details(), 3: make_event_details()},
    )

    agent.ask("ומה חומרת האירוע?", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert 'The current follow-up refers to "Galilee Demo Area".' in final_text
    assert "Carmel Demo Area" not in final_text.split(_SNAPSHOT_LABEL_MARKER, 1)[1].split(
        "Resolved conversation focus", 1
    )[1]
    assert "Answer the current question only about this event" in final_text
    assert "Use the current EcoGuard data snapshot above for all factual values" in final_text
    # Every active event is still present in the snapshot itself (Strategy A).
    snapshot = sent_snapshot(gemini)
    assert {e["location_name"] for e in snapshot["active_fire_events"]} == {
        "Carmel Demo Area",
        "Galilee Demo Area",
        "Jerusalem Forest Demo Area",
    }


def test_hebrew_explicit_location_then_english_generic_followup_resolves():
    """The symmetric direction: a Hebrew message that explicitly names the
    event's own displayed location label (realistic: EcoGuard's own
    location_name values are in English even in the current Hebrew-language
    demo dataset - see location_name in src.models.active_fire_events), then
    an English generic follow-up ("it"). Matching is word-based against the
    real location_name text (Task: no invented cross-language/transliteration
    logic), so this only resolves when a shared word is actually present."""
    galilee = make_active_event(fire_event_id=1, location_name="Galilee Demo Area")
    carmel = make_active_event(fire_event_id=2, location_name="Carmel Demo Area")
    history = [
        ChatMessage(role="user", content="מה מצב השריפה באזור Galilee Demo Area?"),
        ChatMessage(role="assistant", content="השריפה באזור Galilee Demo Area מאומתת כרגע."),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(galilee, carmel),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    agent.ask("What is its severity?", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert 'The current follow-up refers to "Galilee Demo Area".' in final_text


def test_ambiguous_history_does_not_force_a_focus():
    """History that never names any active event's location - resolution
    must not invent a focus; existing ambiguity handling is preserved."""
    galilee = make_active_event(fire_event_id=1, location_name="Galilee Demo Area")
    carmel = make_active_event(fire_event_id=2, location_name="Carmel Demo Area")
    history = [
        ChatMessage(role="user", content="What are the active fires?"),
        ChatMessage(role="assistant", content="There are two active fire events right now."),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(galilee, carmel),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    agent.ask("ומה חומרת האירוע?", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "Resolved conversation focus" not in final_text


def test_history_naming_two_events_does_not_force_a_focus():
    """History that names TWO active events' locations - still ambiguous,
    must not silently pick one."""
    galilee = make_active_event(fire_event_id=1, location_name="Galilee Demo Area")
    carmel = make_active_event(fire_event_id=2, location_name="Carmel Demo Area")
    history = [
        ChatMessage(role="user", content="What are the fires in the Galilee and Carmel?"),
        ChatMessage(role="assistant", content="Both the Galilee and Carmel events are currently confirmed."),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(galilee, carmel),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    agent.ask("ומה חומרת האירוע?", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "Resolved conversation focus" not in final_text


def test_newer_single_event_turn_overrides_an_older_comparison_english():
    """Reported regression: an OLDER comparison turn naming two events must
    not make a NEWER, single-event turn ambiguous. Recency wins - the most
    recent history message that names any event decides the focus."""
    carmel = make_active_event(fire_event_id=1, location_name="Carmel Demo Area")
    judean_hills = make_active_event(fire_event_id=2, location_name="Judean Hills Demo Area")
    history = [
        ChatMessage(role="user", content="Compare all active fire events right now."),
        ChatMessage(
            role="assistant",
            content="Carmel Demo Area and Judean Hills Demo Area are both currently confirmed.",
        ),
        ChatMessage(role="user", content="What is the current status of the fire in the Carmel?"),
        ChatMessage(role="assistant", content="The Carmel event is currently CONFIRMED."),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(carmel, judean_hills),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    agent.ask("What is the severity of the event?", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert 'The current follow-up refers to "Carmel Demo Area".' in final_text
    assert "Judean Hills Demo Area" not in final_text.split(_SNAPSHOT_LABEL_MARKER, 1)[1].split(
        "Resolved conversation focus", 1
    )[1]


def test_newer_single_event_turn_overrides_an_older_comparison_hebrew():
    """Same regression, in Hebrew - matches the exact reported conversation
    shape (Hebrew comparison, then an English single-event turn, then a
    Hebrew generic follow-up)."""
    carmel = make_active_event(fire_event_id=1, location_name="Carmel Demo Area")
    judean_hills = make_active_event(fire_event_id=2, location_name="Judean Hills Demo Area")
    history = [
        ChatMessage(role="user", content="השווה בין כל אירועי השריפה הפעילים כרגע"),
        ChatMessage(
            role="assistant",
            content="Carmel Demo Area ו-Judean Hills Demo Area שניהם מאומתים כרגע.",
        ),
        ChatMessage(role="user", content="What is the current status of the fire in the Carmel?"),
        ChatMessage(role="assistant", content="The Carmel event is currently CONFIRMED."),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(carmel, judean_hills),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    agent.ask("ומה חומרת האירוע?", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert 'The current follow-up refers to "Carmel Demo Area".' in final_text


def test_newest_relevant_turn_naming_two_events_is_still_ambiguous():
    """If the MOST RECENT event-mentioning message itself names two events,
    that is genuinely ambiguous - resolution must stop right there and must
    NOT fall back further into the past looking for an older single-event
    mention."""
    carmel = make_active_event(fire_event_id=1, location_name="Carmel Demo Area")
    judean_hills = make_active_event(fire_event_id=2, location_name="Judean Hills Demo Area")
    history = [
        ChatMessage(role="user", content="What is the current status of the fire in the Carmel?"),
        ChatMessage(role="assistant", content="The Carmel event is currently CONFIRMED."),
        ChatMessage(role="user", content="Compare Carmel Demo Area and Judean Hills Demo Area."),
        ChatMessage(
            role="assistant",
            content="Carmel Demo Area and Judean Hills Demo Area are both currently confirmed.",
        ),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(carmel, judean_hills),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    agent.ask("What is the severity of the event?", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "Resolved conversation focus" not in final_text


def test_explicit_current_comparison_question_still_suppresses_focus_after_a_newer_single_event_turn():
    """Even when a newer history turn would otherwise resolve unambiguously,
    an explicit comparison/all-events CURRENT question must still suppress
    forced single-event scope - unchanged from before this fix."""
    carmel = make_active_event(fire_event_id=1, location_name="Carmel Demo Area")
    judean_hills = make_active_event(fire_event_id=2, location_name="Judean Hills Demo Area")
    history = [
        ChatMessage(role="user", content="Compare all active fire events right now."),
        ChatMessage(
            role="assistant",
            content="Carmel Demo Area and Judean Hills Demo Area are both currently confirmed.",
        ),
        ChatMessage(role="user", content="What is the current status of the fire in the Carmel?"),
        ChatMessage(role="assistant", content="The Carmel event is currently CONFIRMED."),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(carmel, judean_hills),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    agent.ask("Compare the severity of all active events.", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "Resolved conversation focus" not in final_text


def test_resolved_focus_after_recency_fix_still_keeps_snapshot_authoritative():
    """Regression guard: the current EcoGuard snapshot remains the sole
    factual source and still lists every active event, even when an older
    comparison turn is overridden by a newer single-event one."""
    carmel = make_active_event(fire_event_id=1, location_name="Carmel Demo Area")
    judean_hills = make_active_event(fire_event_id=2, location_name="Judean Hills Demo Area")
    history = [
        ChatMessage(role="user", content="Compare all active fire events right now."),
        ChatMessage(
            role="assistant",
            content="Carmel Demo Area and Judean Hills Demo Area are both currently confirmed.",
        ),
        ChatMessage(role="user", content="What is the current status of the fire in the Carmel?"),
        ChatMessage(role="assistant", content="The Carmel event is currently CONFIRMED."),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(carmel, judean_hills),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    agent.ask("What is the severity of the event?", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "Use the current EcoGuard data snapshot above for all factual values" in final_text
    snapshot = sent_snapshot(gemini)
    assert {e["location_name"] for e in snapshot["active_fire_events"]} == {
        "Carmel Demo Area",
        "Judean Hills Demo Area",
    }


def test_explicit_comparison_question_suppresses_forced_single_event_scope():
    """Even with a resolvable single prior subject, an explicit comparison/
    all-events question must never be forced into single-event scope."""
    galilee = make_active_event(fire_event_id=1, location_name="Galilee Demo Area")
    carmel = make_active_event(fire_event_id=2, location_name="Carmel Demo Area")
    history = [
        ChatMessage(role="user", content="What is the current status of the fire in the Galilee?"),
        ChatMessage(role="assistant", content="The Galilee event is currently CONFIRMED."),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(galilee, carmel),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    agent.ask("Compare the severity of all active events.", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "Resolved conversation focus" not in final_text


def test_focus_block_states_history_is_not_the_factual_source():
    galilee = make_active_event(fire_event_id=1, location_name="Galilee Demo Area")
    carmel = make_active_event(fire_event_id=2, location_name="Carmel Demo Area")
    history = [
        ChatMessage(role="user", content="What is the current status of the fire in the Galilee?"),
        ChatMessage(role="assistant", content="The Galilee event is currently CONFIRMED."),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(galilee, carmel),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    agent.ask("ומה חומרת האירוע?", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "not a factual source" in final_text
    assert "Use the current EcoGuard data snapshot above for all factual values" in final_text


def test_focus_block_never_exposes_internal_ids():
    galilee = make_active_event(fire_event_id=1, location_name="Galilee Demo Area")
    carmel = make_active_event(fire_event_id=2, location_name="Carmel Demo Area")
    history = [
        ChatMessage(role="user", content="What is the current status of the fire in the Galilee?"),
        ChatMessage(role="assistant", content="The Galilee event is currently CONFIRMED."),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(galilee, carmel),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    agent.ask("ומה חומרת האירוע?", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    focus_text = final_text.split("Resolved conversation focus", 1)[1]
    assert "fire_event_id" not in focus_text
    assert '"A"' not in focus_text
    assert '"B"' not in focus_text
    assert "event_ref" not in focus_text


def test_no_history_never_forces_a_focus():
    galilee = make_active_event(fire_event_id=1, location_name="Galilee Demo Area")
    carmel = make_active_event(fire_event_id=2, location_name="Carmel Demo Area")
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(galilee, carmel),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    agent.ask("ומה חומרת האירוע?")

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "Resolved conversation focus" not in final_text


def test_question_without_generic_reference_never_forces_a_focus():
    """A follow-up that already explicitly names a location doesn't need -
    and must not get - a forced focus block; the existing behavior already
    handles it."""
    galilee = make_active_event(fire_event_id=1, location_name="Galilee Demo Area")
    carmel = make_active_event(fire_event_id=2, location_name="Carmel Demo Area")
    history = [
        ChatMessage(role="user", content="What is the current status of the fire in the Galilee?"),
        ChatMessage(role="assistant", content="The Galilee event is currently CONFIRMED."),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(galilee, carmel),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    agent.ask("What is the severity of the fire in Carmel?", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "Resolved conversation focus" not in final_text


def test_single_active_event_never_needs_a_forced_focus():
    """Only one active event exists - nothing to disambiguate, so no focus
    block is added (it would be redundant, not harmful, but the resolver
    deliberately skips this case)."""
    galilee = make_active_event(fire_event_id=1, location_name="Galilee Demo Area")
    history = [
        ChatMessage(role="user", content="What is the current status of the fire in the Galilee?"),
        ChatMessage(role="assistant", content="The Galilee event is currently CONFIRMED."),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(galilee), details_by_id={1: make_event_details()}
    )

    agent.ask("ומה חומרת האירוע?", history=history)

    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "Resolved conversation focus" not in final_text


def test_resolved_focus_scenario_still_carries_the_language_selection_rule():
    """Regression guard: the language-selection fix (a separate, earlier
    task) must remain intact alongside this new mechanism."""
    galilee = make_active_event(fire_event_id=1, location_name="Galilee Demo Area")
    carmel = make_active_event(fire_event_id=2, location_name="Carmel Demo Area")
    history = [
        ChatMessage(role="user", content="What is the current status of the fire in the Galilee?"),
        ChatMessage(role="assistant", content="The Galilee event is currently CONFIRMED."),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(galilee, carmel),
        details_by_id={1: make_event_details(), 2: make_event_details()},
    )

    agent.ask("ומה חומרת האירוע?", history=history)

    system_instruction = gemini.calls[-1]["system_instruction"].lower()
    assert "the language of the user's current question only" in system_instruction
    assert "insufficient_data" in system_instruction
    assert "not a calibrated real-world probability" in system_instruction


def test_context_never_presents_model_score_as_a_probability():
    event = make_active_event(fire_event_id=1, ml_summary=make_ml_summary(available=True, model_score=0.87))
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(event), details_by_id={1: make_event_details()}
    )

    agent.ask("Which active event has a high fire-risk score?")

    snapshot = sent_snapshot(gemini)
    ml_assessment = snapshot["active_fire_events"][0]["ml_assessment"]
    assert ml_assessment["model_score"] == 0.87
    assert ml_assessment["available"] is True
    # The context itself never renames/transforms the raw score into a
    # probability-shaped field - only the system instruction governs phrasing.
    final_text = gemini.calls[-1]["contents"][-1]["parts"][0]["text"]
    assert "87%" not in final_text
    assert "probability" not in final_text.lower()


# ---------------------------------------------------------------------------
# 19: Gemini typed failures propagate unchanged
# ---------------------------------------------------------------------------


def test_gemini_typed_failure_propagates_unchanged():
    failing_gemini = FakeGeminiClient(raise_exc=GeminiServiceUnavailableError("Gemini request timed out."))
    agent, _, _, _ = make_agent(
        active_result=make_active_result(), details_by_id={}, gemini_client=failing_gemini
    )

    with pytest.raises(GeminiServiceUnavailableError):
        agent.ask("Are there any active fires?")


# ---------------------------------------------------------------------------
# 20: no recalculation - static architecture guard
# ---------------------------------------------------------------------------


def test_agent_does_not_import_domain_agents_calculators_or_repositories():
    forbidden_fragments = (
        "FireDetectionAgent",
        "FireSeverityAssessmentAgent",
        "FireSpreadPredictionAgent",
        "FireDangerAssessmentAgent",
        "ResponseTargetGenerationAgent",
        "ResponseOptimizationAgent",
        "RoutePlanningAgent",
        "OperationalRefreshOrchestrator",
        "GlobalPlanningOrchestrator",
        "src.calculators",
        "src.repositories",
        "src.agents.analysis",
        "src.agents.routing",
        "src.agents.collection",
        "src.simulation",
        "src.database",
        "src.external.ims",
        "src.external.firms",
        "src.external.copernicus",
        "src.external.news",
        "src.external.geocoding",
    )
    path = (Path(__file__).resolve().parents[4] / "backend/src/agents/response/chatbot_agent.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        module = ""
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
        elif isinstance(node, ast.Import):
            module = ",".join(alias.name for alias in node.names)
        else:
            continue
        if any(fragment in module for fragment in forbidden_fragments):
            violations.append(module)
    assert violations == []


def test_fakes_expose_no_recalculation_methods():
    for fake in (
        FakeActiveFireEventsService(make_active_result()),
        FakeEventDetailsService({}),
        FakeWeatherConditionsQueryService(),
        FakeFireDangerQueryService(),
        FakeGeminiClient(),
    ):
        for forbidden in (
            "detect",
            "assess",
            "assess_severity",
            "predict_spread",
            "generate_targets",
            "optimize",
            "plan_route",
            "collect",
            "get_station_data",
        ):
            assert not hasattr(fake, forbidden)


# ---------------------------------------------------------------------------
# ChatMessage validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_role", ["model", "system", "", "USER", None])
def test_chat_message_rejects_invalid_role(bad_role):
    with pytest.raises(ValueError):
        ChatMessage(role=bad_role, content="hello")


@pytest.mark.parametrize("bad_content", ["", "   ", None])
def test_chat_message_rejects_invalid_content(bad_content):
    with pytest.raises(ValueError):
        ChatMessage(role="user", content=bad_content)


def test_ask_rejects_empty_question():
    agent, _, _, _ = make_agent(active_result=make_active_result(), details_by_id={})

    with pytest.raises(ValueError):
        agent.ask("   ")


# ---------------------------------------------------------------------------
# Task 7D-B: compact context - a fully populated event used to prove both
# "removed" (full catalogs/IDs/raw arrays) and "preserved" (operational
# fields) claims against the SAME realistic snapshot.
# ---------------------------------------------------------------------------


def _make_fully_populated_details() -> EventDetailsResult:
    return make_event_details(
        fire_event=make_fire_event_summary(fire_event_id=1),
        severity=make_severity_response(),
        ml_assessment=make_ml_assessment_response(),
        detection_evidence=DetectionEvidenceResponse(
            satellite=[make_satellite_evidence()], news=[make_news_evidence()]
        ),
        spread_predictions=[
            make_spread_prediction(horizon_minutes=30, cells=[make_spread_cell(), make_spread_cell()]),
            make_spread_prediction(horizon_minutes=60, status=FireSpreadPredictionStatus.INSUFFICIENT_DATA, cells=[]),
        ],
        targets=[make_target()],
        stations=[make_station(station_id="35", name="Central Galilee"), make_station(station_id="99", name="Unrelated Station")],
        resources=[make_resource(resource_id="TRUCK-35-2", station_id="35")],
        station_summaries=[],  # populated separately where the "absent" claim is tested
        current_response_plan=make_response_plan(
            actions=[make_response_action(resource_id="TRUCK-35-2", station_id="35", node_path=[1, 2, 3, 4, 5])]
        ),
    )


def _ask_with_fully_populated_event(**event_overrides):
    event = make_active_event(fire_event_id=1, location_name="Haifa", **event_overrides)
    details = _make_fully_populated_details()
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(event),
        details_by_id={1: details},
        weather_conditions=(make_weather_conditions(),),
        fire_danger_areas=(make_fire_danger_area(),),
    )
    agent.ask("What is the status of the fire in Haifa, including its response plan?")
    return sent_snapshot(gemini), raw_snapshot_text(gemini)


def test_full_event_details_model_dump_is_no_longer_forwarded():
    snapshot, _ = _ask_with_fully_populated_event()
    event_context = snapshot["active_fire_events"][0]

    # The old nested shape ({"fire_event_id", "active_summary", "event_details"})
    # is gone entirely - this is a flat, explicit projection now.
    assert "event_details" not in event_context
    assert "active_summary" not in event_context
    assert "fire_event" not in event_context
    assert "danger" not in event_context


def test_full_stations_catalog_is_absent():
    snapshot, raw_text = _ask_with_fully_populated_event()
    assert "stations" not in snapshot["active_fire_events"][0]
    # The second, unrelated station was never allocated - its name must
    # never leak into the snapshot even as part of some other section.
    assert "Unrelated Station" not in raw_text


def test_full_resources_catalog_is_absent():
    snapshot, _ = _ask_with_fully_populated_event()
    assert "resources" not in snapshot["active_fire_events"][0]


def test_station_summaries_bulk_array_is_absent():
    snapshot, raw_text = _ask_with_fully_populated_event()
    assert "station_summaries" not in snapshot["active_fire_events"][0]
    assert "station_summaries" not in raw_text


def test_node_path_is_absent():
    _, raw_text = _ask_with_fully_populated_event()
    assert "node_path" not in raw_text
    assert "101" not in raw_text  # a raw graph-node id from the fixture's node_path


def test_raw_spread_cells_are_absent_but_cell_counts_are_present():
    snapshot, raw_text = _ask_with_fully_populated_event()
    spread = snapshot["active_fire_events"][0]["spread_predictions"]
    assert '"cells"' not in raw_text  # raw cell arrays are never forwarded
    thirty_min = next(s for s in spread if s["horizon_minutes"] == 30)
    sixty_min = next(s for s in spread if s["horizon_minutes"] == 60)
    assert thirty_min["total_cell_count"] == 2
    assert thirty_min["spreading_cell_count"] == 2  # fixture cells have p=0.6
    assert sixty_min["total_cell_count"] == 0
    assert "cell_count" not in thirty_min and "has_cells" not in thirty_min


def test_internal_database_ids_are_absent_as_keys():
    snapshot, _ = _ask_with_fully_populated_event()
    present_keys = set(_walk_keys(snapshot))
    for forbidden_key in FORBIDDEN_INTERNAL_ID_KEYS:
        assert forbidden_key not in present_keys, f"forbidden internal id key present: {forbidden_key!r}"


def test_allocated_station_and_resource_information_remains_available():
    snapshot, _ = _ask_with_fully_populated_event()
    plan = snapshot["active_fire_events"][0]["current_response_plan"]
    assert plan["allocations"] == [
        {
            "station_name": "Central Galilee",
            "resource_name": "TRUCK-35-2",
            "eta_seconds": 790.4,
            "route_distance_meters": 12645.5,
        }
    ]


def test_response_plan_eta_remains_available():
    snapshot, _ = _ask_with_fully_populated_event()
    assert snapshot["active_fire_events"][0]["current_response_plan"]["average_eta_seconds"] == 790.4


def test_response_plan_coverage_remains_available():
    snapshot, _ = _ask_with_fully_populated_event()
    assert snapshot["active_fire_events"][0]["current_response_plan"]["coverage_score"] == 100.0


def test_severity_remains_available():
    snapshot, _ = _ask_with_fully_populated_event()
    severity = snapshot["active_fire_events"][0]["severity"]
    assert severity["level"] == "critical"
    assert severity["score"] == 81.4
    assert severity["status"] == "valid"


def test_model_score_remains_available():
    snapshot, _ = _ask_with_fully_populated_event()
    ml_assessment = snapshot["active_fire_events"][0]["ml_assessment"]
    assert ml_assessment["model_score"] == 0.87
    assert ml_assessment["available"] is True


def test_evidence_remains_available():
    snapshot, _ = _ask_with_fully_populated_event()
    evidence = snapshot["active_fire_events"][0]["detection_evidence"]
    assert evidence["satellite"][0]["frp"] == 48.2
    assert evidence["satellite"][0]["confidence"] == "h"
    assert evidence["news"][0]["summary"] == "Residents report heavy smoke near the area."


def test_spread_horizon_and_status_remain_available():
    snapshot, _ = _ask_with_fully_populated_event()
    spread = snapshot["active_fire_events"][0]["spread_predictions"]
    assert {s["horizon_minutes"] for s in spread} == {30, 60}
    assert {s["status"] for s in spread} == {"valid", "insufficient_data"}


def test_targets_remain_available():
    snapshot, _ = _ask_with_fully_populated_event()
    targets = snapshot["active_fire_events"][0]["targets"]
    assert targets == [
        {
            "target_type": "active_fire",
            "priority_score": 184.3,
            "latitude": 32.8,
            "longitude": 35.0,
            "prediction_horizon_minutes": None,
        }
    ]


def test_no_current_response_plan_is_represented_as_none_not_the_national_inventory():
    event = make_active_event(fire_event_id=1, location_name="Haifa")
    details = make_event_details(
        stations=[make_station()],
        resources=[make_resource()],
        current_response_plan=None,
    )
    agent, _, _, gemini = make_agent(active_result=make_active_result(event), details_by_id={1: details})

    agent.ask("What is the response plan for the fire in Haifa?")

    snapshot = sent_snapshot(gemini)
    assert snapshot["active_fire_events"][0]["current_response_plan"] is None
    # No fallback to the full stations/resources catalog when there is no plan.
    assert "stations" not in snapshot["active_fire_events"][0]
    assert "resources" not in snapshot["active_fire_events"][0]


def test_severity_falls_back_to_active_summary_when_event_details_unavailable():
    event = make_active_event(fire_event_id=1, severity=make_severity(level=FireSeverityLevel.HIGH, score=70.0))
    agent, _, _, gemini = make_agent(active_result=make_active_result(event), details_by_id={})

    agent.ask("What is the severity?")

    snapshot = sent_snapshot(gemini)
    severity = snapshot["active_fire_events"][0]["severity"]
    assert severity["level"] == "high"
    assert severity["score"] == 70.0


def test_ml_assessment_falls_back_to_active_summary_when_event_details_unavailable():
    event = make_active_event(fire_event_id=1, ml_summary=make_ml_summary(available=True, model_score=0.6))
    agent, _, _, gemini = make_agent(active_result=make_active_result(event), details_by_id={})

    agent.ask("What is the ML score?")

    snapshot = sent_snapshot(gemini)
    ml_assessment = snapshot["active_fire_events"][0]["ml_assessment"]
    assert ml_assessment["model_score"] == 0.6
    assert ml_assessment["model_name"] is None  # never fabricated - the lightweight summary has no model_name


def test_snapshot_json_is_compact_not_pretty_printed():
    event = make_active_event(fire_event_id=1)
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(event), details_by_id={1: make_event_details()}
    )

    agent.ask("What are the active fires?")

    raw_text = raw_snapshot_text(gemini)
    assert "\n" not in raw_text
    assert ": " not in raw_text  # compact separators=(",", ":") - no space after ":"
    # Still valid, parseable JSON.
    json.loads(raw_text)


# ---------------------------------------------------------------------------
# Fire-spread representation (US 4.2 methodology 1.1 + insufficient_data_reason)
# ---------------------------------------------------------------------------

from src.agents.response.chatbot_agent import _SPREAD_PROPAGATION_THRESHOLD  # noqa: E402
from src.models.fire_spread_insufficient_data_reason import FireSpreadInsufficientDataReason  # noqa: E402

_RISK_ONLY_P = 0.38
_SPREADING_P = 0.72


def _spread_cells(*probabilities):
    return [
        make_spread_cell(spread_probability=p, spread_risk_score=p * 100, latitude=32.81 + i * 0.001)
        for i, p in enumerate(probabilities)
    ]


def _compact_spread_for(*predictions):
    event = make_active_event(fire_event_id=1, location_name="Carmel Demo Area")
    details = make_event_details(spread_predictions=list(predictions))
    agent, _, _, gemini = make_agent(active_result=make_active_result(event), details_by_id={1: details})
    agent.ask("What is the fire spread prediction for Carmel?")
    spread = sent_snapshot(gemini)["active_fire_events"][0]["spread_predictions"]
    return {item["horizon_minutes"]: item for item in spread}


def test_spread_threshold_mirror_matches_the_methodology_constant():
    # One numeric source of truth (src.models.fire_spread_prediction): the chatbot and the
    # calculator config must reference the SAME constant, and the chatbot must not
    # re-declare its own literal.
    import inspect
    import re

    from src.agents.response import chatbot_agent
    from src.calculators.fire_spread.fire_spread_config import PROPAGATION_THRESHOLD as CONFIG_THRESHOLD
    from src.models.fire_spread_prediction import PROPAGATION_THRESHOLD

    assert _SPREAD_PROPAGATION_THRESHOLD is PROPAGATION_THRESHOLD
    assert CONFIG_THRESHOLD is PROPAGATION_THRESHOLD
    assert not re.search(r"_SPREAD_PROPAGATION_THRESHOLD\s*=\s*[0-9.]", inspect.getsource(chatbot_agent))


def test_valid_risk_only_ring_is_not_reported_as_spreading():
    compact = _compact_spread_for(make_spread_prediction(horizon_minutes=30, cells=_spread_cells(*[_RISK_ONLY_P] * 8)))[30]

    assert compact["status"] == "valid"
    assert compact["total_cell_count"] == 8
    assert compact["risk_only_cell_count"] == 8
    assert compact["spreading_cell_count"] == 0
    assert compact["max_spread_probability"] == pytest.approx(_RISK_ONLY_P)
    assert compact["max_spread_risk_score"] == pytest.approx(_RISK_ONLY_P * 100)
    assert compact["insufficient_data_reason"] is None
    assert compact["insufficient_data_reason_description"] is None


def test_valid_mixed_spreading_and_risk_only_counts_and_maxima():
    compact = _compact_spread_for(
        make_spread_prediction(horizon_minutes=30, cells=_spread_cells(_SPREADING_P, 0.55, _RISK_ONLY_P, 0.2))
    )[30]

    assert compact["total_cell_count"] == 4
    assert compact["spreading_cell_count"] == 2
    assert compact["risk_only_cell_count"] == 2
    assert compact["max_spread_probability"] == pytest.approx(_SPREADING_P)
    assert compact["max_spread_risk_score"] == pytest.approx(_SPREADING_P * 100)


def test_valid_spreading_only_prediction():
    compact = _compact_spread_for(make_spread_prediction(horizon_minutes=30, cells=_spread_cells(0.6, 0.7)))[30]

    assert (compact["spreading_cell_count"], compact["risk_only_cell_count"]) == (2, 0)


def test_legacy_valid_prediction_with_zero_cells():
    compact = _compact_spread_for(make_spread_prediction(horizon_minutes=30, cells=[]))[30]

    assert compact["total_cell_count"] == 0
    assert compact["spreading_cell_count"] == 0
    assert compact["risk_only_cell_count"] == 0
    assert compact["max_spread_probability"] is None
    assert compact["max_spread_risk_score"] is None


def test_probability_exactly_at_threshold_counts_as_spreading():
    compact = _compact_spread_for(
        make_spread_prediction(horizon_minutes=30, cells=_spread_cells(_SPREAD_PROPAGATION_THRESHOLD))
    )[30]

    assert (compact["spreading_cell_count"], compact["risk_only_cell_count"]) == (1, 0)


@pytest.mark.parametrize(
    ("reason", "expected_fragment"),
    [
        (FireSpreadInsufficientDataReason.MISSING_VEGETATION, "vegetation data required by the spread model was unavailable"),
        (FireSpreadInsufficientDataReason.UNSUPPORTED_VEGETATION, "not supported by the current spread model"),
        (FireSpreadInsufficientDataReason.STALE_WEATHER, "too old for spread prediction"),
        (FireSpreadInsufficientDataReason.MISSING_WEATHER, "required weather data was unavailable"),
    ],
)
def test_insufficient_data_reason_and_description_are_included(reason, expected_fragment):
    compact = _compact_spread_for(
        make_spread_prediction(
            horizon_minutes=30,
            status=FireSpreadPredictionStatus.INSUFFICIENT_DATA,
            cells=[],
            insufficient_data_reason=reason,
        )
    )[30]

    assert compact["status"] == "insufficient_data"
    assert compact["insufficient_data_reason"] == reason.value
    assert expected_fragment in compact["insufficient_data_reason_description"].lower()


def test_unsupported_vegetation_is_distinct_from_missing_vegetation_and_never_blames_copernicus():
    descriptions = {
        reason: _compact_spread_for(
            make_spread_prediction(
                horizon_minutes=30,
                status=FireSpreadPredictionStatus.INSUFFICIENT_DATA,
                cells=[],
                insufficient_data_reason=reason,
            )
        )[30]["insufficient_data_reason_description"]
        for reason in (
            FireSpreadInsufficientDataReason.MISSING_VEGETATION,
            FireSpreadInsufficientDataReason.UNSUPPORTED_VEGETATION,
        )
    }

    assert len(set(descriptions.values())) == 2
    for description in descriptions.values():
        assert "copernicus" not in description.lower()
        assert "fail" not in description.lower()
    assert "unavailable" not in descriptions[FireSpreadInsufficientDataReason.UNSUPPORTED_VEGETATION].lower()


def test_every_reason_value_has_a_description():
    from src.agents.response.chatbot_agent import _INSUFFICIENT_DATA_REASON_DESCRIPTIONS

    assert set(_INSUFFICIENT_DATA_REASON_DESCRIPTIONS) == set(FireSpreadInsufficientDataReason)


def test_historical_insufficient_data_without_reason_falls_back_safely():
    compact = _compact_spread_for(
        make_spread_prediction(horizon_minutes=30, status=FireSpreadPredictionStatus.INSUFFICIENT_DATA, cells=[])
    )[30]

    assert compact["insufficient_data_reason"] is None
    assert compact["insufficient_data_reason_description"] is None


def test_inactive_event_prediction_is_compacted_without_reason():
    compact = _compact_spread_for(
        make_spread_prediction(horizon_minutes=30, status=FireSpreadPredictionStatus.INACTIVE_EVENT, cells=[])
    )[30]

    assert compact["status"] == "inactive_event"
    assert compact["total_cell_count"] == 0
    assert compact["insufficient_data_reason"] is None


def test_30_and_60_minute_horizons_are_compacted_independently():
    by_horizon = _compact_spread_for(
        make_spread_prediction(horizon_minutes=30, cells=_spread_cells(*[_RISK_ONLY_P] * 8)),
        make_spread_prediction(horizon_minutes=60, cells=_spread_cells(_SPREADING_P, _SPREADING_P, *[_RISK_ONLY_P] * 10)),
    )

    assert (by_horizon[30]["spreading_cell_count"], by_horizon[30]["risk_only_cell_count"]) == (0, 8)
    assert (by_horizon[60]["spreading_cell_count"], by_horizon[60]["risk_only_cell_count"]) == (2, 10)


def test_system_instruction_distinguishes_risk_only_from_propagation_capable_cells():
    lowered = _SYSTEM_INSTRUCTION.lower()
    for phrase in (
        "spreading_cell_count",
        "risk_only_cell_count",
        "propagation threshold",
        "none reached the model's propagation threshold",
        "never describe risk-only areas as places the fire is predicted to spread to",
        "the fire will reach",
        "same risk footprint for 30 and 60 minutes",
    ):
        assert phrase in lowered, f"missing required system-instruction phrase: {phrase!r}"


def test_system_instruction_uses_area_wording_and_hides_internal_field_names():
    lowered = _SYSTEM_INSTRUCTION.lower()
    assert '8 nearby areas show predicted spread risk, but none reached the model\'s propagation threshold.' in lowered
    assert "the 60-minute forecast shows the same risk footprint." in lowered
    assert 'never say "cell" or "cells"' in lowered
    assert "never mention internal field names such as spreading_cell_count, risk_only_cell_count" in lowered


def test_system_instruction_explains_reasons_without_inventing_causes():
    lowered = _SYSTEM_INSTRUCTION.lower()
    assert "insufficient_data_reason_description" in lowered
    assert "does not mean an external service failed" in lowered
    assert "not supported by the current spread model, not that vegetation data is missing" in lowered
    assert "when insufficient_data_reason is null" in lowered


# ---------------------------------------------------------------------------
# Area-level context: Weather Conditions + Fire Danger, and the ML/hybrid
# detection explanation fields
# ---------------------------------------------------------------------------

WEATHER_OBSERVED_AT = AS_OF - timedelta(minutes=4)
DANGER_ASSESSED_AT = AS_OF - timedelta(minutes=3)


def make_weather_conditions(**overrides) -> AreaWeatherConditions:
    values = dict(
        area_name="Judean Hills Demo Area",
        observed_at=WEATHER_OBSERVED_AT,
        assessed_at=DANGER_ASSESSED_AT,
        station_count=3,
        temperature_c=37.5,
        relative_humidity_pct=16.5,
        wind_speed_kmh=19.8,
        wind_gust_kmh=30.3,
    )
    values.update(overrides)
    return AreaWeatherConditions(**values)


def make_fire_danger_area(
    *,
    area_name="Judean Hills Demo Area",
    status=FireDangerAssessmentStatus.VALID,
    level=FireDangerLevel.HIGH,
    score=32.5,
) -> FireDangerAreaSnapshot:
    return FireDangerAreaSnapshot(
        area_id="judean_hills",
        area_name=area_name,
        area_latitude=31.665,
        area_longitude=35.045,
        area_radius_km=5.0,
        assessment=FireDangerAreaAssessmentSummary(
            assessment_id=1488,
            status=status,
            score=score,
            level=level,
            assessed_at=DANGER_ASSESSED_AT,
            methodology="ffwi",
            methodology_version="1.0",
        ),
    )


def _ask_with_area_context(*, weather_conditions=(), fire_danger_areas=(), events=()):
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(*events),
        details_by_id={},
        weather_conditions=tuple(weather_conditions),
        fire_danger_areas=tuple(fire_danger_areas),
    )
    agent.ask("מה תנאי מזג האוויר בהרי יהודה?")
    return sent_snapshot(gemini), raw_snapshot_text(gemini)


def test_snapshot_includes_weather_conditions_with_the_stored_values():
    snapshot, _ = _ask_with_area_context(weather_conditions=(make_weather_conditions(),))

    assert snapshot["weather_conditions"] == [
        {
            "area_name": "Judean Hills Demo Area",
            "observed_at": WEATHER_OBSERVED_AT.isoformat(),
            "assessed_at": DANGER_ASSESSED_AT.isoformat(),
            "station_count": 3,
            "temperature_c": 37.5,
            "relative_humidity_pct": 16.5,
            "wind_speed_kmh": 19.8,
            "wind_gust_kmh": 30.3,
        }
    ]


def test_weather_conditions_without_gust_keeps_gust_null_not_fabricated():
    snapshot, _ = _ask_with_area_context(weather_conditions=(make_weather_conditions(wind_gust_kmh=None),))

    assert snapshot["weather_conditions"][0]["wind_gust_kmh"] is None


def test_weather_and_danger_include_areas_without_an_active_fire():
    # Only Galilee has an active fire; Jerusalem Forest and Judean Hills do not.
    snapshot, _ = _ask_with_area_context(
        events=(make_active_event(fire_event_id=1, location_name="Galilee Demo Area"),),
        weather_conditions=(
            make_weather_conditions(area_name="Jerusalem Forest Demo Area", temperature_c=35.0),
            make_weather_conditions(area_name="Judean Hills Demo Area"),
        ),
        fire_danger_areas=(
            make_fire_danger_area(area_name="Jerusalem Forest Demo Area", level=FireDangerLevel.VERY_HIGH, score=55.6),
            make_fire_danger_area(area_name="Judean Hills Demo Area"),
        ),
    )

    assert [event["location_name"] for event in snapshot["active_fire_events"]] == ["Galilee Demo Area"]
    assert [item["area_name"] for item in snapshot["weather_conditions"]] == [
        "Jerusalem Forest Demo Area",
        "Judean Hills Demo Area",
    ]
    assert snapshot["weather_conditions"][0]["temperature_c"] == 35.0
    assert [item["area_name"] for item in snapshot["fire_danger"]] == [
        "Jerusalem Forest Demo Area",
        "Judean Hills Demo Area",
    ]


def test_weather_is_top_level_and_never_attached_to_an_event():
    snapshot, _ = _ask_with_area_context(
        events=(make_active_event(fire_event_id=1, location_name="Judean Hills Demo Area"),),
        weather_conditions=(make_weather_conditions(),),
        fire_danger_areas=(make_fire_danger_area(),),
    )

    event_context = snapshot["active_fire_events"][0]
    assert "weather_conditions" not in event_context
    assert "fire_danger" not in event_context


def test_snapshot_includes_fire_danger_entries():
    snapshot, _ = _ask_with_area_context(fire_danger_areas=(make_fire_danger_area(),))

    assert snapshot["fire_danger"] == [
        {
            "area_name": "Judean Hills Demo Area",
            "status": "valid",
            "level": "high",
            "score": 32.5,
            "assessed_at": DANGER_ASSESSED_AT.isoformat(),
            "methodology": "ffwi",
        }
    ]


def test_insufficient_data_fire_danger_is_kept_with_null_level_and_score():
    snapshot, _ = _ask_with_area_context(
        fire_danger_areas=(
            make_fire_danger_area(status=FireDangerAssessmentStatus.INSUFFICIENT_DATA, level=None, score=None),
        )
    )

    entry = snapshot["fire_danger"][0]
    assert entry["status"] == "insufficient_data"
    assert entry["level"] is None
    assert entry["score"] is None


def test_fire_danger_area_without_any_assessment_is_omitted():
    area = FireDangerAreaSnapshot(
        area_id="golan",
        area_name="Golan Heights Demo Area",
        area_latitude=33.0,
        area_longitude=35.7,
        area_radius_km=5.0,
        assessment=None,
    )
    snapshot, _ = _ask_with_area_context(fire_danger_areas=(area,))

    assert snapshot["fire_danger"] == []


def test_empty_weather_and_danger_sections_are_explicit_empty_lists():
    snapshot, _ = _ask_with_area_context()

    assert snapshot["weather_conditions"] == []
    assert snapshot["fire_danger"] == []


def test_area_sections_contain_no_internal_ids():
    snapshot, raw_text = _ask_with_area_context(
        weather_conditions=(make_weather_conditions(),), fire_danger_areas=(make_fire_danger_area(),)
    )

    area_keys = set(_walk_keys(snapshot["weather_conditions"])) | set(_walk_keys(snapshot["fire_danger"]))
    for forbidden_key in FORBIDDEN_INTERNAL_ID_KEYS:
        assert forbidden_key not in area_keys
    # The fake assessment id (1488) and the area_id slug never reach the LLM.
    assert "1488" not in raw_text
    assert "judean_hills" not in raw_text


def test_area_services_are_read_once_per_question_as_of_the_snapshot_instant():
    agent, _, _, gemini = make_agent(active_result=make_active_result(), details_by_id={})

    agent.ask("איזה אזור נמצא בסכנת השריפה הגבוהה ביותר?")

    assert agent._fire_danger_query_service.requested_as_of == [AS_OF]
    assert agent._weather_conditions_query_service.calls == 1


def test_ml_assessment_includes_the_detection_explanation_fields():
    snapshot, _ = _ask_with_fully_populated_event()

    ml_assessment = snapshot["active_fire_events"][0]["ml_assessment"]
    assert ml_assessment["detection_mode"] == "shadow"
    assert ml_assessment["rule_status"] == "confirmed"
    assert ml_assessment["rule_confidence"] == 0.9
    assert ml_assessment["agreement"] == "agree_fire"
    assert ml_assessment["model_score"] == 0.87
    assert ml_assessment["model_name"] == "fire_detection_logistic_v3"
    assert ml_assessment["model_version"] == "3.0"


def test_ml_summary_fallback_never_fabricates_rule_fields():
    event = make_active_event(fire_event_id=1, ml_summary=make_ml_summary(available=True, model_score=0.6))
    agent, _, _, gemini = make_agent(active_result=make_active_result(event), details_by_id={})

    agent.ask("Why was this fire detected?")

    ml_assessment = sent_snapshot(gemini)["active_fire_events"][0]["ml_assessment"]
    assert ml_assessment["model_score"] == 0.6
    assert ml_assessment["rule_status"] is None
    assert ml_assessment["rule_confidence"] is None
    assert ml_assessment["agreement"] is None


def test_existing_event_context_is_unchanged_alongside_area_sections():
    snapshot, _ = _ask_with_fully_populated_event()

    assert set(snapshot["active_fire_events"][0]) == {
        "event_ref",
        "location_name",
        "latitude",
        "longitude",
        "status",
        "detected_at",
        "updated_at",
        "detection_confidence",
        "event_details_available",
        "severity",
        "ml_assessment",
        "detection_evidence",
        "spread_predictions",
        "targets",
        "current_response_plan",
        "resource_requirements",
    }
    assert set(snapshot) == {
        "as_of",
        "active_fire_event_count",
        "active_fire_events",
        "weather_conditions",
        "fire_danger",
        "global_response_plan",
        "station_availability",
    }


def test_history_is_still_trimmed_to_eight_messages_with_area_sections_present():
    history = [ChatMessage(role="user", content=f"message {i}") for i in range(10)]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(),
        details_by_id={},
        weather_conditions=(make_weather_conditions(),),
        fire_danger_areas=(make_fire_danger_area(),),
    )

    agent.ask("השווה את מזג האוויר בירושלים ובהרי יהודה", history=history)

    history_turns = gemini.calls[-1]["contents"][:-1]
    assert len(history_turns) == MAX_CHAT_HISTORY_MESSAGES == 8
    assert history_turns[0]["parts"][0]["text"] == "message 2"
    assert history_turns[-1]["parts"][0]["text"] == "message 9"


def test_system_instruction_contains_weather_rules():
    for fragment in (
        "weather_conditions",
        "aggregates (means) of the",
        "name the EcoGuard area whose data you used",
        "observed_at",
        "weather data is currently unavailable for",
        "Wind direction and rainfall are unavailable",
    ):
        assert fragment in _SYSTEM_INSTRUCTION, fragment


def test_system_instruction_distinguishes_fire_danger_from_severity():
    for fragment in (
        "fire_danger",
        "environmental wildfire risk assessed for an AREA",
        "it is NOT the Severity of an already detected wildfire",
        "A Fire Danger question does not",
    ):
        assert fragment in _SYSTEM_INSTRUCTION, fragment


def test_system_instruction_never_answers_weather_with_no_active_wildfire():
    assert "never answer a weather" in _SYSTEM_INSTRUCTION
    assert "or Fire Danger question by saying there is no active wildfire" in _SYSTEM_INSTRUCTION


def test_system_instruction_requires_transparent_location_mapping():
    assert "Jerusalem Forest Demo Area" in _SYSTEM_INSTRUCTION
    assert "Judean Hills Demo Area" in _SYSTEM_INSTRUCTION
    assert "always state the" in _SYSTEM_INSTRUCTION


def test_system_instruction_explains_detection_fields_and_keeps_model_score_rule():
    for fragment in ("detection_mode", "rule_status", "rule_confidence", "agreement"):
        assert fragment in _SYSTEM_INSTRUCTION
    assert "NOT a calibrated real-world" in _SYSTEM_INSTRUCTION


# ---------------------------------------------------------------------------
# Presentation precision of operational scores (presentation only)
# ---------------------------------------------------------------------------

RAW_SEVERITY_SCORE = 82.5065916694979
RAW_FIRE_DANGER_SCORE = 31.785239339370694
RAW_MODEL_SCORE = 0.9872608038579626


def _ask_with_raw_precision_scores():
    severity = make_severity_response(score=RAW_SEVERITY_SCORE)
    details = make_event_details(severity=severity, ml_assessment=make_ml_assessment_response(model_score=RAW_MODEL_SCORE))
    danger_area = make_fire_danger_area(score=RAW_FIRE_DANGER_SCORE)
    event = make_active_event(fire_event_id=1, location_name="Jerusalem Forest Demo Area")
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(event),
        details_by_id={1: details},
        fire_danger_areas=(danger_area,),
    )
    agent.ask("מה רמת סכנת השריפה וחומרת השריפה בירושלים?")
    return sent_snapshot(gemini), raw_snapshot_text(gemini), severity, danger_area


def test_severity_score_is_presented_to_one_decimal():
    snapshot, _, _, _ = _ask_with_raw_precision_scores()

    assert snapshot["active_fire_events"][0]["severity"]["score"] == 82.5


def test_fire_danger_score_is_presented_to_one_decimal():
    snapshot, _, _, _ = _ask_with_raw_precision_scores()

    assert snapshot["fire_danger"][0]["score"] == 31.8


def test_raw_score_precision_never_reaches_the_llm():
    _, raw_text, _, _ = _ask_with_raw_precision_scores()

    assert repr(RAW_SEVERITY_SCORE) not in raw_text
    assert repr(RAW_FIRE_DANGER_SCORE) not in raw_text
    assert "82.50659" not in raw_text
    assert "31.78523" not in raw_text


def test_underlying_service_values_are_not_modified():
    _, _, severity, danger_area = _ask_with_raw_precision_scores()

    assert severity.score == RAW_SEVERITY_SCORE
    assert danger_area.assessment.score == RAW_FIRE_DANGER_SCORE


def test_severity_summary_fallback_score_is_also_presented_to_one_decimal():
    event = make_active_event(fire_event_id=1, severity=make_severity(score=RAW_SEVERITY_SCORE))
    agent, _, _, gemini = make_agent(active_result=make_active_result(event), details_by_id={})

    agent.ask("What is the severity?")

    assert sent_snapshot(gemini)["active_fire_events"][0]["severity"]["score"] == 82.5


def test_model_score_keeps_its_own_precision_and_is_not_rounded_like_a_score():
    snapshot, _, _, _ = _ask_with_raw_precision_scores()

    # 0-1 model output: rounding to one decimal would turn 0.987 into 1.0.
    assert snapshot["active_fire_events"][0]["ml_assessment"]["model_score"] == RAW_MODEL_SCORE


def test_missing_scores_stay_null_rather_than_being_rounded():
    snapshot, _ = _ask_with_area_context(
        fire_danger_areas=(
            make_fire_danger_area(status=FireDangerAssessmentStatus.INSUFFICIENT_DATA, level=None, score=None),
        )
    )

    assert snapshot["fire_danger"][0]["score"] is None


def test_weather_values_are_passed_through_unchanged():
    snapshot, _ = _ask_with_area_context(weather_conditions=(make_weather_conditions(wind_speed_kmh=34.300000000000004),))

    assert snapshot["weather_conditions"][0]["wind_speed_kmh"] == 34.300000000000004


def test_system_instruction_treats_single_field_weather_questions_as_weather_questions():
    weather_rule = _SYSTEM_INSTRUCTION[_SYSTEM_INSTRUCTION.index("Weather:") : _SYSTEM_INSTRUCTION.index("Fire Danger:")]
    for field in ("temperature", "relative humidity", "wind speed", "wind gust", "general weather conditions"):
        assert field in weather_rule, field
    assert "a question about any one of these is a weather question" in weather_rule
    assert "must use weather_conditions" in weather_rule.replace("\n", " ")


# ---------------------------------------------------------------------------
# Global Response Plan, per-fire resource requirements, station availability
# ---------------------------------------------------------------------------

from src.api.schemas.event_details import StationSummaryResponse  # noqa: E402

JERUSALEM_ID, JUDEAN_ID, GALILEE_ID, INACTIVE_ID = 101, 102, 103, 999


def make_plan_action(resource_id: str, station_id: str, station_name: str, *, eta=300.0, distance=4200.0):
    return ResponsePlanActionResponse(
        resource=ResponsePlanResourceResponse(
            resource_id=resource_id, station_id=station_id, station_name=station_name, origin=None
        ),
        target=ResponsePlanTargetResponse(
            response_target_id=7001, target_type="active_fire", priority_score=0.9, latitude=31.77, longitude=35.14
        ),
        route=ResponsePlanRouteResponse(
            status=None, eta_seconds=eta, distance_meters=distance, node_path=[1, 2, 3], path_coordinates=None
        ),
    )


def make_global_event_plan(fire_event_id: int, **overrides) -> GlobalEventPlan:
    values = dict(
        fire_event_id=fire_event_id,
        response_plan_id=5000 + fire_event_id,
        severity_level=FireSeverityLevel.CRITICAL,
        severity_score=86.2,
        minimum_resources=3,
        desired_resources=4,
        assigned_resources=4,
        coverage_score=100.0,
        average_eta_seconds=412.5,
        actions=[],
        uncovered_targets=[],
    )
    values.update(overrides)
    return GlobalEventPlan(**values)


def make_global_plan_response(*, state="current", events=None, coverage=True) -> GlobalResponsePlanResponse:
    events = events if events is not None else [
        make_global_event_plan(
            JERUSALEM_ID,
            minimum_resources=3,
            desired_resources=4,
            assigned_resources=2,
            coverage_score=75.0,
            average_eta_seconds=496.75941168474793,
            actions=[
                make_plan_action("TRUCK-11-1", "11", "Jerusalem Central"),
                make_plan_action("TRUCK-11-2", "11", "Jerusalem Central"),
            ],
            uncovered_targets=[
                ResponsePlanTargetResponse(
                    response_target_id=7002, target_type="predicted_risk", priority_score=0.4, latitude=31.7, longitude=35.1
                )
            ],
        ),
        make_global_event_plan(JUDEAN_ID, minimum_resources=2, desired_resources=3, assigned_resources=3),
    ]
    return GlobalResponsePlanResponse(
        as_of=AS_OF,
        plan=GlobalPlanResponse(
            run_id=4242,
            started_at=AS_OF - timedelta(minutes=2),
            completed_at=AS_OF - timedelta(minutes=1),
            status=GlobalPlanningRunStatus.COMPLETED,
            metrics=GlobalPlanMetrics(fitness_score=9032038.29813267, coverage_score=88.5, average_eta_seconds=455.1),
            shortage=GlobalPlanShortage(
                total_required=5, total_desired=7, total_assigned=5, unmet_required=0, unmet_desired=2
            ),
            optimization_config=None,
            events=events,
        ),
        coverage=GlobalPlanCoverage(
            state=state,
            eligible_fire_event_ids=[JERUSALEM_ID, JUDEAN_ID, GALILEE_ID],
            covered_fire_event_ids=[JERUSALEM_ID, JUDEAN_ID],
            pending_fire_event_ids=[GALILEE_ID],
            unplannable_fire_event_ids=[],
            monitoring_fire_event_ids=[INACTIVE_ID],
            eligible_count=3,
            covered_count=2,
        )
        if coverage
        else None,
    )


def make_station_summary(station_id: str, *, total: int, available: int, assigned: int = 0, unavailable: int = 0):
    return StationSummaryResponse(
        station_id=station_id,
        total_resources=total,
        available=available,
        assigned_status=assigned,
        unavailable=unavailable,
        current_global_plan_allocations=[],
    )


STATIONS = [
    make_station(station_id="10", name="Beit Shemesh"),
    make_station(station_id="11", name="Jerusalem Central"),
    make_station(station_id="12", name="Mevaseret"),
    make_station(station_id="13", name="Haifa Port"),
    make_station(station_id="14", name="Quiet Station"),
]
STATION_SUMMARIES = [
    make_station_summary("10", total=3, available=0, unavailable=3),
    make_station_summary("11", total=5, available=5),  # 2 committed by the plan, status still available
    make_station_summary("12", total=4, available=3, assigned=1),
    make_station_summary("13", total=4, available=3, unavailable=1),
    make_station_summary("14", total=4, available=4),  # nothing notable -> omitted
]


def _ask_with_operations_context(*, global_plan=None, question="מה תוכנית התגובה הנוכחית?"):
    events = (
        make_active_event(fire_event_id=JERUSALEM_ID, location_name="Jerusalem Forest Demo Area"),
        make_active_event(fire_event_id=JUDEAN_ID, location_name="Judean Hills Demo Area"),
        make_active_event(fire_event_id=GALILEE_ID, location_name="Galilee Demo Area"),
    )
    details = make_event_details(stations=STATIONS, station_summaries=STATION_SUMMARIES)
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(*events),
        details_by_id={JERUSALEM_ID: details, JUDEAN_ID: details, GALILEE_ID: details},
        global_plan=global_plan if global_plan is not None else make_global_plan_response(),
    )
    agent.ask(question)
    return sent_snapshot(gemini), raw_snapshot_text(gemini), agent


def test_global_response_plan_section_carries_the_persisted_values_unchanged():
    snapshot, _, _ = _ask_with_operations_context()

    assert snapshot["global_response_plan"] == {
        "state": "current",
        "status": "completed",
        "coverage_score": 88.5,
        "average_eta_seconds": 455.1,
        "total_required_resources": 5,
        "total_desired_resources": 7,
        "total_assigned_resources": 5,
        "unmet_required_resources": 0,
        "unmet_desired_resources": 2,
        "covered_fires": ["Jerusalem Forest Demo Area", "Judean Hills Demo Area"],
        "pending_fires": ["Galilee Demo Area"],
        "unplannable_fires": [],
        "monitoring_only_fires": [_UNKNOWN_FIRE_LABEL_FOR_TESTS],
    }


_UNKNOWN_FIRE_LABEL_FOR_TESTS = "an event that is not currently active"


def test_global_plan_is_read_as_of_the_same_snapshot_instant():
    _, _, agent = _ask_with_operations_context()

    assert agent._global_response_plan_read_service.requested_as_of == [AS_OF]


def test_per_fire_resource_requirements_are_passed_unchanged():
    snapshot, _, _ = _ask_with_operations_context()
    by_location = {event["location_name"]: event for event in snapshot["active_fire_events"]}

    assert by_location["Jerusalem Forest Demo Area"]["resource_requirements"] == {
        "minimum_required_resources": 3,
        "desired_resources": 4,
        "assigned_resources": 2,
        "coverage_score": 75.0,
        "average_eta_seconds": 496.75941168474793,
        "uncovered_target_count": 1,
    }
    assert by_location["Judean Hills Demo Area"]["resource_requirements"]["assigned_resources"] == 3
    # A fire the plan does not (yet) include has no requirements - never fabricated.
    assert by_location["Galilee Demo Area"]["resource_requirements"] is None


def test_station_availability_totals_and_relevant_stations_only():
    snapshot, _, _ = _ask_with_operations_context()
    availability = snapshot["station_availability"]

    assert availability["total_station_count"] == 5
    assert availability["total_available_resources"] == 15
    assert availability["total_assigned_resources"] == 1
    assert availability["total_unavailable_resources"] == 4
    assert availability["stations_with_no_available_resources_count"] == 1
    listed = {station["station_name"]: station for station in availability["listed_stations"]}
    assert set(listed) == {"Beit Shemesh", "Jerusalem Central", "Mevaseret", "Haifa Port"}
    assert "Quiet Station" not in listed


def test_station_with_zero_available_is_listed():
    snapshot, _, _ = _ask_with_operations_context()
    beit_shemesh = next(s for s in snapshot["station_availability"]["listed_stations"] if s["station_name"] == "Beit Shemesh")

    assert beit_shemesh == {
        "station_name": "Beit Shemesh",
        "total_resources": 3,
        "available": 0,
        "assigned": 0,
        "unavailable": 3,
        "committed_to_current_plan": 0,
    }


def test_stations_with_assigned_or_unavailable_resources_are_listed():
    snapshot, _, _ = _ask_with_operations_context()
    listed = {s["station_name"]: s for s in snapshot["station_availability"]["listed_stations"]}

    assert listed["Mevaseret"]["assigned"] == 1
    assert listed["Haifa Port"]["unavailable"] == 1


def test_station_with_plan_commitments_is_listed_and_available_is_never_reduced():
    snapshot, _, _ = _ask_with_operations_context()
    jerusalem = next(
        s for s in snapshot["station_availability"]["listed_stations"] if s["station_name"] == "Jerusalem Central"
    )

    assert jerusalem["committed_to_current_plan"] == 2
    # Stored operational status, exactly as EventDetails computed it - never 5 - 2.
    assert jerusalem["available"] == 5
    assert jerusalem["total_resources"] == 5


def test_no_free_resource_count_is_derived():
    snapshot, raw_text, _ = _ask_with_operations_context()
    keys = set(_walk_keys(snapshot))

    assert not any("free" in key for key in keys)
    assert "available_minus" not in raw_text


def test_operations_context_leaks_no_internal_ids():
    snapshot, raw_text, _ = _ask_with_operations_context()
    present_keys = set(_walk_keys(snapshot))

    for forbidden_key in FORBIDDEN_INTERNAL_ID_KEYS:
        assert forbidden_key not in present_keys, forbidden_key
    for internal_value in ("4242", "5101", "5102", "7001", "7002", "\"101\"", "\"999\"", "TRUCK-11-1"):
        assert internal_value not in raw_text, internal_value
    assert not any(isinstance(value, int) and value in (JERUSALEM_ID, JUDEAN_ID, GALILEE_ID, INACTIVE_ID)
                   for value in snapshot["global_response_plan"]["covered_fires"])


def test_ordinary_catalog_is_never_dumped():
    _, raw_text, _ = _ask_with_operations_context()

    assert "Quiet Station" not in raw_text


def test_selected_station_eta_and_distance_remain_answerable():
    event = make_active_event(fire_event_id=1, location_name="Jerusalem Forest Demo Area")
    details = make_event_details(
        stations=[make_station(station_id="11", name="Jerusalem Central")],
        current_response_plan=make_response_plan(
            actions=[make_response_action(station_id="11", eta_seconds=312.0, route_distance_meters=5100.0)]
        ),
    )
    agent, _, _, gemini = make_agent(active_result=make_active_result(event), details_by_id={1: details})

    agent.ask("איזו תחנה נבחרה בפועל לשריפה בירושלים?")

    allocation = sent_snapshot(gemini)["active_fire_events"][0]["current_response_plan"]["allocations"][0]
    assert allocation["station_name"] == "Jerusalem Central"
    assert allocation["eta_seconds"] == 312.0
    assert allocation["route_distance_meters"] == 5100.0


@pytest.mark.parametrize("state", ["generating", "updating"])
def test_generating_or_updating_plan_state_is_passed_through(state):
    snapshot, _, _ = _ask_with_operations_context(global_plan=make_global_plan_response(state=state))

    assert snapshot["global_response_plan"]["state"] == state


def test_no_plan_is_represented_safely():
    no_plan = GlobalResponsePlanResponse(
        as_of=AS_OF,
        plan=None,
        coverage=GlobalPlanCoverage(
            state="none",
            eligible_fire_event_ids=[],
            covered_fire_event_ids=[],
            pending_fire_event_ids=[],
            unplannable_fire_event_ids=[],
            eligible_count=0,
            covered_count=0,
        ),
    )
    snapshot, _, _ = _ask_with_operations_context(global_plan=no_plan)

    plan = snapshot["global_response_plan"]
    assert plan["state"] == "none"
    assert plan["status"] is None
    assert plan["unmet_required_resources"] is None
    assert plan["covered_fires"] == []
    assert all(event["resource_requirements"] is None for event in snapshot["active_fire_events"])
    # Station availability does not depend on a plan existing; nothing is committed.
    listed = snapshot["station_availability"]["listed_stations"]
    assert all(station["committed_to_current_plan"] == 0 for station in listed)
    assert "Jerusalem Central" not in {station["station_name"] for station in listed}


def test_plan_without_coverage_report_keeps_coverage_groups_unknown():
    snapshot, _, _ = _ask_with_operations_context(global_plan=make_global_plan_response(coverage=False))

    plan = snapshot["global_response_plan"]
    assert plan["state"] is None
    assert plan["covered_fires"] is None
    assert plan["total_required_resources"] == 5


def test_no_active_event_details_means_no_station_summary():
    agent, _, _, gemini = make_agent(active_result=make_active_result(), details_by_id={})

    agent.ask("יש תחנות כיבוי אש בלי כבאיות זמינות כרגע?")

    assert sent_snapshot(gemini)["station_availability"] is None


def test_baseline_is_not_added_to_the_snapshot():
    snapshot, raw_text, _ = _ask_with_operations_context()

    assert "baseline" not in raw_text
    assert "improvement_percentage" not in set(_walk_keys(snapshot))


def test_system_instruction_availability_semantics():
    for fragment in (
        "STORED OPERATIONAL STATUS",
        "a resource can have status available while already being",
        "Never subtract committed_to_current_plan from",
        "never compute a number of \"free\" resources",
        "EcoGuard does not",
        "store that derived value directly",
    ):
        assert fragment in _SYSTEM_INSTRUCTION, fragment


def test_system_instruction_constrains_closest_station_claims():
    for fragment in (
        "never claim that a",
        "selected station is the closest station overall",
        "shortest ETA among",
        "but not the full",
        "comparison against all stations",
    ):
        assert fragment in _SYSTEM_INSTRUCTION, fragment


def test_system_instruction_forbids_inventing_a_plan_rationale():
    for fragment in (
        "rationale are not stored",
        "never",
        "invent why one station or one plan was chosen instead of another",
    ):
        assert fragment in _SYSTEM_INSTRUCTION, fragment


def test_system_instruction_keeps_baseline_unavailable():
    assert "baseline comparison is currently unavailable" in _SYSTEM_INSTRUCTION
    assert "never infer or estimate an" in _SYSTEM_INSTRUCTION


def test_system_instruction_covers_global_plan_states():
    assert "global_response_plan is null or its state is \"none\", \"generating\" or" in _SYSTEM_INSTRUCTION


# ---------------------------------------------------------------------------
# Area follow-ups, Fire Danger freshness, optimization-method wording
# ---------------------------------------------------------------------------


def _instruction_section(start_marker: str, end_marker: str) -> str:
    start = _SYSTEM_INSTRUCTION.index(start_marker)
    return _SYSTEM_INSTRUCTION[start : _SYSTEM_INSTRUCTION.index(end_marker, start)].replace("\n", " ")


def _area_follow_up_parts() -> tuple[str, str, str]:
    """(whole rule, the 'inherits the area' examples, the 'general question' part)."""
    rule = _instruction_section("Area follow-ups", "Global response plan:")
    inherit_start = rule.index("these follow-ups refer to")
    general_start = rule.index("is a GENERAL question")
    return rule, rule[inherit_start:general_start], rule[general_start:]


def test_system_instruction_has_area_follow_up_rule_for_weather_and_fire_danger():
    rule, _, _ = _area_follow_up_parts()

    assert "ONLY when its wording clearly refers back to that area" in rule
    assert "weather_conditions and fire_danger, not only to active fire events" in rule
    # History resolves only WHICH area; values always come from the current snapshot.
    assert "never supplies or overrides values" in rule
    assert "always take the values from the current snapshot" in rule


def test_humidity_continuation_after_judean_hills_weather_inherits_judean_hills():
    rule, inherits, general = _area_follow_up_parts()

    assert 'after "מה מזג האוויר בהרי יהודה?", these follow-ups refer to Judean Hills Demo Area' in rule
    assert '"ומה הלחות?"' in inherits and '"ומה הלחות?"' not in general


def test_danger_there_after_judean_hills_weather_inherits_judean_hills():
    _, inherits, general = _area_follow_up_parts()

    assert '"ומה הסכנה שם?"' in inherits and '"ומה הסכנה שם?"' not in general


@pytest.mark.parametrize(
    "follow_up", ['"ומה הטמפרטורה?"', '"ומה רמת הסכנה באזור הזה?"', '"ומה המצב שם כרגע?"']
)
def test_explicit_back_references_inherit_the_previous_area(follow_up):
    _, inherits, general = _area_follow_up_parts()

    assert follow_up in inherits and follow_up not in general


@pytest.mark.parametrize("general_question", ['"מה הסכנה כרגע?"', '"מה מצב הסכנה עכשיו?"', '"מה רמת הסכנה כרגע?"'])
def test_general_current_danger_questions_do_not_inherit_the_previous_area(general_question):
    rule, inherits, general = _area_follow_up_parts()

    assert general_question in general and general_question not in inherits
    assert "does NOT inherit the previous area, even right after an area-specific question" in rule


def test_general_fire_danger_questions_answer_from_all_monitored_area_entries():
    _, _, general = _area_follow_up_parts()

    assert "general current Fire Danger questions" in general
    assert "answer them from the current fire_danger entries for all relevant monitored areas" in general


def test_general_and_back_referring_follow_ups_both_reach_gemini_with_all_areas_and_history():
    history = [
        ChatMessage(role="user", content="מה מזג האוויר בהרי יהודה?"),
        ChatMessage(role="assistant", content="ב-Judean Hills Demo Area: 37.5°C."),
    ]
    areas = (
        make_fire_danger_area(area_name="Jerusalem Forest Demo Area", level=FireDangerLevel.VERY_HIGH, score=55.6),
        make_fire_danger_area(area_name="Judean Hills Demo Area"),
    )
    for question in ("מה הסכנה כרגע?", "ומה הסכנה שם?"):
        agent, _, _, gemini = make_agent(
            active_result=make_active_result(), details_by_id={}, fire_danger_areas=areas
        )

        agent.ask(question, history=history)

        contents = gemini.calls[-1]["contents"]
        assert [turn["parts"][0]["text"] for turn in contents[:-1]] == [message.content for message in history]
        # Every monitored area's current entry is available, so a general question can cover them all.
        assert [entry["area_name"] for entry in sent_snapshot(gemini)["fire_danger"]] == [
            "Jerusalem Forest Demo Area",
            "Judean Hills Demo Area",
        ]
        assert question in contents[-1]["parts"][0]["text"]


def test_ambiguous_area_follow_up_must_not_be_guessed():
    rule = _instruction_section("Area follow-ups", "Global response plan:")

    assert "does not identify exactly one area unambiguously" in rule
    assert "do not guess an area" in rule
    assert "answer for the relevant available areas, or ask the user" in rule


def test_area_follow_up_history_and_snapshot_are_both_sent_to_gemini():
    history = [
        ChatMessage(role="user", content="מה מזג האוויר בהרי יהודה?"),
        ChatMessage(role="assistant", content="ב-Judean Hills Demo Area: 37.5°C."),
    ]
    agent, _, _, gemini = make_agent(
        active_result=make_active_result(),
        details_by_id={},
        weather_conditions=(make_weather_conditions(),),
        fire_danger_areas=(make_fire_danger_area(),),
    )

    agent.ask("מה הסכנה כרגע?", history=history)

    contents = gemini.calls[-1]["contents"]
    assert [turn["parts"][0]["text"] for turn in contents[:-1]] == [message.content for message in history]
    snapshot = sent_snapshot(gemini)
    assert snapshot["fire_danger"][0]["area_name"] == "Judean Hills Demo Area"
    assert snapshot["weather_conditions"][0]["area_name"] == "Judean Hills Demo Area"


def test_fire_danger_assessed_at_remains_available_and_assessments_may_update():
    snapshot, _ = _ask_with_area_context(fire_danger_areas=(make_fire_danger_area(),))
    assert snapshot["fire_danger"][0]["assessed_at"] == DANGER_ASSESSED_AT.isoformat()

    rule = _instruction_section("Fire Danger: answer", "Location matching:")
    assert "mention its assessment time (assessed_at) when useful" in rule
    assert "updated over time" in rule
    assert "a newer answer may legitimately differ from an earlier one" in rule


def test_documented_optimization_method_is_represented_accurately():
    rule = _instruction_section("Why a plan or station was selected", "Baseline comparison:")

    for fragment in (
        "required resource slots are prioritized first",
        "then desired slots",
        "then optional predicted-risk coverage",
        "under scarcity, higher-severity needs are favored",
        "shorter ETA / route distance and target priority affect the score",
        "keeping existing valid assignments receives a stability preference",
        "it is NOT a stored per-selection rationale",
    ):
        assert fragment in rule, fragment


def test_instruction_never_asserts_guaranteed_coverage_or_eta_minimization_as_the_objective():
    lowered = _SYSTEM_INSTRUCTION.lower().replace("\n", " ")
    prohibitions = (
        "never say the optimizer guarantees full or 100% coverage",
        "never describe minimizing average eta as the sole optimization objective",
    )
    for prohibition in prohibitions:
        assert prohibition in lowered, prohibition

    # Outside those two prohibitions, no affirmative claim of either idea exists.
    without_prohibitions = lowered
    for prohibition in prohibitions:
        without_prohibitions = without_prohibitions.replace(prohibition, "")
    for affirmative in (
        "guarantees",
        "guarantee 100% coverage",
        "100% coverage",
        "the optimizer minimizes",
        "aims to minimize",
        "minimizing average eta",
    ):
        assert affirmative not in without_prohibitions, affirmative


def test_no_per_station_causal_rationale_may_be_invented():
    rule = _instruction_section("Why a plan or station was selected", "Baseline comparison:")

    assert "Never claim that a particular station was chosen specifically because of one of these factors" in rule
    assert "result metrics of the selected plan, not proof of why a particular station won over another" in rule
    for part in (
        "which station/resource was selected",
        "the stored metrics of that result",
        "the general optimization method above",
        "the exact alternative-by-alternative rationale is not persisted",
    ):
        assert part in rule, part


def test_system_instruction_says_scores_are_already_rounded():
    assert "already rounded for" in _SYSTEM_INSTRUCTION
    assert "never add decimal places" in _SYSTEM_INSTRUCTION

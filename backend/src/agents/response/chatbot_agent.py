"""EcoGuard's AI decision-support chatbot (Task 4).

`ChatbotAgent` answers a user's natural-language question using ONLY
information EcoGuard's existing agents/calculators have already computed and
persisted. It performs no wildfire calculation of its own: for every
question it reads the current active-FireEvents snapshot via
`ActiveFireEventsService` and each active event's full detail via
`EventDetailsService`, plus every area's latest Weather Conditions via
`WeatherConditionsQueryService` and latest Fire Danger via
`FireDangerQueryService` (all already strictly read-only, no-recalculation
services - see their own module docstrings), assembles a deterministic
structured EcoGuard context from that data, and forwards it - together with
a fixed EcoGuard system instruction, optional trimmed conversation history,
and the user's question - to `GeminiClient` (Task 3). It never invokes a
domain agent/calculator/orchestrator and never queries a repository
directly.

Location resolution is intentionally NOT performed here: every active
event's `location_name` (and coordinates) is included in the context, and
Gemini itself matches the user's natural-language location reference
against that supplied data (see the system instruction below) - no new
backend fuzzy-matching or geocoding is introduced.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from src.api.schemas.event_details import EventDetailsResult, FireEventMLAssessmentResponse, SeverityAssessmentResponse
from src.api.schemas.global_response_plan import GlobalEventPlan, GlobalResponsePlanResponse
from src.external.gemini.gemini_client import GeminiClient
from src.models.active_fire_events import (
    ActiveFireEventMLSummary,
    ActiveFireEventSeveritySummary,
    ActiveFireEventsResult,
    ActiveFireEventSummary,
)
from src.models.fire_danger_areas import FireDangerAreaSnapshot
from src.models.fire_spread_insufficient_data_reason import FireSpreadInsufficientDataReason
from src.models.fire_spread_prediction import PROPAGATION_THRESHOLD
from src.services.fire_danger.fire_danger_query_service import FireDangerQueryService
from src.services.fire_event_read.active_fire_events_service import ActiveFireEventsService
from src.services.fire_event_read.event_details_service import EventDetailsService
from src.services.global_planning.global_response_plan_read_service import GlobalResponsePlanReadService
from src.services.weather.weather_conditions_query_service import (
    AreaWeatherConditions,
    WeatherConditionsQueryService,
)

_EVENT_REF_ALPHABET_SIZE = 26

MAX_CHAT_HISTORY_MESSAGES = 8

_VALID_CHAT_ROLES = ("user", "assistant")

# Gemini's own `contents` turn roles are "user"/"model" - never "assistant".
# This mapping is a Gemini API protocol detail, not EcoGuard domain logic,
# but it stays here (not in GeminiClient) because GeminiClient never
# interprets `contents`, only forwards it - see gemini_client.py's docstring.
_GEMINI_ROLE_BY_CHAT_ROLE = {"user": "user", "assistant": "model"}

# US 4.2 methodology: a stored spread cell with spread_probability >= this
# threshold is propagation-capable ("spreading"); below it (but > 0) it is a
# risk-only cell. The same single constant the calculator uses - imported from
# src.models (this module may not import src.calculators, architecture guard).
_SPREAD_PROPAGATION_THRESHOLD = PROPAGATION_THRESHOLD

# User-facing meaning of each stored FireSpreadInsufficientDataReason. Worded
# to claim no more than the reason itself proves.
_INSUFFICIENT_DATA_REASON_DESCRIPTIONS: dict[FireSpreadInsufficientDataReason, str] = {
    FireSpreadInsufficientDataReason.EVENT_UNAVAILABLE: (
        "The fire event was not available in a usable state for spread prediction."
    ),
    FireSpreadInsufficientDataReason.MISSING_SEVERITY: "No fire severity assessment was available for this event.",
    FireSpreadInsufficientDataReason.SEVERITY_NOT_VALID: (
        "The latest fire severity assessment was not valid, so spread could not be predicted."
    ),
    FireSpreadInsufficientDataReason.MISSING_WEATHER: "Required weather data was unavailable.",
    FireSpreadInsufficientDataReason.INCOMPLETE_WEATHER: (
        "Available weather observations were missing values the spread model requires "
        "(temperature, humidity, wind speed or wind direction)."
    ),
    FireSpreadInsufficientDataReason.STALE_WEATHER: (
        "Available weather observations were too old for spread prediction."
    ),
    FireSpreadInsufficientDataReason.FUTURE_WEATHER: (
        "Available weather observations were timestamped after the prediction time, so they could not be used."
    ),
    FireSpreadInsufficientDataReason.MISSING_VEGETATION: (
        "Vegetation data required by the spread model was unavailable for this location."
    ),
    FireSpreadInsufficientDataReason.UNSUPPORTED_VEGETATION: (
        "The vegetation category at this location is not supported by the current spread model."
    ),
}

_SNAPSHOT_LABEL = "Current EcoGuard data snapshot (authoritative JSON):"
_QUESTION_LABEL = "User's current question:"
_FOCUS_HEADER = "Resolved conversation focus (from conversation history - not a factual source):"

# Deterministic, LLM-free follow-up-referent resolution (no vague extra
# system-prompt sentence - see _resolve_conversation_focus below). A
# generic reference ("the event"/"it"/"האירוע"/...) in the CURRENT question
# only triggers resolution; an explicit multi-event/comparison question
# always suppresses it, regardless of a generic word also appearing in it.
_GENERIC_EVENT_REFERENCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bthe event\b",
        r"\bthat event\b",
        r"\bits\b",
        r"\bit\b",
        r"\bהאירוע\b",
        r"\bאותו אירוע\b",
        r"\bשלו\b",
        r"\bשלה\b",
    )
)
_MULTI_EVENT_QUESTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\ball (?:the )?(?:active )?(?:fires|events)\b",
        r"\bboth events\b",
        r"\beach event\b",
        r"\bcompar\w*",
        r"\bכל השריפות\b",
        r"\bכל האירועים\b",
        r"\bשני האירועים\b",
        r"\bהשווא?ה\b",
        r"\bלהשוות\b",
    )
)
_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)

_SYSTEM_INSTRUCTION = """\
You are EcoGuard's AI decision-support assistant.

Answer ONLY using the EcoGuard information supplied in this request's current \
EcoGuard data snapshot. Do not use outside knowledge to invent facts about \
current fires, and do not fabricate any missing data. If the user asks about \
something that is not present in the supplied EcoGuard data, explicitly say \
that it is unavailable rather than guessing.

When a field is null, missing, an empty list, or has a status such as \
"insufficient_data": report that status, and give an explicit reason for it \
ONLY if the supplied EcoGuard data itself states that reason. Never infer or \
guess a cause from related fields (e.g. a zero count, an empty array, another \
status field, or missing evidence) - if the data does not state why, say the \
current data does not specify a reason.

Fire-spread predictions (spread_predictions) are model outputs, given per horizon (30 and 60 minutes); describe each horizon from its own values. Each counted unit is a nearby area around the fire. In a horizon, spreading_cell_count is the number of nearby areas whose predicted spread probability reached the model's propagation threshold, and risk_only_cell_count is the number of nearby areas with predicted spread risk below that threshold, which the model does not treat as spreading further. When talking to the user, call these "areas", "nearby areas" or "areas with predicted spread risk" - never say "cell" or "cells", and never mention internal field names such as spreading_cell_count, risk_only_cell_count or total_cell_count. Never describe risk-only areas as places the fire is predicted to spread to or will reach: for example, with risk_only_cell_count 8 and spreading_cell_count 0, say "8 nearby areas show predicted spread risk, but none reached the model's propagation threshold." Only when spreading_cell_count is greater than 0 may you say that some nearby areas reached the propagation threshold (predicted spread according to the model). Never state certainty such as "the fire will reach" or "the fire will definitely spread". A valid prediction with total_cell_count 0 means the model predicts no spread. If both horizons show the same counts, say the model predicts the same risk footprint for 30 and 60 minutes, concisely - for example: "At 30 minutes, 8 nearby areas show predicted spread risk, but none reached the model's propagation threshold. The 60-minute forecast shows the same risk footprint." Describe a difference between horizons only when their values actually differ.

For a spread prediction whose status is "insufficient_data", when \
insufficient_data_reason_description is provided, use it to explain why the \
prediction is unavailable, without adding causes it does not state - a \
missing-vegetation reason does not mean an external service failed, and an \
unsupported vegetation category means the vegetation type is not supported \
by the current spread model, not that vegetation data is missing. When \
insufficient_data_reason is null, say the current data does not specify why \
the spread prediction is unavailable.

Do not perform any new wildfire calculations, and do not replace or override \
EcoGuard's existing algorithms or already-stored decisions - you only explain \
and summarize what EcoGuard has already determined.

When answering, clearly distinguish between:
- observed/detected facts (e.g. detection evidence, event status, coordinates)
- ML model outputs (e.g. model_score)
- severity assessments
- spread predictions
- response targets
- response plans
- area weather conditions
- area Fire Danger assessments

To explain why a fire was detected, use the event's detection_confidence, \
its detection_evidence, and its ml_assessment fields: detection_mode (how \
the detection decision was made), rule_status and rule_confidence (the \
deterministic detection rule's own result), agreement (whether the model \
agreed with the rule), and model_score.

Severity and Fire Danger scores in the snapshot are already rounded for \
presentation - quote them exactly as given and never add decimal places.

Critical rule about model_score: `model_score` is a model output / fire-risk \
score on a 0-1 scale and is NOT a calibrated real-world probability of \
wildfire occurrence. Always describe it as a model score or fire-risk score \
(for example: "the ML model returned a high fire-risk score of 0.87"), and \
NEVER as a percentage probability (for example, never say "there is an 87% \
probability of a wildfire").

Always answer in the language of the user's CURRENT question only - never \
the dominant language of the conversation history. If the current question \
is in English, answer in English; if it is in Hebrew, answer in Hebrew. A \
follow-up question may switch language from the previous turn, and your \
response must switch with it. Keep technical identifiers/field names (e.g. \
model_score) as-is, but write the explanatory prose in the current \
question's language.

The current EcoGuard data snapshot supplied with this request is always \
authoritative. Conversation history, when supplied, is provided only for \
conversational context and reference resolution (e.g. understanding "it" or \
"that fire" from an earlier message). If the conversation history conflicts \
with the current EcoGuard data snapshot, always prefer the current snapshot.

When a follow-up uses a referring expression such as "the event", "that \
event", "it", "its" (or Hebrew equivalents such as "האירוע", "אותו אירוע", \
"היא", "שלו"/"שלה"), use the conversation history ONLY to determine WHICH \
previously-discussed event it refers to - never to supply its facts. If that \
referent is unambiguous, answer about that one event only, using its current \
values from the snapshot, and do not expand the answer to other active \
events the user did not ask about. If the referent cannot be resolved \
unambiguously from the history, say so explicitly rather than silently \
guessing which event is meant.

If multiple active events share the same or a similar location name, \
describe all of the matching events clearly and separately - never silently \
pick one of them on the user's behalf.

The current EcoGuard data snapshot's active_fire_events list is the \
COMPLETE list of currently active wildfire events - no other active event \
exists beyond what it contains. Distinguish these two situations precisely \
and never confuse them:
- If the location, area, or event the user asks about does NOT match any \
event in that list, state clearly that there is currently no active \
wildfire event there (in the language of the current question). Never \
phrase this as the data being "unavailable" or "missing" - that would \
wrongly imply such an event exists but its details are absent. You may \
mention which locations ARE currently active, if useful.
- If the user asks about an event that DOES match one in the list, but a \
specific field for that event is null, missing, or has a status such as \
"insufficient_data", say that specific field's data is unavailable for \
that event - never say no event exists there.
Never invent an event, a location, or any detail that is not present in \
the supplied snapshot.
The two situations above apply only to questions about wildfire events. \
They never apply to weather or Fire Danger questions: never answer a weather \
or Fire Danger question by saying there is no active wildfire in that area.

Weather: answer weather questions from the snapshot's weather_conditions \
list. Weather questions include questions about temperature, relative \
humidity, wind speed, wind gust, and general weather conditions - a question \
about any one of these is a weather question and must use \
weather_conditions. It reflects EcoGuard's current stored operational/simulation weather \
state, one entry per area. Each entry's values are aggregates (means) of the \
weather stations used for that area (station_count), not a single \
measurement. Always name the EcoGuard area whose data you used, and mention \
the observation time (observed_at) when relevant. If an area is not present \
in weather_conditions, say that weather data is currently unavailable for \
that area. Wind direction and rainfall are unavailable at the area-summary \
level unless explicitly present in the snapshot - say so if asked, and never \
estimate them.

Fire Danger: answer Fire Danger questions from the snapshot's fire_danger \
list. Fire Danger is the environmental wildfire risk assessed for an AREA \
from its weather; it is NOT the Severity of an already detected wildfire \
(the severity field of an active event), and the two must never be \
confused or substituted for each other. A Fire Danger question does not \
require an active fire in that area. When an entry's status is \
"insufficient_data", say its Fire Danger level is currently unavailable. \
When reporting a Fire Danger level or score - especially for a "current" or \
"now" question - mention its assessment time (assessed_at) when useful. Fire \
Danger assessments are updated over time as new weather arrives, so a newer \
answer may legitimately differ from an earlier one in the conversation; \
always report the snapshot's current values rather than repeating an \
earlier answer.

Location matching: area and event names are EcoGuard area names such as \
"Jerusalem Forest Demo Area" or "Judean Hills Demo Area". Match the user's \
natural wording (e.g. "ירושלים" or "Jerusalem", "הרי יהודה" or "Judean \
Hills") to the closest matching EcoGuard area name, and always state the \
EcoGuard area name whose data you used so the mapping is transparent. If no \
area plausibly matches, say that EcoGuard has no data for that location.

Area follow-ups (Weather and Fire Danger): a question inherits the \
geographic area discussed in the immediately preceding conversation ONLY \
when its wording clearly refers back to that area - a continuation such as \
"ומה ...?" asking about another attribute of the same place, or an explicit \
back-reference such as "שם", "באזור הזה", "there" or "in that area". For \
example, after "מה מזג האוויר בהרי יהודה?", these follow-ups refer to \
Judean Hills Demo Area: "ומה הלחות?", "ומה הטמפרטורה?", "ומה הסכנה שם?", \
"ומה רמת הסכנה באזור הזה?", "ומה המצב שם כרגע?". A question that neither \
names a location nor refers back to one is a GENERAL question about the \
current system and does NOT inherit the previous area, even right after an \
area-specific question - for example "מה הסכנה כרגע?", "מה מצב הסכנה \
עכשיו?" and "מה רמת הסכנה כרגע?" are general current Fire Danger \
questions: answer them from the current fire_danger entries for all \
relevant monitored areas. Conversation history may resolve WHICH area is \
meant only when the user's wording clearly refers back to it, and it never \
supplies or overrides values - always take the values from the current \
snapshot. This applies to weather_conditions and fire_danger, not only to \
active fire events. If the wording refers back to an area but the previous \
context does not identify exactly one area unambiguously, do not guess an \
area: answer for the relevant available areas, or ask the user which area \
they mean.

Global response plan: answer questions about overall resource sufficiency, \
shortages and which fires are covered from global_response_plan (state, \
status, coverage_score, average_eta_seconds, total_required_resources, \
total_desired_resources, total_assigned_resources, unmet_required_resources, \
unmet_desired_resources, and the covered_fires / pending_fires / \
unplannable_fires / monitoring_only_fires lists). Each active event's \
resource_requirements gives that fire's own minimum required, desired and \
assigned resources, coverage, average ETA and uncovered target count. When \
global_response_plan is null or its state is "none", "generating" or \
"updating", say that a current global plan is not yet available or is being \
updated.

Stations and resources: station_availability summarizes fire-station \
resources by their STORED OPERATIONAL STATUS (available / assigned / \
unavailable), plus committed_to_current_plan - the number of that station's \
resources the current response plan commits. Status and plan commitment are \
separate facts: a resource can have status available while already being \
committed to a response plan. Never subtract committed_to_current_plan from \
available, never compute a number of "free" resources, and never claim that \
"available" means free for a new fire. listed_stations contains only \
stations with no available resources, with assigned or unavailable \
resources, or with plan commitments; every other station has all of its \
resources available and none committed. You may answer which stations have \
zero available resources from the stored available counts. If asked how many \
resources are truly free for a new assignment, explain that EcoGuard does not \
store that derived value directly, and report the available and committed \
counts separately instead.

Selected versus closest station: the snapshot contains the station and \
resource actually selected for each allocation, with its stored ETA and route \
distance - you may report those. EcoGuard stores routes only for the selected \
resources, not a comparison against all stations, so never claim that a \
selected station is the closest station overall or has the shortest ETA among \
all stations. If asked which station is closest, explain that the stored data \
contains the selected station and its ETA/distance, but not the full \
comparison against all stations.

Why a plan or station was selected: the optimizer's alternative ranking and \
rationale are not stored. You may describe the selected result and the \
stored metrics of the selected plan (coverage, ETA, shortages), but never \
invent why one station or one plan was chosen instead of another - say that \
the detailed alternative-selection rationale is not stored. The documented \
optimization METHOD (general, not specific to any selection) is: required \
resource slots are prioritized first, then desired slots, then optional \
predicted-risk coverage; under scarcity, higher-severity needs are favored; \
within the applicable tier, shorter ETA / route distance and target \
priority affect the score; and keeping existing valid assignments receives \
a stability preference. This describes how plans are scored in general - it \
is NOT a stored per-selection rationale. Never claim that a particular \
station was chosen specifically because of one of these factors, since no \
such causal reason is stored. Never say the optimizer guarantees full or \
100% coverage, and never describe minimizing average ETA as the sole \
optimization objective. Stored coverage, ETA and shortage values are result \
metrics of the selected plan, not proof of why a particular station won \
over another. When asked why a station or plan was selected, answer in four \
distinct parts: which station/resource was selected; the stored metrics of \
that result; the general optimization method above; and that the exact \
alternative-by-alternative rationale is not persisted.

Baseline comparison: no baseline comparison is present in the snapshot. If \
asked how much better the optimized plan is than a baseline, say that the \
baseline comparison is currently unavailable, and never infer or estimate an \
improvement percentage.

Each active event in the snapshot carries an internal `event_ref` label \
(e.g. "A", "B") used only to distinguish events inside this system. Never \
expose an `event_ref` value in a user-facing answer - refer to events by \
their location_name and other distinguishing details instead.
"""


@dataclass(frozen=True)
class ChatMessage:
    """One prior turn of conversation history, supplied by the caller.

    `role` is EcoGuard/chatbot-facing vocabulary ("user"/"assistant"), not
    Gemini's own "user"/"model" turn roles - `ChatbotAgent` translates
    between the two when building the Gemini request.
    """

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in _VALID_CHAT_ROLES:
            raise ValueError(f"role must be one of {_VALID_CHAT_ROLES}, got {self.role!r}")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError(f"content must be a non-empty string, got {self.content!r}")


class ChatbotAgent:
    """Answers a user's question from EcoGuard's already-persisted state via Gemini.

    Depends only on `ActiveFireEventsService`, `EventDetailsService`, and
    `GeminiClient` - no repository is injected or accessed directly, and no
    new aggregation service/orchestrator is introduced. Raises whatever
    typed `GeminiClientError` subclass (see `src.external.gemini.exceptions`)
    `GeminiClient.generate_content` raises; mapping that to an HTTP response
    is left to the future API router task.
    """

    def __init__(
        self,
        *,
        active_fire_events_service: ActiveFireEventsService | None = None,
        event_details_service: EventDetailsService | None = None,
        weather_conditions_query_service: WeatherConditionsQueryService | None = None,
        fire_danger_query_service: FireDangerQueryService | None = None,
        global_response_plan_read_service: GlobalResponsePlanReadService | None = None,
        gemini_client: GeminiClient | None = None,
    ) -> None:
        self._active_fire_events_service = active_fire_events_service or ActiveFireEventsService()
        self._event_details_service = event_details_service or EventDetailsService()
        self._weather_conditions_query_service = weather_conditions_query_service or WeatherConditionsQueryService()
        self._fire_danger_query_service = fire_danger_query_service or FireDangerQueryService()
        # Production wiring (src/api/dependencies.py) passes the API's own
        # instance, which also reports plan coverage; this default does not.
        self._global_response_plan_read_service = (
            global_response_plan_read_service or GlobalResponsePlanReadService()
        )
        self._gemini_client = gemini_client or GeminiClient()

    def ask(self, question: str, *, history: Sequence[ChatMessage] = ()) -> str:
        """Answer `question` using EcoGuard's current persisted state.

        `history`, if given, is trimmed to its latest `MAX_CHAT_HISTORY_MESSAGES`
        entries (oldest-first order preserved) before being sent - conversational
        context only, never a substitute for the current EcoGuard snapshot built
        fresh on every call (see the system instruction).
        """
        self._validate_question(question)
        trimmed_history = tuple(history)[-MAX_CHAT_HISTORY_MESSAGES:]

        active_result = self._active_fire_events_service.get_active_events()
        event_details_by_id = {
            summary.fire_event_id: self._event_details_service.get_event_details(summary.fire_event_id)
            for summary in active_result.items
        }
        weather_conditions = self._weather_conditions_query_service.get_latest_for_all_areas()
        fire_danger_areas = self._fire_danger_query_service.get_latest_for_all_areas(as_of=active_result.as_of).areas
        global_plan = self._global_response_plan_read_service.get_current(as_of=active_result.as_of)
        snapshot_text = _build_snapshot_text(
            active_result, event_details_by_id, weather_conditions, fire_danger_areas, global_plan
        )
        focus_location_name = _resolve_conversation_focus(question, trimmed_history, active_result)
        contents = _build_contents(trimmed_history, snapshot_text, question, focus_location_name)

        return self._gemini_client.generate_content(contents, system_instruction=_SYSTEM_INSTRUCTION)

    @staticmethod
    def _validate_question(question: object) -> None:
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question must be a non-empty string, got {question!r}")


# ---------------------------------------------------------------------------
# Structured EcoGuard context assembly (pure mapping - no calculation)
#
# Task 7D-B: this builds an explicit COMPACT projection, never the full
# `EventDetailsResult.model_dump()` - see backend/docs (Task 7D investigation)
# for why: the full model carries the entire national stations/resources
# catalog (repeated per event, byte-identical every time), a 138-entry
# station_summaries array (>98% irrelevant per event), raw routing
# node_path arrays, and internal DB ids - none of which any supported
# chatbot question needs. Only actually-allocated stations/resources (read
# from the event's own `current_response_plan.actions`, joined against
# `stations` purely as an in-memory name lookup) are ever emitted; the full
# catalogs themselves are never serialized. This changes ONLY what
# ChatbotAgent sends to Gemini - `ActiveFireEventsService`/
# `EventDetailsService` are still called exactly as before (see `ask()`
# above), so this is a size optimization only, not a latency one.
# ---------------------------------------------------------------------------


def _event_ref_for_index(index: int) -> str:
    """A short, opaque per-request label ("A", "B", ...) - never the real
    database `fire_event_id`. Falls back to a numbered label past the
    26-letter alphabet (not expected in practice for "active" events)."""
    if index < _EVENT_REF_ALPHABET_SIZE:
        return chr(ord("A") + index)
    return f"E{index + 1}"


def _build_snapshot_text(
    active_result: ActiveFireEventsResult,
    event_details_by_id: dict[int, EventDetailsResult | None],
    weather_conditions: Sequence[AreaWeatherConditions] = (),
    fire_danger_areas: Sequence[FireDangerAreaSnapshot] = (),
    global_plan: GlobalResponsePlanResponse | None = None,
) -> str:
    event_plans_by_id: dict[int, GlobalEventPlan] = (
        {event_plan.fire_event_id: event_plan for event_plan in global_plan.plan.events}
        if global_plan is not None and global_plan.plan is not None
        else {}
    )
    snapshot = {
        "as_of": active_result.as_of.isoformat(),
        "active_fire_event_count": len(active_result.items),
        "active_fire_events": [
            {
                **_build_event_context(
                    _event_ref_for_index(index), summary, event_details_by_id.get(summary.fire_event_id)
                ),
                "resource_requirements": _compact_event_resource_requirements(
                    event_plans_by_id.get(summary.fire_event_id)
                ),
            }
            for index, summary in enumerate(active_result.items)
        ],
        "global_response_plan": _compact_global_response_plan(global_plan, active_result),
        "station_availability": _compact_station_availability(event_details_by_id, event_plans_by_id),
        # Area-level sections: every assessed area, independent of whether
        # it currently has an active fire.
        "weather_conditions": [_compact_weather_conditions(conditions) for conditions in weather_conditions],
        "fire_danger": [
            _compact_fire_danger(area) for area in fire_danger_areas if area.assessment is not None
        ],
    }
    # Compact (no indent/extra whitespace) - Gemini needs no pretty-printing,
    # and indentation alone was ~47% of the previous payload's size.
    return json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))


def _build_event_context(
    event_ref: str,
    summary: ActiveFireEventSummary,
    details: EventDetailsResult | None,
) -> dict[str, Any]:
    """One active event's compact projection: the dashboard summary (source
    of `location_name`) combined with only the operationally-useful subset
    of its full `EventDetailsResult`, if still available.

    `details` is `None` only in the rare case an event stopped being active
    between the two read calls in `ChatbotAgent.ask` - represented
    explicitly via `event_details_available: false` (and severity/ML
    falling back to the still-available dashboard summary) rather than
    silently dropped or fabricated.
    """
    severity_source = (
        details.severity if details is not None and details.severity is not None else summary.severity
    )
    ml_source = (
        details.ml_assessment if details is not None and details.ml_assessment is not None else summary.ml_summary
    )
    return {
        "event_ref": event_ref,
        # Never fabricated: None here means EcoGuard has no trustworthy name
        # for this event (see ActiveFireEventsService._resolve_location_name)
        # - only coordinates/region info from the rest of this context apply.
        "location_name": summary.location_name,
        "latitude": summary.latitude,
        "longitude": summary.longitude,
        "status": summary.status.value,
        "detected_at": summary.detected_at.isoformat(),
        "updated_at": summary.updated_at.isoformat(),
        "detection_confidence": summary.detection_confidence,
        "event_details_available": details is not None,
        "severity": _compact_severity(severity_source),
        "ml_assessment": _compact_ml_assessment(ml_source),
        "detection_evidence": _compact_detection_evidence(details) if details is not None else None,
        "spread_predictions": _compact_spread_predictions(details) if details is not None else [],
        "targets": _compact_targets(details) if details is not None else [],
        "current_response_plan": _compact_response_plan(details) if details is not None else None,
    }


def _compact_severity(
    severity: ActiveFireEventSeveritySummary | SeverityAssessmentResponse | None,
) -> dict[str, Any] | None:
    if severity is None:
        return None
    return {
        "status": severity.status.value,
        "level": severity.level.value if severity.level is not None else None,
        "score": _present_score(severity.score),
    }


# User-facing 0-100 operational scores (Fire Severity, Fire Danger) are sent
# to the LLM at this precision, so answers never quote raw floats like
# 82.5065916694979. Presentation only: the persisted/service values are never
# modified. 0-1 values (model_score, confidences) are deliberately left as is.
_PRESENTED_SCORE_DECIMALS = 1


def _present_score(score: float | None) -> float | None:
    return round(score, _PRESENTED_SCORE_DECIMALS) if score is not None else None


def _compact_ml_assessment(
    ml_source: ActiveFireEventMLSummary | FireEventMLAssessmentResponse | None,
) -> dict[str, Any] | None:
    if ml_source is None:
        return None
    return {
        # `model_score` is a 0-1 model output, not a probability - the
        # system instruction (not this mapping) is what tells Gemini how to
        # phrase it; this function only copies the persisted value as-is.
        "available": ml_source.available,
        "model_score": ml_source.model_score,
        # Only present when the full EventDetailsResult (not just the
        # lightweight dashboard summary) was available - never invented.
        "model_name": getattr(ml_source, "model_name", None),
        "model_version": getattr(ml_source, "model_version", None),
        # How the detection decision was made: the decision mode, the
        # deterministic rule's own status/confidence, and whether the model
        # agreed with the rule - copied as persisted.
        "detection_mode": _enum_value(getattr(ml_source, "mode", None)),
        "rule_status": _enum_value(getattr(ml_source, "rule_status", None)),
        "rule_confidence": getattr(ml_source, "rule_confidence", None),
        "agreement": _enum_value(getattr(ml_source, "agreement", None)),
    }


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _compact_weather_conditions(conditions: AreaWeatherConditions) -> dict[str, Any]:
    # No observation/assessment/station ids and no SIM-* station names.
    return {
        "area_name": conditions.area_name,
        "observed_at": conditions.observed_at.isoformat(),
        "assessed_at": conditions.assessed_at.isoformat(),
        "station_count": conditions.station_count,
        "temperature_c": conditions.temperature_c,
        "relative_humidity_pct": conditions.relative_humidity_pct,
        "wind_speed_kmh": conditions.wind_speed_kmh,
        "wind_gust_kmh": conditions.wind_gust_kmh,
    }


_UNKNOWN_FIRE_LABEL = "an event that is not currently active"


def _fire_labels_by_id(active_result: ActiveFireEventsResult) -> dict[int, str]:
    """fire_event_id -> the label the LLM already knows the event by (its
    location_name, else its internal event_ref) - so global-plan groups never
    carry a fire_event_id."""
    return {
        summary.fire_event_id: summary.location_name or _event_ref_for_index(index)
        for index, summary in enumerate(active_result.items)
    }


def _compact_global_response_plan(
    global_plan: GlobalResponsePlanResponse | None, active_result: ActiveFireEventsResult
) -> dict[str, Any] | None:
    """The current global response plan's state, shortage and coverage groups -
    copied as persisted (no recalculation), with no run/plan/fire ids."""
    if global_plan is None:
        return None
    plan = global_plan.plan
    coverage = global_plan.coverage
    labels = _fire_labels_by_id(active_result)

    def fire_names(fire_event_ids: Sequence[int]) -> list[str]:
        return [labels.get(fire_event_id, _UNKNOWN_FIRE_LABEL) for fire_event_id in fire_event_ids]

    return {
        "state": coverage.state if coverage is not None else ("none" if plan is None else None),
        "status": plan.status.value if plan is not None else None,
        "coverage_score": plan.metrics.coverage_score if plan is not None else None,
        "average_eta_seconds": plan.metrics.average_eta_seconds if plan is not None else None,
        "total_required_resources": plan.shortage.total_required if plan is not None else None,
        "total_desired_resources": plan.shortage.total_desired if plan is not None else None,
        "total_assigned_resources": plan.shortage.total_assigned if plan is not None else None,
        "unmet_required_resources": plan.shortage.unmet_required if plan is not None else None,
        "unmet_desired_resources": plan.shortage.unmet_desired if plan is not None else None,
        "covered_fires": fire_names(coverage.covered_fire_event_ids) if coverage is not None else None,
        "pending_fires": fire_names(coverage.pending_fire_event_ids) if coverage is not None else None,
        "unplannable_fires": fire_names(coverage.unplannable_fire_event_ids) if coverage is not None else None,
        "monitoring_only_fires": fire_names(coverage.monitoring_fire_event_ids) if coverage is not None else None,
    }


def _compact_event_resource_requirements(event_plan: GlobalEventPlan | None) -> dict[str, Any] | None:
    """This fire's persisted global-plan membership values; None when the
    current global plan does not include it."""
    if event_plan is None:
        return None
    return {
        "minimum_required_resources": event_plan.minimum_resources,
        "desired_resources": event_plan.desired_resources,
        "assigned_resources": event_plan.assigned_resources,
        "coverage_score": event_plan.coverage_score,
        "average_eta_seconds": event_plan.average_eta_seconds,
        "uncovered_target_count": len(event_plan.uncovered_targets),
    }


def _compact_station_availability(
    event_details_by_id: dict[int, EventDetailsResult | None],
    event_plans_by_id: dict[int, GlobalEventPlan],
) -> dict[str, Any] | None:
    """Station counts as already computed in EventDetailsResult.station_summaries
    (stored operational status), plus how many of each station's resources the
    current global plan's actions commit. Only stations with no available
    resources, any assigned/unavailable resources, or any commitment are
    listed - never the full national catalog. `available` is never reduced by
    `committed_to_current_plan`: they are separate facts. None when no active
    event's details (the only source of station summaries) are available.
    """
    details = next(
        (item for item in event_details_by_id.values() if item is not None and item.station_summaries), None
    )
    if details is None:
        return None
    station_names = {station.station_id: station.name for station in details.stations}
    committed_by_station_id: dict[str, int] = {}
    for event_plan in event_plans_by_id.values():
        for action in event_plan.actions:
            station_id = action.resource.station_id
            committed_by_station_id[station_id] = committed_by_station_id.get(station_id, 0) + 1

    summaries = details.station_summaries
    relevant = [
        {
            "station_name": station_names.get(summary.station_id, "unnamed station"),
            "total_resources": summary.total_resources,
            "available": summary.available,
            "assigned": summary.assigned_status,
            "unavailable": summary.unavailable,
            "committed_to_current_plan": committed_by_station_id.get(summary.station_id, 0),
        }
        for summary in summaries
        if summary.available == 0
        or summary.assigned_status > 0
        or summary.unavailable > 0
        or committed_by_station_id.get(summary.station_id, 0) > 0
    ]
    return {
        "total_station_count": len(summaries),
        "total_available_resources": sum(summary.available for summary in summaries),
        "total_assigned_resources": sum(summary.assigned_status for summary in summaries),
        "total_unavailable_resources": sum(summary.unavailable for summary in summaries),
        "stations_with_no_available_resources_count": sum(1 for summary in summaries if summary.available == 0),
        "listed_stations": sorted(relevant, key=lambda station: station["station_name"]),
    }


def _compact_fire_danger(area: FireDangerAreaSnapshot) -> dict[str, Any]:
    # No assessment/area ids - area_name is what a conversational answer needs.
    assessment = area.assessment
    return {
        "area_name": area.area_name,
        "status": assessment.status.value,
        "level": assessment.level.value if assessment.level is not None else None,
        "score": _present_score(assessment.score),
        "assessed_at": assessment.assessed_at.isoformat(),
        "methodology": assessment.methodology,
    }


def _compact_detection_evidence(details: EventDetailsResult) -> dict[str, Any]:
    evidence = details.detection_evidence
    return {
        "satellite": [
            {
                "satellite": item.satellite,
                "detected_at": item.detected_at.isoformat(),
                "confidence": item.confidence,
                "frp": item.frp,
                "brightness": item.brightness,
            }
            for item in evidence.satellite
        ],
        "news": [
            {
                "source": item.source,
                "observed_at": item.observed_at.isoformat(),
                "title": item.title,
                "summary": item.summary,
            }
            for item in evidence.news
        ],
    }


def _compact_spread_predictions(details: EventDetailsResult) -> list[dict[str, Any]]:
    # Never the raw `cells` array (can be large and is not needed for
    # conversational Q&A) - only per-horizon counts and maxima, with cells
    # classified (not recalculated) as spreading vs risk-only.
    return [_compact_spread_prediction(prediction) for prediction in details.spread_predictions]


def _compact_spread_prediction(prediction) -> dict[str, Any]:  # noqa: ANN001 - SpreadPredictionResponse
    probabilities = [cell.spread_probability for cell in prediction.cells]
    risk_scores = [cell.spread_risk_score for cell in prediction.cells]
    reason = getattr(prediction, "insufficient_data_reason", None)
    return {
        "horizon_minutes": prediction.horizon_minutes,
        "status": prediction.status.value,
        "predicted_at": prediction.predicted_at.isoformat(),
        "total_cell_count": len(probabilities),
        "spreading_cell_count": sum(1 for p in probabilities if p >= _SPREAD_PROPAGATION_THRESHOLD),
        "risk_only_cell_count": sum(1 for p in probabilities if 0 < p < _SPREAD_PROPAGATION_THRESHOLD),
        "max_spread_probability": max(probabilities) if probabilities else None,
        "max_spread_risk_score": max(risk_scores) if risk_scores else None,
        # Null for valid/inactive_event predictions, and for historical
        # insufficient_data rows stored before the reason existed.
        "insufficient_data_reason": reason.value if reason is not None else None,
        "insufficient_data_reason_description": (
            _INSUFFICIENT_DATA_REASON_DESCRIPTIONS.get(reason) if reason is not None else None
        ),
    }


def _compact_targets(details: EventDetailsResult) -> list[dict[str, Any]]:
    # Never the raw internal response_target_id - target_type + location +
    # priority is what a conversational answer needs.
    return [
        {
            "target_type": target.target_type.value,
            "priority_score": target.priority_score,
            "latitude": target.latitude,
            "longitude": target.longitude,
            "prediction_horizon_minutes": target.prediction_horizon_minutes,
        }
        for target in details.targets
    ]


def _compact_response_plan(details: EventDetailsResult) -> dict[str, Any] | None:
    """The current response plan, compacted to what a conversational answer
    needs. `details.stations` is used ONLY as an in-memory station_id ->
    name lookup for the allocations below - the full stations catalog (and
    the full resources catalog, and the 138-entry station_summaries array)
    are never themselves serialized into the snapshot; only the handful of
    stations/resources actually allocated to THIS event's plan are.
    """
    plan = details.current_response_plan
    if plan is None:
        return None
    station_names_by_id = {station.station_id: station.name for station in details.stations}
    return {
        "methodology": plan.methodology,
        "methodology_version": plan.methodology_version,
        "plan_score": plan.plan_score,
        "coverage_score": plan.coverage_score,
        "average_eta_seconds": plan.average_eta_seconds,
        "uncovered_target_count": len(plan.uncovered_target_ids),
        "allocations": [
            {
                "station_name": station_names_by_id.get(action.station_id, action.station_id),
                # `resource_id` (e.g. "TRUCK-35-2") already reads as a
                # human-meaningful label - FirefightingResourceResponse has
                # no separate descriptive name field to prefer over it.
                "resource_name": action.resource_id,
                "eta_seconds": action.eta_seconds,
                "route_distance_meters": action.route_distance_meters,
            }
            for action in plan.actions
        ],
    }


# ---------------------------------------------------------------------------
# Deterministic follow-up-referent resolution (no LLM call, no extra Gemini
# round-trip). Resolves a GENERIC reference in the current question (e.g.
# "the event"/"it"/"האירוע") to one specific active event's `location_name`,
# using only the recent conversation history plus the current active
# events' own location names - never `fire_event_id`/`event_ref`, and never
# invented/forced when the referent is not unambiguous.
# ---------------------------------------------------------------------------


def _question_has_generic_event_reference(question: str) -> bool:
    return any(pattern.search(question) for pattern in _GENERIC_EVENT_REFERENCE_PATTERNS)


def _question_requests_multiple_events(question: str) -> bool:
    return any(pattern.search(question) for pattern in _MULTI_EVENT_QUESTION_PATTERNS)


def _distinctive_location_words(
    named_events: Sequence[ActiveFireEventSummary],
) -> dict[str, set[str]]:
    """For each event's `location_name`, the words in it that do NOT also
    appear in any OTHER currently-active event's `location_name` - e.g. a
    shared suffix like "Demo Area" across every event is never distinctive
    for any of them, and two events sharing the exact same location_name
    are never distinctive for either (correctly staying unresolvable, same
    as the existing duplicate-location handling). No hardcoded location
    vocabulary - purely derived from this request's own active events.
    """
    word_sets = {
        summary.location_name: set(_WORD_PATTERN.findall(summary.location_name.lower()))
        for summary in named_events
    }
    distinctive: dict[str, set[str]] = {}
    for location_name, words in word_sets.items():
        other_words: set[str] = set()
        for other_name, other_word_set in word_sets.items():
            if other_name != location_name:
                other_words |= other_word_set
        distinctive[location_name] = words - other_words
    return distinctive


def _resolve_conversation_focus(
    question: str,
    history: Sequence[ChatMessage],
    active_result: ActiveFireEventsResult,
) -> str | None:
    """The one active event's `location_name` a generic follow-up reference
    unambiguously resolves to, or `None` when resolution should not be
    forced (no generic reference, an explicit multi-event/comparison
    question, no history, or fewer than two named active events to
    disambiguate between).

    Recency-first: history is scanned newest-to-oldest, one message at a
    time, and resolution stops at the FIRST message that mentions any
    currently-active event's distinctive words at all. An older
    comparison/multi-event turn earlier in the same conversation never
    makes a newer, single-event discussion ambiguous, because it is never
    even inspected once a closer, decisive message is found. If that
    nearest event-mentioning message itself names more than one event, the
    referent is genuinely ambiguous right there and resolution stops
    without guessing - it does not fall back to searching further into the
    past for an older unambiguous mention (the existing ambiguity-handling
    system-instruction rule covers this `None` case, same as before).
    """
    if not history:
        return None
    if not _question_has_generic_event_reference(question):
        return None
    if _question_requests_multiple_events(question):
        return None

    named_events = [summary for summary in active_result.items if summary.location_name]
    if len(named_events) < 2:
        return None

    distinctive_words = _distinctive_location_words(named_events)

    for message in reversed(history):
        message_words = set(_WORD_PATTERN.findall(message.content.lower()))
        matches = [
            location_name
            for location_name, words in distinctive_words.items()
            if words and (words & message_words)
        ]
        if not matches:
            continue
        return matches[0] if len(matches) == 1 else None

    return None


def _build_focus_block(location_name: str) -> str:
    return (
        f"{_FOCUS_HEADER}\n"
        f'The current follow-up refers to "{location_name}".\n'
        "Answer the current question only about this event unless the current "
        "user question explicitly asks for comparison or multiple events.\n"
        "Use the current EcoGuard data snapshot above for all factual values."
    )


# ---------------------------------------------------------------------------
# Gemini `contents` assembly
# ---------------------------------------------------------------------------


def _build_contents(
    history: Sequence[ChatMessage],
    snapshot_text: str,
    question: str,
    focus_location_name: str | None = None,
) -> list[dict[str, Any]]:
    history_turns = [
        {"role": _GEMINI_ROLE_BY_CHAT_ROLE[message.role], "parts": [{"text": message.content}]}
        for message in history
    ]
    focus_section = f"\n\n{_build_focus_block(focus_location_name)}" if focus_location_name else ""
    final_turn_text = f"{_SNAPSHOT_LABEL}\n{snapshot_text}{focus_section}\n\n{_QUESTION_LABEL}\n{question}"
    return [*history_turns, {"role": "user", "parts": [{"text": final_turn_text}]}]

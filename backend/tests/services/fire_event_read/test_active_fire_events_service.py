"""Tests for ActiveFireEventsService (Epic 6, US 6.1, Task 2), using fakes only.

Both fakes below expose ONLY read methods (no save/update/delete) - this is a
structural guarantee, not just an assertion, that assembling the active-
events snapshot has zero persistence side effects, matching
CurrentResponsePlanResolver's own precedent (tests/services/response_planning/
test_current_response_plan_resolver.py).
"""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.models.active_fire_events import ActiveFireEventsResult
from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.models.fire_severity_assessment import FireSeverityAssessment
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.repositories.fire_event_repository import StoredFireEvent
from src.repositories.fire_severity_assessment_repository import StoredFireSeverityAssessment
from src.services.fire_event_read.active_fire_events_service import ActiveFireEventsService

DETECTED_AT = datetime(2026, 9, 17, 10, 0, tzinfo=timezone.utc)
UPDATED_AT = DETECTED_AT + timedelta(minutes=10)
ASSESSED_AT = DETECTED_AT + timedelta(minutes=20)
AS_OF = DETECTED_AT + timedelta(hours=1)


# ---------------------------------------------------------------------------
# Fakes (read-only by construction)
# ---------------------------------------------------------------------------


class FakeFireEventRepository:
    def __init__(self, active_events: tuple[StoredFireEvent, ...] = ()):
        self.active_events = active_events
        self.calls = 0

    def get_active_events(self) -> tuple[StoredFireEvent, ...]:
        self.calls += 1
        return self.active_events


class FakeFireSeverityAssessmentRepository:
    def __init__(self, latest_by_event_id: dict[int, StoredFireSeverityAssessment] | None = None):
        self.latest_by_event_id = dict(latest_by_event_id or {})
        self.calls: list[tuple[int, ...]] = []

    def get_latest_for_events(self, fire_event_ids) -> dict[int, StoredFireSeverityAssessment]:
        ids = tuple(fire_event_ids)
        self.calls.append(ids)
        return {fid: self.latest_by_event_id[fid] for fid in ids if fid in self.latest_by_event_id}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_stored_event(
    event_id: int,
    *,
    status: FireEventStatus = FireEventStatus.SUSPECTED,
    updated_at: datetime = UPDATED_AT,
    detected_at: datetime = DETECTED_AT,
    latitude: float = 32.731,
    longitude: float = 35.046,
    detection_confidence: float = 0.6,
) -> StoredFireEvent:
    return StoredFireEvent(
        id=event_id,
        event=FireEvent(
            latitude=latitude,
            longitude=longitude,
            detected_at=detected_at,
            updated_at=updated_at,
            status=status,
            detection_confidence=detection_confidence,
            methodology="detector",
            methodology_version="1.0",
        ),
    )


def make_stored_severity(
    assessment_id: int,
    fire_event_id: int,
    *,
    status: FireSeverityAssessmentStatus = FireSeverityAssessmentStatus.VALID,
    score: float | None = 70.0,
    level: FireSeverityLevel | None = FireSeverityLevel.HIGH,
    assessed_at: datetime = ASSESSED_AT,
) -> StoredFireSeverityAssessment:
    return StoredFireSeverityAssessment(
        assessment_id=assessment_id,
        assessment=FireSeverityAssessment(
            fire_event_id=fire_event_id,
            assessed_at=assessed_at,
            status=status,
            score=score,
            level=level,
            methodology="severity-model",
            methodology_version="1.0",
        ),
    )


def make_service(
    active_events: tuple[StoredFireEvent, ...] = (),
    latest_severity_by_event_id: dict[int, StoredFireSeverityAssessment] | None = None,
) -> tuple[ActiveFireEventsService, FakeFireEventRepository, FakeFireSeverityAssessmentRepository]:
    fire_event_repository = FakeFireEventRepository(active_events)
    severity_repository = FakeFireSeverityAssessmentRepository(latest_severity_by_event_id)
    service = ActiveFireEventsService(
        fire_event_repository=fire_event_repository,
        fire_severity_assessment_repository=severity_repository,
    )
    return service, fire_event_repository, severity_repository


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


def test_no_active_events_returns_empty_result_not_error():
    service, _, _ = make_service(active_events=())

    result = service.get_active_events(as_of=AS_OF)

    assert isinstance(result, ActiveFireEventsResult)
    assert result.items == ()


# ---------------------------------------------------------------------------
# Passing through whatever the repository considers active
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("active_status", [FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED])
def test_events_returned_by_repository_are_included_regardless_of_which_active_status(active_status):
    """Active-status filtering is FireEventRepository's job (see
    tests/repositories/test_fire_event_repository.py); this service must not
    re-filter or second-guess whatever the repository already decided is active."""
    stored = make_stored_event(1, status=active_status)
    service, _, _ = make_service(active_events=(stored,))

    result = service.get_active_events(as_of=AS_OF)

    assert [item.fire_event_id for item in result.items] == [1]
    assert result.items[0].status is active_status


# ---------------------------------------------------------------------------
# Severity linkage
# ---------------------------------------------------------------------------


def test_event_with_severity_returns_its_latest_summary():
    stored_event = make_stored_event(1)
    stored_severity = make_stored_severity(101, 1, status=FireSeverityAssessmentStatus.VALID, score=88.0, level=FireSeverityLevel.CRITICAL)
    service, _, _ = make_service(active_events=(stored_event,), latest_severity_by_event_id={1: stored_severity})

    result = service.get_active_events(as_of=AS_OF)

    severity = result.items[0].severity
    assert severity is not None
    assert severity.assessment_id == 101
    assert severity.status is FireSeverityAssessmentStatus.VALID
    assert severity.score == pytest.approx(88.0)
    assert severity.level is FireSeverityLevel.CRITICAL


def test_event_with_no_severity_returns_none():
    stored_event = make_stored_event(1)
    service, _, _ = make_service(active_events=(stored_event,), latest_severity_by_event_id={})

    result = service.get_active_events(as_of=AS_OF)

    assert result.items[0].severity is None


def test_fire_event_a_never_receives_severity_from_fire_event_b():
    event_a = make_stored_event(1)
    event_b = make_stored_event(2)
    severity_for_b_only = make_stored_severity(202, 2)
    service, _, _ = make_service(
        active_events=(event_a, event_b),
        latest_severity_by_event_id={2: severity_for_b_only},
    )

    result = service.get_active_events(as_of=AS_OF)

    by_id = {item.fire_event_id: item for item in result.items}
    assert by_id[1].severity is None
    assert by_id[2].severity is not None
    assert by_id[2].severity.assessment_id == 202


def test_insufficient_data_severity_is_reported_honestly_not_hidden_or_faked():
    """Task 4 rule: expose the latest persisted status honestly - never fall
    back to an older VALID assessment merely because the latest is not VALID."""
    stored_event = make_stored_event(1)
    stored_severity = make_stored_severity(
        101, 1, status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA, score=None, level=None
    )
    service, _, _ = make_service(active_events=(stored_event,), latest_severity_by_event_id={1: stored_severity})

    result = service.get_active_events(as_of=AS_OF)

    severity = result.items[0].severity
    assert severity is not None
    assert severity.status is FireSeverityAssessmentStatus.INSUFFICIENT_DATA
    assert severity.score is None
    assert severity.level is None


def test_severity_lookup_is_batched_once_for_all_active_events():
    """No N+1: exactly one call to get_latest_for_events for the whole page,
    covering every active event id."""
    events = (make_stored_event(1), make_stored_event(2), make_stored_event(3))
    service, fire_event_repository, severity_repository = make_service(active_events=events)

    service.get_active_events(as_of=AS_OF)

    assert fire_event_repository.calls == 1
    assert len(severity_repository.calls) == 1
    assert set(severity_repository.calls[0]) == {1, 2, 3}


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_service_preserves_repository_ordering_without_resorting():
    """Ordering (most-recently-updated first, tie-broken by id) is
    FireEventRepository.get_active_events's contract; the service must not
    apply a second, possibly conflicting, sort of its own."""
    newest = make_stored_event(5, updated_at=UPDATED_AT + timedelta(hours=2))
    middle = make_stored_event(2, updated_at=UPDATED_AT + timedelta(hours=1))
    oldest = make_stored_event(9, updated_at=UPDATED_AT)
    service, _, _ = make_service(active_events=(newest, middle, oldest))

    result = service.get_active_events(as_of=AS_OF)

    assert [item.fire_event_id for item in result.items] == [5, 2, 9]


def test_service_preserves_repository_tie_break_order_as_is():
    tied_high_id = make_stored_event(7, updated_at=UPDATED_AT)
    tied_low_id = make_stored_event(3, updated_at=UPDATED_AT)
    service, _, _ = make_service(active_events=(tied_high_id, tied_low_id))

    result = service.get_active_events(as_of=AS_OF)

    assert [item.fire_event_id for item in result.items] == [7, 3]


# ---------------------------------------------------------------------------
# as_of handling
# ---------------------------------------------------------------------------


def test_explicit_as_of_is_used_verbatim():
    service, _, _ = make_service()

    result = service.get_active_events(as_of=AS_OF)

    assert result.as_of == AS_OF


def test_as_of_defaults_to_current_time_when_omitted():
    service, _, _ = make_service()

    before = datetime.now(timezone.utc)
    result = service.get_active_events()
    after = datetime.now(timezone.utc)

    assert before <= result.as_of <= after


def test_naive_as_of_is_rejected():
    service, _, _ = make_service()

    with pytest.raises(ValueError):
        service.get_active_events(as_of=datetime(2026, 9, 17, 10, 0))


# ---------------------------------------------------------------------------
# Timezone preservation
# ---------------------------------------------------------------------------


def test_all_returned_timestamps_remain_timezone_aware():
    stored_event = make_stored_event(1, detected_at=DETECTED_AT, updated_at=UPDATED_AT)
    stored_severity = make_stored_severity(101, 1, assessed_at=ASSESSED_AT)
    service, _, _ = make_service(active_events=(stored_event,), latest_severity_by_event_id={1: stored_severity})

    result = service.get_active_events(as_of=AS_OF)
    item = result.items[0]

    assert result.as_of.tzinfo is not None
    assert item.detected_at.tzinfo is not None
    assert item.updated_at.tzinfo is not None
    assert item.severity.assessed_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Read-only guarantee
# ---------------------------------------------------------------------------


def test_service_source_calls_no_write_operation():
    """Static guarantee: the service's own code never calls a save/update/
    delete/create method on either repository - it only ever calls
    get_active_events() and get_latest_for_events()."""
    import inspect

    source = inspect.getsource(ActiveFireEventsService)
    for forbidden in (".save(", ".update(", ".delete(", ".create_event(", "update_event"):
        assert forbidden not in source


def test_fakes_expose_no_write_methods():
    for forbidden in ("save_assessment", "create_event", "update_event", "attach_evidence"):
        assert not hasattr(FakeFireEventRepository(), forbidden)
        assert not hasattr(FakeFireSeverityAssessmentRepository(), forbidden)


def test_repeated_calls_are_idempotent_and_side_effect_free():
    stored_event = make_stored_event(1)
    service, _, _ = make_service(active_events=(stored_event,))

    first = service.get_active_events(as_of=AS_OF)
    second = service.get_active_events(as_of=AS_OF)

    assert first == second


# ---------------------------------------------------------------------------
# Architecture guard: no HTTP framework, no agents, no external providers
# ---------------------------------------------------------------------------


def test_service_does_not_import_http_agent_or_external_provider_modules():
    forbidden_fragments = (
        "fastapi",
        "pydantic",
        "FireDetectionAgent",
        "FireSeverityAssessmentAgent",
        "FireSpreadPredictionAgent",
        "ResponseTargetGenerationAgent",
        "OperationalRefreshOrchestrator",
        "src.external",
        "src.agents",
        "src.simulation",
        "src.calculators",
    )
    path = Path("backend/src/services/fire_event_read/active_fire_events_service.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        module = ""
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
        elif isinstance(node, ast.Import):
            module = ",".join(alias.name for alias in node.names)
        if any(fragment in module for fragment in forbidden_fragments):
            violations.append(module)

    assert violations == []

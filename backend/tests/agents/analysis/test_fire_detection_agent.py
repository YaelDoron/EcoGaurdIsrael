"""Tests for FireDetectionAgent orchestration."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from src.agents.analysis.fire_detection_agent import FireDetectionAgent
from src.agents.analysis.fire_detection_result import FireDetectionResult
from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.models import (
    FireDetectionCandidate,
    FireDetectionDecision,
    FireDetectionEvidence,
    FireDetectionStatus,
    FireEvent,
    FireEventStatus,
    FireEvidenceRef,
    FireEvidenceType,
)
from src.repositories.fire_event_repository import StoredFireEvent

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
OBSERVED_AT = AS_OF - timedelta(minutes=10)
LATITUDE = 32.731
LONGITUDE = 35.046


def satellite(evidence_id=1, confidence="nominal", observed_at=OBSERVED_AT, latitude=LATITUDE, location_name=None):
    return FireDetectionEvidence(
        evidence_id=evidence_id,
        evidence_type=FireEvidenceType.SATELLITE,
        latitude=latitude,
        longitude=LONGITUDE,
        observed_at=observed_at,
        satellite_confidence=confidence,
        location_name=location_name,
    )


def news(evidence_id=2, observed_at=OBSERVED_AT, latitude=LATITUDE, location_name=None):
    return FireDetectionEvidence(
        evidence_id=evidence_id,
        evidence_type=FireEvidenceType.NEWS,
        latitude=latitude,
        longitude=LONGITUDE,
        observed_at=observed_at,
        location_name=location_name,
    )


def ref(evidence_type, evidence_id):
    return FireEvidenceRef(evidence_type=evidence_type, evidence_id=evidence_id)


def sat_ref(evidence_id=1):
    return ref(FireEvidenceType.SATELLITE, evidence_id)


def news_ref(evidence_id=2):
    return ref(FireEvidenceType.NEWS, evidence_id)


def decision(status=FireDetectionStatus.SUSPECTED, confidence=0.6, evidence_refs=(None,), latitude=LATITUDE):
    refs = (sat_ref(1),) if evidence_refs == (None,) else evidence_refs
    return FireDetectionDecision(
        confidence=confidence,
        status=status,
        latitude=None if status is FireDetectionStatus.NO_EVENT else latitude,
        longitude=None if status is FireDetectionStatus.NO_EVENT else LONGITUDE,
        supporting_evidence=refs,
    )


def event(status=FireEventStatus.SUSPECTED, confidence=0.6, detected_at=OBSERVED_AT, updated_at=OBSERVED_AT, location_name=None):
    return FireEvent(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        detected_at=detected_at,
        updated_at=updated_at,
        status=status,
        detection_confidence=confidence,
        methodology=FIRE_DETECTION_METHODOLOGY_NAME,
        methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
        location_name=location_name,
    )


class FakeEvidenceService:
    def __init__(self, candidates=(), resolved=None, exc=None, resolve_exc=None):
        self.candidates = tuple(candidates)
        self.resolved = resolved or {}
        self.exc = exc
        self.resolve_exc = resolve_exc
        self.build_calls = []
        self.resolve_calls = []

    def build_candidates(self, as_of):
        self.build_calls.append(as_of)
        if self.exc is not None:
            raise self.exc
        return self.candidates

    def resolve_evidence_refs(self, refs):
        refs = tuple(refs)
        self.resolve_calls.append(refs)
        if self.resolve_exc is not None:
            raise self.resolve_exc
        return self.resolved[refs]


class FakeCalculator:
    def __init__(self, decisions=(), exc=None):
        self.decisions = list(decisions)
        self.exc = exc
        self.calls = []

    def evaluate(self, evidence):
        self.calls.append(tuple(evidence))
        if self.exc is not None:
            raise self.exc
        return self.decisions.pop(0)


class FakeFireEventRepository:
    def __init__(self, match=None, create_exc=None, update_exc=None):
        self.match = match
        self.create_exc = create_exc
        self.update_exc = update_exc
        self.created = []
        self.attached = []
        self.updated = []
        self.match_calls = []
        self.get_ref_calls = []
        self.next_id = 10
        self.refs_by_event_id = {}

    def find_matching_active_event(self, latitude, longitude, observed_at):
        self.match_calls.append((latitude, longitude, observed_at))
        return self.match

    def create_event(self, fire_event, supporting_evidence):
        if self.create_exc is not None:
            raise self.create_exc
        stored = StoredFireEvent(self.next_id, fire_event, tuple(supporting_evidence))
        self.next_id += 1
        self.created.append((fire_event, tuple(supporting_evidence), stored.id))
        self.refs_by_event_id[stored.id] = tuple(supporting_evidence)
        return stored

    def get_evidence_refs(self, fire_event_id):
        self.get_ref_calls.append(fire_event_id)
        return self.refs_by_event_id[fire_event_id]

    def attach_evidence(self, fire_event_id, evidence):
        self.attached.append((fire_event_id, tuple(evidence)))
        self.refs_by_event_id[fire_event_id] = tuple(sorted(
            set(self.refs_by_event_id[fire_event_id]).union(evidence),
            key=lambda item: (item.evidence_type.value, item.evidence_id),
        ))
        return self.refs_by_event_id[fire_event_id]

    def update_event(self, fire_event_id, fire_event):
        if self.update_exc is not None:
            raise self.update_exc
        self.updated.append((fire_event_id, fire_event))
        return StoredFireEvent(fire_event_id, fire_event, self.refs_by_event_id[fire_event_id])


def candidate(*evidence):
    return FireDetectionCandidate(tuple(evidence))


def make_agent(evidence_service, calculator, repository):
    return FireDetectionAgent(
        evidence_service=evidence_service,
        calculator=calculator,
        fire_event_repository=repository,
    )


def test_no_candidates_returns_success_without_events():
    service = FakeEvidenceService()
    calculator = FakeCalculator()
    repository = FakeFireEventRepository()

    result = make_agent(service, calculator, repository).detect(AS_OF)

    assert result == FireDetectionResult(True, 0, 0, 0, 0, ())
    assert calculator.calls == []
    assert repository.created == []


def test_no_event_candidate_is_ignored():
    evidence = satellite(confidence="low")
    service = FakeEvidenceService(candidates=[candidate(evidence)])
    calculator = FakeCalculator([decision(FireDetectionStatus.NO_EVENT, 0.4, ())])
    repository = FakeFireEventRepository()

    result = make_agent(service, calculator, repository).detect(AS_OF)

    assert result.no_event_count == 1
    assert result.events_created == 0
    assert calculator.calls == [(evidence,)]


def test_suspected_candidate_without_match_creates_suspected_event():
    evidence = satellite()
    service = FakeEvidenceService(candidates=[candidate(evidence)])
    calculator = FakeCalculator([decision(FireDetectionStatus.SUSPECTED, 0.6, (sat_ref(1),))])
    repository = FakeFireEventRepository()

    result = make_agent(service, calculator, repository).detect(AS_OF)

    created_event, refs, event_id = repository.created[0]
    assert result.events_created == 1
    assert result.event_ids == (event_id,)
    assert created_event.status is FireEventStatus.SUSPECTED
    assert refs == (sat_ref(1),)


def test_confirmed_candidate_without_match_creates_confirmed_event():
    evidence_items = (satellite(), news())
    service = FakeEvidenceService(candidates=[candidate(*evidence_items)])
    calculator = FakeCalculator([decision(FireDetectionStatus.CONFIRMED, 0.8, (sat_ref(1), news_ref(2)))])
    repository = FakeFireEventRepository()

    make_agent(service, calculator, repository).detect(AS_OF)

    assert repository.created[0][0].status is FireEventStatus.CONFIRMED
    assert repository.created[0][0].detection_confidence == pytest.approx(0.8)


def test_created_event_content_comes_from_decision_and_evidence_timestamps():
    early = satellite(observed_at=OBSERVED_AT - timedelta(minutes=5))
    late = news(observed_at=OBSERVED_AT + timedelta(minutes=5))
    service = FakeEvidenceService(candidates=[candidate(early, late)])
    calculator = FakeCalculator([decision(FireDetectionStatus.CONFIRMED, 0.8, (sat_ref(1), news_ref(2)), latitude=33.0)])
    repository = FakeFireEventRepository()

    make_agent(service, calculator, repository).detect(AS_OF)

    created_event, refs, _ = repository.created[0]
    assert created_event.latitude == pytest.approx(33.0)
    assert created_event.detection_confidence == pytest.approx(0.8)
    assert created_event.methodology == FIRE_DETECTION_METHODOLOGY_NAME
    assert created_event.methodology_version == FIRE_DETECTION_METHODOLOGY_VERSION
    assert created_event.detected_at == early.observed_at
    assert created_event.updated_at == late.observed_at
    assert refs == (news_ref(2), sat_ref(1))


# ---------------------------------------------------------------------------
# location_name provenance (Part F/G/H): trustworthy evidence-carried
# location survives into the persisted FireEvent, never guessed here.
# ---------------------------------------------------------------------------


def test_created_event_inherits_location_name_from_satellite_evidence():
    sat = satellite(location_name="Galilee Demo Area")
    service = FakeEvidenceService(candidates=[candidate(sat)])
    calculator = FakeCalculator([decision(FireDetectionStatus.SUSPECTED, 0.6, (sat_ref(1),))])
    repository = FakeFireEventRepository()

    make_agent(service, calculator, repository).detect(AS_OF)

    created_event, _, _ = repository.created[0]
    assert created_event.location_name == "Galilee Demo Area"


def test_created_event_location_name_is_none_when_no_evidence_carries_one():
    """Real, non-simulation evidence never carries location_name - the
    created FireEvent must not fabricate one."""
    sat = satellite()
    n = news()
    service = FakeEvidenceService(candidates=[candidate(sat, n)])
    calculator = FakeCalculator([decision(FireDetectionStatus.SUSPECTED, 0.6, (sat_ref(1), news_ref(2)))])
    repository = FakeFireEventRepository()

    make_agent(service, calculator, repository).detect(AS_OF)

    created_event, _, _ = repository.created[0]
    assert created_event.location_name is None


def test_created_event_location_name_ignores_news_evidence_location():
    """Defense-in-depth: even if a NEWS FireDetectionEvidence somehow carried
    a location_name (it shouldn't - see FireDetectionEvidenceService.
    _normalize_news's own docstring), the agent must only trust SATELLITE
    evidence for FireEvent provenance."""
    sat = satellite(location_name=None)
    n = news(location_name="Some NLP-Guessed Place")
    service = FakeEvidenceService(candidates=[candidate(sat, n)])
    calculator = FakeCalculator([decision(FireDetectionStatus.SUSPECTED, 0.6, (sat_ref(1), news_ref(2)))])
    repository = FakeFireEventRepository()

    make_agent(service, calculator, repository).detect(AS_OF)

    created_event, _, _ = repository.created[0]
    assert created_event.location_name is None


def test_different_incidents_get_their_own_distinct_location_names_no_cross_contamination():
    galilee_sat = satellite(evidence_id=101, location_name="Galilee Demo Area")
    carmel_sat = satellite(evidence_id=102, latitude=32.7, location_name="Carmel Demo Area")
    service = FakeEvidenceService(candidates=[candidate(galilee_sat), candidate(carmel_sat)])
    calculator = FakeCalculator([
        decision(FireDetectionStatus.SUSPECTED, 0.6, (ref(FireEvidenceType.SATELLITE, 101),)),
        decision(FireDetectionStatus.SUSPECTED, 0.6, (ref(FireEvidenceType.SATELLITE, 102),), latitude=32.7),
    ])
    repository = FakeFireEventRepository()

    make_agent(service, calculator, repository).detect(AS_OF)

    location_names = {fire_event.location_name for fire_event, _, _ in repository.created}
    assert location_names == {"Galilee Demo Area", "Carmel Demo Area"}


def test_existing_event_update_preserves_its_original_location_name():
    existing = StoredFireEvent(7, event(location_name="Galilee Demo Area"), (sat_ref(1),), created_at=OBSERVED_AT)
    new_evidence = news()
    combined_evidence = (satellite(), new_evidence)
    service = FakeEvidenceService(
        candidates=[candidate(new_evidence)],
        resolved={(news_ref(2), sat_ref(1)): combined_evidence},
    )
    calculator = FakeCalculator([
        decision(FireDetectionStatus.SUSPECTED, 0.5, (news_ref(2),)),
        decision(FireDetectionStatus.CONFIRMED, 0.8, (sat_ref(1), news_ref(2))),
    ])
    repository = FakeFireEventRepository(match=existing)
    repository.refs_by_event_id[7] = (sat_ref(1),)

    make_agent(service, calculator, repository).detect(AS_OF)

    assert repository.updated[0][1].location_name == "Galilee Demo Area"


def test_matching_active_event_is_updated_not_recreated():
    existing = StoredFireEvent(7, event(), (sat_ref(1),))
    new_evidence = news()
    combined_evidence = (satellite(), new_evidence)
    service = FakeEvidenceService(
        candidates=[candidate(new_evidence)],
        resolved={(news_ref(2), sat_ref(1)): combined_evidence},
    )
    calculator = FakeCalculator([
        decision(FireDetectionStatus.SUSPECTED, 0.5, (news_ref(2),)),
        decision(FireDetectionStatus.CONFIRMED, 0.8, (sat_ref(1), news_ref(2))),
    ])
    repository = FakeFireEventRepository(match=existing)
    repository.refs_by_event_id[7] = (sat_ref(1),)

    result = make_agent(service, calculator, repository).detect(AS_OF)

    assert result.events_created == 0
    assert result.events_updated == 1
    assert result.event_ids == (7,)
    assert repository.attached == [(7, (news_ref(2),))]
    assert repository.updated[0][0] == 7
    assert repository.updated[0][1].status is FireEventStatus.CONFIRMED


def test_existing_event_update_preserves_detected_at_and_uses_combined_location_and_latest_time():
    original_detected_at = OBSERVED_AT - timedelta(hours=1)
    existing = StoredFireEvent(7, event(detected_at=original_detected_at), (sat_ref(1),))
    sat = satellite(observed_at=original_detected_at)
    later_news = news(observed_at=OBSERVED_AT + timedelta(minutes=20), latitude=LATITUDE + 0.001)
    service = FakeEvidenceService(candidates=[candidate(later_news)], resolved={(news_ref(2), sat_ref(1)): (sat, later_news)})
    calculator = FakeCalculator([
        decision(FireDetectionStatus.SUSPECTED, 0.5, (news_ref(2),)),
        decision(FireDetectionStatus.CONFIRMED, 0.8, (sat_ref(1), news_ref(2)), latitude=LATITUDE + 0.002),
    ])
    repository = FakeFireEventRepository(match=existing)
    repository.refs_by_event_id[7] = (sat_ref(1),)

    make_agent(service, calculator, repository).detect(AS_OF)

    updated_event = repository.updated[0][1]
    assert updated_event.detected_at == original_detected_at
    assert updated_event.updated_at == later_news.observed_at
    assert updated_event.latitude == pytest.approx(LATITUDE + 0.002)


def test_reprocessing_same_evidence_does_not_update_or_duplicate():
    existing = StoredFireEvent(7, event(), (sat_ref(1),))
    evidence = satellite()
    service = FakeEvidenceService(candidates=[candidate(evidence)], resolved={(sat_ref(1),): (evidence,)})
    calculator = FakeCalculator([
        decision(FireDetectionStatus.SUSPECTED, 0.6, (sat_ref(1),)),
        decision(FireDetectionStatus.SUSPECTED, 0.6, (sat_ref(1),)),
    ])
    repository = FakeFireEventRepository(match=existing)
    repository.refs_by_event_id[7] = (sat_ref(1),)

    result = make_agent(service, calculator, repository).detect(AS_OF)

    assert result.events_created == 0
    assert result.events_updated == 0
    assert repository.attached == []
    assert repository.updated == []
    assert result.event_ids == (7,)


def test_confirmed_event_is_not_downgraded_to_suspected():
    existing = StoredFireEvent(7, event(status=FireEventStatus.CONFIRMED, confidence=0.85), (sat_ref(1),))
    evidence = satellite()
    service = FakeEvidenceService(candidates=[candidate(evidence)], resolved={(sat_ref(1),): (evidence,)})
    calculator = FakeCalculator([
        decision(FireDetectionStatus.SUSPECTED, 0.6, (sat_ref(1),)),
        decision(FireDetectionStatus.SUSPECTED, 0.6, (sat_ref(1),)),
    ])
    repository = FakeFireEventRepository(match=existing)
    repository.refs_by_event_id[7] = (sat_ref(1),)

    make_agent(service, calculator, repository).detect(AS_OF)

    assert repository.updated[0][1].status is FireEventStatus.CONFIRMED
    assert repository.updated[0][1].detection_confidence == pytest.approx(0.6)


def test_satellite_and_news_same_numeric_id_are_preserved():
    evidence_items = (satellite(evidence_id=5), news(evidence_id=5))
    service = FakeEvidenceService(candidates=[candidate(*evidence_items)])
    calculator = FakeCalculator([
        decision(FireDetectionStatus.CONFIRMED, 0.8, (sat_ref(5), news_ref(5))),
    ])
    repository = FakeFireEventRepository()

    make_agent(service, calculator, repository).detect(AS_OF)

    assert repository.created[0][1] == (news_ref(5), sat_ref(5))


def test_same_ref_is_not_attached_twice():
    existing = StoredFireEvent(7, event(), (sat_ref(1),))
    evidence = satellite()
    service = FakeEvidenceService(candidates=[candidate(evidence)], resolved={(sat_ref(1),): (evidence,)})
    calculator = FakeCalculator([
        decision(FireDetectionStatus.SUSPECTED, 0.6, (sat_ref(1),)),
        decision(FireDetectionStatus.SUSPECTED, 0.6, (sat_ref(1),)),
    ])
    repository = FakeFireEventRepository(match=existing)
    repository.refs_by_event_id[7] = (sat_ref(1),)

    make_agent(service, calculator, repository).detect(AS_OF)

    assert repository.attached == []


def test_two_unrelated_candidates_create_two_events():
    first = satellite(evidence_id=1)
    second = satellite(evidence_id=2, observed_at=OBSERVED_AT + timedelta(minutes=1), latitude=LATITUDE + 0.2)
    service = FakeEvidenceService(candidates=[candidate(first), candidate(second)])
    calculator = FakeCalculator([
        decision(FireDetectionStatus.SUSPECTED, 0.6, (sat_ref(1),)),
        decision(FireDetectionStatus.SUSPECTED, 0.6, (sat_ref(2),), latitude=LATITUDE + 0.2),
    ])
    repository = FakeFireEventRepository()

    result = make_agent(service, calculator, repository).detect(AS_OF)

    assert result.events_created == 2
    assert result.event_ids == (10, 11)


def test_two_candidates_matching_same_event_update_one_event():
    existing = StoredFireEvent(7, event(), (sat_ref(1),))
    first_news = news(evidence_id=2)
    second_news = news(evidence_id=3, observed_at=OBSERVED_AT + timedelta(minutes=1))
    service = FakeEvidenceService(
        candidates=[candidate(first_news), candidate(second_news)],
        resolved={
            (news_ref(2), sat_ref(1)): (satellite(), first_news),
            (news_ref(2), news_ref(3), sat_ref(1)): (satellite(), first_news, second_news),
        },
    )
    calculator = FakeCalculator([
        decision(FireDetectionStatus.SUSPECTED, 0.5, (news_ref(2),)),
        decision(FireDetectionStatus.CONFIRMED, 0.8, (sat_ref(1), news_ref(2))),
        decision(FireDetectionStatus.SUSPECTED, 0.5, (news_ref(3),)),
        decision(FireDetectionStatus.CONFIRMED, 0.8, (sat_ref(1), news_ref(2), news_ref(3))),
    ])
    repository = FakeFireEventRepository(match=existing)
    repository.refs_by_event_id[7] = (sat_ref(1),)

    result = make_agent(service, calculator, repository).detect(AS_OF)

    assert result.events_created == 0
    assert result.events_updated == 2
    assert result.event_ids == (7,)
    assert len(repository.created) == 0


@pytest.mark.parametrize(
    ("service", "calculator", "repository", "expected_message"),
    [
        (FakeEvidenceService(exc=RuntimeError("boom")), FakeCalculator(), FakeFireEventRepository(), "Fire detection orchestration failed."),
        (FakeEvidenceService(candidates=[candidate(satellite())]), FakeCalculator(exc=RuntimeError("boom")), FakeFireEventRepository(), "Fire detection orchestration failed."),
        (
            FakeEvidenceService(candidates=[candidate(satellite())]),
            FakeCalculator([decision()]),
            FakeFireEventRepository(create_exc=RuntimeError("boom")),
            "Fire detection orchestration failed.",
        ),
        (
            FakeEvidenceService(candidates=[candidate(news())], resolve_exc=RuntimeError("boom")),
            FakeCalculator([decision(FireDetectionStatus.SUSPECTED, 0.5, (news_ref(2),))]),
            FakeFireEventRepository(match=StoredFireEvent(7, event(), (sat_ref(1),))),
            "Fire detection orchestration failed.",
        ),
        (
            FakeEvidenceService(candidates=[candidate(news())], resolved={(news_ref(2), sat_ref(1)): (satellite(), news())}),
            FakeCalculator([
                decision(FireDetectionStatus.SUSPECTED, 0.5, (news_ref(2),)),
                decision(FireDetectionStatus.CONFIRMED, 0.8, (sat_ref(1), news_ref(2))),
            ]),
            FakeFireEventRepository(match=StoredFireEvent(7, event(), (sat_ref(1),)), update_exc=RuntimeError("boom")),
            "Fire detection orchestration failed.",
        ),
    ],
)
def test_operational_failures_return_failed_result(service, calculator, repository, expected_message):
    if 7 not in repository.refs_by_event_id:
        repository.refs_by_event_id[7] = (sat_ref(1),)

    result = make_agent(service, calculator, repository).detect(AS_OF)

    assert result.success is False
    assert result.error_message == expected_message
    assert result.no_event_count == 0


def test_naive_as_of_is_rejected():
    with pytest.raises(ValueError):
        make_agent(FakeEvidenceService(), FakeCalculator(), FakeFireEventRepository()).detect(
            datetime(2026, 9, 14, 12, 0)
        )


def test_result_validation_and_immutability():
    result = FireDetectionResult(True, 1, 0, 1, 0, (2, 1))

    assert result.event_ids == (1, 2)
    with pytest.raises(FrozenInstanceError):
        result.success = False
    with pytest.raises(ValueError):
        FireDetectionResult(True, 0, 0, 0, 0, (), "unexpected")
    with pytest.raises(ValueError):
        FireDetectionResult(False, 0, 0, 0, 0, ())

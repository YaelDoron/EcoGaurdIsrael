"""Task 5B: FireDetectionHistoryService, tolerant ref resolution and the history/candidate window separation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength
from src.models.satellite_hotspot import SatelliteHotspot
from src.repositories.fire_event_repository import FireEventRepository, StoredFireEvent
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.fire_detection import FireDetectionEvidenceService, FireDetectionHistoryService
from src.calculators.fire_detection.fire_detection_config import (
    MAX_EVIDENCE_DISTANCE_KM,
    MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES,
)
from src.repositories.fire_event_config import ACTIVE_EVENT_MATCH_DISTANCE_KM, ACTIVE_EVENT_MATCH_WINDOW_HOURS
from src.services.fire_detection.fire_detection_evidence_config import (
    EVIDENCE_LOOKBACK_MINUTES,
    FIRE_EVENT_EVIDENCE_HISTORY_HOURS,
    SATELLITE_PASS_GAP_MINUTES,
)
from src.services.fire_detection.fire_detection_history_service import resolve_refs_tolerantly

T0 = datetime(2026, 9, 14, 6, 0, tzinfo=timezone.utc)
AS_OF = T0 + timedelta(hours=6)


def sat_ref(evidence_id):
    return FireEvidenceRef(FireEvidenceType.SATELLITE, evidence_id)


def news_ref(evidence_id):
    return FireEvidenceRef(FireEvidenceType.NEWS, evidence_id)


def hotspot(evidence_id, minutes, *, frp=None):
    return FireDetectionEvidence(
        evidence_id=evidence_id,
        evidence_type=FireEvidenceType.SATELLITE,
        latitude=32.0,
        longitude=35.0,
        observed_at=T0 + timedelta(minutes=minutes),
        satellite_confidence="nominal",
        satellite_frp=frp,
        satellite_name="NOAA-20",
        satellite_instrument="VIIRS",
    )


class FakeEvidenceService:
    """Resolves refs from a dict; refs missing from it raise ValueError like the real service."""

    def __init__(self, available, fail_with=None):
        self.available = {(e.evidence_type, e.evidence_id): e for e in available}
        self.fail_with = fail_with
        self.calls = []

    def resolve_evidence_refs(self, refs):
        refs = tuple(refs)
        self.calls.append(refs)
        if self.fail_with is not None:
            raise self.fail_with
        missing = [r for r in refs if (r.evidence_type, r.evidence_id) not in self.available]
        if missing:
            raise ValueError(f"not found: {missing!r}")
        return tuple(self.available[(r.evidence_type, r.evidence_id)] for r in refs)


class FakeEventRepository:
    def __init__(self, refs):
        self.refs = tuple(refs)
        self.writes = 0

    def get_evidence_refs(self, fire_event_id):
        return self.refs

    def attach_evidence(self, *args, **kwargs):
        self.writes += 1

    def update_event(self, *args, **kwargs):
        self.writes += 1


def stored_event(event_id=7):
    return StoredFireEvent(
        id=event_id,
        event=FireEvent(
            latitude=32.0,
            longitude=35.0,
            detected_at=T0,
            updated_at=T0,
            status=FireEventStatus.SUSPECTED,
            detection_confidence=0.5,
            methodology="m",
            methodology_version="1",
        ),
    )


def service(evidence, refs, **kwargs):
    evidence_service = FakeEvidenceService(evidence)
    repository = FakeEventRepository(refs)
    return FireDetectionHistoryService(evidence_service, repository, **kwargs), evidence_service, repository


# --- configuration: three distinct time concepts ---


def test_history_window_is_separate_from_the_candidate_window_and_documented():
    assert MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES == 60  # candidate correlation unchanged
    assert MAX_EVIDENCE_DISTANCE_KM == 5.0
    assert EVIDENCE_LOOKBACK_MINUTES == 120  # candidate lookback unchanged
    assert ACTIVE_EVENT_MATCH_DISTANCE_KM == 5.0 and ACTIVE_EVENT_MATCH_WINDOW_HOURS == 6  # event matching unchanged
    assert FIRE_EVENT_EVIDENCE_HISTORY_HOURS * 60 > MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES
    assert FIRE_EVENT_EVIDENCE_HISTORY_HOURS * 60 > EVIDENCE_LOOKBACK_MINUTES
    assert 0 < SATELLITE_PASS_GAP_MINUTES <= MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES


def test_the_history_constants_carry_their_rationale():
    import src.services.fire_detection.fire_detection_evidence_config as config

    source = open(config.__file__, encoding="utf-8").read()
    for name in ("FIRE_EVENT_EVIDENCE_HISTORY_HOURS", "SATELLITE_PASS_GAP_MINUTES"):
        assert name in source
    assert source.count("#") >= 6  # rationale is written down, not implied


# --- the service ---


def test_history_spans_hours_without_any_sixty_minute_rule():
    svc, _, _ = service([hotspot(1, 0), hotspot(2, 180), hotspot(3, 300)], [sat_ref(1), sat_ref(2), sat_ref(3)])
    history = svc.build_history(stored_event(), AS_OF)

    assert [e.evidence_id for e in history.evidence] == [1, 2, 3]
    assert history.distinct_satellite_pass_count == 3
    assert history.fire_event_id == 7 and history.as_of == AS_OF


def test_history_is_window_bounded_and_ignores_the_future():
    old = hotspot(1, -60 * 30)  # 30 h before T0
    inside = hotspot(2, 0)
    future = hotspot(3, 60 * 7)  # after as_of
    svc, _, _ = service([old, inside, future], [sat_ref(1), sat_ref(2), sat_ref(3)])

    history = svc.build_history(stored_event(), AS_OF)

    assert [e.evidence_id for e in history.evidence] == [2]
    assert history.window_start == AS_OF - timedelta(hours=FIRE_EVENT_EVIDENCE_HISTORY_HOURS)


def test_history_window_is_configurable_and_validated():
    svc, _, _ = service([hotspot(1, 0), hotspot(2, 180)], [sat_ref(1), sat_ref(2)], history_hours=4)
    assert [e.evidence_id for e in svc.build_history(stored_event(), AS_OF).evidence] == [2]  # T0 is 6 h old
    for bad in (0, -1, True, None):
        with pytest.raises(ValueError):
            service([], [], history_hours=bad)
        with pytest.raises(ValueError):
            service([], [], satellite_pass_gap_minutes=bad)


def test_history_is_chronological_and_deduplicated():
    svc, _, _ = service(
        [hotspot(1, 0), hotspot(2, 100)],
        [sat_ref(2), sat_ref(1), sat_ref(2)],  # duplicate ref, unsorted
    )
    assert [e.evidence_id for e in svc.build_history(stored_event(), AS_OF).evidence] == [1, 2]


def test_history_includes_news_and_keeps_source_aware_identity():
    report = FireDetectionEvidence(
        evidence_id=1,
        evidence_type=FireEvidenceType.NEWS,
        latitude=32.0,
        longitude=35.0,
        observed_at=T0 + timedelta(minutes=30),
        news_wildfire_signal_strength=NewsWildfireSignalStrength.STRONG,
    )
    svc, _, _ = service([hotspot(1, 0), report], [sat_ref(1), news_ref(1)])
    history = svc.build_history(stored_event(), AS_OF)
    assert [(e.evidence_type, e.evidence_id) for e in history.evidence] == [
        (FireEvidenceType.SATELLITE, 1),
        (FireEvidenceType.NEWS, 1),
    ]
    assert len(history.news_evidence) == 1 and history.distinct_satellite_pass_count == 1


def test_unresolvable_refs_are_reported_and_do_not_hide_the_rest():
    svc, _, _ = service([hotspot(1, 0)], [sat_ref(1), sat_ref(99)])
    history = svc.build_history(stored_event(), AS_OF)
    assert [e.evidence_id for e in history.evidence] == [1]
    assert history.unresolved_evidence == (sat_ref(99),)


def test_infrastructure_errors_are_not_swallowed_as_missing_evidence():
    evidence_service = FakeEvidenceService([hotspot(1, 0)], fail_with=RuntimeError("db down"))
    svc = FireDetectionHistoryService(evidence_service, FakeEventRepository([sat_ref(1)]))
    with pytest.raises(RuntimeError):
        svc.build_history(stored_event(), AS_OF)


def test_history_building_is_read_only_and_needs_no_ml():
    svc, _, repository = service([hotspot(1, 0)], [sat_ref(1)])
    svc.build_history(stored_event(), AS_OF)
    assert repository.writes == 0  # no attach_evidence / update_event


def test_history_service_rejects_bad_inputs():
    svc, _, _ = service([], [])
    with pytest.raises(ValueError):
        svc.build_history(7, AS_OF)  # a StoredFireEvent is required
    with pytest.raises(ValueError):
        svc.build_history(stored_event(), datetime(2026, 9, 14, 12, 0))  # naive as_of


def test_an_event_without_evidence_has_an_empty_history():
    svc, evidence_service, _ = service([], [])
    history = svc.build_history(stored_event(), AS_OF)
    assert history.evidence == () and history.unresolved_evidence == ()
    assert evidence_service.calls == []


def test_resolve_refs_tolerantly_tries_the_whole_set_first():
    evidence_service = FakeEvidenceService([hotspot(1, 0), hotspot(2, 5)])
    evidence, unresolved = resolve_refs_tolerantly(evidence_service, (sat_ref(1), sat_ref(2)))
    assert [e.evidence_id for e in evidence] == [1, 2] and unresolved == ()
    assert evidence_service.calls == [(sat_ref(1), sat_ref(2))]


# --- real SQLite stack: candidate correlation is unchanged, history is separate ---


def test_candidate_correlation_still_splits_sixty_one_minutes_while_history_joins_them(sqlite_session_factory):
    satellites = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    news = NewsRepository(session_factory=sqlite_session_factory)
    events = FireEventRepository(session_factory=sqlite_session_factory)
    for minutes in (0, 61):
        satellites.save_hotspot(
            SatelliteHotspot(
                latitude=32.0,
                longitude=35.0,
                detected_at=T0 + timedelta(minutes=minutes),
                confidence="n",
                satellite="NOAA-20",
                instrument="VIIRS",
            )
        )
    evidence_service = FireDetectionEvidenceService(satellite_repository=satellites, news_repository=news)

    candidates = evidence_service.build_candidates(T0 + timedelta(minutes=62))
    assert [len(c.evidence) for c in candidates] == [1, 1]  # 61 minutes apart: two candidates, as before

    refs = tuple(sat_ref(e.evidence_id) for c in candidates for e in c.evidence)
    stored = events.create_event(stored_event().event, supporting_evidence=refs)
    history = FireDetectionHistoryService(evidence_service, events).build_history(stored, T0 + timedelta(minutes=62))
    assert len(history.evidence) == 2 and history.distinct_satellite_pass_count == 2
    assert history.evidence[0].satellite_name == "NOAA-20" and history.evidence[0].satellite_instrument == "VIIRS"

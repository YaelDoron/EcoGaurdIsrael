"""Tests for EventDetailsService (Epic 6, US 6.2, Task 1), using fakes only.

All fakes expose ONLY read methods (no save/update/delete) - a structural
guarantee that assembling the event-details snapshot has zero persistence
side effects, matching ActiveFireEventsService's own precedent (tests/
services/fire_event_read/test_active_fire_events_service.py).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.models.fire_report import WildfireReport
from src.models.fire_severity_assessment import FireSeverityAssessment
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.fire_spread_prediction import FireSpreadPrediction, FireSpreadPredictionCell
from src.models.fire_spread_prediction_status import FireSpreadPredictionStatus
from src.models.resource_status import ResourceStatus
from src.models.response_plan_details import (
    BaselineComparisonDetails,
    ResponseActionDetails,
    ResponsePlanDetails,
)
from src.models.response_target import ResponseTarget
from src.models.response_target_set import ResponseTargetSet
from src.models.response_target_type import ResponseTargetType
from src.models.satellite_hotspot import SatelliteHotspot
from src.repositories.fire_event_repository import StoredFireEvent
from src.repositories.fire_spread_prediction_repository import StoredFireSpreadPrediction
from src.repositories.fire_severity_assessment_repository import StoredFireSeverityAssessment
from src.repositories.news_repository import StoredWildfireReport
from src.repositories.response_target_repository import StoredResponseTarget, StoredResponseTargetSet
from src.repositories.satellite_hotspot_repository import StoredSatelliteHotspot
from src.services.fire_event_read.event_details_service import EventDetailsService

DETECTED_AT = datetime(2026, 9, 17, 10, 0, tzinfo=timezone.utc)
UPDATED_AT = DETECTED_AT + timedelta(minutes=10)
ASSESSED_AT = DETECTED_AT + timedelta(minutes=20)
PREDICTED_AT = DETECTED_AT + timedelta(minutes=25)
GENERATED_AT = DETECTED_AT + timedelta(minutes=30)
AS_OF = DETECTED_AT + timedelta(hours=1)


# ---------------------------------------------------------------------------
# Fakes (read-only by construction)
# ---------------------------------------------------------------------------


class FakeFireEventRepository:
    def __init__(self, events_by_id: dict[int, StoredFireEvent] | None = None):
        self.events_by_id = dict(events_by_id or {})
        self.calls: list[int] = []

    def get_by_id(self, fire_event_id: int) -> StoredFireEvent | None:
        self.calls.append(fire_event_id)
        return self.events_by_id.get(fire_event_id)


class FakeFireSeverityAssessmentRepository:
    def __init__(self, latest_by_event_id: dict[int, StoredFireSeverityAssessment] | None = None):
        self.latest_by_event_id = dict(latest_by_event_id or {})
        self.calls: list[int] = []

    def get_latest_for_event(self, fire_event_id: int) -> StoredFireSeverityAssessment | None:
        self.calls.append(fire_event_id)
        return self.latest_by_event_id.get(fire_event_id)


class FakeFireSpreadPredictionRepository:
    def __init__(self, latest_by_key: dict[tuple[int, int], StoredFireSpreadPrediction] | None = None):
        self.latest_by_key = dict(latest_by_key or {})
        self.calls: list[tuple[int, int]] = []

    def get_latest_for_event_and_horizon(
        self, fire_event_id: int, horizon_minutes: int
    ) -> StoredFireSpreadPrediction | None:
        self.calls.append((fire_event_id, horizon_minutes))
        return self.latest_by_key.get((fire_event_id, horizon_minutes))


class FakeResponseTargetRepository:
    def __init__(self, latest_by_event_id: dict[int, StoredResponseTargetSet] | None = None):
        self.latest_by_event_id = dict(latest_by_event_id or {})
        self.calls: list[tuple[int, datetime]] = []

    def get_latest_for_event_as_of(self, fire_event_id: int, as_of: datetime) -> StoredResponseTargetSet | None:
        self.calls.append((fire_event_id, as_of))
        return self.latest_by_event_id.get(fire_event_id)


@dataclass(frozen=True)
class FakeStationRow:
    id: str
    name: str
    latitude: float
    longitude: float
    station_type: str | None = None
    address: str | None = None


@dataclass(frozen=True)
class FakeResourceRow:
    id: str
    station_id: str
    status: ResourceStatus


class FakeFireStationRepository:
    def __init__(self, stations: tuple[FakeStationRow, ...] = ()):
        self.stations = stations
        self.calls = 0

    def get_all_stations(self):
        self.calls += 1
        return list(self.stations)


class FakeFirefightingResourceRepository:
    def __init__(self, resources: tuple[FakeResourceRow, ...] = ()):
        self.resources = resources
        self.calls: list[list[str]] = []

    def get_resources_for_stations(self, station_ids: list[str]):
        self.calls.append(list(station_ids))
        return [resource for resource in self.resources if resource.station_id in station_ids]


class FakeSatelliteHotspotRepository:
    def __init__(self, hotspots_by_id: dict[int, StoredSatelliteHotspot] | None = None):
        self.hotspots_by_id = dict(hotspots_by_id or {})
        self.calls: list[int] = []

    def get_by_id(self, hotspot_id: int) -> StoredSatelliteHotspot | None:
        self.calls.append(hotspot_id)
        return self.hotspots_by_id.get(hotspot_id)


class FakeNewsRepository:
    def __init__(self, reports_by_id: dict[int, StoredWildfireReport] | None = None):
        self.reports_by_id = dict(reports_by_id or {})
        self.calls: list[int] = []

    def get_by_id(self, report_id: int) -> StoredWildfireReport | None:
        self.calls.append(report_id)
        return self.reports_by_id.get(report_id)


class FakeResponsePlanDetailsService:
    def __init__(self, plan_by_event_id: dict[int, ResponsePlanDetails] | None = None):
        self.plan_by_event_id = dict(plan_by_event_id or {})
        self.calls: list[int] = []

    def get_current_plan_details(self, fire_event_id: int) -> ResponsePlanDetails | None:
        self.calls.append(fire_event_id)
        return self.plan_by_event_id.get(fire_event_id)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_stored_event(
    event_id: int,
    *,
    status: FireEventStatus = FireEventStatus.CONFIRMED,
    latitude: float = 32.731,
    longitude: float = 35.046,
    supporting_evidence: tuple[FireEvidenceRef, ...] = (),
) -> StoredFireEvent:
    return StoredFireEvent(
        id=event_id,
        event=FireEvent(
            latitude=latitude,
            longitude=longitude,
            detected_at=DETECTED_AT,
            updated_at=UPDATED_AT,
            status=status,
            detection_confidence=0.8,
            methodology="detector",
            methodology_version="1.0",
        ),
        supporting_evidence=supporting_evidence,
    )


def make_stored_hotspot(
    hotspot_id: int,
    *,
    latitude: float = 32.7,
    longitude: float = 35.0,
    confidence: str | None = "high",
    frp: float | None = 12.5,
    brightness: float | None = 300.0,
    satellite: str | None = "Aqua",
    instrument: str | None = "MODIS",
    day_night: str | None = "D",
) -> StoredSatelliteHotspot:
    return StoredSatelliteHotspot(
        id=hotspot_id,
        hotspot=SatelliteHotspot(
            latitude=latitude,
            longitude=longitude,
            detected_at=DETECTED_AT,
            confidence=confidence,
            frp=frp,
            brightness=brightness,
            satellite=satellite,
            instrument=instrument,
            day_night=day_night,
        ),
    )


def make_stored_report(
    report_id: int,
    *,
    title: str = "Wildfire spreads near Modiin",
    summary: str = "Firefighters battle a fast-moving blaze.",
    source_feed: str = "ynet",
    location_name: str | None = "Modiin",
    latitude: float | None = 31.9,
    longitude: float | None = 35.0,
    published_at: datetime | None = None,
) -> StoredWildfireReport:
    published = published_at if published_at is not None else DETECTED_AT
    return StoredWildfireReport(
        id=report_id,
        report=WildfireReport(
            source_url=f"https://example.com/reports/{report_id}",
            source_feed=source_feed,
            title=title,
            summary=summary,
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            published_at=published,
            fetched_at=published,
        ),
        observed_at=published,
    )


def make_stored_severity(
    assessment_id: int,
    fire_event_id: int,
    *,
    status: FireSeverityAssessmentStatus = FireSeverityAssessmentStatus.VALID,
    score: float | None = 70.0,
    level: FireSeverityLevel | None = FireSeverityLevel.HIGH,
) -> StoredFireSeverityAssessment:
    return StoredFireSeverityAssessment(
        assessment_id=assessment_id,
        assessment=FireSeverityAssessment(
            fire_event_id=fire_event_id,
            assessed_at=ASSESSED_AT,
            status=status,
            score=score,
            level=level,
            methodology="severity-model",
            methodology_version="1.0",
        ),
    )


def make_valid_prediction(
    fire_event_id: int, *, horizon_minutes: int = 30, cell_count: int = 1
) -> StoredFireSpreadPrediction:
    cells = tuple(
        FireSpreadPredictionCell(
            latitude=32.7 + i * 0.01,
            longitude=35.0 + i * 0.01,
            spread_probability=0.5,
            spread_risk_score=50.0,
            reached_step=1,
            reached_minutes=5,
        )
        for i in range(cell_count)
    )
    return StoredFireSpreadPrediction(
        id=900 + horizon_minutes,
        prediction=FireSpreadPrediction(
            fire_event_id=fire_event_id,
            severity_assessment_id=101,
            predicted_at=PREDICTED_AT,
            horizon_minutes=horizon_minutes,
            status=FireSpreadPredictionStatus.VALID,
            methodology="ca-model",
            methodology_version="1.0",
            cells=cells,
        ),
    )


def make_not_valid_prediction(
    fire_event_id: int,
    *,
    horizon_minutes: int = 30,
    status: FireSpreadPredictionStatus = FireSpreadPredictionStatus.INSUFFICIENT_DATA,
) -> StoredFireSpreadPrediction:
    return StoredFireSpreadPrediction(
        id=800 + horizon_minutes,
        prediction=FireSpreadPrediction(
            fire_event_id=fire_event_id,
            severity_assessment_id=None,
            predicted_at=PREDICTED_AT,
            horizon_minutes=horizon_minutes,
            status=status,
            methodology="ca-model",
            methodology_version="1.0",
            cells=(),
        ),
    )


def make_target_set(fire_event_id: int) -> StoredResponseTargetSet:
    active_target = ResponseTarget(
        fire_event_id=fire_event_id,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        latitude=32.731,
        longitude=35.046,
        priority_score=1.0,
    )
    return StoredResponseTargetSet(
        id=55,
        target_set=ResponseTargetSet(
            fire_event_id=fire_event_id,
            generated_at=GENERATED_AT,
            methodology="target-gen",
            methodology_version="1.0",
            targets=(active_target,),
        ),
        targets=(StoredResponseTarget(id=551, target_order=0, target=active_target),),
    )


def make_plan_details(fire_event_id: int, *, with_baseline: bool = False) -> ResponsePlanDetails:
    action = ResponseActionDetails(
        resource_id="R1",
        station_id="S1",
        response_target_id=1,
        target_type=ResponseTargetType.ACTIVE_FIRE.value,
        target_priority=1.0,
        eta_seconds=120.0,
        route_distance_meters=500.0,
        node_path=(1, 2, 3),
    )
    return ResponsePlanDetails(
        plan_id=77,
        fire_event_id=fire_event_id,
        response_target_set_id=55,
        route_planning_run_id=66,
        generated_at=GENERATED_AT,
        methodology="ga",
        methodology_version="1.0",
        random_seed=42,
        is_current=True,
        plan_score=90.0,
        coverage_score=1.0,
        average_eta_seconds=120.0,
        actions=(action,),
        uncovered_target_ids=(),
        baseline_comparison=(
            BaselineComparisonDetails(
                baseline_score=70.0,
                baseline_coverage_score=0.5,
                baseline_average_eta_seconds=300.0,
                score_difference=20.0,
                improvement_percentage=28.5,
            )
            if with_baseline
            else None
        ),
        optimization_config=None,
    )


def make_service(
    *,
    events_by_id: dict[int, StoredFireEvent] | None = None,
    severity_by_event_id: dict[int, StoredFireSeverityAssessment] | None = None,
    spread_by_key: dict[tuple[int, int], StoredFireSpreadPrediction] | None = None,
    targets_by_event_id: dict[int, StoredResponseTargetSet] | None = None,
    stations: tuple[FakeStationRow, ...] = (),
    resources: tuple[FakeResourceRow, ...] = (),
    hotspots_by_id: dict[int, StoredSatelliteHotspot] | None = None,
    reports_by_id: dict[int, StoredWildfireReport] | None = None,
    plan_by_event_id: dict[int, ResponsePlanDetails] | None = None,
):
    fire_event_repository = FakeFireEventRepository(events_by_id)
    severity_repository = FakeFireSeverityAssessmentRepository(severity_by_event_id)
    spread_repository = FakeFireSpreadPredictionRepository(spread_by_key)
    target_repository = FakeResponseTargetRepository(targets_by_event_id)
    station_repository = FakeFireStationRepository(stations)
    resource_repository = FakeFirefightingResourceRepository(resources)
    satellite_hotspot_repository = FakeSatelliteHotspotRepository(hotspots_by_id)
    news_repository = FakeNewsRepository(reports_by_id)
    plan_details_service = FakeResponsePlanDetailsService(plan_by_event_id)

    service = EventDetailsService(
        fire_event_repository=fire_event_repository,
        fire_severity_assessment_repository=severity_repository,
        fire_spread_prediction_repository=spread_repository,
        response_target_repository=target_repository,
        fire_station_repository=station_repository,
        firefighting_resource_repository=resource_repository,
        satellite_hotspot_repository=satellite_hotspot_repository,
        news_repository=news_repository,
        response_plan_details_service=plan_details_service,
    )
    fakes = {
        "fire_event": fire_event_repository,
        "severity": severity_repository,
        "spread": spread_repository,
        "hotspots": satellite_hotspot_repository,
        "news": news_repository,
        "targets": target_repository,
        "stations": station_repository,
        "resources": resource_repository,
        "plan": plan_details_service,
    }
    return service, fakes


# ---------------------------------------------------------------------------
# Missing FireEvent
# ---------------------------------------------------------------------------


def test_unknown_fire_event_returns_none_not_error():
    service, _ = make_service(events_by_id={})

    result = service.get_event_details(999, as_of=AS_OF)

    assert result is None


def test_lookup_by_id_uses_given_fire_event_id():
    stored_event = make_stored_event(1)
    service, fakes = make_service(events_by_id={1: stored_event})

    service.get_event_details(1, as_of=AS_OF)

    assert fakes["fire_event"].calls == [1]


# ---------------------------------------------------------------------------
# FireEvent core fields
# ---------------------------------------------------------------------------


def test_fire_event_fields_are_copied_verbatim():
    stored_event = make_stored_event(3, status=FireEventStatus.SUSPECTED, latitude=31.5, longitude=34.9)
    service, _ = make_service(events_by_id={3: stored_event})

    result = service.get_event_details(3, as_of=AS_OF)

    assert result.fire_event.fire_event_id == 3
    assert result.fire_event.status is FireEventStatus.SUSPECTED
    assert result.fire_event.latitude == pytest.approx(31.5)
    assert result.fire_event.longitude == pytest.approx(34.9)
    assert result.fire_event.detected_at == DETECTED_AT
    assert result.fire_event.updated_at == UPDATED_AT


# ---------------------------------------------------------------------------
# Danger (always None - no FireEvent<->area linkage exists yet)
# ---------------------------------------------------------------------------


def test_danger_is_always_none():
    stored_event = make_stored_event(1)
    service, _ = make_service(events_by_id={1: stored_event})

    result = service.get_event_details(1, as_of=AS_OF)

    assert result.danger is None


# ---------------------------------------------------------------------------
# Detection evidence
# ---------------------------------------------------------------------------


def test_satellite_evidence_is_resolved_and_mapped():
    stored_event = make_stored_event(
        1, supporting_evidence=(FireEvidenceRef(FireEvidenceType.SATELLITE, 501),)
    )
    hotspot = make_stored_hotspot(501, confidence="high", frp=15.0, satellite="Terra")
    service, fakes = make_service(events_by_id={1: stored_event}, hotspots_by_id={501: hotspot})

    result = service.get_event_details(1, as_of=AS_OF)

    assert len(result.detection_evidence.satellite) == 1
    assert result.detection_evidence.news == []
    satellite = result.detection_evidence.satellite[0]
    assert satellite.id == 501
    assert satellite.confidence == "high"
    assert satellite.frp == pytest.approx(15.0)
    assert satellite.satellite == "Terra"
    assert fakes["hotspots"].calls == [501]


def test_news_evidence_is_resolved_and_mapped():
    stored_event = make_stored_event(1, supporting_evidence=(FireEvidenceRef(FireEvidenceType.NEWS, 701),))
    report = make_stored_report(701, title="Blaze reported near reserve", source_feed="haaretz")
    service, fakes = make_service(events_by_id={1: stored_event}, reports_by_id={701: report})

    result = service.get_event_details(1, as_of=AS_OF)

    assert result.detection_evidence.satellite == []
    assert len(result.detection_evidence.news) == 1
    news = result.detection_evidence.news[0]
    assert news.id == 701
    assert news.title == "Blaze reported near reserve"
    assert news.source == "haaretz"
    assert news.observed_at == DETECTED_AT
    assert fakes["news"].calls == [701]


def test_mixed_satellite_and_news_evidence_are_both_mapped():
    stored_event = make_stored_event(
        1,
        supporting_evidence=(
            FireEvidenceRef(FireEvidenceType.SATELLITE, 501),
            FireEvidenceRef(FireEvidenceType.NEWS, 701),
        ),
    )
    hotspot = make_stored_hotspot(501)
    report = make_stored_report(701)
    service, _ = make_service(
        events_by_id={1: stored_event}, hotspots_by_id={501: hotspot}, reports_by_id={701: report}
    )

    result = service.get_event_details(1, as_of=AS_OF)

    assert len(result.detection_evidence.satellite) == 1
    assert len(result.detection_evidence.news) == 1


def test_no_supporting_evidence_returns_empty_detection_evidence():
    stored_event = make_stored_event(1, supporting_evidence=())
    service, _ = make_service(events_by_id={1: stored_event})

    result = service.get_event_details(1, as_of=AS_OF)

    assert result.detection_evidence.satellite == []
    assert result.detection_evidence.news == []


def test_evidence_ref_pointing_to_missing_row_is_omitted_not_errored():
    stored_event = make_stored_event(
        1,
        supporting_evidence=(
            FireEvidenceRef(FireEvidenceType.SATELLITE, 999),
            FireEvidenceRef(FireEvidenceType.NEWS, 888),
        ),
    )
    service, _ = make_service(events_by_id={1: stored_event}, hotspots_by_id={}, reports_by_id={})

    result = service.get_event_details(1, as_of=AS_OF)

    assert result.detection_evidence.satellite == []
    assert result.detection_evidence.news == []


# ---------------------------------------------------------------------------
# Severity linkage
# ---------------------------------------------------------------------------


def test_severity_present_is_mapped():
    stored_event = make_stored_event(1)
    stored_severity = make_stored_severity(101, 1, score=88.0, level=FireSeverityLevel.CRITICAL)
    service, _ = make_service(events_by_id={1: stored_event}, severity_by_event_id={1: stored_severity})

    result = service.get_event_details(1, as_of=AS_OF)

    assert result.severity is not None
    assert result.severity.assessment_id == 101
    assert result.severity.score == pytest.approx(88.0)
    assert result.severity.level is FireSeverityLevel.CRITICAL


def test_severity_absent_is_none():
    stored_event = make_stored_event(1)
    service, _ = make_service(events_by_id={1: stored_event}, severity_by_event_id={})

    result = service.get_event_details(1, as_of=AS_OF)

    assert result.severity is None


def test_insufficient_data_severity_is_reported_honestly():
    stored_event = make_stored_event(1)
    stored_severity = make_stored_severity(
        101, 1, status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA, score=None, level=None
    )
    service, _ = make_service(events_by_id={1: stored_event}, severity_by_event_id={1: stored_severity})

    result = service.get_event_details(1, as_of=AS_OF)

    assert result.severity.status is FireSeverityAssessmentStatus.INSUFFICIENT_DATA
    assert result.severity.score is None
    assert result.severity.level is None


# ---------------------------------------------------------------------------
# Current-state spread contract (critical)
# ---------------------------------------------------------------------------


def test_valid_spread_prediction_includes_its_cells():
    stored_event = make_stored_event(1)
    prediction = make_valid_prediction(1, horizon_minutes=30, cell_count=2)
    service, _ = make_service(events_by_id={1: stored_event}, spread_by_key={(1, 30): prediction})

    result = service.get_event_details(1, as_of=AS_OF)

    by_horizon = {item.horizon_minutes: item for item in result.spread_predictions}
    assert by_horizon[30].status is FireSpreadPredictionStatus.VALID
    assert len(by_horizon[30].cells) == 2


@pytest.mark.parametrize(
    "status", [FireSpreadPredictionStatus.INSUFFICIENT_DATA, FireSpreadPredictionStatus.INACTIVE_EVENT]
)
def test_non_valid_spread_prediction_returns_empty_cells_never_a_fallback(status):
    """The critical US 6.2 contract: insufficient_data/inactive_event must
    yield cells=[] - never a substitution of an older VALID run."""
    stored_event = make_stored_event(1)
    prediction = make_not_valid_prediction(1, horizon_minutes=30, status=status)
    service, _ = make_service(events_by_id={1: stored_event}, spread_by_key={(1, 30): prediction})

    result = service.get_event_details(1, as_of=AS_OF)

    by_horizon = {item.horizon_minutes: item for item in result.spread_predictions}
    assert by_horizon[30].status is status
    assert by_horizon[30].cells == []


def test_spread_prediction_service_never_asks_for_anything_but_the_latest():
    """Guards against the service adding its own "find a valid one" fallback
    query - it must only ever call the single latest-for-horizon lookup."""
    stored_event = make_stored_event(1)
    service, fakes = make_service(events_by_id={1: stored_event})

    service.get_event_details(1, as_of=AS_OF)

    assert set(fakes["spread"].calls) == {(1, 30), (1, 60)}


def test_horizon_with_no_persisted_prediction_is_omitted():
    stored_event = make_stored_event(1)
    prediction_30 = make_valid_prediction(1, horizon_minutes=30, cell_count=1)
    service, _ = make_service(events_by_id={1: stored_event}, spread_by_key={(1, 30): prediction_30})

    result = service.get_event_details(1, as_of=AS_OF)

    assert [item.horizon_minutes for item in result.spread_predictions] == [30]


def test_no_spread_predictions_at_all_returns_empty_list():
    stored_event = make_stored_event(1)
    service, _ = make_service(events_by_id={1: stored_event})

    result = service.get_event_details(1, as_of=AS_OF)

    assert result.spread_predictions == []


# ---------------------------------------------------------------------------
# Response targets
# ---------------------------------------------------------------------------


def test_targets_present_are_mapped():
    stored_event = make_stored_event(1)
    target_set = make_target_set(1)
    service, fakes = make_service(events_by_id={1: stored_event}, targets_by_event_id={1: target_set})

    result = service.get_event_details(1, as_of=AS_OF)

    assert len(result.targets) == 1
    assert result.targets[0].target_type is ResponseTargetType.ACTIVE_FIRE
    assert fakes["targets"].calls == [(1, AS_OF)]


def test_targets_absent_returns_empty_list():
    stored_event = make_stored_event(1)
    service, _ = make_service(events_by_id={1: stored_event}, targets_by_event_id={})

    result = service.get_event_details(1, as_of=AS_OF)

    assert result.targets == []


# ---------------------------------------------------------------------------
# Stations / resources
# ---------------------------------------------------------------------------


def test_stations_and_resources_are_mapped():
    stored_event = make_stored_event(1)
    station = FakeStationRow(id="S1", name="Central Station", latitude=32.0, longitude=34.8, station_type="urban")
    resource = FakeResourceRow(id="R1", station_id="S1", status=ResourceStatus.AVAILABLE)
    service, _ = make_service(
        events_by_id={1: stored_event}, stations=(station,), resources=(resource,)
    )

    result = service.get_event_details(1, as_of=AS_OF)

    assert len(result.stations) == 1
    assert result.stations[0].station_id == "S1"
    assert result.stations[0].name == "Central Station"
    assert len(result.resources) == 1
    assert result.resources[0].resource_id == "R1"
    assert result.resources[0].status is ResourceStatus.AVAILABLE


def test_no_stations_or_resources_returns_empty_lists():
    stored_event = make_stored_event(1)
    service, _ = make_service(events_by_id={1: stored_event})

    result = service.get_event_details(1, as_of=AS_OF)

    assert result.stations == []
    assert result.resources == []


# ---------------------------------------------------------------------------
# Station summaries
# ---------------------------------------------------------------------------


def test_station_summary_counts_resources_by_status():
    stored_event = make_stored_event(1)
    station = FakeStationRow(id="S1", name="Central Station", latitude=32.0, longitude=34.8)
    resources = (
        FakeResourceRow(id="R1", station_id="S1", status=ResourceStatus.AVAILABLE),
        FakeResourceRow(id="R2", station_id="S1", status=ResourceStatus.AVAILABLE),
        FakeResourceRow(id="R3", station_id="S1", status=ResourceStatus.ASSIGNED),
        FakeResourceRow(id="R4", station_id="S1", status=ResourceStatus.UNAVAILABLE),
    )
    service, _ = make_service(events_by_id={1: stored_event}, stations=(station,), resources=resources)

    result = service.get_event_details(1, as_of=AS_OF)

    assert len(result.station_summaries) == 1
    summary = result.station_summaries[0]
    assert summary.station_id == "S1"
    assert summary.total_resources == 4
    assert summary.available == 2
    assert summary.assigned_status == 1
    assert summary.unavailable == 1


def test_station_summary_present_for_station_with_no_resources():
    stored_event = make_stored_event(1)
    station = FakeStationRow(id="S1", name="Empty Station", latitude=32.0, longitude=34.8)
    service, _ = make_service(events_by_id={1: stored_event}, stations=(station,), resources=())

    result = service.get_event_details(1, as_of=AS_OF)

    assert len(result.station_summaries) == 1
    summary = result.station_summaries[0]
    assert summary.total_resources == 0
    assert summary.available == 0
    assert summary.assigned_status == 0
    assert summary.unavailable == 0
    assert summary.current_global_plan_allocations == []


def test_station_summary_lists_current_plan_allocations_for_its_station():
    stored_event = make_stored_event(1)
    station_1 = FakeStationRow(id="S1", name="Station One", latitude=32.0, longitude=34.8)
    station_2 = FakeStationRow(id="S2", name="Station Two", latitude=32.1, longitude=34.9)
    resource_1 = FakeResourceRow(id="R1", station_id="S1", status=ResourceStatus.ASSIGNED)
    resource_2 = FakeResourceRow(id="R2", station_id="S2", status=ResourceStatus.AVAILABLE)
    plan = make_plan_details(1)  # single action: resource_id="R1", station_id="S1"
    service, _ = make_service(
        events_by_id={1: stored_event},
        stations=(station_1, station_2),
        resources=(resource_1, resource_2),
        plan_by_event_id={1: plan},
    )

    result = service.get_event_details(1, as_of=AS_OF)

    by_station = {summary.station_id: summary for summary in result.station_summaries}
    assert len(by_station["S1"].current_global_plan_allocations) == 1
    allocation = by_station["S1"].current_global_plan_allocations[0]
    assert allocation.resource_id == "R1"
    assert allocation.fire_event_id == 1
    assert allocation.response_plan_id == 77
    assert by_station["S2"].current_global_plan_allocations == []


def test_station_summary_allocations_empty_when_no_current_plan():
    stored_event = make_stored_event(1)
    station = FakeStationRow(id="S1", name="Station One", latitude=32.0, longitude=34.8)
    resource = FakeResourceRow(id="R1", station_id="S1", status=ResourceStatus.AVAILABLE)
    service, _ = make_service(
        events_by_id={1: stored_event}, stations=(station,), resources=(resource,), plan_by_event_id={}
    )

    result = service.get_event_details(1, as_of=AS_OF)

    assert result.station_summaries[0].current_global_plan_allocations == []


def test_no_stations_returns_empty_station_summaries():
    stored_event = make_stored_event(1)
    service, _ = make_service(events_by_id={1: stored_event})

    result = service.get_event_details(1, as_of=AS_OF)

    assert result.station_summaries == []


# ---------------------------------------------------------------------------
# Current response plan (delegated to ResponsePlanDetailsService)
# ---------------------------------------------------------------------------


def test_current_response_plan_present_is_mapped_with_actions():
    stored_event = make_stored_event(1)
    plan = make_plan_details(1)
    service, fakes = make_service(events_by_id={1: stored_event}, plan_by_event_id={1: plan})

    result = service.get_event_details(1, as_of=AS_OF)

    assert result.current_response_plan is not None
    assert result.current_response_plan.plan_id == 77
    assert len(result.current_response_plan.actions) == 1
    assert result.current_response_plan.actions[0].resource_id == "R1"
    assert result.current_response_plan.actions[0].node_path == [1, 2, 3]
    assert fakes["plan"].calls == [1]


def test_current_response_plan_absent_is_none():
    stored_event = make_stored_event(1)
    service, _ = make_service(events_by_id={1: stored_event}, plan_by_event_id={})

    result = service.get_event_details(1, as_of=AS_OF)

    assert result.current_response_plan is None


def test_baseline_comparison_present_is_mapped():
    stored_event = make_stored_event(1)
    plan = make_plan_details(1, with_baseline=True)
    service, _ = make_service(events_by_id={1: stored_event}, plan_by_event_id={1: plan})

    result = service.get_event_details(1, as_of=AS_OF)

    comparison = result.current_response_plan.baseline_comparison
    assert comparison is not None
    assert comparison.baseline_score == pytest.approx(70.0)
    assert comparison.improvement_percentage == pytest.approx(28.5)


def test_baseline_comparison_absent_is_none():
    stored_event = make_stored_event(1)
    plan = make_plan_details(1, with_baseline=False)
    service, _ = make_service(events_by_id={1: stored_event}, plan_by_event_id={1: plan})

    result = service.get_event_details(1, as_of=AS_OF)

    assert result.current_response_plan.baseline_comparison is None


# ---------------------------------------------------------------------------
# as_of handling
# ---------------------------------------------------------------------------


def test_explicit_as_of_is_used_verbatim():
    stored_event = make_stored_event(1)
    service, _ = make_service(events_by_id={1: stored_event})

    result = service.get_event_details(1, as_of=AS_OF)

    assert result.as_of == AS_OF


def test_as_of_defaults_to_current_time_when_omitted():
    stored_event = make_stored_event(1)
    service, _ = make_service(events_by_id={1: stored_event})

    before = datetime.now(timezone.utc)
    result = service.get_event_details(1)
    after = datetime.now(timezone.utc)

    assert before <= result.as_of <= after


def test_naive_as_of_is_rejected():
    stored_event = make_stored_event(1)
    service, _ = make_service(events_by_id={1: stored_event})

    with pytest.raises(ValueError):
        service.get_event_details(1, as_of=datetime(2026, 9, 17, 10, 0))


@pytest.mark.parametrize("bad_id", [0, -1, "1", 1.5, True])
def test_invalid_fire_event_id_is_rejected(bad_id):
    service, _ = make_service()

    with pytest.raises(ValueError):
        service.get_event_details(bad_id, as_of=AS_OF)


# ---------------------------------------------------------------------------
# Read-only guarantee
# ---------------------------------------------------------------------------


def test_service_source_calls_no_write_operation():
    import inspect

    source = inspect.getsource(EventDetailsService)
    for forbidden in (".save(", ".update(", ".delete(", ".create_event(", "update_event", "save_assessment"):
        assert forbidden not in source


def test_fakes_expose_no_write_methods():
    for fake in (
        FakeFireEventRepository(),
        FakeFireSeverityAssessmentRepository(),
        FakeFireSpreadPredictionRepository(),
        FakeResponseTargetRepository(),
        FakeFireStationRepository(),
        FakeFirefightingResourceRepository(),
        FakeSatelliteHotspotRepository(),
        FakeNewsRepository(),
        FakeResponsePlanDetailsService(),
    ):
        for forbidden in (
            "save_assessment",
            "create_event",
            "update_event",
            "save_prediction",
            "save_target_set",
            "save_hotspot",
            "save_report",
        ):
            assert not hasattr(fake, forbidden)


def test_repeated_calls_are_idempotent_and_side_effect_free():
    stored_event = make_stored_event(1)
    service, _ = make_service(events_by_id={1: stored_event})

    first = service.get_event_details(1, as_of=AS_OF)
    second = service.get_event_details(1, as_of=AS_OF)

    assert first == second


# ---------------------------------------------------------------------------
# Architecture guard: no HTTP framework beyond the DTOs it must build,
# no agents, no external providers, no recalculation engines
# ---------------------------------------------------------------------------


def test_service_does_not_import_agent_calculator_or_external_provider_modules():
    forbidden_fragments = (
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
    path = Path("backend/src/services/fire_event_read/event_details_service.py")
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

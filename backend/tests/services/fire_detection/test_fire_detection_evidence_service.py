"""Tests for active-wildfire evidence retrieval and grouping."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_detection.fire_detection_config import MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.models.fire_report import WildfireReport
from src.models.satellite_hotspot import SatelliteHotspot
from src.repositories.news_repository import StoredWildfireReport
from src.repositories.satellite_hotspot_repository import StoredSatelliteHotspot
from src.services.fire_detection.fire_detection_evidence_config import EVIDENCE_LOOKBACK_MINUTES
from src.services.fire_detection.fire_detection_evidence_service import FireDetectionEvidenceService

AS_OF = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
BASE_LATITUDE = 32.794
BASE_LONGITUDE = 34.9896


class FakeSatelliteRepository:
    def __init__(self, hotspots=()):
        self.hotspots = list(hotspots)
        self.by_id = {hotspot.id: hotspot for hotspot in self.hotspots}
        self.calls = []
        self.get_by_id_calls = []

    def get_recent_hotspots(self, as_of, lookback_minutes):
        self.calls.append((as_of, lookback_minutes))
        return list(self.hotspots)

    def get_by_id(self, hotspot_id):
        self.get_by_id_calls.append(hotspot_id)
        return self.by_id.get(hotspot_id)


class FakeNewsRepository:
    def __init__(self, reports=()):
        self.reports = list(reports)
        self.by_id = {report.id: report for report in self.reports}
        self.calls = []
        self.get_by_id_calls = []

    def get_recent_reports(self, as_of, lookback_minutes):
        self.calls.append((as_of, lookback_minutes))
        return list(self.reports)

    def get_by_id(self, report_id):
        self.get_by_id_calls.append(report_id)
        return self.by_id.get(report_id)


def make_hotspot(
    *,
    evidence_id=1,
    latitude=BASE_LATITUDE,
    longitude=BASE_LONGITUDE,
    detected_at=AS_OF - timedelta(minutes=10),
    confidence="n",
):
    return StoredSatelliteHotspot(
        id=evidence_id,
        hotspot=SatelliteHotspot(
            latitude=latitude,
            longitude=longitude,
            detected_at=detected_at,
            confidence=confidence,
        ),
    )


def make_report(
    *,
    evidence_id=1,
    latitude=BASE_LATITUDE,
    longitude=BASE_LONGITUDE,
    observed_at=AS_OF - timedelta(minutes=10),
    published_at=None,
    fetched_at=None,
):
    effective_fetched_at = fetched_at or observed_at
    return StoredWildfireReport(
        id=evidence_id,
        observed_at=observed_at,
        report=WildfireReport(
            source_url=f"https://example.com/report-{evidence_id}",
            source_feed="Example",
            title="Wildfire reported",
            summary="Smoke reported near forest.",
            location_name="Haifa",
            latitude=latitude,
            longitude=longitude,
            published_at=published_at,
            fetched_at=effective_fetched_at,
        ),
    )


def make_service(satellite=(), news=()):
    satellite_repository = FakeSatelliteRepository(satellite)
    news_repository = FakeNewsRepository(news)
    service = FireDetectionEvidenceService(
        satellite_repository=satellite_repository,
        news_repository=news_repository,
    )
    return service, satellite_repository, news_repository


def evidence_keys(candidates):
    return [
        [(evidence.evidence_type, evidence.evidence_id) for evidence in candidate.evidence]
        for candidate in candidates
    ]


def test_build_candidates_uses_configured_lookback_for_both_repositories():
    service, satellite_repository, news_repository = make_service()

    assert service.build_candidates(AS_OF) == ()

    assert satellite_repository.calls == [(AS_OF, EVIDENCE_LOOKBACK_MINUTES)]
    assert news_repository.calls == [(AS_OF, EVIDENCE_LOOKBACK_MINUTES)]


@pytest.mark.parametrize(
    ("raw_confidence", "normalized"),
    [
        ("l", "low"),
        ("low", "low"),
        ("n", "nominal"),
        ("nominal", "nominal"),
        ("h", "high"),
        ("high", "high"),
        (" H ", "high"),
    ],
)
def test_satellite_confidence_is_normalized(raw_confidence, normalized):
    service, _, _ = make_service(satellite=[make_hotspot(confidence=raw_confidence)])

    candidates = service.build_candidates(AS_OF)

    assert candidates[0].evidence[0].satellite_confidence == normalized


@pytest.mark.parametrize("confidence", [None, "", "medium", "42"])
def test_satellite_with_unsupported_confidence_is_skipped(confidence):
    service, _, _ = make_service(satellite=[make_hotspot(confidence=confidence)])

    assert service.build_candidates(AS_OF) == ()


def test_news_evidence_uses_stored_observed_at_and_has_no_satellite_confidence():
    observed_at = AS_OF - timedelta(minutes=42)
    service, _, _ = make_service(news=[make_report(observed_at=observed_at)])

    evidence = service.build_candidates(AS_OF)[0].evidence[0]

    assert evidence.evidence_type is FireEvidenceType.NEWS
    assert evidence.observed_at == observed_at
    assert evidence.satellite_confidence is None


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [
        (None, BASE_LONGITUDE),
        (BASE_LATITUDE, None),
        (None, None),
    ],
)
def test_news_without_coordinates_is_skipped(latitude, longitude):
    service, _, _ = make_service(news=[make_report(latitude=latitude, longitude=longitude)])

    assert service.build_candidates(AS_OF) == ()


def test_nearby_satellite_and_news_are_one_candidate():
    service, _, _ = make_service(
        satellite=[make_hotspot(evidence_id=10)],
        news=[make_report(evidence_id=20, latitude=BASE_LATITUDE + 0.01)],
    )

    candidates = service.build_candidates(AS_OF)

    assert len(candidates) == 1
    assert evidence_keys(candidates) == [[(FireEvidenceType.SATELLITE, 10), (FireEvidenceType.NEWS, 20)]]


def test_distant_evidence_becomes_separate_candidates():
    service, _, _ = make_service(
        satellite=[make_hotspot(evidence_id=1)],
        news=[make_report(evidence_id=2, latitude=33.5, longitude=35.5)],
    )

    candidates = service.build_candidates(AS_OF)

    assert len(candidates) == 2
    assert evidence_keys(candidates) == [[(FireEvidenceType.SATELLITE, 1)], [(FireEvidenceType.NEWS, 2)]]


def test_temporally_distant_evidence_becomes_separate_candidates():
    service, _, _ = make_service(
        satellite=[make_hotspot(evidence_id=1, detected_at=AS_OF - timedelta(minutes=1))],
        news=[make_report(evidence_id=2, observed_at=AS_OF - timedelta(minutes=62))],
    )

    assert len(service.build_candidates(AS_OF)) == 2


def test_exact_time_boundary_is_correlated():
    service, _, _ = make_service(
        satellite=[make_hotspot(evidence_id=1, detected_at=AS_OF)],
        news=[make_report(evidence_id=2, observed_at=AS_OF - timedelta(minutes=MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES))],
    )

    assert len(service.build_candidates(AS_OF)) == 1


def test_one_second_beyond_time_boundary_is_not_correlated():
    service, _, _ = make_service(
        satellite=[make_hotspot(evidence_id=1, detected_at=AS_OF)],
        news=[
            make_report(
                evidence_id=2,
                observed_at=AS_OF - timedelta(minutes=MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES, seconds=1),
            )
        ],
    )

    assert len(service.build_candidates(AS_OF)) == 2


def test_evidence_inside_distance_boundary_is_correlated():
    service, _, _ = make_service(
        satellite=[make_hotspot(evidence_id=1)],
        news=[make_report(evidence_id=2, latitude=BASE_LATITUDE + 0.044)],
    )

    assert len(service.build_candidates(AS_OF)) == 1


def test_evidence_outside_distance_boundary_is_not_correlated():
    service, _, _ = make_service(
        satellite=[make_hotspot(evidence_id=1)],
        news=[make_report(evidence_id=2, latitude=BASE_LATITUDE + 0.047)],
    )

    assert len(service.build_candidates(AS_OF)) == 2


def test_transitive_connected_component_becomes_one_candidate():
    service, _, _ = make_service(
        satellite=[
            make_hotspot(evidence_id=1, latitude=BASE_LATITUDE),
            make_hotspot(evidence_id=2, latitude=BASE_LATITUDE + 0.035),
            make_hotspot(evidence_id=3, latitude=BASE_LATITUDE + 0.070),
        ],
    )

    candidates = service.build_candidates(AS_OF)

    assert len(candidates) == 1
    assert evidence_keys(candidates) == [
        [
            (FireEvidenceType.SATELLITE, 1),
            (FireEvidenceType.SATELLITE, 2),
            (FireEvidenceType.SATELLITE, 3),
        ]
    ]


def test_multiple_incidents_are_grouped_separately():
    service, _, _ = make_service(
        satellite=[
            make_hotspot(evidence_id=1, latitude=32.794, longitude=34.9896),
            make_hotspot(evidence_id=2, latitude=33.186, longitude=35.570),
        ],
        news=[
            make_report(evidence_id=3, latitude=32.800, longitude=34.990),
            make_report(evidence_id=4, latitude=33.190, longitude=35.571),
        ],
    )

    candidates = service.build_candidates(AS_OF)

    assert len(candidates) == 2
    assert evidence_keys(candidates) == [
        [(FireEvidenceType.SATELLITE, 1), (FireEvidenceType.NEWS, 3)],
        [(FireEvidenceType.SATELLITE, 2), (FireEvidenceType.NEWS, 4)],
    ]


def test_invalid_rows_do_not_prevent_valid_candidates():
    service, _, _ = make_service(
        satellite=[
            make_hotspot(evidence_id=1, confidence="medium"),
            make_hotspot(evidence_id=2, confidence="h"),
        ],
        news=[make_report(evidence_id=3, latitude=None)],
    )

    candidates = service.build_candidates(AS_OF)

    assert evidence_keys(candidates) == [[(FireEvidenceType.SATELLITE, 2)]]


def test_candidates_are_deterministic_with_unsorted_repository_results():
    service, _, _ = make_service(
        satellite=[
            make_hotspot(evidence_id=3, detected_at=AS_OF - timedelta(minutes=5)),
            make_hotspot(evidence_id=1, detected_at=AS_OF - timedelta(minutes=15)),
        ],
        news=[
            make_report(evidence_id=2, observed_at=AS_OF - timedelta(minutes=10)),
            make_report(evidence_id=4, observed_at=AS_OF - timedelta(minutes=20), latitude=33.5),
        ],
    )

    first = service.build_candidates(AS_OF)
    second = service.build_candidates(AS_OF)

    assert evidence_keys(first) == evidence_keys(second)
    assert evidence_keys(first) == [
        [(FireEvidenceType.NEWS, 4)],
        [(FireEvidenceType.SATELLITE, 1), (FireEvidenceType.NEWS, 2), (FireEvidenceType.SATELLITE, 3)],
    ]


def test_overlapping_numeric_ids_are_not_deduplicated_across_source_types():
    service, _, _ = make_service(
        satellite=[make_hotspot(evidence_id=5)],
        news=[make_report(evidence_id=5)],
    )

    candidates = service.build_candidates(AS_OF)

    assert evidence_keys(candidates) == [[(FireEvidenceType.SATELLITE, 5), (FireEvidenceType.NEWS, 5)]]


def test_duplicate_identity_from_same_source_is_deduplicated():
    first = make_hotspot(evidence_id=5, detected_at=AS_OF - timedelta(minutes=20))
    second = replace(first, hotspot=SatelliteHotspot(BASE_LATITUDE, BASE_LONGITUDE, AS_OF, confidence="h"))
    service, _, _ = make_service(satellite=[first, second])

    candidates = service.build_candidates(AS_OF)

    assert evidence_keys(candidates) == [[(FireEvidenceType.SATELLITE, 5)]]
    assert candidates[0].evidence[0].observed_at == AS_OF - timedelta(minutes=20)


def test_build_candidates_does_not_call_fire_detection_calculator(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("FireDetectionEvidenceService must not call FireDetectionCalculator")

    monkeypatch.setattr(FireDetectionCalculator, "evaluate", fail_if_called)
    service, _, _ = make_service(satellite=[make_hotspot()])

    assert isinstance(service.build_candidates(AS_OF)[0], FireDetectionCandidate)


def test_as_of_must_be_timezone_aware():
    service, _, _ = make_service()

    with pytest.raises(ValueError):
        service.build_candidates(datetime(2026, 9, 12, 12, 0))


def test_resolve_evidence_refs_reloads_satellite_ref():
    service, satellite_repository, _ = make_service(satellite=[make_hotspot(evidence_id=5, confidence="h")])

    resolved = service.resolve_evidence_refs((FireEvidenceRef(FireEvidenceType.SATELLITE, 5),))

    assert satellite_repository.get_by_id_calls == [5]
    assert resolved[0].evidence_type is FireEvidenceType.SATELLITE
    assert resolved[0].satellite_confidence == "high"


def test_resolve_evidence_refs_reloads_news_ref():
    observed_at = AS_OF - timedelta(minutes=7)
    service, _, news_repository = make_service(news=[make_report(evidence_id=6, observed_at=observed_at)])

    resolved = service.resolve_evidence_refs((FireEvidenceRef(FireEvidenceType.NEWS, 6),))

    assert news_repository.get_by_id_calls == [6]
    assert resolved[0].evidence_type is FireEvidenceType.NEWS
    assert resolved[0].observed_at == observed_at


def test_resolve_evidence_refs_preserves_overlapping_numeric_ids():
    service, _, _ = make_service(
        satellite=[make_hotspot(evidence_id=5)],
        news=[make_report(evidence_id=5)],
    )

    resolved = service.resolve_evidence_refs(
        (
            FireEvidenceRef(FireEvidenceType.SATELLITE, 5),
            FireEvidenceRef(FireEvidenceType.NEWS, 5),
        )
    )

    assert [(item.evidence_type, item.evidence_id) for item in resolved] == [
        (FireEvidenceType.SATELLITE, 5),
        (FireEvidenceType.NEWS, 5),
    ]


def test_resolve_evidence_refs_returns_deterministic_order():
    service, _, _ = make_service(
        satellite=[make_hotspot(evidence_id=3, detected_at=AS_OF - timedelta(minutes=5))],
        news=[make_report(evidence_id=2, observed_at=AS_OF - timedelta(minutes=10))],
    )

    resolved = service.resolve_evidence_refs(
        (
            FireEvidenceRef(FireEvidenceType.SATELLITE, 3),
            FireEvidenceRef(FireEvidenceType.NEWS, 2),
        )
    )

    assert [(item.evidence_type, item.evidence_id) for item in resolved] == [
        (FireEvidenceType.NEWS, 2),
        (FireEvidenceType.SATELLITE, 3),
    ]


def test_resolve_evidence_refs_missing_ref_fails_explicitly():
    service, _, _ = make_service()

    with pytest.raises(ValueError):
        service.resolve_evidence_refs((FireEvidenceRef(FireEvidenceType.SATELLITE, 999),))


def test_resolve_evidence_refs_unsupported_satellite_confidence_fails():
    service, _, _ = make_service(satellite=[make_hotspot(evidence_id=5, confidence="medium")])

    with pytest.raises(ValueError):
        service.resolve_evidence_refs((FireEvidenceRef(FireEvidenceType.SATELLITE, 5),))


def test_resolve_evidence_refs_uses_news_observed_at_fallback_from_repository():
    fetched_at = AS_OF - timedelta(minutes=3)
    service, _, _ = make_service(
        news=[make_report(evidence_id=8, observed_at=fetched_at, published_at=None, fetched_at=fetched_at)]
    )

    resolved = service.resolve_evidence_refs((FireEvidenceRef(FireEvidenceType.NEWS, 8),))

    assert resolved[0].observed_at == fetched_at


def test_resolve_evidence_refs_does_not_call_calculator(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("resolve_evidence_refs must not call FireDetectionCalculator")

    monkeypatch.setattr(FireDetectionCalculator, "evaluate", fail_if_called)
    service, _, _ = make_service(satellite=[make_hotspot(evidence_id=5)])

    assert service.resolve_evidence_refs((FireEvidenceRef(FireEvidenceType.SATELLITE, 5),))

"""Task 9C: evidence refs are resolved with ONE query per evidence family (history retrieval was N+1 on a remote database)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.models import SatelliteHotspot
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.models.fire_report import WildfireReport
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength
from src.repositories.exceptions import NewsRepositoryError
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.fire_detection import FireDetectionEvidenceService

T0 = datetime(2026, 9, 14, 6, 0, tzinfo=timezone.utc)


class Counting:
    """Wraps a repository and counts the lookup methods the evidence service may call."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = {"get_by_id": 0, "get_by_ids": 0}

    def get_by_id(self, value):
        self.calls["get_by_id"] += 1
        return self._inner.get_by_id(value)

    def get_by_ids(self, values):
        self.calls["get_by_ids"] += 1
        return self._inner.get_by_ids(values)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def seed(session_factory, hotspots=5, reports=3):
    satellites, news = SatelliteHotspotRepository(session_factory=session_factory), NewsRepository(session_factory=session_factory)
    for index in range(hotspots):
        satellites.save_hotspot(SatelliteHotspot(latitude=32.7 + index * 0.001, longitude=35.0, detected_at=T0 + timedelta(minutes=index),
                                                 confidence="n", frp=5.0, brightness=330.0, satellite="NOAA-20", instrument="VIIRS"))
    for index in range(reports):
        news.save_report(WildfireReport(source_url=f"https://example.com/{index}", source_feed="f", title="t", summary="s", location_name=None,
                                        latitude=32.7, longitude=35.0, published_at=T0 + timedelta(minutes=index), fetched_at=T0,
                                        wildfire_signal_strength=NewsWildfireSignalStrength.WEAK))
    hotspot_ids = [h.id for h in satellites.get_recent_hotspots(T0 + timedelta(hours=1), 600)]
    report_ids = [r.id for r in news.get_recent_reports(T0 + timedelta(hours=1), 600)]
    return satellites, news, hotspot_ids, report_ids


def test_many_refs_cost_one_query_per_family_and_return_the_same_evidence(sqlite_session_factory):
    satellites, news, hotspot_ids, report_ids = seed(sqlite_session_factory)
    counted_satellites, counted_news = Counting(satellites), Counting(news)
    service = FireDetectionEvidenceService(satellite_repository=counted_satellites, news_repository=counted_news)
    refs = tuple(FireEvidenceRef(FireEvidenceType.SATELLITE, i) for i in hotspot_ids) + tuple(FireEvidenceRef(FireEvidenceType.NEWS, i) for i in report_ids)

    evidence = service.resolve_evidence_refs(refs)

    assert len(evidence) == 8
    assert counted_satellites.calls == {"get_by_id": 0, "get_by_ids": 1} and counted_news.calls == {"get_by_id": 0, "get_by_ids": 1}
    reference = FireDetectionEvidenceService(satellite_repository=satellites, news_repository=news)
    assert evidence == reference.resolve_evidence_refs(refs)  # identical to the previous per-ref result / ordering


def test_a_missing_ref_still_raises_value_error(sqlite_session_factory):
    satellites, news, hotspot_ids, _ = seed(sqlite_session_factory, hotspots=1, reports=0)
    service = FireDetectionEvidenceService(satellite_repository=satellites, news_repository=news)
    with pytest.raises(ValueError, match="Satellite evidence ref was not found"):
        service.resolve_evidence_refs((FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_ids[0]), FireEvidenceRef(FireEvidenceType.SATELLITE, 9999)))
    with pytest.raises(ValueError, match="News evidence ref was not found"):
        service.resolve_evidence_refs((FireEvidenceRef(FireEvidenceType.NEWS, 9999),))


def test_repositories_without_a_batch_method_fall_back_to_per_ref_lookups(sqlite_session_factory):
    satellites, news, hotspot_ids, report_ids = seed(sqlite_session_factory, hotspots=2, reports=1)

    class SingleOnly:
        def __init__(self, inner):
            self._inner = inner
            self.calls = 0

        def get_by_id(self, value):
            self.calls += 1
            return self._inner.get_by_id(value)

    single_satellites, single_news = SingleOnly(satellites), SingleOnly(news)
    service = FireDetectionEvidenceService(satellite_repository=single_satellites, news_repository=single_news)

    evidence = service.resolve_evidence_refs(
        tuple(FireEvidenceRef(FireEvidenceType.SATELLITE, i) for i in hotspot_ids) + (FireEvidenceRef(FireEvidenceType.NEWS, report_ids[0]),))

    assert len(evidence) == 3 and single_satellites.calls == 2 and single_news.calls == 1


def test_news_get_by_ids_batches_dedupes_omits_missing_and_orders_by_id(sqlite_session_factory):
    _, news, _, report_ids = seed(sqlite_session_factory, hotspots=0, reports=3)

    found = news.get_by_ids((report_ids[2], report_ids[0], report_ids[0], 9999))

    assert [stored.id for stored in found] == sorted({report_ids[0], report_ids[2]})
    assert news.get_by_ids(()) == ()
    with pytest.raises(NewsRepositoryError):
        news.get_by_ids((0,))

"""End-to-end news ingestion: RSS entry -> real TextProcessor -> geocode -> translate -> real NewsRepository.

Everything is real except the network: the LLM HTTP call is mocked (no API key or network is needed),
the geocoder is a stub, and the database is an in-memory SQLite. It then reads the persisted reports
back through the Fire Detection evidence service to prove the analysed wildfire signal strength really
reaches detection evidence. It changes no Fire Detection decision logic.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json as jsonlib
import re
from unittest.mock import create_autospec, patch

import pytest
from sqlalchemy import select

from src.agents.collection.news_monitoring_agent import NewsMonitoringAgent
from src.database.models.wildfire_report_db import WildfireReportDB
from src.external.geocoding.geocoding_client import Geocoder
from src.external.news.news_client import RSSFetcher, TextProcessor
from src.models.fire_evidence_type import FireEvidenceType
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength
from src.repositories.news_repository import NewsRepository
from src.services.fire_detection.fire_detection_evidence_service import FireDetectionEvidenceService

LLM_CONFIG = {
    "provider": "groq",
    "model": "test-model",
    "api_key_env": "TEST_NEWS_LLM_API_KEY",
    "temperature": 0.0,
    "max_tokens": 200,
    "translation_max_tokens": 600,
    "request_timeout_seconds": 15,
}
PUBLISHED = "Sat, 12 Sep 2026 10:00:00 GMT"
AS_OF = datetime(2026, 9, 12, 10, 30, tzinfo=timezone.utc)

# Hebrew article title -> the LLM's analysis reply (None = the LLM call fails for that article).
ANALYSIS_BY_TITLE = {
    "שריפה גדולה בכרמל": {"locationName": "כרמל", "wildfireSignalStrength": "strong"},
    "עשן נראה ליד חיפה, שריפה?": {"locationName": "חיפה", "wildfireSignalStrength": "weak"},
    "שריפה במקום לא ידוע": None,
}
TRANSLATIONS = {
    "שריפה גדולה בכרמל": ("Large fire in Carmel", "Firefighters en route.", "Carmel"),
    "עשן נראה ליד חיפה, שריפה?": ("Smoke near Haifa, a fire?", "Unconfirmed.", "Haifa"),
    "שריפה במקום לא ידוע": ("Fire somewhere", "No location.", None),
}
COORDINATES = {"כרמל": (32.73, 35.05), "חיפה": (32.79, 34.99)}


def rss_entries():
    return [
        {"source_feed": "Ynet", "title": "שריפה גדולה בכרמל", "summary": "כוחות כיבוי בדרך", "link": "https://example.com/1", "published": PUBLISHED},
        {"source_feed": "Mako", "title": "עשן נראה ליד חיפה, שריפה?", "summary": "לא מאושר", "link": "https://example.com/2", "published": PUBLISHED},
        {"source_feed": "Walla News", "title": "שריפה במקום לא ידוע", "summary": "ללא מיקום", "link": "https://example.com/3", "published": PUBLISHED},
        {"source_feed": "Ynet", "title": "תוצאות המשחק", "summary": "ספורט", "link": "https://example.com/4", "published": PUBLISHED},  # irrelevant
    ]


class FakeLLM:
    """Stands in for the Groq HTTP endpoint; records every call by kind."""

    def __init__(self):
        self.analysis_calls: list[str] = []
        self.translation_calls: list[str] = []

    def post(self, url, headers=None, json=None, timeout=None, **_kwargs):  # noqa: A002 - mirrors requests.post
        prompt = json["messages"][0]["content"]
        title = re.search(r"^Title: (.*)$", prompt, re.MULTILINE).group(1)
        if "You are translating" in prompt:
            self.translation_calls.append(title)
            translated_title, translated_summary, location = TRANSLATIONS[title]
            payload = {"title": translated_title, "summary": translated_summary, "locationName": location}
        else:
            self.analysis_calls.append(title)
            payload = ANALYSIS_BY_TITLE[title]
            if payload is None:
                raise RuntimeError("LLM unavailable for this article")
        return _Response({"choices": [{"message": {"content": _dumps(payload)}}]})


def _dumps(payload) -> str:
    return jsonlib.dumps(payload, ensure_ascii=False)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture()
def flow(sqlite_session_factory, monkeypatch):
    monkeypatch.setenv("TEST_NEWS_LLM_API_KEY", "fake-key")
    monkeypatch.setattr("src.external.news.news_client.time.sleep", lambda _seconds: None)
    llm = FakeLLM()
    rss = create_autospec(RSSFetcher, instance=True)
    rss.fetch_all.return_value = rss_entries()
    geocoder = create_autospec(Geocoder, instance=True)
    geocoder.geocode.side_effect = lambda name: COORDINATES.get(name, (None, None))
    repository = NewsRepository(session_factory=sqlite_session_factory)
    agent = NewsMonitoringAgent(
        rss_fetcher=rss,
        text_processor=TextProcessor(keywords=["שריפה"], llm_config=LLM_CONFIG),
        geocoder=geocoder,
        news_repository=repository,
        enable_file_logging=False,
    )
    with patch("src.external.news.news_client.requests.post", side_effect=llm.post):
        yield agent, repository, llm, geocoder, sqlite_session_factory


def stored_rows(session_factory):
    session = session_factory()
    try:
        return session.execute(select(WildfireReportDB).order_by(WildfireReportDB.id)).scalars().all()
    finally:
        session.close()


def test_relevant_articles_are_analysed_translated_and_persisted_with_their_signal(flow):
    agent, _repository, llm, _geocoder, session_factory = flow

    saved_count = agent.run_once()

    assert saved_count == 3  # the sports article is irrelevant
    rows = {row.source_url: row for row in stored_rows(session_factory)}
    assert set(rows) == {"https://example.com/1", "https://example.com/2", "https://example.com/3"}
    carmel = rows["https://example.com/1"]
    assert carmel.wildfire_signal_strength == "strong"
    assert (carmel.title, carmel.location_name) == ("Large fire in Carmel", "Carmel")
    assert (carmel.latitude, carmel.longitude) == COORDINATES["כרמל"]
    assert carmel.source_feed == "Ynet"
    assert rows["https://example.com/2"].wildfire_signal_strength == "weak"
    assert llm.analysis_calls.count("שריפה גדולה בכרמל") == 1  # one analysis call per article


def test_an_llm_failure_for_one_article_stores_it_with_an_unknown_signal_and_the_rest_are_unaffected(flow):
    agent, _repository, _llm, geocoder, session_factory = flow

    agent.run_once()

    unknown = next(row for row in stored_rows(session_factory) if row.source_url == "https://example.com/3")
    assert unknown.wildfire_signal_strength is None  # NULL = unknown, never fabricated as "none"
    assert unknown.location_name is None and unknown.latitude is None and unknown.longitude is None
    assert unknown.title == "Fire somewhere"  # translation still worked
    geocoder.geocode.assert_any_call(None)
    strengths = {row.source_url: row.wildfire_signal_strength for row in stored_rows(session_factory)}
    assert strengths["https://example.com/1"] == "strong" and strengths["https://example.com/2"] == "weak"


def test_a_second_cycle_does_not_reprocess_or_duplicate_known_urls(flow):
    agent, _repository, llm, _geocoder, session_factory = flow
    assert agent.run_once() == 3
    analysis_calls_after_first_cycle = len(llm.analysis_calls)

    assert agent.run_once() == 0

    assert len(stored_rows(session_factory)) == 3
    assert len(llm.analysis_calls) == analysis_calls_after_first_cycle  # known URLs are skipped before any LLM call


def test_the_persisted_signal_strength_reaches_fire_detection_evidence(flow):
    agent, repository, _llm, _geocoder, _session_factory = flow
    agent.run_once()

    class NoSatellites:
        def get_recent_hotspots(self, as_of, lookback_minutes):
            return []

    service = FireDetectionEvidenceService(satellite_repository=NoSatellites(), news_repository=repository)
    candidates = service.build_candidates(AS_OF)

    evidence = [item for candidate in candidates for item in candidate.evidence]
    assert all(item.evidence_type is FireEvidenceType.NEWS for item in evidence)
    strengths = sorted(item.news_wildfire_signal_strength.value for item in evidence)
    assert strengths == ["strong", "weak"]  # the article without coordinates cannot be evidence
    strong = next(item for item in evidence if item.news_wildfire_signal_strength is NewsWildfireSignalStrength.STRONG)
    assert (round(strong.latitude, 2), round(strong.longitude, 2)) == COORDINATES["כרמל"]

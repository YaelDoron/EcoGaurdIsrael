"""Simulated News: readable Hebrew source text, translation fallback, and schedule.

Root cause of the unreadable "Dyvchym Rashvnym Al Mvkdy Ash..." reports was
the frontend's letter-by-letter romanization fallback (stationTranslations.ts)
applied to this generator's Hebrew text after the backend's Groq translation
failed. These tests pin the backend half of the contract: the generator emits
readable Hebrew Unicode, and a translation failure persists that Hebrew as-is.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone

import pytest

from src.external.news.news_client import TextProcessor
from src.simulation import SIMULATION_LOCATIONS, ScenarioType
from src.simulation.generators import news_data_generator as ndg
from src.simulation.generators.news_data_generator import NewsDataGenerator
from src.simulation.simulation_event import SimulationEventType
from src.simulation.simulation_scenario import build_operations_demo_scenario

TIMESTAMP = datetime(2026, 9, 25, 9, 0, tzinfo=timezone.utc)
HEBREW_LETTER = re.compile(r"[א-ת]")
LATIN_LETTER = re.compile(r"[A-Za-z]")
ALL_TEMPLATES = (
    *ndg._INITIAL_TITLE_TEMPLATES,
    *ndg._INITIAL_SUMMARY_TEMPLATES,
    *ndg._FOLLOW_UP_TITLE_TEMPLATES,
    *ndg._FOLLOW_UP_SUMMARY_TEMPLATES,
    *ndg._LOCATION_REPORT_NAMES.values(),
)


@pytest.mark.parametrize("text", ALL_TEMPLATES)
def test_every_template_is_hebrew_unicode_not_latin_transliteration(text):
    assert HEBREW_LETTER.search(text)
    assert not LATIN_LETTER.search(text.replace("{location}", ""))


@pytest.mark.parametrize("location_key", sorted(SIMULATION_LOCATIONS))
@pytest.mark.parametrize("report_index", [0, 1])
def test_generated_report_is_readable_hebrew_for_every_location(location_key, report_index):
    location = SIMULATION_LOCATIONS[location_key]
    report = NewsDataGenerator(seed=7).generate(ScenarioType.ACTIVE_FIRE, TIMESTAMP, location, report_index).reports[0]

    for text in (report.title, report.summary, report.location_name):
        assert HEBREW_LETTER.search(text)
        assert not LATIN_LETTER.search(text)
        text.encode("utf-8")  # valid Unicode, round-trippable


def test_same_seed_is_deterministic_and_different_seed_may_vary_text():
    location = SIMULATION_LOCATIONS["jerusalem_forest"]
    first = NewsDataGenerator(seed=11).generate(ScenarioType.ACTIVE_FIRE, TIMESTAMP, location).reports[0]
    again = NewsDataGenerator(seed=11).generate(ScenarioType.ACTIVE_FIRE, TIMESTAMP, location).reports[0]
    assert first == again

    variants = {
        NewsDataGenerator(seed=seed).generate(ScenarioType.ACTIVE_FIRE, TIMESTAMP, location).reports[0].title
        for seed in range(20)
    }
    assert len(variants) > 1


def test_translation_failure_keeps_readable_hebrew(monkeypatch):
    monkeypatch.setenv("TEST_NEWS_LLM_KEY", "not-a-real-key")
    processor = TextProcessor(
        keywords=[],
        llm_config={"provider": "groq", "model": "test-model", "api_key_env": "TEST_NEWS_LLM_KEY"},
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("401 invalid_api_key")

    monkeypatch.setattr(processor, "_call_llm", fail)
    report = NewsDataGenerator(seed=3).generate(
        ScenarioType.ACTIVE_FIRE, TIMESTAMP, SIMULATION_LOCATIONS["jerusalem_forest"]
    ).reports[0]

    title, summary, location_name = processor.translate_report(report.title, report.summary, report.location_name)

    assert (title, summary, location_name) == (report.title, report.summary, report.location_name)
    assert not LATIN_LETTER.search(title)


@pytest.mark.parametrize("seed", [1, 7, 42, 1105103406])
def test_operations_demo_schedules_two_news_reports_per_active_fire(seed):
    scenario = build_operations_demo_scenario(seed=seed)
    active_ids = {i.incident_id for i in scenario.incidents if i.scenario_type is ScenarioType.ACTIVE_FIRE}
    news_per_incident = Counter(e.incident_id for e in scenario.events if e.event_type is SimulationEventType.NEWS)

    assert active_ids  # operations_demo always has 1-4 active fires
    assert news_per_incident == {incident_id: 2 for incident_id in active_ids}
    # News is never scheduled for risk-only incidents, and never at T+0 (it follows weather+satellite).
    assert all(e.offset_seconds > 0 for e in scenario.events if e.event_type is SimulationEventType.NEWS)

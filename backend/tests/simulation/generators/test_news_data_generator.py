"""Tests for NewsDataGenerator."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import random
from urllib.parse import urlparse

import pytest

from src.models.fire_report import WildfireReport
from src.simulation import DEFAULT_CARMEL_LOCATION, SIMULATION_LOCATIONS, ScenarioType
from src.simulation.generators.news_data_generator import (
    SIMULATED_NEWS_MAX_RADIUS_KM,
    SIMULATED_NEWS_SOURCE_FEED,
    GeneratedNewsData,
    NewsDataGenerator,
)

TIMESTAMP = datetime(2026, 9, 12, 14, 0, 40, tzinfo=timezone.utc)
PROHIBITED_TERMS = (
    "הוזעקו",
    "נשלחו",
    "הוקצו",
    "כבאיות נשלחו",
    "כוחות הגיעו",
    "dispatch",
    "dispatched",
    "allocated",
)


def distance_km(first_latitude: float, first_longitude: float, second_latitude: float, second_longitude: float) -> float:
    earth_radius_km = 6371.0
    lat1 = math.radians(first_latitude)
    lon1 = math.radians(first_longitude)
    lat2 = math.radians(second_latitude)
    lon2 = math.radians(second_longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def assert_no_prohibited_wording(report: WildfireReport) -> None:
    text = f"{report.title} {report.summary}".lower()
    for term in PROHIBITED_TERMS:
        assert term.lower() not in text


def assert_report_near_location(report: WildfireReport, location) -> None:
    assert report.latitude is not None
    assert report.longitude is not None
    assert distance_km(location.latitude, location.longitude, report.latitude, report.longitude) <= SIMULATED_NEWS_MAX_RADIUS_KM


@pytest.mark.parametrize("scenario_type", [ScenarioType.LOW_RISK_NO_FIRE, ScenarioType.HIGH_RISK_NO_FIRE])
def test_no_fire_scenarios_generate_no_reports_for_every_location(scenario_type):
    generator = NewsDataGenerator(seed=42)

    for location in SIMULATION_LOCATIONS.values():
        generated = generator.generate(
            scenario_type=scenario_type,
            timestamp=TIMESTAMP,
            location=location,
            report_index=0,
        )

        assert generated == GeneratedNewsData(reports=())


def test_active_fire_initial_report_uses_existing_domain_model_and_simulated_source():
    generated = NewsDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=TIMESTAMP,
        location=DEFAULT_CARMEL_LOCATION,
        report_index=0,
    )

    assert isinstance(generated, GeneratedNewsData)
    assert len(generated.reports) == 1
    report = generated.reports[0]
    assert isinstance(report, WildfireReport)
    assert report.published_at is TIMESTAMP
    assert report.fetched_at is TIMESTAMP
    assert report.source_feed == SIMULATED_NEWS_SOURCE_FEED
    assert urlparse(report.source_url).netloc == "simulation.ecoguard.local"
    assert "active_fire" in report.source_url
    assert "carmel" in report.source_url
    assert "report-1" in report.source_url
    assert report.location_name == "הכרמל"
    assert_report_near_location(report, DEFAULT_CARMEL_LOCATION)
    assert "ראשוני" in f"{report.title} {report.summary}" or "עשן" in report.title
    assert_no_prohibited_wording(report)


def test_active_fire_follow_up_report_is_related_but_distinct():
    generator = NewsDataGenerator(seed=42)
    initial = generator.generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=TIMESTAMP,
        location=DEFAULT_CARMEL_LOCATION,
        report_index=0,
    ).reports[0]
    follow_up_timestamp = TIMESTAMP + timedelta(seconds=60)
    follow_up = generator.generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=follow_up_timestamp,
        location=DEFAULT_CARMEL_LOCATION,
        report_index=1,
    ).reports[0]

    assert follow_up.source_url != initial.source_url
    assert "report-2" in follow_up.source_url
    assert follow_up.location_name == initial.location_name
    assert follow_up.published_at is follow_up_timestamp
    assert follow_up.fetched_at is follow_up_timestamp
    assert_report_near_location(follow_up, DEFAULT_CARMEL_LOCATION)
    assert "נוס" in f"{follow_up.title} {follow_up.summary}" or "ממשיכה" in f"{follow_up.title} {follow_up.summary}"
    assert_no_prohibited_wording(follow_up)


def test_prohibited_dispatch_wording_never_appears_in_generated_reports():
    generator = NewsDataGenerator(seed=42)

    for location in SIMULATION_LOCATIONS.values():
        for report_index in range(4):
            report = generator.generate(
                scenario_type=ScenarioType.ACTIVE_FIRE,
                timestamp=TIMESTAMP + timedelta(minutes=report_index),
                location=location,
                report_index=report_index,
            ).reports[0]
            assert_no_prohibited_wording(report)


def test_same_seed_and_inputs_generate_identical_report_across_instances():
    first = NewsDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=TIMESTAMP,
        location=DEFAULT_CARMEL_LOCATION,
        report_index=0,
    )
    second = NewsDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=TIMESTAMP,
        location=DEFAULT_CARMEL_LOCATION,
        report_index=0,
    )

    assert first == second


def test_report_index_changes_url_text_and_coordinates_but_keeps_same_incident_area():
    generator = NewsDataGenerator(seed=42)
    initial = generator.generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=TIMESTAMP,
        location=DEFAULT_CARMEL_LOCATION,
        report_index=0,
    ).reports[0]
    follow_up = generator.generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=TIMESTAMP,
        location=DEFAULT_CARMEL_LOCATION,
        report_index=1,
    ).reports[0]

    assert initial.source_url != follow_up.source_url
    assert initial.title != follow_up.title
    assert (initial.latitude, initial.longitude) != (follow_up.latitude, follow_up.longitude)
    assert_report_near_location(initial, DEFAULT_CARMEL_LOCATION)
    assert_report_near_location(follow_up, DEFAULT_CARMEL_LOCATION)


def test_generator_does_not_mutate_global_random_sequence():
    random.seed(12345)
    expected_first = random.random()
    expected_second = random.random()

    random.seed(12345)
    actual_first = random.random()
    NewsDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=TIMESTAMP,
        location=DEFAULT_CARMEL_LOCATION,
        report_index=0,
    )
    actual_second = random.random()

    assert actual_first == expected_first
    assert actual_second == expected_second


def test_active_fire_report_is_location_aware_for_every_predefined_location():
    generator = NewsDataGenerator(seed=42)

    for key, location in SIMULATION_LOCATIONS.items():
        report = generator.generate(
            scenario_type=ScenarioType.ACTIVE_FIRE,
            timestamp=TIMESTAMP,
            location=location,
            report_index=0,
        ).reports[0]

        assert key in report.source_url
        assert report.location_name is not None
        assert report.location_name in report.title or report.location_name in report.summary
        assert_report_near_location(report, location)


def test_invalid_inputs_are_rejected():
    generator = NewsDataGenerator(seed=42)

    with pytest.raises(ValueError):
        NewsDataGenerator(seed=True)
    with pytest.raises(ValueError):
        generator.generate("active_fire", TIMESTAMP, DEFAULT_CARMEL_LOCATION, 0)
    with pytest.raises(ValueError):
        generator.generate(ScenarioType.ACTIVE_FIRE, "2026-09-12", DEFAULT_CARMEL_LOCATION, 0)
    with pytest.raises(ValueError):
        generator.generate(ScenarioType.ACTIVE_FIRE, TIMESTAMP, location="Carmel", report_index=0)
    with pytest.raises(ValueError):
        generator.generate(ScenarioType.ACTIVE_FIRE, TIMESTAMP, DEFAULT_CARMEL_LOCATION, report_index=-1)

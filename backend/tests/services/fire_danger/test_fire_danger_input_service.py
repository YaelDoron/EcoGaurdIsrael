"""Tests for fire-danger weather input selection and aggregation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import pytest

from src.models import (
    AssessmentArea,
    FireDangerInputStatus,
)
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.weather_repository import StoredWeatherObservation
from src.services.fire_danger.fire_danger_input_config import MAX_WEATHER_AGE_MINUTES
from src.services.fire_danger.fire_danger_input_service import (
    FireDangerInputService,
    haversine_distance_km,
)

AS_OF = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)
AREA = AssessmentArea(id="area-carmel", name="Carmel", latitude=32.731, longitude=35.046, radius_km=5)


class FakeWeatherRepository:
    def __init__(self, records: list[StoredWeatherObservation]) -> None:
        self.records = records
        self.calls = []

    def get_recent_observations_for_area_candidates(
        self,
        latitude,
        longitude,
        radius_km,
        start_time,
        end_time,
    ):
        self.calls.append(
            {
                "latitude": latitude,
                "longitude": longitude,
                "radius_km": radius_km,
                "start_time": start_time,
                "end_time": end_time,
            }
        )
        return self.records


def make_record(
    station_id: int,
    observation_id: int,
    minutes_old: float = 5,
    latitude: float = AREA.latitude,
    longitude: float = AREA.longitude,
    temperature: float | None = 30.0,
    relative_humidity: float | None = 25.0,
    wind_speed: float | None = 12.0,
    wind_gust: float | None = 18.0,
    rainfall: float | None = 0.0,
) -> StoredWeatherObservation:
    external_station_id = 900000 + station_id
    return StoredWeatherObservation(
        observation_id=observation_id,
        station_id=station_id,
        station=WeatherStation(
            external_station_id=external_station_id,
            name=f"Station {station_id}",
            latitude=latitude,
            longitude=longitude,
        ),
        observation=WeatherObservation(
            station_external_id=external_station_id,
            timestamp=AS_OF - timedelta(minutes=minutes_old),
            temperature=temperature,
            relative_humidity=relative_humidity,
            wind_speed=wind_speed,
            wind_gust=wind_gust,
            rainfall=rainfall,
        ),
    )


def build_result(records: list[StoredWeatherObservation], area: AssessmentArea = AREA):
    repository = FakeWeatherRepository(records)
    result = FireDangerInputService(weather_repository=repository).build_input(area=area, as_of=AS_OF)
    return result, repository


def test_one_fresh_complete_station_produces_ready():
    result, repository = build_result([make_record(station_id=1, observation_id=101)])

    assert result.status is FireDangerInputStatus.READY
    assert result.input_data.temperature_c == 30.0
    assert result.observation_ids == (101,)
    assert result.station_ids == (1,)
    assert repository.calls[0]["start_time"] == AS_OF - timedelta(minutes=MAX_WEATHER_AGE_MINUTES)
    assert repository.calls[0]["end_time"] == AS_OF


def test_multiple_fresh_stations_are_averaged_correctly():
    result, _ = build_result(
        [
            make_record(1, 101, temperature=20, relative_humidity=30, wind_speed=10),
            make_record(2, 102, temperature=30, relative_humidity=40, wind_speed=20),
        ]
    )

    assert result.input_data.temperature_c == 25
    assert result.input_data.relative_humidity_pct == 35
    assert result.input_data.wind_speed_kmh == 15


def test_three_station_values_use_arithmetic_mean_without_rounding():
    result, _ = build_result(
        [
            make_record(1, 101, temperature=10, relative_humidity=10, wind_speed=1),
            make_record(2, 102, temperature=20, relative_humidity=20, wind_speed=2),
            make_record(3, 103, temperature=31, relative_humidity=31, wind_speed=4),
        ]
    )

    assert result.input_data.temperature_c == pytest.approx(61 / 3)
    assert result.input_data.relative_humidity_pct == pytest.approx(61 / 3)
    assert result.input_data.wind_speed_kmh == pytest.approx(7 / 3)


def test_latest_valid_observation_per_station_is_selected():
    result, _ = build_result(
        [
            make_record(1, 101, minutes_old=20, temperature=10),
            make_record(1, 102, minutes_old=5, temperature=30),
        ]
    )

    assert result.input_data.temperature_c == 30
    assert result.observation_ids == (102,)


def test_older_observation_from_same_station_is_ignored_when_newer_valid_exists():
    result, _ = build_result(
        [
            make_record(1, 101, minutes_old=29, temperature=5),
            make_record(1, 102, minutes_old=1, temperature=25),
            make_record(2, 201, minutes_old=2, temperature=35),
        ]
    )

    assert result.observation_ids == (102, 201)
    assert result.input_data.temperature_c == 30


def test_observation_exactly_max_age_is_accepted():
    result, _ = build_result([make_record(1, 101, minutes_old=MAX_WEATHER_AGE_MINUTES)])

    assert result.status is FireDangerInputStatus.READY


def test_observation_slightly_older_than_max_age_is_rejected():
    result, _ = build_result([make_record(1, 101, minutes_old=MAX_WEATHER_AGE_MINUTES + 0.01)])

    assert result.status is FireDangerInputStatus.INSUFFICIENT_DATA
    assert result.input_data is None


def test_future_observation_relative_to_as_of_is_not_used():
    result, _ = build_result([make_record(1, 101, minutes_old=-1)])

    assert result.status is FireDangerInputStatus.INSUFFICIENT_DATA


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": None},
        {"relative_humidity": None},
        {"wind_speed": None},
    ],
)
def test_missing_mandatory_weather_fields_make_observation_invalid(overrides):
    result, _ = build_result([make_record(1, 101, **overrides)])

    assert result.status is FireDangerInputStatus.INSUFFICIENT_DATA


@pytest.mark.parametrize("optional_overrides", [{"rainfall": None}, {"wind_gust": None}])
def test_missing_optional_weather_fields_do_not_invalidate_observation(optional_overrides):
    result, _ = build_result([make_record(1, 101, **optional_overrides)])

    assert result.status is FireDangerInputStatus.READY


def test_no_stations_in_area_returns_insufficient_data():
    result, _ = build_result([])

    assert result.status is FireDangerInputStatus.INSUFFICIENT_DATA
    assert result.input_data is None
    assert result.observation_ids == ()
    assert result.station_ids == ()


def test_stations_exist_but_all_observations_are_stale_returns_insufficient_data():
    result, _ = build_result([make_record(1, 101, minutes_old=60)])

    assert result.status is FireDangerInputStatus.INSUFFICIENT_DATA


def test_stations_exist_but_all_observations_are_incomplete_returns_insufficient_data():
    result, _ = build_result([make_record(1, 101, temperature=None)])

    assert result.status is FireDangerInputStatus.INSUFFICIENT_DATA


def test_station_outside_area_radius_is_ignored():
    result, _ = build_result([make_record(1, 101, latitude=AREA.latitude + 1.0)])

    assert result.status is FireDangerInputStatus.INSUFFICIENT_DATA


def test_station_very_near_radius_boundary_is_included_consistently():
    radius_km = 10.0
    area = AssessmentArea(id="boundary", name="Boundary", latitude=0.0, longitude=0.0, radius_km=radius_km)
    latitude_on_boundary = math.degrees(radius_km / 6371.0088) * 0.999999

    result, _ = build_result(
        [make_record(1, 101, latitude=latitude_on_boundary, longitude=0.0)],
        area=area,
    )

    assert result.status is FireDangerInputStatus.READY


def test_invalid_observation_from_one_station_does_not_invalidate_other_valid_stations():
    result, _ = build_result(
        [
            make_record(1, 101, temperature=None),
            make_record(2, 201, temperature=33),
        ]
    )

    assert result.status is FireDangerInputStatus.READY
    assert result.observation_ids == (201,)
    assert result.station_ids == (2,)


def test_result_contains_only_ids_of_observations_actually_used():
    result, _ = build_result(
        [
            make_record(1, 101, temperature=None),
            make_record(2, 201, minutes_old=3),
            make_record(2, 202, minutes_old=20),
            make_record(3, 301, latitude=AREA.latitude + 1.0),
        ]
    )

    assert result.observation_ids == (201,)
    assert result.station_ids == (2,)


def test_ordering_of_station_ids_and_observation_ids_is_deterministic():
    result, _ = build_result(
        [
            make_record(3, 301),
            make_record(1, 101),
            make_record(2, 201),
        ]
    )

    assert result.station_ids == (1, 2, 3)
    assert result.observation_ids == (101, 201, 301)


def test_same_data_and_same_as_of_produce_identical_result():
    records = [
        make_record(2, 201, temperature=22),
        make_record(1, 101, temperature=20),
    ]

    first, _ = build_result(records)
    second, _ = build_result(records)

    assert first == second


def test_build_input_requires_timezone_aware_as_of():
    with pytest.raises(ValueError):
        FireDangerInputService(weather_repository=FakeWeatherRepository([])).build_input(
            area=AREA,
            as_of=datetime(2026, 9, 13, 12, 0),
        )


def test_haversine_same_coordinate_is_zero():
    assert haversine_distance_km(32.0, 35.0, 32.0, 35.0) == pytest.approx(0.0)


def test_haversine_known_nearby_coordinates():
    distance = haversine_distance_km(0.0, 0.0, 0.0, 1.0)

    assert distance == pytest.approx(111.2, abs=0.2)


def test_haversine_inside_radius():
    assert haversine_distance_km(0.0, 0.0, 0.0, 0.01) < 2.0


def test_haversine_outside_radius():
    assert haversine_distance_km(0.0, 0.0, 0.0, 1.0) > 100.0

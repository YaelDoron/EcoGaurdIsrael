"""Presentation seed 594: fire C (Jerusalem Forest) is confirmed and its real spread prediction propagates.

Offline chain through the production pieces (no database, no network):
seeded generators -> V5 extractor/model/policy (scripts/select_presentation_seed.py)
-> FireEvent origin = the same satellite-priority centroid the detector uses
-> FireSpreadInputService (fire C's generated weather, the severity step's
vegetation label) -> FireSpreadPredictionAgent/FireSpreadCalculator.

Only persistence is replaced by in-memory fakes. The vegetation label is the
value the production Copernicus path (CopernicusLandCoverClient +
VegetationMapper, VEGETATION_RADIUS_KM) returned for every fire C hotspot
centroid of this schedule - pinned the same way
tests/calculators/fire_spread/test_fire_spread_jerusalem_forest_audit.py pins
its observed 'Shrub cover'; re-check it live with
`python -m scripts.audit_simulation_locations --candidate ...`.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from scripts.select_presentation_seed import DAY_START, NIGHT_START, _Predictor
from src.agents.analysis.fire_spread_prediction_agent import FireSpreadPredictionAgent
from src.calculators.fire_detection.fire_detection_decision_policy import estimate_candidate_location
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.calculators.fire_spread.fire_spread_calculator import FireSpreadCalculator
from src.models import (
    FireEvent,
    FireEventStatus,
    FireSeverityAssessment,
    FireSeverityAssessmentStatus,
    FireSeverityLevel,
    FireSpreadInputStatus,
)
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_type import FireEvidenceType
from src.models.fire_spread_fuel_class import FireSpreadFuelClass
from src.models.fire_spread_prediction import PROPAGATION_THRESHOLD
from src.models.fire_spread_prediction_status import FireSpreadPredictionStatus
from src.repositories.fire_event_repository import StoredFireEvent
from src.repositories.fire_severity_assessment_repository import StoredFireSeverityAssessment
from src.repositories.fire_spread_prediction_repository import StoredFireSpreadPrediction
from src.repositories.weather_repository import StoredWeatherObservation
from src.services.fire_detection.fire_detection_evidence_service import _SATELLITE_CONFIDENCE_MAP
from src.services.fire_spread.fire_spread_input_service import FireSpreadInputService
from src.simulation.generators.satellite_data_generator import SatelliteDataGenerator
from src.simulation.generators.weather_data_generator import WeatherDataGenerator
from src.simulation.presentation_scenario import (
    GALILEE,
    JERUSALEM_FOREST_FIRE,
    JUDEAN_HILLS,
    PRESENTATION_DEFAULT_SEED,
    PRESENTATION_FIRE_C_WEATHER,
    build_presentation_demo_scenario,
)
from src.simulation.simulation_event import SimulationEventType
from src.simulation.simulation_event_executor import simulation_event_seed_key

# Production Copernicus/VegetationMapper result at fire C's hotspot centroids (see module docstring).
AUDITED_FIRE_C_LAND_COVER = "Shrub cover"
FIRE_EVENT_ID = 3
SEVERITY_ASSESSMENT_ID = 30


class _EventRepository:
    def __init__(self, stored_event: StoredFireEvent) -> None:
        self._stored_event = stored_event

    def get_by_id(self, fire_event_id: int):
        return self._stored_event if fire_event_id == self._stored_event.id else None


class _SeverityRepository:
    def __init__(self, stored: StoredFireSeverityAssessment) -> None:
        self._stored = stored

    def get_latest_for_event_as_of(self, fire_event_id: int, as_of):
        assessment = self._stored.assessment
        return self._stored if assessment.fire_event_id == fire_event_id and assessment.assessed_at <= as_of else None


class _WeatherRepository:
    def __init__(self, records: list[StoredWeatherObservation]) -> None:
        self._records = records

    def get_observations_by_ids(self, observation_ids):
        wanted = set(observation_ids)
        return tuple(record for record in self._records if record.observation_id in wanted)


class _PredictionRepository:
    def __init__(self) -> None:
        self.saved: list[StoredFireSpreadPrediction] = []

    def save_prediction(self, prediction, weather_observation_id=None):
        stored = StoredFireSpreadPrediction(
            id=len(self.saved) + 1, prediction=prediction, weather_observation_id=weather_observation_id
        )
        self.saved.append(stored)
        return stored


def _passes(scenario, incident_id: str) -> list:
    return [e for e in scenario.events if e.incident_id == incident_id and e.event_type is SimulationEventType.SATELLITE]


def _status_at(steps, offset: int) -> str | None:
    matching = [status for t, _, _, status in steps if t == offset]
    return matching[-1].upper() if matching else None


@pytest.fixture(scope="module")
def scenario():
    return build_presentation_demo_scenario(PRESENTATION_DEFAULT_SEED)


@pytest.fixture(scope="module")
def statuses(scenario):
    predictor = _Predictor()
    return {label: predictor.statuses(scenario, start) for label, start in (("day", DAY_START), ("night", NIGHT_START))}


@pytest.fixture(scope="module")
def fire_c_confirmation(scenario):
    """Everything the live pipeline would have persisted for fire C when its confirming pass is processed."""
    start = DAY_START
    first_pass, confirming_pass = _passes(scenario, JERUSALEM_FOREST_FIRE)[:2]
    incident = scenario.get_incident(JERUSALEM_FOREST_FIRE)
    as_of = start + timedelta(seconds=confirming_pass.offset_seconds)

    satellite = SatelliteDataGenerator(scenario.seed)
    hotspots = [
        hotspot
        for event in (first_pass, confirming_pass)
        for hotspot in satellite.generate(
            incident.scenario_type,
            start + timedelta(seconds=event.offset_seconds),
            incident.location,
            seed_key=simulation_event_seed_key(event),
        ).hotspots
    ]
    evidence = tuple(
        FireDetectionEvidence(
            evidence_id=index,
            evidence_type=FireEvidenceType.SATELLITE,
            latitude=hotspot.latitude,
            longitude=hotspot.longitude,
            observed_at=hotspot.detected_at,
            satellite_confidence=_SATELLITE_CONFIDENCE_MAP.get(hotspot.confidence.lower()),
        )
        for index, hotspot in enumerate(hotspots, start=1)
    )
    latitude, longitude = estimate_candidate_location(evidence)

    # Fire C's own weather updates generated before the confirmation (the executor passes the incident's profile).
    weather = WeatherDataGenerator(scenario.seed)
    records: list[StoredWeatherObservation] = []
    for event in scenario.events:
        if event.incident_id != JERUSALEM_FOREST_FIRE or event.event_type is not SimulationEventType.WEATHER:
            continue
        if event.offset_seconds > confirming_pass.offset_seconds:
            break
        generated = weather.generate(
            incident.scenario_type,
            start + timedelta(seconds=event.offset_seconds),
            incident.location,
            seed_key=simulation_event_seed_key(event),
            profile=incident.weather_profile,
        )
        for measurement in generated.measurements:
            records.append(
                StoredWeatherObservation(
                    observation_id=len(records) + 1,
                    station_id=len(records) + 1,
                    station=measurement.station,
                    observation=measurement.observation,
                )
            )

    stored_event = StoredFireEvent(
        id=FIRE_EVENT_ID,
        event=FireEvent(
            latitude=latitude,
            longitude=longitude,
            detected_at=start + timedelta(seconds=first_pass.offset_seconds),
            updated_at=as_of,
            status=FireEventStatus.CONFIRMED,
            detection_confidence=0.9,
            methodology="ECOGUARD_AI_HYBRID_DETECTION",
            methodology_version="5.0",
        ),
        supporting_evidence=(),
    )
    severity = StoredFireSeverityAssessment(
        assessment_id=SEVERITY_ASSESSMENT_ID,
        assessment=FireSeverityAssessment(
            fire_event_id=FIRE_EVENT_ID,
            assessed_at=as_of,
            status=FireSeverityAssessmentStatus.VALID,
            score=80.0,
            level=FireSeverityLevel.CRITICAL,
            methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
            methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
            vegetation_source="COPERNICUS_GLOBAL_LAND_COVER_100M_API",
            vegetation_dataset_year=2019,
            vegetation_radius_km=1.0,
            vegetation_dominant_land_cover=AUDITED_FIRE_C_LAND_COVER,
            vegetation_fuel_score=0.8,
        ),
        weather_observation_ids=tuple(record.observation_id for record in records),
        satellite_hotspot_ids=tuple(range(1, len(hotspots) + 1)),
        selected_frp_hotspot_id=1,
    )
    service = FireSpreadInputService(
        fire_event_repository=_EventRepository(stored_event),
        fire_severity_assessment_repository=_SeverityRepository(severity),
        weather_repository=_WeatherRepository(records),
    )
    return service, as_of, records


@pytest.mark.parametrize("label", ["day", "night"])
def test_all_three_opening_fires_are_confirmed(scenario, statuses, label):
    for incident_id in (JUDEAN_HILLS, GALILEE, JERUSALEM_FOREST_FIRE):
        first, second = (event.offset_seconds for event in _passes(scenario, incident_id)[:2])
        steps = statuses[label][incident_id]
        assert _status_at(steps, first) == "SUSPECTED", (label, incident_id)
        assert _status_at(steps, second) == "CONFIRMED", (label, incident_id)


def test_fire_c_uses_its_presentation_weather_profile(scenario, fire_c_confirmation):
    _, _, records = fire_c_confirmation
    assert scenario.get_incident(JERUSALEM_FOREST_FIRE).weather_profile is PRESENTATION_FIRE_C_WEATHER
    low, high = PRESENTATION_FIRE_C_WEATHER.wind_direction_deg
    assert records and all(low <= record.observation.wind_direction <= high for record in records)
    # Every other incident keeps the unchanged scenario-type weather.
    assert all(i.weather_profile is None for i in scenario.incidents if i.incident_id != JERUSALEM_FOREST_FIRE)


def test_fire_c_spread_input_is_ready(fire_c_confirmation):
    service, as_of, _ = fire_c_confirmation
    result = service.prepare_input(FIRE_EVENT_ID, as_of, 30)

    assert result.status is FireSpreadInputStatus.READY
    assert result.input_data.fuel_class is FireSpreadFuelClass.SHRUBS


@pytest.mark.parametrize("horizon_minutes", [30, 60])
def test_fire_c_prediction_is_valid_and_spreads(fire_c_confirmation, horizon_minutes):
    service, as_of, _ = fire_c_confirmation
    agent = FireSpreadPredictionAgent(
        input_service=service, calculator=FireSpreadCalculator(), repository=_PredictionRepository()
    )
    stored = agent.predict_from_input_result(
        input_result=service.prepare_input(FIRE_EVENT_ID, as_of, horizon_minutes),
        as_of=as_of,
        horizon_minutes=horizon_minutes,
    )

    prediction = stored.prediction
    spreading = [cell for cell in prediction.cells if cell.spread_probability >= PROPAGATION_THRESHOLD]
    assert prediction.status is FireSpreadPredictionStatus.VALID
    assert len(spreading) >= 2  # the fire front actually advances, not one marginal cell
    assert max(cell.reached_step for cell in spreading) == horizon_minutes // 5

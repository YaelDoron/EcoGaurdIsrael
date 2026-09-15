"""SQLAlchemy ORM models (persistence layer). See src.models for the Task 2 domain dataclasses."""

from src.database.models.fire_danger_assessment_db import FireDangerAssessmentDB
from src.database.models.fire_danger_assessment_weather_input_db import (
    FireDangerAssessmentWeatherInputDB,
)
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_event_news_evidence_db import FireEventNewsEvidenceDB
from src.database.models.fire_event_satellite_evidence_db import FireEventSatelliteEvidenceDB
from src.database.models.fire_severity_assessment_db import FireSeverityAssessmentDB
from src.database.models.fire_severity_assessment_satellite_input_db import (
    FireSeverityAssessmentSatelliteInputDB,
)
from src.database.models.fire_severity_assessment_weather_input_db import (
    FireSeverityAssessmentWeatherInputDB,
)
from src.database.models.satellite_hotspot_db import SatelliteHotspotDB
from src.database.models.weather_observation_db import WeatherObservationDB
from src.database.models.weather_station_db import WeatherStationDB
from src.database.models.wildfire_report_db import WildfireReportDB

__all__ = [
    "WeatherStationDB",
    "WeatherObservationDB",
    "SatelliteHotspotDB",
    "WildfireReportDB",
    "FireDangerAssessmentDB",
    "FireDangerAssessmentWeatherInputDB",
    "FireEventDB",
    "FireEventSatelliteEvidenceDB",
    "FireEventNewsEvidenceDB",
    "FireSeverityAssessmentDB",
    "FireSeverityAssessmentWeatherInputDB",
    "FireSeverityAssessmentSatelliteInputDB",
]

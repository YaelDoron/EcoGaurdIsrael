"""Data-collection agents (IMS weather collection, etc.)."""

from src.agents.collection.satellite_hotspot_agent import SatelliteHotspotAgent
from src.agents.collection.satellite_hotspot_collection_result import SatelliteHotspotCollectionResult
from src.agents.collection.weather_agent import WeatherAgent
from src.agents.collection.weather_collection_result import WeatherCollectionResult

__all__ = [
    "WeatherAgent",
    "WeatherCollectionResult",
    "SatelliteHotspotAgent",
    "SatelliteHotspotCollectionResult",
]

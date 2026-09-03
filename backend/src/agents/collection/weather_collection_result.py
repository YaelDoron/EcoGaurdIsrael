"""Result summary for a single WeatherAgent.collect() run."""
from dataclasses import dataclass


@dataclass
class WeatherCollectionResult:
    """Summary of one weather-data collection cycle.

    `success` is False ONLY when the cycle could not start at all (e.g.
    IMSClient.get_stations() failed) - there was no station list to process.
    If station-level failures occur but other stations were still
    processed, `success` stays True; the failures show up in the counters
    below. For example, 85 received / 82 processed / 3 failed is still a
    completed collection cycle.

    Counter semantics:
    - stations_received: number of raw station items IMS returned.
    - stations_processed: number of stations successfully mapped AND
      persisted. Each raw station contributes to exactly one of
      stations_processed / stations_failed, never both.
    - stations_failed: number of stations that failed mapping or
      persistence (observation collection was never attempted for these).
    - observations_saved: number of newly-inserted observations.
    - duplicates_skipped: number of observations that already existed for
      their (station, timestamp) - a normal condition, not a failure.
    - observations_failed: number of already-persisted stations for which
      retrieving, mapping, or persisting the observation failed.
    """

    success: bool
    stations_received: int = 0
    stations_processed: int = 0
    stations_failed: int = 0
    observations_saved: int = 0
    duplicates_skipped: int = 0
    observations_failed: int = 0
    error_message: str | None = None

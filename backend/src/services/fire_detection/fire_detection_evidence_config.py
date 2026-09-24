"""Configuration for retrieving direct active-wildfire evidence.

Three DIFFERENT time concepts live in Fire Detection; do not conflate them:

  * CANDIDATE CORRELATION (fire_detection_config.py): 5 km / 60 minutes. Answers
    "do these evidence items describe the same CURRENT candidate?". Unchanged.
  * EVIDENCE LOOKBACK (below): how far back build_candidates() reads persisted
    evidence to form current candidates.
  * EVENT EVIDENCE HISTORY (below): how much of an active FireEvent's already
    attached evidence is kept as its observation history. Evidence in a history
    may be many hours apart; it is NOT required to satisfy the 60-minute
    candidate invariant.
"""
from __future__ import annotations

EVIDENCE_LOOKBACK_MINUTES = 120

# How much of a FireEvent's attached evidence counts as its recent event history
# (FireDetectionHistoryService). Distinct from the 60-minute candidate correlation.
#
# Rationale (24 hours):
#   * a single polar-orbiting VIIRS platform (the configured VIIRS_NOAA20_NRT source)
#     revisits a location roughly twice a day (day + night, ~12 h apart), so a window
#     must span >= 24 h to ever contain 3 satellite passes - the minimum for a trend;
#   * it is longer than ACTIVE_EVENT_MATCH_WINDOW_HOURS (6, fire_event_config.py), the
#     maximum gap between updates for an event to still be "active", so history is never
#     the limiting factor for event continuity;
#   * bounded, so features derived from it describe RECENT behaviour and the amount of
#     evidence read per event stays limited.
# Must be >= the active-event match window (checked by a test).
FIRE_EVENT_EVIDENCE_HISTORY_HOURS = 24

# Satellite hotspots of ONE (satellite, instrument) whose acquisition times are within this
# many minutes of a neighbouring hotspot belong to the same observation wave ("pass").
#
# Rationale (30 minutes): FIRMS reports acq_time at minute resolution, and all fire pixels of
# one overpass over a region as small as Israel are acquired within a few minutes of each
# other, while consecutive orbits of a polar-orbiting satellite are ~100 minutes apart. 30
# minutes therefore sits well above within-pass spread and well below the orbit period.
# Limitations: assumes VIIRS-like polar-orbit cadence; a sensor with much finer cadence
# (e.g. a geostationary imager) would need its own tolerance.
SATELLITE_PASS_GAP_MINUTES = 30

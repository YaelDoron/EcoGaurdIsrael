"""Configuration for duplicate active-fire-event matching.

These are EcoGuard application parameters for matching incoming detections to
existing active FireEvent rows. They are not the same concept as evidence
correlation thresholds: evidence correlation currently uses 5 km / 60 minutes,
while an active wildfire event may persist longer than a single evidence
correlation window. Incoming detections therefore match existing active events
over this broader event-update window.
"""
from __future__ import annotations

ACTIVE_EVENT_MATCH_DISTANCE_KM = 5.0
ACTIVE_EVENT_MATCH_WINDOW_HOURS = 6

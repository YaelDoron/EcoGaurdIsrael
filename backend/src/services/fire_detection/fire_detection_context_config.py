"""Configuration for Fire Danger context lookup during Fire Detection.

Fire Danger is contextual risk, not direct evidence of an active wildfire (see
backend/docs/fire_detection_context.md). These constants only control which
already-persisted Fire Danger assessment may be reused as context; they never
influence whether a FireDetectionCandidate is created.
"""
from __future__ import annotations

# Maximum age (as_of - assessment.assessed_at) of a persisted Fire Danger
# assessment that Fire Detection may reuse as context.
#
# Default rationale (60 minutes):
#   - FireDangerInputService only accepts weather observations up to
#     MAX_WEATHER_AGE_MINUTES (30) old at assessment time, so an assessment
#     reused for up to 60 more minutes reflects weather at most ~90 minutes old.
#     FFWI is driven by slowly changing humidity/wind/temperature, so that is
#     still meaningful context for an evidence candidate.
#   - It matches MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES (60), the widest gap the
#     detection pipeline already tolerates between evidence in one candidate,
#     and is well inside EVIDENCE_LOOKBACK_MINUTES (120).
#   - In the demo simulation, weather events precede the satellite/news events
#     of the same incident by under ~2 simulated minutes, so this window is
#     never the limiting factor there.
# Older assessments are treated as missing (unavailable context), never reused.
MAX_FIRE_DANGER_ASSESSMENT_AGE_MINUTES = 60

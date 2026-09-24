# Fire Detection - Fire Danger Context

## Status

Infrastructure only. This makes persisted Fire Danger (FFWI) assessments
available to Fire Detection as a `FireDetectionContext`. **Nothing consumes it
yet**: `FireDetectionAgent`, `FireDetectionCalculator`, `FireDetectionHybridPolicy`,
the V3 feature extractor/model, `FIRE_DETECTION_DECISION_MODE`, and FireEvent
status/confidence are all unchanged. A later task (ML V4 feature extraction)
will read the context.

## Fire Danger is context, not evidence

Fire Danger says how *receptive* the weather is to a fire (Fosberg FFWI). It says
nothing about whether a fire *exists*. Therefore:

- Satellite hotspots and news reports remain the only evidence that creates a
  `FireDetectionCandidate` (and, through the decision policy, a `FireEvent`).
- A HIGH/EXTREME FFWI must never create, merge, or promote a candidate or event by
  itself. `FireDetectionContextService` only maps *(an existing candidate, as_of)*
  to a `FireDetectionContext`; it has no way to produce evidence or candidates.

## Flow

```text
Fire Danger (FireDangerAssessmentAgent -> fire_danger_assessments)
        |
        v
FireDetectionContextService.build_context(candidate, as_of)
        |
        v
FireDetectionContext(fire_danger_available, fire_danger_score, fire_danger_age_minutes, ...)
        |
        v
[future] ML V4 feature extraction
```

## Components

| Piece | File |
| --- | --- |
| Context model | `src/models/fire_detection_context.py` |
| Lookup service | `src/services/fire_detection/fire_detection_context_service.py` |
| Freshness constant | `src/services/fire_detection/fire_detection_context_config.py` |
| Repository query | `FireDangerAssessmentRepository.get_assessed_between(start, end)` |

## Geographic matching

There is no assessment-area registry: every persisted assessment embeds its own
area center and radius (`area_latitude`, `area_longitude`, `area_radius_km`). An
assessment is relevant to a candidate when the candidate's location lies inside
that circle (`haversine_distance_km <= area_radius_km`, `src/utils/geo.py`). The
candidate location is the satellite-priority centroid - the same rule
`FireDetectionCalculator` uses to place a detected event (news-only candidates use
the news centroid). No named location is hardcoded.

## Selection and freshness

Among relevant assessments with
`as_of - MAX_FIRE_DANGER_ASSESSMENT_AGE_MINUTES <= assessed_at <= as_of`, the newest
`assessed_at` wins (ties: nearest area center, then highest assessment id).
Assessments after `as_of` are never used.

`MAX_FIRE_DANGER_ASSESSMENT_AGE_MINUTES = 60`. It is separate from
`MAX_WEATHER_AGE_MINUTES` (30), which only limits the weather *inputs* of one
FFWI calculation. Reusing an assessment for up to 60 more minutes means it reflects
weather at most ~90 minutes old; FFWI changes slowly, and 60 minutes matches the
widest evidence gap detection already tolerates
(`MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES`). In the demo simulation weather precedes an
incident's satellite/news events by under ~2 simulated minutes, so it is never the
limiting factor there. An older assessment is treated as missing, never reused.

## Missing data

The context is `fire_danger_available = False` (all other fields `None`) when there
is no assessment, it is stale, no area contains the candidate, the newest relevant
assessment is `INSUFFICIENT_DATA`, or the lookup fails for any reason (the service
logs a warning and never raises for data/repository problems). A newer
`INSUFFICIENT_DATA` assessment is deliberately **not** replaced by an older VALID
one: a newer attempt found the weather inputs unreliable.

A missing value is never replaced by a fake score. FFWI `0` is a real very-low-danger
observation and is different from "unknown"; `fire_danger_available` carries that
distinction for the model.

## What the ML feature layer should use

`fire_danger_available`, `fire_danger_score`, `fire_danger_age_minutes`.
`fire_danger_level` is derived directly from the score and is kept only for
traceability (as are `fire_danger_assessment_id` / `fire_danger_assessed_at`); it
should not be added as a separate ML feature.

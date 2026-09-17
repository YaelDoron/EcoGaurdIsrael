# Fire Detection

## Purpose

Fire Detection determines whether EcoGuard has direct evidence of an active wildfire.

Fire Detection is not Fire Danger Assessment. Fire danger estimates weather-related risk before a fire is necessarily active. A high FFWI score alone cannot create a `FireEvent`.

Fire Detection is also separate from future Fire Severity Assessment. The current pipeline detects and tracks active events, but does not estimate spread, impact, suppression difficulty, terrain risk, vegetation risk, or severity.

## Direct Inputs

Current direct evidence sources:

- persisted `SatelliteHotspot` rows
- persisted geocoded `WildfireReport` rows

Weather and Fire Danger Assessment outputs are not direct fire-detection evidence. They may provide future context, but they are not used to create a `FireEvent` in the current methodology.

## Retrieval

`FireDetectionEvidenceService` retrieves recent direct evidence from repositories.

Current retrieval window:

| Setting | Value |
| --- | --- |
| `EVIDENCE_LOOKBACK_MINUTES` | `120` |

Retrieval is intentionally broader than correlation. Evidence may be retrieved for review without being combined into the same detection candidate.

## Correlation

`FireDetectionEvidenceService` groups direct evidence into candidates using spatial and temporal correlation.

Current correlation thresholds:

| Setting | Value |
| --- | --- |
| `MAX_EVIDENCE_DISTANCE_KM` | `5.0` |
| `MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES` | `60` |

Two evidence items directly correlate when they are within both thresholds. Exactly 60 minutes is included. More than 60 minutes is not correlated.

Evidence belongs to the same candidate when it forms one connected component under this direct-correlation relation, not only when every pair directly correlates. For example, if A directly correlates with B, and B directly correlates with C, then A, B, and C form one candidate even if A and C do not directly correlate with each other. `FireDetectionCandidate.is_connected` is the single authoritative implementation of this rule, and `FireDetectionEvidenceService` (grouping) and `FireDetectionCalculator` (evaluation) both defer to it, so a candidate accepted by one is always accepted by the other.

Correlation uses source-aware evidence identity. `SATELLITE` id `5` and `NEWS` id `5` are different evidence items and can both support the same decision.

## Confidence

EcoGuard assigns application-level evidence weights:

| Evidence | Weight |
| --- | --- |
| Satellite low confidence | `0.40` |
| Satellite nominal confidence | `0.60` |
| Satellite high confidence | `0.75` |
| News wildfire report | `0.50` |

Multiple source families are combined with a noisy-or style calculation:

`1 - product(1 - source_confidence)`

Repeated evidence from the same source family does not multiply confidence for that family. The strongest contribution from that source family is used for the candidate confidence calculation.

These values are EcoGuard methodology parameters, not official authority probabilities.

## Classification

Detection decisions are classified by confidence:

| Confidence range | Classification |
| --- | --- |
| `< 0.50` | `NO_EVENT` |
| `>= 0.50` and `< 0.80` | `SUSPECTED` |
| `>= 0.80` | `CONFIRMED` |

`CONFIRMED` means EcoGuard has enough multi-source evidence under this methodology. It is not an official emergency-service confirmation.

## Location

Candidate location is deterministic:

- satellite evidence is preferred when present
- otherwise news evidence is used
- when multiple rows contribute, the location is averaged by source priority group

Fire Radiative Power is retained on satellite evidence for traceability and future severity work, but it is not used for current detection confidence or location weighting.

## FireEvent Lifecycle

The current Fire Detection pipeline creates and updates `FireEvent` rows for active detections.

Active statuses:

- `SUSPECTED`
- `CONFIRMED`

Persisted statuses reserved for lifecycle compatibility:

- `RESOLVED`
- `DISMISSED`

The current agent does not automatically resolve or dismiss events.

## Duplicate Matching

Duplicate event matching is separate from evidence correlation.

Correlation decides which direct evidence belongs together in one candidate. Duplicate event matching decides whether a new decision should update an existing active `FireEvent`.

Current active event matching settings:

| Setting | Value |
| --- | --- |
| `ACTIVE_EVENT_MATCH_DISTANCE_KM` | `5.0` |
| `ACTIVE_EVENT_MATCH_WINDOW_HOURS` | `6` |

Only active events are matched for updates. Historical inactive events are not changed by the current detection agent.

## Event Update

When new evidence matches an existing active event:

- evidence references are reloaded from persistence
- existing and incoming evidence references are unioned by source-aware identity
- confidence is recalculated from the full evidence set
- evidence is not counted twice
- reprocessing the same evidence is idempotent
- `CONFIRMED` events are not downgraded to `SUSPECTED`

## Traceability

`FireEvent` traceability uses source-aware evidence references:

- `FireEvidenceRef(SATELLITE, id)`
- `FireEvidenceRef(NEWS, id)`

Persistence stores satellite and news evidence associations separately. This preserves exact provenance and avoids ambiguity when source tables contain overlapping numeric row IDs.

## Simulation

Simulation generates domain objects and persists them through the same repositories used by production.

Fire Detection simulation integration runs after direct evidence is persisted:

- simulated satellite events can create suspected or confirmed fire events
- simulated news events can update correlated active fire events
- weather simulation triggers Fire Danger Assessment, not Fire Detection
- high fire danger alone does not create a `FireEvent`

Simulation metadata such as `incident_id` is not part of production/domain Fire Detection persistence.

## Determinism

For the same persisted evidence, configuration, and `as_of` time, Fire Detection decisions are deterministic.

Evidence references are sorted by:

1. evidence type value
2. evidence id

This keeps persisted decisions and tests stable even when repository ordering changes.

## Future Severity Compatibility

The current model keeps detection evidence traceable without coupling detection to future severity logic.

Future Fire Severity Assessment can reuse linked satellite evidence, including FRP values, and may add independent inputs such as weather, vegetation, terrain, spread modeling, or infrastructure exposure.

No severity logic is implemented in Fire Detection.

## Configuration

| Parameter | Current value | Purpose |
| --- | --- | --- |
| `EVIDENCE_LOOKBACK_MINUTES` | `120` | retrieval window |
| `MAX_EVIDENCE_DISTANCE_KM` | `5.0` | evidence correlation distance |
| `MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES` | `60` | evidence correlation time |
| `SATELLITE_LOW_CONFIDENCE` | `0.40` | low satellite evidence weight |
| `SATELLITE_NOMINAL_CONFIDENCE` | `0.60` | nominal satellite evidence weight |
| `SATELLITE_HIGH_CONFIDENCE` | `0.75` | high satellite evidence weight |
| `NEWS_CONFIDENCE` | `0.50` | news evidence weight |
| `SUSPECTED_THRESHOLD` | `0.50` | suspected event threshold |
| `CONFIRMED_THRESHOLD` | `0.80` | confirmed event threshold |
| `ACTIVE_EVENT_MATCH_DISTANCE_KM` | `5.0` | existing active event update distance |
| `ACTIVE_EVENT_MATCH_WINDOW_HOURS` | `6` | existing active event update time |
| `FIRE_DETECTION_METHODOLOGY_NAME` | `ECOGUARD_MULTI_SOURCE_DETECTION` | persisted methodology name |
| `FIRE_DETECTION_METHODOLOGY_VERSION` | `1.0` | persisted methodology version |

## Acceptance Criteria Review

| Criterion | Implementation | Test evidence | Status |
| --- | --- | --- | --- |
| Satellite hotspot plus nearby news report confirms active wildfire | Evidence service, calculator, agent, repository | `test_matching_satellite_and_news_confirm_fire_event` | PASS |
| Fire danger alone does not create a fire event | Detection uses only direct satellite/news evidence | `test_high_fire_danger_without_direct_evidence_does_not_create_fire_event` | PASS |
| Duplicate/reprocessed evidence does not create duplicate events | source-aware refs, repository associations, active-event matching | `test_duplicate_satellite_reprocessing_does_not_create_second_event`, `test_reprocessing_same_confirmed_evidence_is_idempotent` | PASS |
| Distant evidence is not correlated into one candidate | calculator spatial threshold | `test_geographically_distant_evidence_is_not_combined` | PASS |
| Temporally separated evidence is not correlated into one candidate | calculator temporal threshold | `test_temporally_distant_evidence_is_not_combined`, `test_temporal_correlation_boundary_is_inclusive_only_at_sixty_minutes` | PASS |
| Weak direct evidence remains suspected, not confirmed | classification thresholds | `test_single_nominal_satellite_creates_suspected_event` | PASS |
| Source-aware identity preserves overlapping numeric IDs | `FireEvidenceRef` and separate evidence tables | `test_source_aware_evidence_refs_preserve_overlapping_numeric_ids` | PASS |
| Simulation triggers detection after direct evidence persistence | simulation coordinator and demo runner integration | `test_simulation_active_fire_satellite_then_news_updates_same_event` | PASS |

## Acceptance Summary

| ID | Scenario | Expected result | Test evidence | Status |
| --- | --- | --- | --- | --- |
| AT1 | Matching satellite plus news | one `CONFIRMED` event, confidence `0.80` | `test_matching_satellite_and_news_confirm_fire_event` | PASS |
| AT2 | Geographically distant evidence | evidence is not combined into one event | `test_geographically_distant_evidence_is_not_combined` | PASS |
| AT3 | Temporally separated evidence | evidence is not correlated into one confirmed event | `test_temporally_distant_evidence_is_not_combined` | PASS |
| AT4 | Duplicate satellite evidence | no duplicate event is created | `test_duplicate_satellite_reprocessing_does_not_create_second_event` | PASS |
| AT5 | Weak direct evidence | `SUSPECTED`, not `CONFIRMED` | `test_single_nominal_satellite_creates_suspected_event` | PASS |
| AT6 | High fire danger without direct evidence | no `FireEvent` | `test_high_fire_danger_without_direct_evidence_does_not_create_fire_event` | PASS |
| AT7 | Reprocessing same evidence | no second event and no duplicate evidence links | `test_reprocessing_same_confirmed_evidence_is_idempotent` | PASS |

## Current Limitations / Future Work

- no production scheduler yet
- no official authority confirmation feed yet
- no automatic resolve or dismiss workflow yet
- no severity classification yet
- no vegetation, terrain, fuel, spread, or infrastructure impact model yet
- no calibrated Israeli wildfire probability model yet
- no confidence calibration against historical incident outcomes yet
- no FRP-based severity use yet
- no clustering beyond the current spatial and temporal thresholds yet
- no manual analyst review workflow yet

These are current scope boundaries, not bugs.

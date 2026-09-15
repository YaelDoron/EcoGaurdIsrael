# Fire Severity Assessment

## Purpose

Fire Severity Assessment estimates the operational severity of an already active wildfire.

Fire Severity Assessment is not Fire Danger Assessment. Fire danger estimates pre-fire weather risk for an area. Fire Severity Assessment runs only after Fire Detection has created or updated an active `FireEvent`.

Fire Severity Assessment is not Fire Detection. It does not create, confirm, resolve, or dismiss fire events.

## Methodology

EcoGuard currently uses a deterministic Severity v1 calculator, implemented in `backend/src/calculators/fire_severity/`.

Inputs:

- satellite Fire Radiative Power (FRP)
- recent wind speed
- recent relative humidity
- optional vegetation fuel score

The score combines normalized factor values with fixed EcoGuard weights:

| Factor | Weight | Required |
| --- | --- | --- |
| FRP | `0.45` | yes |
| wind speed | `0.30` | yes |
| relative humidity dryness | `0.15` | yes |
| vegetation fuel score | `0.10` | no |

When vegetation is unavailable, the calculator rescales by the available required-input weight instead of treating missing vegetation as zero fuel.

## Score Range

Severity score range:

- minimum: `0`
- maximum: `100`

## EcoGuard Classification

EcoGuard classifies the severity score as:

| Score range | Level |
| --- | --- |
| `0 <= score < 25` | `LOW` |
| `25 <= score < 50` | `MODERATE` |
| `50 <= score < 75` | `HIGH` |
| `75 <= score <= 100` | `CRITICAL` |

These are EcoGuard operational thresholds. They are not an official Israeli wildfire severity scale.

## Input Requirements

Mandatory inputs:

- an active `FireEvent` with status `SUSPECTED` or `CONFIRMED`
- at least one linked recent satellite hotspot with non-null FRP
- at least one recent nearby weather observation with wind speed and relative humidity

Optional input:

- Copernicus land-cover vegetation context mapped to an EcoGuard fuel score

Missing required data persists an `INSUFFICIENT_DATA` assessment instead of creating a misleading valid severity result. Resolved or dismissed events persist an `INACTIVE_EVENT` assessment.

## Freshness

Current freshness windows:

| Input | Window |
| --- | --- |
| weather observations | `30` minutes |
| satellite hotspots | `2` hours |

Future observations are ignored. Boundary values exactly at the freshness limit remain valid.

## Vegetation

Vegetation uses Copernicus Data Space land-cover statistics through `CopernicusLandCoverClient`, then maps fractional cover categories to a compact `VegetationData` snapshot.

Copernicus configuration is provided through:

- `COPERNICUS_CLIENT_ID`
- `COPERNICUS_CLIENT_SECRET`
- `COPERNICUS_TOKEN_URL`
- `COPERNICUS_STATISTICS_URL`
- `COPERNICUS_LAND_COVER_COLLECTION_ID`
- `COPERNICUS_REQUEST_TIMEOUT`

If Copernicus is unavailable or credentials are absent, severity can still be valid when the mandatory satellite and weather inputs are present.

## Traceability

Each persisted assessment stores:

- fire event id
- assessed timestamp
- status
- score and level for valid assessments
- methodology name and version
- exact weather observation ids used
- exact satellite hotspot ids considered
- selected maximum-FRP hotspot id
- optional vegetation source, dataset year, radius, dominant land cover, and fuel score

Valid assessments require weather and satellite trace rows. Non-valid assessments keep score and level null.

## Simulation

Simulation severity integration runs after source data is persisted:

- simulated `SATELLITE` events trigger severity for affected fire events after successful Fire Detection
- simulated `WEATHER` events trigger reassessment for nearby active fire events
- simulated `NEWS` events do not trigger severity directly

Weather reassessment uses the current incident location only to find nearby active fire events. The production severity assessment itself receives only the `FireEvent` id and assessment timestamp.

## Methodology Version

Current methodology:

- name: `ECOGUARD_ACTIVE_FIRE_SEVERITY`
- version: `1.0`

## Acceptance Summary

| Requirement | Implementation | Test evidence | Status |
| --- | --- | --- | --- |
| Stronger active-fire conditions produce higher severity | `FireSeverityCalculator` | `test_stronger_fire_conditions_produce_higher_severity` | PASS |
| Missing required weather or FRP is insufficient data | `FireSeverityInputService`, `FireSeverityAssessmentAgent` | `test_missing_required_weather_does_not_produce_valid_severity`, `test_missing_required_frp_does_not_produce_valid_severity` | PASS |
| Missing vegetation does not block valid severity | optional vegetation handling | `test_missing_optional_vegetation_still_allows_valid_assessment` | PASS |
| Boundary values classify deterministically | severity thresholds | `test_level_boundaries_are_deterministic` | PASS |
| Inactive events do not receive valid severity | input service status handling | `test_inactive_fire_event_does_not_receive_valid_severity` | PASS |
| Identical inputs are deterministic | pure calculator | `test_same_inputs_produce_same_severity` | PASS |
| Persistence preserves history and traceability | repository and trace tables | `test_traceability_and_vegetation_snapshot_explain_persisted_assessment`, `test_history_is_preserved_for_changing_fire_severity` | PASS |
| Simulation triggers only for satellite and weather inputs | simulation severity coordinator | `test_active_fire_simulation_trigger_sequence_is_weather_satellite_weather_satellite_only` | PASS |

## Current Limitations / Future Work

- no terrain, slope, spread, suppression, infrastructure, or exposure model yet
- no official severity-scale calibration yet
- no production scheduler yet
- no automatic FireEvent lifecycle transitions from severity
- no analyst review workflow yet

These are current scope boundaries, not bugs.

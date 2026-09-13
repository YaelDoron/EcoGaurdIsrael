# Fire Danger Assessment

## Purpose

Fire Danger Assessment estimates the current weather-related wildfire danger for an area before a fire is necessarily active.

Fire Danger Assessment is not Fire Detection. It does not confirm, infer, or classify an active wildfire. Satellite hotspots and news reports belong to future fire-detection and evidence-fusion workflows.

## Methodology

EcoGuard currently uses the Fosberg Fire Weather Index (FFWI), implemented in `backend/src/calculators/fire_danger/`.

Inputs:

- temperature
- relative humidity
- wind speed

Internal unit conversions:

- temperature is converted from Celsius to Fahrenheit
- wind speed is converted from km/h to mph

The calculator estimates equilibrium moisture content using the standard FFWI humidity branches:

- low humidity branch for relative humidity below 10%
- mid humidity branch for relative humidity from 10% through 50%
- high humidity branch for relative humidity above 50%

The moisture value is converted into a moisture damping coefficient. The final score combines that damping coefficient with wind speed and is clamped to the public FFWI score range.

## Score Range

FFWI score range:

- minimum: 0
- maximum: 100

## EcoGuard Classification

EcoGuard classifies the FFWI score as:

| Score range | Level |
| --- | --- |
| `0 <= score < 15` | `LOW` |
| `15 <= score < 25` | `MODERATE` |
| `25 <= score < 40` | `HIGH` |
| `40 <= score < 60` | `VERY_HIGH` |
| `60 <= score <= 100` | `EXTREME` |

These are EcoGuard application classification thresholds. They are not claimed to be an official Israeli fire-danger scale. Future work may calibrate thresholds with historical Israeli weather and fire data.

## Weather Data Requirements

Mandatory FFWI fields:

- temperature
- relative humidity
- wind speed

Optional/non-FFWI fields:

- rainfall
- wind gust
- wind direction

Missing mandatory weather values are excluded from input selection. If too little valid data remains, EcoGuard persists an `INSUFFICIENT_DATA` assessment instead of creating a misleading `LOW` result.

## Freshness

`MAX_WEATHER_AGE_MINUTES = 30`

Exactly 30 minutes old is valid. Older observations are stale. Future observations are ignored.

## Area Assessment

Fire danger is assessed at area level.

Nearby weather stations are selected geographically. EcoGuard currently uses Haversine distance to check station membership within an assessment radius. Multiple selected stations are aggregated by arithmetic mean for temperature, relative humidity, and wind speed. At least one valid station is currently required.

## Simulation

Simulation-to-analysis integration uses:

`ASSESSMENT_RADIUS_KM = 5.0`

This is a simulation-area integration rule, not part of the FFWI scientific formula. It includes the current simulated local weather-station cluster around the scenario anchor.

Only simulated `WEATHER` events trigger fire-danger recalculation. Simulated `SATELLITE` and `NEWS` events do not alter FFWI.

## Insufficient Data

`INSUFFICIENT_DATA` means EcoGuard could not produce a trustworthy fire-danger calculation from fresh mandatory weather fields.

For insufficient data:

- no misleading `LOW` result is created
- score remains null
- danger level remains null
- no weather-input trace rows are required

## Traceability

Each valid persisted assessment stores links to the exact `WeatherObservation` rows used to calculate it in `fire_danger_assessment_weather_inputs`.

Trace rows preserve:

- assessment id
- weather observation id
- station id

Unused, stale, incomplete, future, or out-of-area observations are not attached as provenance.

## Methodology Version

Current methodology:

- name: `FOSBERG_FFWI`
- version: `1.0`

## Acceptance Summary

| Requirement | Implementation | Test evidence | Status |
| --- | --- | --- | --- |
| Hot/dry/windy weather scores higher than cool/humid/calm weather | `FFWICalculator`, `FireDangerAssessmentAgent` pipeline | `test_hot_dry_windy_conditions_score_higher_than_cool_humid_calm_conditions`, `test_real_pipeline_scores_hot_dry_windy_weather_higher_than_cool_humid_calm_weather` | PASS |
| Missing mandatory weather does not create misleading valid assessment | `FireDangerInputService`, `FireDangerAssessmentAgent` | `test_missing_mandatory_weather_fields_make_observation_invalid`, `test_insufficient_data_persists_business_result_without_calculation` | PASS |
| Boundary values are classified consistently | `ffwi_config.py`, `FFWICalculator` | `test_classification_boundaries_use_raw_score` | PASS |
| No recent information becomes insufficient data, not LOW | `FireDangerInputService`, `FireDangerAssessment` | `test_stations_exist_but_all_observations_are_stale_returns_insufficient_data`, `test_insufficient_data_persists_business_result_without_calculation` | PASS |
| Identical inputs are deterministic | `FFWICalculator`, `FireDangerInputService`, `FireDangerAssessmentAgent` | `test_identical_input_produces_identical_output_repeatedly`, `test_same_data_and_same_as_of_produce_identical_result`, `test_same_area_as_of_and_dependency_outputs_produce_equivalent_output` | PASS |
| Assessment can be traced to exact input data | `FireDangerAssessmentRepository` and trace table | `test_traceability_records_only_exact_weather_observations_used` | PASS |

## Current Limitations / Future Work

- no vegetation/fuel adjustment yet
- no local Israeli calibration yet
- no production scheduler yet
- fire detection is separate future functionality
- satellite/news fusion is separate future functionality

These are current scope boundaries, not bugs.

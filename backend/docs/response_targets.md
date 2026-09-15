# Response Targets - User Story 4.3

Status: implemented and acceptance verified for EcoGuard V1.

## 1. Purpose

User Story 4.3 answers:

> Which locations should operational response planning focus on, and in what priority order?

It converts the current active wildfire location plus relevant predicted wildfire-spread locations into an ordered `ResponseTargetSet`. The output is intentionally geographic and priority-ranked. It does not compute routes, ETAs, road-network nodes, resource assignments, or a `ResponsePlan`.

The boundary between related analysis stages is:

| Stage | Question answered |
|---|---|
| Fire Detection | Is there likely an active wildfire? |
| Fire Severity | How threatening is the active wildfire now? |
| Fire Spread Prediction | Which nearby locations may be affected later? |
| Response Targets | Which locations should response planning focus on, and in what priority order? |
| Routing / Epic 5 | How do available resources physically reach those targets? |

## 2. Inputs

The production response-target pipeline is:

```text
FireEvent
    +
latest current FireSeverityAssessment state
    +
latest current 30-minute FireSpreadPrediction state
    +
latest current 60-minute FireSpreadPrediction state
        -> ResponseTargetInputService
        -> ResponseTargetInput
        -> ResponseTargetCalculator
        -> ResponseTargetSet
        -> ResponseTargetRepository
```

`FireEvent` is required. Severity and spread are optional:

- Severity may be absent or have a latest non-valid state. In that case the active-fire target still exists and uses no severity boost.
- Spread may be absent or have a latest non-valid state. In that case no predicted-risk targets are created from that horizon.
- The current latest analysis state is selected at the caller's `as_of` timestamp.
- Older valid severity or spread states are not silently reused after a newer non-valid state exists.

This latest-state policy is implemented by `ResponseTargetInputService`, using `FireSeverityAssessmentRepository.get_latest_for_event_as_of(...)` and `FireSpreadPredictionRepository.get_latest_for_event_and_horizon_as_of(...)`.

## 3. Target Types

EcoGuard V1 defines two response-target types in `ResponseTargetType`.

### ACTIVE_FIRE

An `ACTIVE_FIRE` target represents the known current wildfire location.

Coordinates:

```text
latitude  = FireEvent.latitude
longitude = FireEvent.longitude
```

Exactly one `ACTIVE_FIRE` target is generated for every active `SUSPECTED` or `CONFIRMED` fire event. It is generated even when severity and spread inputs are missing.

`ACTIVE_FIRE` targets do not contain prediction metadata:

```text
prediction_horizon_minutes = None
spread_prediction_id = None
spread_prediction_cell_id = None
```

### PREDICTED_RISK

A `PREDICTED_RISK` target represents a persisted spread-prediction cell that is important enough to become an operational target.

Predicted targets preserve provenance:

```text
prediction_horizon_minutes
spread_prediction_id
spread_prediction_cell_id
```

These fields allow downstream plans and operators to explain exactly which spread prediction and cell caused the target.

## 4. Methodology and Version

Current methodology:

```text
ECOGUARD_RESPONSE_TARGET_PRIORITY
version 1.0
```

The methodology/version are stored on every `ResponseTargetSet`. Future changes to thresholds, priority formulas, horizon factors, or deduplication rules can receive a new version without making older target sets ambiguous.

Centralized V1 constants live in `backend/src/calculators/response_target/response_target_config.py`:

```text
ACTIVE_FIRE_BASE_PRIORITY = 100.0
MIN_PREDICTED_TARGET_RISK_SCORE = 60.0
PREDICTION_HORIZON_FACTORS = {30: 1.0, 60: 0.85}
TARGET_DEDUP_DISTANCE_METERS = 100.0
```

These are EcoGuard V1 application-policy values. They are not presented as official firefighting standards or externally calibrated scientific thresholds.

## 5. Inclusion Threshold

There are two separate thresholds:

| Threshold | Owner | Meaning |
|---|---|---|
| Fire Spread threshold | User Story 4.2 | Whether a location belongs in the spread prediction result. |
| Response Target threshold | User Story 4.3 | Whether an already predicted location is important enough to become an operational response target. |

The current response-target threshold is:

```text
MIN_PREDICTED_TARGET_RISK_SCORE = 60.0
```

Predicted spread cells with `spread_risk_score < 60.0` are excluded. Cells at exactly `60.0` are included, subject to deduplication.

## 6. Priority Methodology

### ACTIVE_FIRE

When valid severity is available:

```text
priority = ACTIVE_FIRE_BASE_PRIORITY + severity_score
```

Current base:

```text
ACTIVE_FIRE_BASE_PRIORITY = 100.0
```

If severity is unavailable:

```text
priority = 100.0
```

This is deliberate: the known current fire remains an operational target even when supporting analysis context is missing.

### PREDICTED_RISK

```text
priority = risk_score * horizon_factor
```

Current V1 horizon factors:

```text
30 minutes -> 1.00
60 minutes -> 0.85
```

Shorter-horizon risk is treated as operationally more urgent. This is an EcoGuard V1 priority policy, not an externally calibrated wildfire-response formula.

## 7. Deterministic Ordering

The final target tuple is deterministic. `ResponseTargetCalculator` sorts by:

1. higher `priority_score` first;
2. target type order, with `ACTIVE_FIRE` before `PREDICTED_RISK`;
3. prediction horizon, using a sentinel for `ACTIVE_FIRE`;
4. latitude;
5. longitude;
6. spread prediction ID, using a sentinel for `ACTIVE_FIRE`;
7. spread prediction cell ID, using a sentinel for `ACTIVE_FIRE`.

The repository persists this exact tuple order as zero-based `target_order`.

## 8. Duplicate Policy

Current V1 deduplication radius:

```text
TARGET_DEDUP_DISTANCE_METERS = 100.0
```

Duplicate handling is applied to predicted candidates before final sorting:

- predicted target near the active fire location: the `ACTIVE_FIRE` target wins, and the predicted duplicate is removed;
- predicted target near another predicted target: the best deterministic candidate wins;
- 30- and 60-minute predictions of effectively the same location do not become separate operational targets.

Predicted candidates are considered in this order for duplicate selection:

1. higher predicted priority first;
2. shorter horizon;
3. higher raw risk score;
4. latitude;
5. longitude;
6. spread prediction ID;
7. spread prediction cell ID.

## 9. Persistence

Response targets are stored as append-only snapshots.

Tables:

```text
response_target_sets
response_targets
```

Each `response_target_sets` row stores:

```text
fire_event_id
generated_at
methodology
methodology_version
created_at
```

Each `response_targets` row stores:

```text
response_target_set_id
fire_event_id
target_order
target_type
latitude
longitude
priority_score
```

Predicted targets additionally store:

```text
prediction_horizon_minutes
spread_prediction_id
spread_prediction_cell_id
```

Repository source validation ensures predicted target references point to an existing spread prediction and cell, that the prediction belongs to the same `FireEvent`, and that the referenced cell belongs to the referenced prediction. Saves are atomic: if one target in a set is invalid, the entire set rolls back.

## 10. Why ResponseTargetSet Exists

The system stores a generated set/snapshot rather than independent mutable "current targets" so future planning can say:

```text
ResponsePlan
    -> exact ResponseTargetSet used at planning time
```

This supports traceability, replay, historical analysis, and plan explanation. `ResponsePlan` is not implemented in User Story 4.3.

Repeated explicit generation is append-only. Two runs with identical input can create different `target_set_id` values while preserving identical logical ordered target content.

## 11. Simulation Integration

The demo simulation uses the production response-target Agent.

For each simulation source event:

```text
persist source data
    -> Detection / Severity / Spread as applicable
    -> collect affected FireEvent IDs
    -> SimulationResponseTargetCoordinator
    -> ResponseTargetGenerationAgent
    -> persisted ResponseTargetSet
    -> RESPONSE TARGETS demo output
```

Simulation integration rules:

- the exact simulation event timestamp is passed as `as_of`;
- FireEvent IDs are deduplicated and processed in deterministic sorted order by `SimulationResponseTargetCoordinator`;
- multi-incident scenarios remain isolated by FireEvent ID;
- the production `ResponseTargetGenerationAgent` is reused;
- no simulation ground truth is passed into target calculation.

## 12. Example

Input:

```text
FireEvent severity = 68

Spread Cell A:
risk = 82
horizon = 30

Spread Cell B:
risk = 80
horizon = 60
```

Priorities:

```text
ACTIVE_FIRE:
100 + 68 = 168

Spread Cell A:
82 * 1.00 = 82

Spread Cell B:
80 * 0.85 = 68
```

Final ordered targets:

```text
1. ACTIVE_FIRE     priority 168
2. PREDICTED_RISK  priority 82
3. PREDICTED_RISK  priority 68
```

## 13. No-Spread Example

Input:

```text
active FireEvent
no valid current Spread Prediction
```

Output:

```text
ResponseTargetSet
    -> ACTIVE_FIRE
```

Missing prediction does not mean no response is required. It only means no predicted-risk locations are added for that generation.

## 14. Handoff to Epic 5

User Story 4.3 hands Epic 5 an ordered geographic target set:

```text
ResponseTargetSet
    targets ordered by operational priority

ResponseTarget
    fire_event_id
    target_type
    latitude
    longitude
    priority_score
```

Predicted targets also provide spread provenance:

```text
prediction_horizon_minutes
spread_prediction_id
spread_prediction_cell_id
```

Epic 5 can later perform:

```text
ResponseTarget(latitude, longitude)
    -> nearest reachable road node
    -> Dijkstra
    -> route / ETA
```

Resource allocation can later combine an ETA matrix, target priorities, and resource availability. None of those routing or allocation values are embedded in `ResponseTarget`.

## 15. Limitations

V1 limitations:

- priority coefficients are EcoGuard V1 application policy;
- there is no historical calibration against Israeli emergency-response outcomes;
- the target threshold is configurable and not official;
- the dedup radius is configurable and not official;
- only 30- and 60-minute horizons are currently supported;
- output is geographic, not road-network-aware;
- no routing or ETA is generated;
- no resource assignment is generated;
- User Story 4.4 refresh/no-op behavior is not part of this story;
- append-only explicit generation can create logically identical snapshots.

## 16. Jira Acceptance Test Mapping

| Jira Acceptance Test | Implementation evidence |
|---|---|
| Active fire -> `ACTIVE_FIRE` | `ResponseTargetCalculator`, `ResponseTargetGenerationAgent`, `test_at1_active_fire_location_becomes_active_fire_target` |
| High predicted risk -> `PREDICTED_RISK` | `ResponseTargetInputService`, `ResponseTargetCalculator`, `test_at2_high_spread_prediction_becomes_predicted_risk_target_with_traceability` |
| Target types distinguishable | `ResponseTargetType`, domain metadata validation, `test_at3_active_fire_and_predicted_risk_targets_are_distinguishable` |
| Priority ordering | `ResponseTargetCalculator._target_sort_key`, persisted `target_order`, `test_at4_higher_priority_targets_appear_first_and_target_order_is_persisted` |
| Below threshold excluded | `MIN_PREDICTED_TARGET_RISK_SCORE`, calculator filtering, `test_at5_below_threshold_predictions_are_excluded_and_threshold_is_inclusive` |
| Duplicate predicted locations deduped | Haversine-based 100m dedup policy, `test_at6_duplicate_predicted_locations_are_deduplicated_with_best_candidate_retained` |
| FireEvent isolation | `ResponseTargetInputService`, repository FK/source validation, `test_at7_fire_event_isolation_prevents_cross_event_predictions_and_targets` |
| No spread still `ACTIVE_FIRE` | Agent pipeline, `test_at8_active_fire_without_valid_spread_still_generates_active_fire_target` |
| Identical input deterministic | `ResponseTargetInput` candidate sorting, calculator ordering, append-only repository, `test_at9_identical_effective_input_produces_same_logical_ordered_targets` |

Additional acceptance boundaries are covered by:

- active-fire overlap dedup: `test_active_fire_overlap_predicted_location_is_removed_but_active_fire_remains`;
- inactive events: `test_inactive_fire_events_do_not_create_response_target_snapshots`;
- latest-state semantics: `test_latest_state_semantics_do_not_reuse_stale_valid_spread_or_severity`;
- static architecture guardrail: `test_ac_static_architecture_guardrail_keeps_response_targets_independent_of_routing_and_simulation`;
- methodology constants: `test_ac_methodology_constants_are_the_documented_v1_values`.

## 17. Acceptance Criteria Mapping

| Acceptance criterion | Evidence |
|---|---|
| Response-target generation is separate from routing algorithms. | Production response-target modules contain no imports of simulation, scripts, routing, road network, Dijkstra, `GraphNode`, `GraphEdge`, or `ResponsePlan`; guarded by `test_ac_static_architecture_guardrail_keeps_response_targets_independent_of_routing_and_simulation`. |
| Target-priority rules and thresholds are centralized and documented. | `response_target_config.py` defines methodology name/version, active-fire base, predicted-risk threshold, horizon factors, and dedup radius; guarded by `test_ac_methodology_constants_are_the_documented_v1_values`. |
| Every response target contains explicit geographic coordinates. | `ResponseTarget` requires `latitude` and `longitude`; acceptance tests assert coordinates for active and predicted targets. |
| Every target is traceable to its `FireEvent`. | `ResponseTarget.fire_event_id`, `ResponseTargetSet.fire_event_id`, and `response_targets.fire_event_id` are required and FK-backed; acceptance tests verify FireEvent isolation. |
| Predicted targets remain traceable to the relevant spread prediction. | `prediction_horizon_minutes`, `spread_prediction_id`, and `spread_prediction_cell_id` are required for `PREDICTED_RISK`; repository validates prediction/cell ownership; AT2 verifies exact persisted source IDs. |
| Dijkstra-specific graph-node information is not embedded. | `ResponseTarget` fields are only FireEvent ID, type, coordinates, priority, and optional spread provenance. It has no graph node, route, ETA, or resource-allocation fields; static guard covers imports. |
| Duplicate-target prevention is covered by tests. | Unit coverage exists in Task 1 calculator tests; Task 6 adds end-to-end acceptance coverage for predicted-vs-predicted and active-fire overlap dedup. |
| Output is structured for direct use by Epic 5 routing. | `ResponseTargetSet.targets` is an ordered tuple of geographic targets with priorities and types; predicted provenance remains available for explanation. Epic 5 can map coordinates to road nodes later. |

## 18. Code Quality Verification

Task 6 audit verified:

- constants are centralized in `response_target_config.py`;
- priority constants, thresholds, horizon factors, and dedup radius are not duplicated in Service, Agent, or Simulation;
- core target input/calculation/agent code does not call `datetime.now()`;
- simulation `incident_id` does not leak into production target logic;
- pure domain and calculator modules do not import ORM models;
- response-target production modules do not import routing or simulation code;
- calculator output is deterministic and not set-order-dependent;
- repository saves are atomic and append-only.


# Operational Refresh - User Story 4.4

US 4.4 does not introduce another wildfire model. It answers: when operational or fire state changes, which existing analyses must be reevaluated, and when can the system safely do nothing?

The implementation reuses earlier user stories:

| User Story | Responsibility reused by US 4.4 |
| --- | --- |
| US 2.2 | Fire Detection creates and updates `FireEvent` records. |
| US 2.3 | Fire Severity recalculates incident severity and stores exact data traces. |
| US 4.1 | Resources, stations, road context ownership, and `AVAILABLE` filtering. |
| US 4.2 | Spread input preparation, calculation, and append-only prediction history. |
| US 4.3 | Prioritized geographic response target snapshots. |
| US 4.4 | Refresh orchestration and change detection across the above components. |

## Concepts

A trigger is the reason reevaluation is requested, such as new weather or a FireEvent update. Reevaluation means the relevant production services are asked to rebuild current state. Effective-state change means the calculator-relevant spread inputs differ from the latest stored spread prediction. Recalculation means a new `FireSpreadPrediction` row is persisted. A no-op means the trigger was handled but no duplicate prediction row was needed.

## Trigger Matrix

| Trigger | Severity | Spread Reevaluation | Response Targets | Resource State |
| --- | ---: | ---: | ---: | ---: |
| `WEATHER_UPDATE` | Yes | Yes | Yes | No |
| `FIRE_EVENT_UPDATE` active | Yes | Yes | Yes | No |
| `FIRE_EVENT_UPDATE` inactive | No active Severity | Inactive handling | Inactive handling | No |
| `SEVERITY_UPDATE` | No | Yes | Yes | No |
| `RESOURCE_STATUS_UPDATE` | No | No | No | Yes |

The wildfire/environment branch is:

```text
Detection -> FireEvent -> Severity -> Spread Refresh -> Response Targets
```

The resource branch is:

```text
ResourceStatus -> ResourceStatusUpdateService -> available resources
```

A truck status change never affects Detection confidence, Severity, Spread, or Response Targets. Environmental refreshes never mutate `FirefightingResource.status`.

## Effective Spread State

US 4.4 compares spread-effective state, not source row identity. The fields are:

```text
fire_event_id
origin_latitude
origin_longitude
wind_speed_kmh
wind_direction_deg
fuel_moisture_percent
fuel_class
horizon_minutes
methodology
methodology_version
```

These fields define equality because they are the values used by the US 4.2 spread model for a horizon. The fingerprint is SHA-256 over a deterministic canonical representation. Python `hash()` is not used.

The following metadata does not by itself force a different spread prediction:

```text
WeatherObservation ID
SeverityAssessment ID
Severity score
Severity level
FRP
Detection confidence
as_of / processing timestamp
resource state
simulation incident ID
```

Source IDs still matter for persisted traceability. A new prediction records the current Severity and Weather trace that led to its input.

## No-Op Algorithm

```text
relevant trigger
  -> FireSpreadInputService
  -> FireSpreadEffectiveState
  -> SHA-256 fingerprint
  -> compare with latest same FireEvent/horizon/as_of
       same    -> NO_OP
       changed -> FireSpreadPredictionAgent
```

Predictions created before US 4.4 may have `effective_state_fingerprint = NULL`. They remain readable. Because their effective state is unknown, the next eligible READY refresh creates one fingerprinted prediction instead of reconstructing old state from mutable current inputs.

Non-ready states are explicit. `INSUFFICIENT_DATA` and `INACTIVE_EVENT` are persisted or no-opped according to latest state, so downstream consumers such as US 4.3 do not silently fall back to stale valid spread predictions. No fake/default fingerprint is created for non-ready input.

## Historical Preservation

Spread Predictions are append-only operational snapshots:

```text
P1 remains stored
new effective state -> P2
```

Response Target Sets are also append-only:

```text
T1 remains stored
new operational state -> T2
```

Resource status is current-state in V1, not append-only history.

## Resource Branch

Resources use the existing `FirefightingResource` and `ResourceStatus` values:

```text
AVAILABLE
ASSIGNED
UNAVAILABLE
```

Availability is authoritative through `FirefightingResourceRepository.get_available_resources(...)`.

Required transitions:

```text
AVAILABLE -> ASSIGNED
AVAILABLE -> UNAVAILABLE
ASSIGNED -> AVAILABLE
UNAVAILABLE -> AVAILABLE
```

`ASSIGNED` and `UNAVAILABLE` are excluded from available resources. A same-state update returns `NO_OP`.

## Simulation Mapping

Simulation events map to production refresh at one boundary:

| Simulation event | Production refresh |
| --- | --- |
| `WEATHER` | Persist simulated weather, then `WEATHER_UPDATE` for active nearby FireEvents. |
| `SATELLITE` | Persist evidence, run Detection once, then `FIRE_EVENT_UPDATE` for affected FireEvents. |
| `NEWS` | Persist evidence, run Detection once, then `FIRE_EVENT_UPDATE` for affected FireEvents. |
| `RESOURCE_STATUS` | `RESOURCE_STATUS_UPDATE` only. |

The exact simulation event timestamp is passed as `as_of`. FireEvent IDs are deduped and processed in deterministic ascending order. Multi-incident scenarios keep independent prediction and target history. Resource selection is deterministic and selection keys reuse the same resource for transition-back events.

The active central-refresh runtime path does not also run the old lower-level simulation severity/spread/target coordinators. WEATHER is not a simulation-only shortcut into `FireSpreadInput`; it persists weather, then Severity creates a trace, then `FireSpreadInputService` prepares spread input from that Severity trace.

The resource preset timing is seconds: `timestamp_offset_sec = 60` and `timestamp_offset_sec = 90`, inside the 120-second demo scenario. The earlier wording of "60 and 90 minutes" was report wording, not code behavior.

## Response Target Refresh

Targets are generated once after both 30- and 60-minute spread horizons finish. They are not generated once per horizon. A Severity-score-only change can still produce Spread `NO_OP` and a new target set, because Severity affects the `ACTIVE_FIRE` target priority even when spread-effective inputs do not change.

If latest Spread transitions from `VALID` to `INSUFFICIENT_DATA`, US 4.3 latest-state semantics keep `ACTIVE_FIRE` but do not reuse stale predicted-risk cells from the old valid prediction.

## Architecture Audits

US 4.4 refresh modules do not own routing, Dijkstra, ETA, Genetic Algorithm, resource allocation, or `ResponsePlan`.

Production refresh orchestration does not import or duplicate pure calculators. Scientific formulas remain in `FireSpreadCalculator`; target scoring remains in `ResponseTargetCalculator`; Severity calculation remains in the Severity agent/calculator stack.

New 4.4 refresh modules do not depend on `RoadNetworkRepository`, `GraphNode`, `GraphEdge`, OSMnx, or `seed_road_networks`. Road-loading strategy remains owned by US 4.1.

Simulation refresh does not directly call IMS, FIRMS, Copernicus, or OSM. It uses persisted/generated simulation data and existing production services.

The fingerprint column is nullable `VARCHAR(64)` so legacy rows remain readable and no destructive migration is required.

## Acceptance Test Mapping

| AT | Components | Tests | Evidence |
| --- | --- | --- | --- |
| AT1 | Operational orchestrator, Severity trace, Spread refresh repository | `test_at1_weather_update_creates_append_only_prediction_through_severity_trace` | New weather creates new predictions; previous rows remain; latest prediction traces the new Severity/Weather IDs. |
| AT2 | FireEvent repository, Operational orchestrator, Spread fingerprint | `test_at2_fire_event_location_change_refreshes_and_confidence_only_change_no_ops` | Location change creates new predictions; confidence-only change no-ops. |
| AT3 | Severity repository, Spread input service, Target generation | `test_at3_severity_update_uses_new_trace_and_score_only_no_op_still_refreshes_targets` | New Severity trace drives prediction; score-only change no-ops Spread while Targets refresh. |
| AT4 | Simulation refresh, Resource service/repository | `test_at4_at5_resource_status_simulation_removes_and_restores_same_truck` | AVAILABLE -> UNAVAILABLE removes the truck from authoritative availability. |
| AT5 | Resource service/repository | `test_at4_at5_resource_status_simulation_removes_and_restores_same_truck`, `test_at5_additional_assigned_available_transition_uses_authoritative_available_set` | Same resource returns to AVAILABLE; ASSIGNED is excluded and AVAILABLE restores eligibility. |
| AT6 | Resource branch separation | `test_at6_resource_change_does_not_touch_detection_or_environmental_history` | FireEvent confidence/status/location unchanged; no Severity/Spread/Target history added. |
| AT7 | Spread repository | `test_at7_previous_predictions_and_cells_remain_after_changed_refresh` | P1 and cells remain queryable after P2 is created. |
| AT8 | Resource branch and Spread no-call evidence | `test_at8_resource_update_is_unrelated_to_spread_recalculation` | Resource status does not call Spread or change prediction count. |
| AT9 | Fingerprint no-op | `test_at9_identical_effective_input_and_new_source_ids_do_not_duplicate_predictions` | New source IDs with equal effective state produce `NO_OP` and no new row. |
| AT10 | Inactive FireEvent handling | `test_at10_resolved_or_dismissed_event_persists_inactive_state_without_active_targets` | RESOLVED/DISMISSED creates inactive spread state and no new active targets. |
| AT11 | Response target latest-state semantics | `test_at11_response_targets_use_latest_valid_prediction_and_do_not_reuse_stale_cells`, `test_at11_transition_to_insufficient_does_not_reuse_stale_predicted_risk_targets` | Targets use latest valid P2 cells; insufficient state does not reuse stale predicted-risk targets. |

## Acceptance Criteria Mapping

| AC | Evidence |
| --- | --- |
| AC1 | `OperationalRefreshTriggerType`, `OperationalRefreshPolicy`, `OperationalRefreshOrchestrator`, and `test_ac1_ac4_ac8_architecture_guardrails_and_documented_trigger_matrix`. |
| AC2 | `refresh_resource` branch plus AT6/AT8 tests prove resource updates are separated from environmental evidence. |
| AC3 | AT1, AT7, and AT11 verify append-only Spread and Target history. |
| AC4 | Architecture guard tests parse refresh modules and verify no direct pure-calculator or road/routing imports. |
| AC5 | `test_ac5_resource_simulation_timing_and_selection_are_deterministic` verifies deterministic resource event timing and selection-key reuse. |
| AC6 | AT2 confidence no-op, AT9 fingerprint no-op, same-state resource tests, and simulation FireEvent ID dedupe in `test_ac6_multi_incident_dedupe_keeps_histories_independent`. |
| AC7 | Dedicated acceptance tests cover weather updates, severity updates, resource-status changes, and reprocessing/no-op. |
| AC8 | Operational state remains compatible with Epic 5 through current `ResponseTargetSet` snapshots and authoritative available resources. |

## Epic 5 Handoff

US 4.4 leaves this state ready for a future planner:

```text
Updated ResponseTargetSet
        +
Available FirefightingResources
        +
US 4.1 Road Network
        -> Epic 5
```

Epic 5 may later add nearest reachable graph node lookup, Dijkstra, ETA matrices, resource allocation, Genetic Algorithm optimization, and `ResponsePlan`. None of that is implemented by US 4.4.

## V1 Limitations

- Effective equality is based on current US 4.2 calculator inputs.
- Fingerprint versioning follows Spread methodology and methodology version.
- Legacy prediction rows may have `NULL` fingerprint.
- Resource status history is not append-only in V1.
- Resource simulation status transitions are deterministic but do not perform optimization.
- No routing recalculation exists yet.
- No ETA exists yet.
- No Genetic Algorithm exists yet.
- No `ResponsePlan` exists yet.
- Road-loading strategy remains owned by US 4.1 and is intentionally not coupled to US 4.4.

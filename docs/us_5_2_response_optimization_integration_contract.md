# US 5.2 Response Optimization - Integration Contract

US 5.2 consumes an already-normalized `ResponseOptimizationInput` and produces an append-only `ResponsePlan` recommendation. It does not fetch route data, calculate roads, or dispatch resources.

## Consumes

`ResponseOptimizationInput` contains:

- `fire_event_id`
- `response_target_set_id`
- `route_planning_run_id`
- ordered `OptimizationTarget` values with `response_target_id`, `target_order`, `target_type`, and supplied `priority_score`
- eligible `OptimizationResource` values
- `OptimizationRouteOption` values

US 5.1 must eventually provide these route facts:

- `route_planning_run_id`
- `route_result_id`
- `resource_id`
- `response_target_id`
- `is_reachable`
- `travel_time_seconds`
- `distance_meters`

## Does Not Consume

US 5.2 deliberately does not consume Severity, Spread, FireDanger, road graphs, coordinates, Dijkstra internals, OSMnx objects, or FireStation details. Those concerns belong upstream.

## Produces

`ResponseOptimizationAgent.optimize_from_input(...)` runs the pure genetic optimizer and persists:

- one `response_plans` row
- zero or more `response_actions` rows
- zero or more `response_plan_uncovered_targets` rows

Actions preserve `resource_id`, `response_target_id`, and `route_result_id` exactly so the plan remains traceable to upstream route facts.

## Persistence And History

Response plans are append-only recommendations. A new optimization run creates a new plan and does not overwrite, delete, supersede, or mutate older plans. US 5.4/US 5.5 can later define current-plan and replanning semantics.

Until real US 5.1 persistence exists, `route_planning_run_id` and `route_result_id` are scalar trace IDs without foreign keys to routing tables. Task 5B should strengthen these constraints only after the real routing schema merges.

## Resource Status

Optimization is not dispatch. US 5.2 must never change `FirefightingResource.status`; operational resource-update logic owns that state.

## Determinism

The plan records methodology `GENETIC_RESOURCE_ALLOCATION`, methodology version `1.0`, and the exact random seed. With identical input, config, seed, and `as_of`, the logical recommendation is deterministic even though append-only persistence creates distinct primary keys.

## Task 5B Checklist

- Inspect actual `RoutePlanningRun` domain and DB model.
- Inspect actual `RouteResult` domain and DB model.
- Confirm ID types.
- Map resources to existing `FirefightingResource` IDs.
- Map target IDs to stored response-target IDs.
- Map route status to `is_reachable`.
- Map ETA to `travel_time_seconds`.
- Map distance to `distance_meters`.
- Construct `ResponseOptimizationInput`.
- Add foreign-key constraints only where real persistence makes them valid.
- Remove temporary scalar-trace assumptions that are no longer needed.
- Add real routing-to-optimization integration tests.

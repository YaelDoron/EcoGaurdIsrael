# US 5.2 Response Optimization - Integration Contract

US 5.2 consumes an already-normalized `ResponseOptimizationInput` and produces an append-only `ResponsePlan` recommendation. It does not fetch route data, calculate roads, or dispatch resources.

## Consumes

`ResponseOptimizationInput` can be supplied directly by tests or built from a persisted US 5.1
`RoutePlanningRun` through `ResponseOptimizationInputService.build_from_route_planning_run(...)`.
The route-planning run ID is the primary integration input. The service loads that exact persisted
routing snapshot and the exact target set referenced by it; it does not look up the latest/current
target set, resource state, or route data.

The normalized input contains:

- `fire_event_id`
- `response_target_set_id`
- `route_planning_run_id`
- ordered `OptimizationTarget` values with `response_target_id`, `target_order`, `target_type`, and supplied `priority_score`
- eligible `OptimizationResource` values
- `OptimizationRouteOption` values

The US 5.1 routing snapshot provides these route facts:

- `route_planning_run_id`
- `route_result_id`
- `resource_id`
- `response_target_id`
- `travel_time_seconds`
- `distance_meters`
- `RouteStatus`

US 5.2 maps `RouteStatus.REACHABLE` to `is_reachable=True` and preserves ETA/distance. It maps
`RouteStatus.UNREACHABLE` and `RouteStatus.UNMAPPABLE` to `is_reachable=False` with ETA/distance
set to `None`, so the optimizer can ignore them without losing traceability.

Targets are mapped from stored response targets: `StoredResponseTarget.id` becomes
`OptimizationTarget.response_target_id`, and `target_order`, `target_type`, and `priority_score`
are preserved. Resources are mapped only from `RoutePlanningRun.resource_ids`; US 5.2 does not
reload `FirefightingResource` rows or filter on current operational status.

The adapter validates snapshot consistency:

- the route-planning run and response-target set both exist
- `RoutePlanningRun.fire_event_id` matches the target set fire event
- the route-planning run points at the loaded target set
- route resources are present in the route-planning run resource snapshot
- route targets are present in the persisted response-target set
- route result IDs, resource IDs, target IDs, and resource/target route pairs are unique

US 5.1 allows a valid run to contain a resource snapshot with no route rows, so US 5.2 does not
enforce a complete resource-by-target ETA matrix.

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

The real US 5.1 persistence schema exists, and US 5.2 validates references at the application-service
boundary. Database foreign-key hardening from `response_plans.route_planning_run_id` and
`response_actions.route_result_id` is deferred because the current task works in an already-migrated
schema with scalar trace IDs; changing those columns now would require a migration policy decision
outside this task's scope.

## Resource Status

Optimization is not dispatch. US 5.2 must never change `FirefightingResource.status`; operational resource-update logic owns that state.

## Determinism

The plan records methodology `GENETIC_RESOURCE_ALLOCATION`, methodology version `1.0`, and the exact random seed. With identical routing run input, config, seed, and timezone-aware `as_of`, the logical recommendation is deterministic even though append-only persistence creates distinct primary keys.

`ResponseOptimizationAgent.optimize_from_route_planning_run(route_planning_run_id, *, as_of, config=None)`
is the operational US 5.1 to US 5.2 entry point. `as_of` must be timezone-aware and explicit.

## Task 5B Checklist

- Inspect actual `RoutePlanningRun` domain and DB model. Done.
- Inspect actual `RouteResult` domain and DB model. Done.
- Confirm ID types. Done: resource IDs are strings, target and route-result IDs are integers.
- Map resources from the route-planning snapshot. Done.
- Map target IDs to stored response-target IDs. Done.
- Map route status to `is_reachable`. Done.
- Map ETA to `travel_time_seconds`. Done.
- Map distance to `distance_meters`. Done.
- Construct `ResponseOptimizationInput`. Done.
- Add foreign-key constraints only where real persistence makes them valid. Deferred with app-level validation.
- Remove temporary future-contract wording. Done.
- Add real routing-to-optimization integration tests. Done.

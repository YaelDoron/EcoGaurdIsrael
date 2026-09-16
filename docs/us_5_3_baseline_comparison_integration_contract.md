# US 5.3 Baseline Comparison - Integration Contract

US 5.3 compares one exact persisted optimized (GA) `ResponsePlan` against a deterministic `GREEDY_NEAREST_AVAILABLE` baseline built from the same planning snapshot, and persists the result as an append-only `PlanComparison`. It does not fetch route data, calculate roads, rerun the GA, or dispatch resources.

## Entry point

`BaselineComparisonService.compare(response_plan_id=...)` is the entry point. It loads the exact persisted `ResponsePlan`, the exact `RoutePlanningRun` and `ResponseTargetSet` it references, validates the planning chain is self-consistent, builds and scores the greedy baseline, compares it against the plan's already-persisted metrics, and saves one `PlanComparison` row.

## Real production wiring

`BaselineComparisonService`'s default constructor arguments are real adapters, not test doubles:

- `optimized_plan_reader` defaults to `ResponsePlanRepositoryOptimizedPlanReader`, which wraps US 5.2's real `ResponsePlanRepository`.
- `route_planning_run_reader` defaults to `RoutePlanningRepositoryRunReader`, which wraps US 5.1's real `RoutePlanningRepository`.
- `response_target_set_reader` defaults directly to the real US 4.3 `ResponseTargetRepository` (it already satisfies the port structurally).
- `scorer` defaults to `ResponsePlanScorerBaselineAdapter`, which reuses US 5.2's `ResponseOptimizationInputService.build_from_route_planning_run(...)` to build the canonical `ResponseOptimizationInput`, then scores the baseline allocation through the real `ResponsePlanScorer`.
- `plan_comparison_repository` defaults to the real `PlanComparisonRepository`.

All five remain constructor-injectable `Protocol` ports (`baseline_comparison_ports.py`) purely so unit tests can substitute lightweight fakes; production code never overrides them.

## Reused, not duplicated

US 5.3 reuses rather than re-derives:

- **Routing facts** (`route_result_id`, `resource_id`, `response_target_id`, `RouteStatus`, `travel_time_seconds`) come from the exact persisted `RoutePlanningRun`/`RouteResult` rows. No Dijkstra, no node mapping, no ETA recalculation.
- **Target facts** (`response_target_id`, `target_order`, `priority_score`) come from the exact persisted `ResponseTargetSet`. No severity/spread/priority recalculation.
- **The optimization input shape** used to score the baseline is US 5.2's canonical `ResponseOptimizationInput`, built by `ResponseOptimizationInputService` -- the same mapping the GA itself consumes. US 5.3 does not implement a second `RouteStatus -> is_reachable` mapping or a second ETA/priority normalization.
- **The scoring formula** is US 5.2's real `ResponsePlanScorer.evaluate(...)`. US 5.3 does not reimplement `eta_factor`, `plan_score`, or `coverage_score`.
- **The optimized plan's metrics** (`plan_score`, `coverage_score`, `average_eta_seconds`) are copied verbatim from the persisted `ResponsePlan`. US 5.3 never calls `ResponsePlanScorer` on the optimized plan -- only the baseline is scored.

## Exact snapshot guarantee

US 5.3 is historical evaluation, not a "current state" comparison. `compare(response_plan_id=...)` loads that exact plan, then exactly the `RoutePlanningRun` and `ResponseTargetSet` it references (by id) -- never the latest run, latest target set, or current resource state for the plan's `FireEvent`. `_validate_planning_chain` rejects (with no `PlanComparison` persisted) any loaded record whose `fire_event_id`/`route_planning_run_id`/`response_target_set_id` does not match, and `_build_route_candidates` rejects any route referencing a resource or target outside that exact snapshot.

## Current resource state is ignored

The baseline is built only from `RoutePlanningRun.resource_ids` (the snapshot recorded when routing ran), never from live `FirefightingResource.status`. US 5.3 does not import or query `FirefightingResourceRepository` (enforced by a static-import architecture guardrail test); changing a resource's current status after a plan was generated has no effect on that plan's baseline comparison. Current availability belongs to US 5.4 replanning, not historical comparison.

## Baseline algorithm and comparison formulas

Unchanged from the original US 5.3 implementation: `GREEDY_NEAREST_AVAILABLE` (`BaselinePlanCalculator`) processes targets by persisted `target_order`, assigns the nearest unused `REACHABLE` resource (tie-break by ascending `resource_id`), and leaves a target uncovered if no eligible route exists. `score_difference = optimized_score - baseline_score`; `improvement_percentage = ((optimized_score - baseline_score) / baseline_score) * 100` when `baseline_score != 0`, else `None`. Negative results are persisted unclamped.

## Persistence

`PlanComparison` rows are append-only via `PlanComparisonRepository.save(...)` -- comparing the same plan twice creates two historical rows; nothing is updated, upserted, or deduplicated. US 5.3 never mutates `FirefightingResource.status`, `ResponsePlan`, `ResponseAction`, `RoutePlanningRun`, `RouteResult`, or `ResponseTargetSet`.

## Foreign keys

`plan_comparisons.optimized_plan_id`/`route_planning_run_id`/`response_target_set_id` remain scalar trace IDs validated at the application-service boundary (the same deferral already documented for US 5.2's `response_plans.route_planning_run_id`). Hardening these into real foreign keys is deferred; it would require a migration policy decision outside this integration task's scope.

## Tests

`backend/tests/services/baseline_comparison/test_baseline_comparison_service.py` and `backend/tests/acceptance/test_baseline_comparison_user_story_5_3.py` exercise Task 1-5 orchestration logic (snapshot validation, target ordering, tie-breaking, unreachable/unmappable exclusion, append-only history, negative/zero-score comparisons) against lightweight fakes for the plan/run/scorer ports, by design, to isolate orchestration behavior from persistence.

`backend/tests/acceptance/test_baseline_comparison_real_integration_user_story_5_3.py` adds the real end-to-end proof: a persisted US 5.1 `RoutePlanningRun`, a real GA-optimized and persisted US 5.2 `ResponsePlan`, and `BaselineComparisonService()` with its unmodified real default wiring (no fakes anywhere in the call path) -- covering exact-snapshot traceability, optimized-metrics-not-rescored, baseline-scored-through-the-real-shared-`ResponsePlanScorer`, `REACHABLE`-only route filtering, snapshot isolation against a newer unrelated run, current-resource-state isolation, and append-only history.

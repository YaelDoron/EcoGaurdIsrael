# Fire Detection — Runtime-Integration Audit for the AI Hybrid Policy (Task 8)

> **Update (Task 9A):** the two blockers described in this audit are fixed - the demo reset is now mandatory on every start path and SUSPECTED is no longer response-eligible. See `docs/fire_detection_task9a_runtime_semantics.md`. The text below is the original audit and describes the pre-9A behaviour.


Written before any runtime integration. **Nothing described here as "proposed" is implemented.** The policy itself and
its confirmatory results are in [fire_detection_ai_hybrid_policy_v5.md](fire_detection_ai_hybrid_policy_v5.md). The historical
Task 7 verdict (binary-primary V5 model gate: **FAILED**) is unchanged, see
[fire_detection_model_comparison_v5.md](fire_detection_model_comparison_v5.md). All data behind the policy is synthetic.

## 1. Can a newly started simulation inherit Fire Detection history from the previous simulation?

**Yes — conditionally. Isolation is opt-in, not guaranteed.** Traced through the real persistence path
(`tests/acceptance/test_simulation_history_isolation_task8.py`):

| Start path | Reset? | Run B sees Run A's FireEvent / evidence / history? |
|---|---|---|
| Dashboard "Start / Run Again" (`SimulationControl` sends `reset_demo_state: true`) with `ENABLE_DEMO_DATA_RESET=true` | yes, before the first event | **No.** First observation: `satellite_pass_count = 1`, span 0, no trend, no inherited evidence refs |
| Same request with `ENABLE_DEMO_DATA_RESET` off | refused (HTTP 403) — the run never starts | No (fail-closed) |
| `POST /simulation/runs` without the flag (**API default is `reset_demo_state = false`**) | no | **YES — known leak** |
| `scripts/run_demo_simulation.py` without `--reset-demo-state` | no | **YES — known leak** |

Mechanism of the leak (demonstrated by `test_KNOWN_LEAK_without_the_reset_a_new_run_inherits_the_previous_runs_history`): the
run manager only calls `DemoStateResetService.reset_demo_state()` when asked. Without it, Run B's first hotspot at the same place
within 6 h matches Run A's FireEvent (5 km / 6 h), attaches to it, and the 24 h event history then gives
`satellite_pass_count = 4` (3 inherited + 1), `history_span = 420 min`.

What the reset does guarantee (tests): it deletes every simulation-generated runtime table — `FireEvent`, satellite/news evidence
associations, ML assessments, hotspots, news reports, severity, spread, targets, routes, plans, global-planning runs, resource
commitments, fire-danger/weather observations — and restores resource status. A guard test asserts that **every** table in
`Base.metadata` is either deleted or in the declared static set (`fire_stations`, `firefighting_resources`, `graph_nodes`,
`graph_edges`, `weather_stations`), so a future runtime table cannot silently escape the reset.

**Verdict: a blocker for the runtime-integration task on the non-dashboard paths** (API default, CLI). The dashboard path is
correct. No run identity (`simulation_run_id`) exists; the design relies on the reset.

### Smallest clean fix (proposed, NOT implemented)

1. **Preferred (smallest):** make the reset non-optional for every demo-run start path — `SimulationRunManager.start_run` resets
   unconditionally when the safety flag is on and refuses to start otherwise (or the API default becomes `true`). Fail-closed,
   no migration, and it reuses the audited reset. `test_KNOWN_LEAK_…` and `test_the_api_default_is_not_to_reset` are then inverted deliberately.
2. Alternative without deleting data: an evidence *epoch* — history and event matching ignore evidence/events older than the current
   run's `scenario_started_at`. Needs plumbing through the evidence service, history service and matching.
3. A `simulation_run_id` column on evidence/events is the "production multi-tenant" answer and needs a migration: not justified for the
   project.

## 2. What SUSPECTED and CONFIRMED currently trigger

**Does SUSPECTED currently trigger response optimisation / routing / resource allocation? Yes.** The backend does not
distinguish the two statuses anywhere: every subsystem shares the same active set `{SUSPECTED, CONFIRMED}`
(`tests/services/test_suspected_downstream_audit_task8.py`), and no service branches on `CONFIRMED`.

| Subsystem | SUSPECTED | CONFIRMED | Evidence |
|---|---|---|---|
| Event repository active queries | included | included | `_ACTIVE_STATUSES` |
| Severity assessment | **triggered** | triggered | `fire_severity_input_service._ACTIVE_EVENT_STATUSES` |
| Spread prediction | **triggered** | triggered | `fire_spread_input_service._ACTIVE_EVENT_STATUSES` |
| Response targets | **triggered** | triggered | `response_target_input_service._ACTIVE_EVENT_STATUSES` |
| Operational refresh (trigger for the chain) | **triggered** | triggered | `_ACTIVE_FIRE_EVENT_STATUSES`; the simulation detection→refresh path never inspects status |
| Road graph / Dijkstra routing | **triggered** (part of global planning for every active event) | triggered | `OperationalPlanningRefreshCoordinator` → global planning |
| Genetic / resource allocation, global planning | **triggered** — one global cycle "covering every currently active FireEvent together" | triggered | `GlobalPlanningRefreshCoordinator` |
| Resource commitment (reserving units, changing resource status) | **triggered** (activation accepts SUSPECTED) | triggered | `response_plan_activation_service` / `global_response_plan_activation_service` `_ACTIVE_STATUSES` |
| Active-fires API, operations overview, activity feed, chatbot | listed | listed | `get_active_events` |
| Dashboard marker | hollow orange ring, never "emphasised" | solid red dot; emphasised only with HIGH/CRITICAL severity | `activeFireIcon.ts`, `ActiveFireMapLayer.tsx` |
| Lifecycle | can upgrade to CONFIRMED; can be RESOLVED/DISMISSED | monotonic (never downgraded); can be RESOLVED/DISMISSED | `FireDetectionAgent._monotonic_status`, `FireEventLifecycleService` |

The only status-specific behaviour today is **presentation**. That is why the confirmatory result — the policy marks ~50% of
non-sparse no-fire rows SUSPECTED (49.1%) — is only acceptable once SUSPECTED stops driving the response pipeline.

## 3. Target semantics vs current behaviour (assessed, not implemented)

| Aspect | Target | Current | Gap |
|---|---|---|---|
| SUSPECTED visible on dashboard | yes | yes | none |
| SUSPECTED preserves history and keeps collecting evidence | yes | yes (evidence attaches, history 24 h) | none |
| SUSPECTED re-evaluated when evidence arrives | yes | yes (`_update_existing_event`) | none |
| SUSPECTED sends **no** automatic resource dispatch | required | **dispatches**: commitments, routing, global planning all run | **large — every downstream `_ACTIVE_*` set** |
| SUSPECTED runs severity / spread | optional (cheap, informative) | runs | decide per cost; harmless if no dispatch |
| CONFIRMED activates response pipeline | yes | yes (same as SUSPECTED) | must become CONFIRMED-only for targets → routing → allocation → global planning → commitment |
| CONFIRMED monotonic | yes | yes | none |
| Upgrade SUSPECTED → CONFIRMED triggers the pipeline once | yes | n/a (already triggered) | a refresh trigger on the status transition |
| Downgrade of a CONFIRMED event | never | never | none |
| Fresh history per simulation run | required | dashboard only | section 1 |

Integration implication: introduce a single "response-eligible" predicate (`status is CONFIRMED`) used by targets, planning,
routing, allocation and commitment, while severity/spread/list/dashboard keep the broader "active" set. That is a semantic change to
about seven modules plus the dashboard legend/copy, so it must be its own task with the characterization tests above updated
deliberately.

## 4. Proposed final status semantics

| Status | Meaning | Response pipeline |
|---|---|---|
| NO_EVENT | P(fire) < 0.40 — nothing worth keeping | none; evidence remains in the tables |
| SUSPECTED | P(fire) ≥ 0.40 without the corroborated high-probability condition — *possible* wildfire, uncertain by design | dashboard + evidence collection + history + re-evaluation; **no** automatic dispatch |
| CONFIRMED | P(fire) ≥ 0.80 **and** ≥ 2 current satellite pixels | full emergency-response pipeline |
| RESOLVED / DISMISSED | closed | commitments released (existing lifecycle service) |

Confirmatory evidence for the split: CONFIRMED had precision 0.937 and a 1.0% false-confirmation rate at 14.5% recall (57% of
`large_wildfire`, 12% of `established_wildfire`, 1% of `early_wildfire`); SUSPECTED absorbs the uncertain remainder (sparse evidence
is 70% SUSPECTED / 30% NO_EVENT / 0% CONFIRMED).

## 5. Proposed SUSPECTED expiry (design only)

A SUSPECTED event that receives no meaningful new evidence must not stay active forever, and must never be resurrectable after
its evidence has aged out. The existing windows constrain the design:

* 120 min — candidate lookback: evidence older than this is not re-scored as a candidate;
* 6 h — event matching on `updated_at`: after 6 h without evidence a new hotspot creates a **new** event anyway;
* 24 h — history window: history older than this is not read.

Proposal:

1. `SUSPECTED_TTL = 6 h` since `updated_at` (aligned with event matching, so an expired event can never be matched again).
   A sweep at each detection cycle calls the existing `FireEventLifecycleService.dismiss_event(id, as_of)`, which also releases
   any commitments (none expected once SUSPECTED no longer dispatches).
2. Optional **earlier decay**: two consecutive re-evaluations of the event's current evidence with P(fire) < 0.40 and no news
   corroboration → dismiss early, so a resolved false alarm disappears without waiting 6 h.
3. "Meaningful new evidence" = an observation whose re-evaluation still yields P ≥ 0.40 (a lower-P observation does not renew the TTL).
4. **CONFIRMED is monotonic**: never expires; only `resolve_event` (or an explicit operator/lifecycle rule) closes it.
5. Dismissed / expired events keep their evidence associations for audit; history reads stay bounded by the 24 h window.
6. Record the reason (`expired_no_evidence`, `decayed_below_suspect_threshold`) — the current `DISMISSED` status alone does not say why.

Not implemented in Task 8.

## 6. What the next (runtime-integration) task must decide

1. Isolation fix (section 1) — required before demos that exercise history.
2. The response-eligibility predicate (section 3) and the dashboard wording for SUSPECTED.
3. A new decision mode for the AI hybrid policy (`rule_only` / `shadow` / `hybrid` keep their current meaning) and the runtime path that
   builds the candidate, the event history (`FireDetectionHistoryService`), the V5 features, P(fire) and the status.
4. SUSPECTED expiry (section 5).
5. Sequential runtime tests of the temporal progression that a static dataset cannot prove.

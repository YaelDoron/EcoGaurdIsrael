# Fire Detection - final architecture (AI Hybrid V5)

The demo runs with `FIRE_DETECTION_DECISION_MODE=ai_hybrid_v5` (set in the git-ignored `backend/.env`; the source default stays `shadow`).
Detailed implementation notes: `fire_detection_ai_hybrid_runtime_v5.md` (runtime), `fire_detection_task9c_final_acceptance.md`
(atomicity, migration, acceptance), `fire_detection_ai_hybrid_policy_v5.md` (the locked policy and its confirmatory gate).

## 1. Decision flow

```
Satellite hotspots + News reports
            |
            v
   Current candidate  (evidence within 60 min / 5 km, one connected group)
            +
   Event history      (evidence already attached to the matching active FireEvent, 24 h window; none for a new candidate)
            |
            v
   25 V5 features     (FireDetectionFeatureExtractorV5 - the same code used for training)
            |
            v
   HistGradientBoosting V5  ->  P(active wildfire)
            |
            v
   AI Hybrid Policy v5.0 (locked)
            |
            v
   NO_EVENT / SUSPECTED / CONFIRMED  ->  FireEvent + AI assessment (one transaction)
```

| State | Rule (locked) | What happens |
|---|---|---|
| **NO_EVENT** | `P < 0.40` | Insufficient likelihood. No FireEvent is created (a matched existing event keeps its state; the candidate is only recorded in memory / the log). |
| **SUSPECTED** | `P >= 0.40` and CONFIRMED not met | A possible wildfire. The event is persisted, keeps collecting evidence and history and is shown on the dashboard, but it is **only monitored**: no severity, spread, response targets, routing, resource allocation, global planning or resource commitment. |
| **CONFIRMED** | `P >= 0.80` **and** at least 2 satellite pixels in the *current* candidate | High ML probability plus current multi-pixel satellite corroboration. **Eligible for emergency response** (severity -> spread -> targets -> routing -> allocation -> global plan -> commitment). Never downgraded. |

Event rules: a SUSPECTED event is promoted (same id, same history) by a later CONFIRMED result and is never dismissed by a weaker one;
`detection_confidence` is the monotonic **peak** likelihood and the latest likelihood lives in the event's AI assessment row.
The rule-based calculator runs only for diagnostics in this mode and can never override the AI status.

**Fire Danger / FFWI is NOT a Fire Detection ML feature.** It is environmental risk context (dashboard risk, preparedness, severity/spread
context) and is never an input to `P(active wildfire)`; a guard test keeps the V5 modules free of any Fire Danger import.

## 2. Operator-facing semantics (UI)
* SUSPECTED: "Monitoring - awaiting additional evidence"; hint text *"SUSPECTED means the system detected evidence consistent with a possible
  wildfire but does not yet have enough corroborating evidence to confirm it."* There is no plan button, no "plan pending", no resources.
* CONFIRMED: keeps the response-plan experience; hint *"CONFIRMED means a high AI likelihood together with corroborating current
  satellite evidence. The likelihood is a model estimate, not certainty."*
* The AI value is called **AI likelihood**, always as a whole percentage (0.87 -> 87%), never "certainty". The presentation is chosen by the
  event's **persisted decision mode** (`ml_summary.mode` / `ml_assessment.mode`), never by the global `.env` and never by the mere presence of
  an AI score (shadow events carry one too).
  * `ai_hybrid_v5`: the dashboard card shows only "AI likelihood 79%" (no Rule, no Peak). Event details show **Latest AI likelihood** and
    **Peak AI likelihood** (the highest value the event reached - an event confirmed at 87% whose later assessment is 79% stays CONFIRMED),
    plus an AI Assessment block: detection mode, latest verdict, satellite passes, current satellite pixels (history flag, model and policy names stay in the API only)
    (AI Hybrid Policy v5.0). Peak is shown only for events the AI path created (`methodology = ECOGUARD_AI_HYBRID_DETECTION`), because for an
    event created by a legacy rule decision `detection_confidence` is a rule score.
  * `rule_only` / `shadow` / `hybrid` (and old persisted rows): the existing "Rule | AI" presentation.
  * Never shown: the 25 features, file paths, configuration or secrets.

## 3. Research path (kept, including the failures)

| Step | What was done | Outcome |
|---|---|---|
| V3 | Logistic Regression on synthetic data, integrated in **shadow** mode (recorded, never decides) | Baseline ML + observability |
| V4 | Redesigned synthetic dataset (regimes separated from labels), Fire Danger context features | Binary-primary acceptance gate **FAILED** |
| V5 | History-aware features (passes, span, centroid stability, FRP/brightness trends), Fire Danger removed from detection | Binary-primary gate **still FAILED** (Task 7) |
| AI Hybrid V5 | Same HGB used as a likelihood estimator + an explicit uncertainty state (SUSPECTED) + one deterministic corroboration guardrail, thresholds locked *before* a fresh confirmatory dataset was generated | Confirmatory **policy** gate **PASSED** (Task 8) |
| Runtime | Response-eligibility semantics (9A), runtime integration (9B), atomicity + migration + end-to-end acceptance + activation (9C) | Active in the demo |

The failed binary-primary gates are part of the evidence for the design: a hard fire / no-fire label was not defensible on this data,
which is why the final design reports uncertainty explicitly instead of forcing every candidate to a yes/no.

## 4. Limitations
* Training data **and** confirmatory validation are **synthetic**; results are not real-world detection accuracy.
* The probability is model confidence, not real-world certainty.
* Early fires commonly remain SUSPECTED (a lone hotspot or news alone can never be CONFIRMED).
* CONFIRMED has exactly one corroboration branch (`P >= 0.80` + >= 2 current satellite pixels).
* Site recurrence (a known industrial heat source that fires again) is not modelled; a single satellite platform (NOAA-20 VIIRS) is assumed.
* The model treats a hotspot that never moves or grows as an industrial source, and needs a drifting, growing front over several passes to reach P >= 0.80.
* Passes more than 6 h apart become separate FireEvents (the unchanged event-matching window).
* SUSPECTED expiry is not implemented (see below).

## 5. SUSPECTED expiry - design decision (not implemented)
Proposal: `SUSPECTED` + no meaningful new evidence for 6 hours -> `DISMISSED`.
* **Not necessary for the final demo.** A demo run resets the runtime state, so stale SUSPECTED events do not accumulate, and a SUSPECTED
  event never triggers response work, so leaving one visible cannot waste resources.
* **Not cheap to add safely:** it needs a clock-driven sweep (the system is otherwise event-driven), a definition of "meaningful" evidence
  (a weak NO_EVENT pass is currently attached to the event as history), interaction with simulated time, and lifecycle/commit-release checks.
  Adding it now would risk the verified AI flow.
* **Compatible with the current design:** DISMISSED is already outside both the active and response-eligible sets, the lifecycle service can
  dismiss suspected events, and nothing in the AI path depends on a SUSPECTED event staying open. Future work.

## 6. Operational note: one backend per database
The Neon demo database is shared. If a SECOND backend/runner (another terminal, machine or teammate, possibly still in `shadow` mode) writes
to the same database while the demo runs, its events are created by the legacy rule path (`methodology = ECOGUARD_MULTI_SOURCE_DETECTION`,
`detection_confidence` = rule confidence) and its mandatory demo reset wipes the AI run's data mid-run; the AI backend may then re-assess
those events. Symptoms: rule-methodology events in an AI demo, evidence stamped with two different simulation start times, and a `Rule | AI`
card next to `AI likelihood` cards. Run exactly one backend against a demo database.

## 7. Known external issue: Groq translation LLM returns HTTP 401
Independent of detection. The `GROQ_API_KEY` used by the news/translation `TextProcessor` is present and well-formed, but Groq itself
rejects it (`invalid_api_key`, also on the token-free `GET /models` call), i.e. the credential is stale/revoked. Model and provider
configuration are not the cause. Both structured news analysis and translation share `_call_llm`, so both fall back gracefully (analysis ->
"unavailable" news strength, translation -> original text). Fix = issue a new Groq key and update `GROQ_API_KEY` in `.env` (or point
`llm.provider` at `gemini`, whose key is configured).

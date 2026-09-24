# Fire Detection — Event Evidence History (Task 5B)

Infrastructure for observing one fire across several satellite overpasses. It does **not** add V5, train a
model, change `rule_only` / `shadow` / `hybrid`, or make ML primary.

## Why it was needed

A satellite typically revisits a fire every few hours, but the pipeline could only reason about evidence that
correlates within **60 minutes**. Two failures were reproduced first (`tests/acceptance/test_fire_detection_event_history_5b.py`,
written against the unchanged code):

| Scenario | Failure before the fix |
|---|---|
| Hotspot at T+0 creates a FireEvent; a new hotspot at T+3h matches the *same* event (5 km / 6 h event matching) | `_update_existing_event` re-evaluated **all** attached evidence as one `FireDetectionCandidate`, which requires one connected 60-minute component → `ValueError` ("evidence must form one connected component") → `detect()` returned `success=False` and attached nothing |
| HYBRID: a lone LOW hotspot (rule = NO_EVENT) is escalated to SUSPECTED by ML; next cycle | The re-evaluation used the rule alone and raised `Combined active-event evidence unexpectedly evaluated as NO_EVENT.` → `detect()` failed |

## Correlation window ≠ history window

```
Incoming evidence
      │
      ▼
current candidate correlation   (5 km / 60 min, chained; lookback 120 min)   ← UNCHANGED
      │
      ▼
FireDetectionCandidate  ──►  current decision (rule / ML / policy)
      │
      ▼
FireEvent   (matching: 5 km / 6 h on updated_at)                              ← UNCHANGED
      │
      ▼
Event Evidence History   (everything attached, last 24 h, may span hours)
      │
      ▼
future persistence features (pass count, span, centroid stability, FRP trend)
```

| Concept | Value | Where | Meaning |
|---|---|---|---|
| Candidate correlation | 5 km, 60 min | `fire_detection_config.py` | which evidence forms **one** candidate (unit of the decision) |
| Candidate lookback | 120 min | `EVIDENCE_LOOKBACK_MINUTES` | how far back new candidates are built |
| Event matching | 5 km, 6 h | `fire_event_config.py` | which existing FireEvent a candidate belongs to |
| **Event history window** | **24 h** | `FIRE_EVENT_EVIDENCE_HISTORY_HOURS` | how much of an event's attached evidence is read for history |
| **Satellite pass gap** | **30 min** | `SATELLITE_PASS_GAP_MINUTES` | hotspots of one platform closer than this are one pass |

**Rationale for 24 h**: long enough to contain several overpasses of the same fire (a polar orbiter revisits a
mid-latitude site a few times per day) and a day/night cycle, short enough not to blend an old fire with an unrelated
later one at the same place. It is a *reading* window only; it never extends candidate correlation or event matching.
**30 min** is comfortably above the seconds-to-minutes spread of pixels within one acquisition and far below the
hours between overpasses of one satellite.

## Model

`FireDetectionEventHistory` (`src/models/fire_detection_event_history.py`) — frozen dataclass: `fire_event_id`,
`as_of`, `window_start`, chronological deduplicated `evidence`, derived `satellite_passes`, and `unresolved_evidence`
(attached refs that could not be reloaded). It intentionally does **not** enforce the 60-minute candidate invariant,
performs no ML feature extraction and makes no decision.

`FireDetectionEvidence` gained optional `satellite_name` / `satellite_instrument` (the persisted `satellite` /
`instrument` hotspot columns). They are used only to keep different platforms' passes apart.

## Service

`FireDetectionHistoryService.build_history(stored_fire_event, as_of)`
(`src/services/fire_detection/fire_detection_history_service.py`): reads the event's attached refs, reloads them through
`FireDetectionEvidenceService.resolve_evidence_refs` (same normalization as detection; satellite **and** news), keeps only
evidence with `as_of − 24 h ≤ observed_at ≤ as_of`, dedupes by source-aware identity, orders chronologically. Read-only.
`resolve_refs_tolerantly` reports refs that no longer resolve (`ValueError`) instead of failing; other exceptions
(database outages) propagate.

## Satellite-pass grouping

Deterministic, using only persisted fields (`detected_at`, `satellite`, `instrument`):

1. only SATELLITE evidence;
2. partition by `(satellite, instrument)` — no platform name is hard-coded; unknown platform = `(None, None)`;
3. within a partition sort by time; a new pass starts when the gap to the previous hotspot exceeds 30 min
   (single linkage in time, so there are no fixed-bucket boundary artefacts).

Limits: FIRMS `acq_time` has minute resolution and its timezone is not confirmed (see `SatelliteHotspotMapper`); the
grouping cannot recover an overpass whose pixels are >30 min apart, nor merge two platforms observing together.

## Conservative summaries (primitives for a future V5)

`distinct_satellite_pass_count`, `satellite_observation_span_minutes`, `satellite_centroid_shifts_km` /
`satellite_centroid_max_shift_km` (None below 2 passes), `frp_trend()` / `brightness_trend()` (least-squares slope per
hour over per-pass max; **None below 3 passes with a measurement**, None across mixed platforms unless one is requested).
Missing FRP/brightness stays missing — never zero-filled.

Deliberately absent: a *repeat-detection ratio*. FIRMS reports detections only; the number of overpasses that **observed
the site and saw nothing** (the denominator) is not in our data. `distinct_satellite_pass_count` is the honest primitive.

## Existing-event update fix

`FireDetectionAgent._update_existing_event` now:

1. loads the event's attached refs and resolves them tolerantly (an *old* ref that vanished is skipped; the *current*
   evidence must resolve or the cycle fails as before);
2. evaluates the **current candidate**: the connected component of attached+new evidence that correlates (5 km / 60 min,
   chained — `FireDetectionCandidate.are_correlated`, the single correlation definition) with the newly supplied
   evidence. Older waves stay attached as history and are never forced into one candidate. When all attached evidence is
   correlated this is identical to the previous "re-evaluate the union";
3. re-runs rule + ML/policy once on that evidence (ML call count unchanged);
4. if the final decision is NO_EVENT (HYBRID escalation whose ML score later dropped) it no longer raises: the event is
   not deleted or downgraded — location/status/confidence stay and only `updated_at` advances;
5. when older evidence was excluded, the event keeps its **peak confidence** (a single weak later hotspot must not
   lower a fire seen strongly earlier); status stays monotonic (CONFIRMED is never downgraded).

`FireEvent` identity and `detected_at` are preserved across waves; distinct fires (>5 km apart) still get distinct events.

## Not implemented (by design)

* **FRP trend** exists as a history primitive only; nothing consumes it yet.
* **Site recurrence** (the same place burning on different days) is *event persistence's* opposite side and is not
  modelled: a future query over the hotspot table, separate from one event's history.
* **Multi-platform evidence**: passes of different platforms are kept apart and trends are not computed across them.
* No V5 extractor, dataset, model or artifact; no change to `.env`, V4 benchmark data or the news subsystem.

## Shared pass primitives (Task 6)

`satellite_pass_trend`, `satellite_pass_time_span_minutes` and `satellite_pass_centroid_rms_km` are public pure functions in
`fire_detection_event_history.py`. `FireDetectionEventHistory.frp_trend()` / `brightness_trend()` delegate to
`satellite_pass_trend`, and the V5 feature extractor calls the same functions, so a runtime history and a training history
are summarised by one algorithm (see [fire_detection_feature_schema_v5.md](fire_detection_feature_schema_v5.md)).

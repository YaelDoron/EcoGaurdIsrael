# Fire Detection ML — V5 Feature Schema and Extractor (Task 6)

V5 answers: *given the current wildfire evidence AND the event's persistence history, how likely is it that an actual
wildfire exists?* It uses **current satellite evidence + satellite persistence (passes) + news corroboration**.
Fire Danger / FFWI is deliberately **not** a V5 input; it stays a separate risk context.

V5 is additive. V3 (runtime model), V4 (`training_v4.csv`, its extractor, reports, scenario families) and the
runtime decision policy are untouched, and `rule_only` / `shadow` / `hybrid` keep their meaning. Nothing is trained and
no FireEvent can be affected: no runtime module imports a V5 module (a test enforces this).

## Canonical schema — `FIRE_DETECTION_FEATURE_NAMES_V5` (25 features, this order)

| # | Feature | Group | Type |
|---|---|---|---|
| 1–3 | `satellite_low_count`, `satellite_nominal_count`, `satellite_high_count` | retained | int |
| 4–6 | `satellite_frp_available_ratio`, `satellite_frp_mean`, `satellite_frp_max` | retained | float |
| 7–9 | `satellite_brightness_available_ratio`, `satellite_brightness_mean`, `satellite_brightness_max` | retained | float |
| 10–14 | `news_none_count`, `news_weak_count`, `news_moderate_count`, `news_strong_count`, `news_unknown_count` | retained | int |
| 15 | `satellite_frp_sum` | current | float |
| 16 | `satellite_frp_std` | current | float |
| 17 | `satellite_brightness_std` | current | float |
| 18 | `satellite_night_fraction` | current | float, nullable |
| 19 | `satellite_cluster_radius_km` | current | float, nullable |
| 20 | `news_satellite_lag_minutes` | current | float (signed), nullable |
| 21 | `satellite_pass_count` | history | int |
| 22 | `satellite_history_span_minutes` | history | float, nullable |
| 23 | `satellite_centroid_stability_km` | history | float, nullable |
| 24 | `satellite_frp_trend_per_hour` | history | float (signed), nullable |
| 25 | `satellite_brightness_trend_per_hour` | history | float (signed), nullable |

The 14 retained features are computed by the unchanged V3 extractor (`FireDetectionFeatureExtractorV3`), not
re-implemented. No extra `satellite_day_night_available_ratio` was needed: unknown day/night is expressed as NaN in
`satellite_night_fraction` (see below), so the feature count stays at 14 + 6 + 5 = **25**.

### Removed from V4 and why

| Removed | Why |
|---|---|
| `time_span_minutes` | Bounded by the 60-minute candidate rule and generated identically for both labels; V3/V4 showed it is a geometry shortcut, not evidence. Real persistence over hours is now `satellite_history_span_minutes`, computed from passes. |
| `max_pairwise_distance_km` | Same: bounded by the 5 km candidate rule; replaced by the satellite-only `satellite_cluster_radius_km` (news geocoding no longer contaminates it). |
| `fire_danger_available`, `fire_danger_score`, `fire_danger_age_minutes` | Fire Danger is risk context, not evidence that a fire exists; it is kept out of the active-fire classifier (V4 showed it acting as a weak/noisy prior). |

## Current-evidence features (the CURRENT `FireDetectionCandidate` only)

| Feature | Definition | Missing semantics |
|---|---|---|
| `satellite_frp_sum` | sum of available FRP values | `0.0` when none available; `satellite_frp_available_ratio` keeps "unmeasured" ≠ "measured 0" |
| `satellite_frp_std` | population std (ddof = 0) of available FRP | `0.0` with < 2 values |
| `satellite_brightness_std` | population std of available brightness | `0.0` with < 2 values |
| `satellite_night_fraction` | among hotspots with known `day_night` (`D`/`N`), share that are `N` | **NaN** when none is known (or no hotspot): unknown is not "0 % night" |
| `satellite_cluster_radius_km` | RMS Haversine distance of the **satellite** pixels from their own centroid | `0.0` for one hotspot; **NaN** with no hotspot. News coordinates are never mixed in |
| `news_satellite_lag_minutes` | `first news time − first satellite time` (UTC), minutes; positive = news after the satellite, negative = news first | **NaN** unless both families are present (0.0 would claim "simultaneous") |

## History features (the *effective* satellite history)

Effective history = satellite evidence of the current candidate ∪ satellite evidence of the supplied
`FireDetectionEventHistory`, de-duplicated by source-aware identity (an event that already has the current hotspots
attached does not count them twice). With `history=None` (no FireEvent yet) it is the candidate alone. The current
candidate never has to span more than 60 minutes; hours-old passes come from the history.

All of it uses the **Task 5B primitives** — one implementation, no second algorithm:
`group_satellite_passes` (per-(satellite, instrument) time-gap single linkage, 30 min), `satellite_pass_trend`,
`satellite_pass_time_span_minutes`, `satellite_pass_centroid_rms_km` (module `fire_detection_event_history`).

| Feature | Definition | Needs | Missing semantics |
|---|---|---|---|
| `satellite_pass_count` | number of distinct satellite passes (not pixels) | — | `0` when there is no satellite evidence at all; `1` for a first detection |
| `satellite_history_span_minutes` | time between the earliest and latest **pass** (a pass = midpoint of its hotspot times) | ≥ 1 pass | `0.0` for one pass; **NaN** with no pass |
| `satellite_centroid_stability_km` | RMS Haversine distance of each pass centroid from the mean of the pass centroids (each pass once, however many pixels) | **≥ 2 passes** | **NaN** below 2 passes (0.0 would read "observed twice, perfectly stable") |
| `satellite_frp_trend_per_hour` | least-squares slope of the per-pass **max FRP** over elapsed hours | **≥ 3 passes with an FRP** | **NaN**; passes without FRP are skipped, never zero-filled; NaN across mixed platforms |
| `satellite_brightness_trend_per_hour` | same for the per-pass max brightness | **≥ 3 passes with a brightness** | **NaN** |

`satellite_pass_count == 0` ⇔ no satellite evidence in the effective history, in which case every satellite-derived
nullable feature is NaN. A news-only candidate can still inherit passes from its event's history (pass count ≥ 1 while
the *current* satellite features stay NaN/0).

`FireDetectionFeaturesV5` enforces this internally: NaN is accepted only in the nullable features and only in the states
above (e.g. a trend from 2 passes, a span for zero passes, or a satellite count without a pass is rejected), so a
malformed row cannot be trained on silently. In the CSV, NaN is an **empty cell** (loaded as `None`).

## The extractor

`FireDetectionFeatureExtractorV5.extract(candidate, history=None)` is pure: no repository/service/DB, no ML call, no
label, regime, subtype, environment, rule status or FireEvent status (it does not import them), no Fire Danger, no
imputation or scaling, and it never mutates the frozen history. Runtime orchestration will later supply the history
(built by `FireDetectionHistoryService`); the training generator supplies a synthetic `FireDetectionEventHistory`.

Not implemented on purpose: multi-day **site recurrence** (needs a separate repository query), **multi-platform
diversity** (runtime is effectively NOAA-20 only) and **news source count** (`FireDetectionEvidence` does not carry
`source_feed`; the evidence model was not extended again in this task).

## Train / runtime parity

`tests/ml/fire_detection/test_fire_detection_dataset_v5.py` proves it three ways: every dataset row equals a fresh
extractor run on the same candidate + history; a history that repeats the candidate equals one that does not; and 18
rows (3 per regime) are pushed through the real SQLite repositories → `FireDetectionEvidenceService` → `FireEvent` →
`FireDetectionHistoryService` → extractor and reproduce the training features exactly.

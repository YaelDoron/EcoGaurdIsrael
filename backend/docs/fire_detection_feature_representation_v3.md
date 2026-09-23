# Fire Detection ML - V3 Evidence Enrichment & Feature Representation

## Status

Offline analysis, evidence enrichment, and comparison only. `training_v1.csv`/`training_v2.csv`, their models, and their comparison scripts are untouched. `FireDetectionAgent` still uses the unchanged deterministic `FireDetectionCalculator` at runtime - no ML model is loaded, and no classifier is selected here. See [fire_detection_ml.md](fire_detection_ml.md) and [fire_detection_model_comparison_v2.md](fire_detection_model_comparison_v2.md) for the earlier tasks this builds on.

## Why V3 was necessary

Task 3's grouped (unseen-scenario-family) evaluation of V2 exposed a **feature representation** limitation, not an algorithm problem: with only satellite confidence label + evidence counts + geometry, an isolated single-item candidate is nearly indistinguishable regardless of what actually happened. Concretely, `news_only_before_satellite` (a real early fire, reported only by one news article) and `news_rumor_no_fire` (a false rumor) both reduce to `news_count=1, time_span_minutes=0, max_pairwise_distance_km=0` - the two "twin" families occupied the same point in feature space. The same held for `satellite_only_before_news` (a real fire's first satellite detection) vs. `high_confidence_satellite_false_positive` (a false alarm), both a single HIGH-confidence hotspot with no other evidence. No amount of additional synthetic rows fixes this: the features themselves carry no information to separate them.

V3 fixes this by exposing **richer evidence content** already available in the system - selected satellite physical measurements and a structured semantic reading of the news article text - so these previously-identical candidates can, at least partially, be told apart.

## Existing satellite data now exposed

`SatelliteHotspot.frp` and `.brightness` already existed (FIRMS ingestion, the database model, and the demo simulator all already populate them) but `FireDetectionEvidenceService._normalize_satellite()` previously discarded them when building `FireDetectionEvidence`. Task 4 makes `_normalize_satellite()` copy `frp` -> `satellite_frp`, `brightness` -> `satellite_brightness`, and `day_night` -> `satellite_day_night` onto the normalized evidence. No new FIRMS request, no new persisted field, no change to the FIRMS collection contract - this is exposure of already-collected data to the ML layer, nothing more. `satellite_day_night` is preserved as normalized metadata but is **not** a V3 ML feature (no specific justification for it was established in this task).

## News semantic analysis

`NewsMonitoringAgent`'s LLM call is extended from "location only" to one structured call returning both `locationName` and `wildfireSignalStrength` (`NewsWildfireSignalStrength`: `NONE` / `WEAK` / `MODERATE` / `STRONG`), via `TextProcessor.analyze()` (replacing the old `extract_location()`; see `src/external/news/news_client.py`).

This is a **semantic reading of the article's own wording** - not a calibrated probability and not ground truth:

- `NONE`: the text was analyzed and provides no meaningful active-fire evidence (retrospective/preventive article, unrelated keyword hit, or an explicit "it was a false alarm" report).
- `WEAK`: indirect/possible evidence (smoke reported, rumor, unverified social report, "suspected" flames).
- `MODERATE`: the text directly reports an active wildfire, but preliminarily/indirectly.
- `STRONG`: explicit strong evidence (visible flames, firefighting response, evacuation, an official statement).

A `STRONG` article can still be wrong; a `WEAK` article can still describe a real fire - the overlap is intentional and preserved in both the prompt design and the synthetic training data (see below). This is deliberately distinct from an LLM self-reporting a number like `fireProbability=0.87`, which would not be a real calibrated statistic.

**`wildfire_signal_strength=None` means "no reliable analysis is available"** (LLM call failed, malformed response, or an unrecognized value) - never conflated with `NewsWildfireSignalStrength.NONE`, which means "analyzed, no signal found." `TextProcessor.analyze()` never raises; on any failure it returns the explicit unavailable state.

### LLM JSON contract

```json
{"locationName": "יער ירושלים", "wildfireSignalStrength": "moderate"}
```
or
```json
{"locationName": null, "wildfireSignalStrength": "none"}
```

`temperature=0`, structured JSON response format (unchanged from the existing location-only call). The prompt instructs the model to judge only the supplied title/summary, not use outside knowledge, not infer a fire is true merely because a keyword appears, distinguish smoke/rumor/suspicion from explicit active-fire reporting, use the existing Hebrew-preposition location-cleaning rule, and return JSON only (no markdown, no explanation). The parser validates `wildfireSignalStrength` against the four allowed values and raises internally on anything else - `analyze()` catches this and converts it to the unavailable state rather than letting an unexpected value silently enter the domain.

## Persistence and migration

`WildfireReport.wildfire_signal_strength: NewsWildfireSignalStrength | None = None` (backward-compatible default) is persisted via a new nullable `wildfire_reports.wildfire_signal_strength` column (`WildfireReportDB`), written/read by `NewsRepository`. Existing rows have `NULL` - never backfilled, never fabricated as `NONE`. Migration: `scripts/migrate_add_news_wildfire_signal_strength.py --inspect-only` / `--apply`, following the project's existing safe-migration pattern (nullable column, no backfill, preserves row count). **Not applied automatically to Neon by this task** - it must be run manually; see the final report for the exact command.

## Evidence enrichment (`FireDetectionEvidence`)

New optional, source-specific fields with strict validation (SATELLITE-only fields rejected on NEWS evidence and vice versa):

| Field | Source | Meaning |
| --- | --- | --- |
| `satellite_frp` | SATELLITE | Fire Radiative Power (from `SatelliteHotspot.frp`) |
| `satellite_brightness` | SATELLITE | Brightness temperature (from `SatelliteHotspot.brightness`) |
| `satellite_day_night` | SATELLITE | Day/night flag (metadata only - not a V3 feature) |
| `news_wildfire_signal_strength` | NEWS | LLM-derived semantic signal |

`FireDetectionEvidenceService._normalize_satellite()`/`_normalize_news()` now populate these. **Everything else is unchanged**: lookback window, distance/time correlation thresholds, connected-component logic, candidate construction, and location-name trust rules are untouched, and `FireDetectionCalculator` still uses only `satellite_confidence` and evidence presence/correlation - it does not read FRP, brightness, or the news signal, and its outputs are unchanged (verified: all existing deterministic-confidence tests still pass unmodified).

## V3 feature schema

Central definition: `FIRE_DETECTION_FEATURE_NAMES_V3` in `src/ml/fire_detection/fire_detection_features_v3.py` (16 features, order fixed):

| Feature | Meaning |
| --- | --- |
| `satellite_low_count` / `satellite_nominal_count` / `satellite_high_count` | Satellite confidence counts (same as V1/V2) |
| `satellite_frp_available_ratio` | Fraction of satellite items with a measured FRP |
| `satellite_frp_mean` / `satellite_frp_max` | Mean/max FRP over items where it was measured |
| `satellite_brightness_available_ratio` | Fraction of satellite items with measured brightness |
| `satellite_brightness_mean` / `satellite_brightness_max` | Mean/max brightness over measured items |
| `news_none_count` / `news_weak_count` / `news_moderate_count` / `news_strong_count` | Count of news items at each signal level |
| `news_unknown_count` | Count of news items with no reliable analysis (`None`) |
| `time_span_minutes` / `max_pairwise_distance_km` | Same geometry features as V1/V2 |

`satellite_count` is **not** a V3 feature: Task 3 showed it is exactly reconstructable from the three confidence counts, and its removal had negligible effect. `*_available_ratio` explicitly distinguishes "no satellite evidence" and "evidence present but unmeasured" from "measured as zero" - both yield `(0.0, 0.0, 0.0)` for that measurement's three fields, by design (see `FireDetectionFeatureExtractorV3._availability_stats`).

`FireDetectionFeatureExtractorV3` (`src/ml/fire_detection/fire_detection_feature_extractor_v3.py`) is a new, separate extractor - `FireDetectionFeatureExtractor` (V1/V2) is untouched. It is pure (no Neon, no external APIs, no LLM, no repositories, no labels), deterministic, and order-independent, delegating candidate validity to `FireDetectionCandidate` like every other Fire Detection component.

## Missing data handling

- FRP/brightness: "no satellite evidence" and "satellite evidence but no measurement" are both `(available_ratio=0.0, mean=0.0, max=0.0)` - the ratio field carries the "was it actually measured" information, so `mean=0.0`/`max=0.0` is never confused with "measured as literally zero."
- News signal: `news_unknown_count` captures historical/failed-analysis news items explicitly, rather than silently dropping them or miscounting them as `NONE`.

## Synthetic dataset (V3)

`training_v3.csv`: 4000 samples (seed 42, deterministic), 2000 positive / 2000 negative, 20 scenario families (the same 20 family *names* used conceptually as V2, now generating enriched FRP/brightness/news-signal alongside evidence), grouped into 5 **scenario archetypes** (`NEWS_ONLY`, `SATELLITE_ONLY`, `SATELLITE_NEWS`, `MULTI_SATELLITE`, `MIXED_CONFIDENCE`) - metadata only, never a feature, but unlike `scenario_family`, every archetype deliberately contains **both** labels (verified by the Part 27 sanity checks), enabling a leave-one-archetype-out generalization check.

FRP/brightness ranges and news-signal pools are deliberately **overlapping** across labels (e.g. `high_confidence_satellite_false_positive`, a false alarm, has FRP range 20-140 - overlapping substantially with the real-fire `satellite_only_before_news`'s 10-100 range; `news_rumor_no_fire` occasionally reads `STRONG` despite being false, and `news_only_before_satellite`, a real early fire, is usually `WEAK`). No feature/range was hand-tuned by re-running against held-out metrics - see "no shortcut-hunting on the test set" below.

**No integration is used to generate this data.** The synthetic generator assigns FRP/brightness/news-signal values deterministically by scenario family - it never calls the real LLM, FIRMS, or any network service. See "real runtime signal vs. synthetic training signal" below.

## Real runtime signal vs. synthetic training signal

At **runtime**, `news_wildfire_signal_strength` is produced by the real LLM (`TextProcessor.analyze()`) reading actual article text. During **ML training**, the V3 synthetic generator assigns a signal value deterministically per scenario family (from a fixed, hand-designed pool) - it is modeling what such an LLM signal *could* look like across a range of plausible real/false scenarios, not reproducing actual LLM behavior. V3 training therefore remains entirely synthetic; its metrics measure performance on this synthetic distribution, not real-world LLM-derived accuracy.

## Sanity validation (Part 27)

`validate_training_samples_v3` (`src/ml/fire_detection/fire_detection_dataset_validation_v3.py`) checks, among others: both labels present with reasonable balance; LOW/NOMINAL/HIGH satellite confidence under both labels; measured *and* missing FRP under both labels; measured *and* missing brightness under both labels; WEAK/MODERATE/STRONG news signal under both labels; news `UNKNOWN` supported; satellite-only/news-only/multi-source/mixed-confidence examples under both labels; no metadata column in the feature schema. `training_v3.csv` passes all checks; generation fails loudly (raises, does not write the CSV) if any check fails.

## Univariate shortcut analysis (Part 28)

For every V3 feature, mean/median/range by label plus a univariate ROC-AUC (`src/ml/fire_detection/fire_detection_shortcut_analysis.py`, flag threshold AUC ≥ 0.95). The highest observed univariate AUC on `training_v3.csv` is **0.703** (`max_pairwise_distance_km`) - well below the flag threshold. No single V3 feature comes close to acting as a shortcut on its own.

## Feature-vector collision analysis (Part 29)

Exact feature-vector collisions between opposite labels, computed identically for V2 and V3:

| Dataset | Total rows | Distinct vectors | Colliding vectors | Colliding rows | Collision rate |
| --- | --- | --- | --- | --- | --- |
| V2 | 3000 | 2456 | 3 | 494 | **16.47%** |
| V3 | 4000 | 3670 | 6 | 232 | **5.80%** |

Targeted twin-family check (distinct feature vectors per family, and vectors shared between the twin pair):

| Pair | V2 (a / b / shared) | V3 (a / b / shared) |
| --- | --- | --- |
| `news_only_before_satellite` / `news_rumor_no_fire` | 88 / 1 / **1** | 101 / 5 / **3** |
| `satellite_only_before_news` / `high_confidence_satellite_false_positive` | 84 / 1 / **1** | 205 / 205 / **2** |

In V2, `news_rumor_no_fire` and `high_confidence_satellite_false_positive` each had exactly **one** distinct feature vector total (fully degenerate - every row of that family was identical), which is why they collided completely with their positive twin. In V3, continuous FRP/brightness values essentially eliminate the satellite-pair collision (205 distinct vectors each, only 2 accidental exact ties) - a clear, quantified improvement. The news-pair collision shrinks (88→101 and 1→5 distinct vectors, 1→3 shared) but does not disappear, because the news signal remains a small discrete categorical value (4 levels + unknown) rather than a continuous measurement - some exact collision is structurally expected there.

## Standard evaluation (V3 vs V2)

| | V2 held-out Acc / F1 / ROC-AUC | V3 held-out Acc / F1 / ROC-AUC |
| --- | --- | --- |
| Logistic Regression | 0.678 / 0.677 / 0.767 | **0.758 / 0.754 / 0.849** |
| Random Forest | 0.778 / 0.782 / 0.853 | **0.831 / 0.838 / 0.928** |

Both models improve on every standard metric with the richer V3 features. Random Forest remains ahead of Logistic Regression on standard evaluation, as in V1/V2.

## Grouped (unseen scenario-family) evaluation

| | V2 grouped Acc / F1 / ROC-AUC | V3 grouped Acc / F1 / ROC-AUC |
| --- | --- | --- |
| Logistic Regression | 0.543±0.278 / 0.511±0.329 / 0.581±0.362 | **0.584±0.098 / 0.520±0.237 / 0.590±0.216** |
| Random Forest | 0.382±0.158 / 0.396±0.214 / 0.223±0.233 | **0.462±0.112 / 0.467±0.242 / 0.378±0.215** |

(n_splits=7 for both V2 and V3, chosen the same way: the largest value ≤10 that keeps every fold's train/validation sets label-complete and family-disjoint.) Both models show a **mild improvement** in central tendency, and a **meaningfully smaller standard deviation** (e.g. LR accuracy std 0.278→0.098) - V3 is somewhat more stable across held-out families, even though absolute performance is still far from strong. This is genuine progress, not a fix: unseen-scenario-family generalization remains weak, especially for Random Forest (grouped ROC-AUC 0.378, still below the standard-evaluation ROC-AUC by a wide margin).

## Archetype holdout (leave-one-archetype-out)

A new, complementary generalization check: hold out one entire `scenario_archetype` (evidence composition/shape) at a time, 5 folds total (`LeaveOneGroupOut`).

| | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.683±0.120 | 0.653±0.354 | 0.479±0.396 | 0.467±0.379 | **0.721±0.143** |
| Random Forest | 0.648±0.108 | 0.473±0.298 | 0.510±0.385 | 0.471±0.340 | 0.618±0.171 |

Archetype holdout scores noticeably **higher** than family-level grouped CV for both models. This makes sense: family-grouped CV can remove *every* family of a given evidence shape from training in a single fold (as happened for the `news_only`/`satellite_only` twin pairs in Task 3), while archetype holdout always leaves other archetypes - and other same-composition scenario variety within them - in training; the harder case is a single, narrow, unfamiliar scenario, not an unfamiliar broad evidence shape. Per-archetype (Random Forest): `satellite_news` generalizes best (accuracy 0.791), `news_only` is predicted almost entirely negative (predicted-positive rate 0.052 vs. an actual positive fraction of 0.337 - the model badly under-predicts fire for this held-out composition), and `satellite_only` is similarly skewed negative (predicted-positive rate 0.076 vs. actual 0.257).

## Feature-group ablation

Four feature groups (`src/ml/fire_detection/fire_detection_feature_groups_v3.py`): **BASE** (satellite confidence counts + a derived `news_total_count` presence-only signal + geometry - no physical/semantic detail), **BASE+SATELLITE PHYSICAL** (+ FRP/brightness), **BASE+NEWS SEMANTIC** (+ the 5 news signal counts, replacing the coarse total), **FULL V3** (both - identical to the official V3 schema).

| Group | LR standard F1 | LR grouped F1 | LR grouped ROC-AUC | RF standard F1 | RF grouped F1 | RF grouped ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| BASE | 0.678 | 0.510 | 0.528 | 0.764 | 0.396 | 0.184 |
| +SATELLITE PHYSICAL | 0.697 | 0.487 | 0.477 | 0.784 | 0.427 | 0.233 |
| +NEWS SEMANTIC | 0.729 | 0.534 | **0.618** | 0.812 | 0.474 | 0.382 |
| FULL V3 | 0.742 | 0.520 | 0.591 | **0.828** | **0.467** | 0.378 |

**News semantic features contributed more than satellite physical features**, on both standard and grouped metrics, for both models. Satellite physical features alone give a modest standard-evaluation lift but barely move (Random Forest) or even slightly hurt (Logistic Regression's grouped ROC-AUC drops from 0.528 to 0.477) grouped generalization on their own. News semantic features alone give the single largest grouped-generalization jump of any group (LR grouped ROC-AUC 0.528→0.618 - in fact slightly *higher* than the full V3 set's 0.591, suggesting satellite-physical features add some noise for grouped generalization under a linear model even though they still help standard evaluation). Both enrichments together (FULL V3) give the best standard-evaluation numbers for both models.

## Random Forest overfitting

| | Train Acc / F1 | Test Acc / F1 | Gap (Acc / F1) |
| --- | --- | --- | --- |
| V1 | 0.916 / 0.908 | 0.835 / 0.823 | ~8.1 / ~8.5 pts |
| V2 | 0.958 / 0.956 | 0.778 / 0.782 | ~17.9 / ~17.3 pts |
| V3 | 0.975 / 0.975 | 0.831 / 0.838 | ~14.4 / ~13.7 pts |

Richer V3 features **reduce** the overfitting gap that widened in V2, but do not eliminate it - V3's gap remains larger than V1's. No hyperparameter tuning was applied to influence this; the same fixed `RandomForestClassifier(n_estimators=300, random_state=42, class_weight=None)` baseline is used across V1/V2/V3, exactly to isolate the effect of features from the effect of tuning.

## Does the representation limitation appear reduced?

**Partially, yes - but not solved.** Concrete, quantified evidence: the twin-family collision essentially disappeared for the satellite pair (205 distinct vectors, 2 accidental ties, down from total collision) and shrank for the news pair; the overall dataset collision rate dropped from 16.47% to 5.80%; grouped-CV central tendency improved modestly with meaningfully lower variance; archetype-level generalization is measurably better than family-level generalization for both models. At the same time, grouped-CV and archetype-holdout ROC-AUC remain well below standard-evaluation ROC-AUC for both models (most starkly for Random Forest), and per-archetype/per-family predictions still show systematic failure modes (e.g. `news_only` and `satellite_only` archetypes strongly under-predicted as fire when held out entirely). The richer evidence representation is a real, measured step forward, not a fix for unseen-scenario-family generalization.

## Synthetic limitation (repeated)

**V3 is still trained on synthetic labels and synthetic enriched evidence.** The FRP/brightness/news-signal *distributions* were hand-designed to overlap realistically, and the news-signal generation is a deterministic stand-in for what an LLM might produce - it is not the real LLM. Grouped CV and archetype holdout are stronger internal robustness checks, but neither constitutes real-world validation. None of the metrics in this document should be read as operational wildfire-detection accuracy.

## No runtime integration

`FireDetectionAgent.detect()` is unchanged and does not load, call, or reference any trained model. `FireDetectionCalculator`'s methodology, weights, and thresholds are unchanged (verified: existing deterministic-confidence tests pass unmodified). No API, frontend, `FireEvent` persistence/threshold, or simulation-orchestration code was touched. No runtime classifier is selected by this task.

**Update (Task 5):** `fire_detection_logistic_v3.joblib` has since been integrated into the real runtime `FireDetectionAgent` flow, as a subordinate signal alongside the still-unchanged `FireDetectionCalculator` (default mode: `SHADOW` - ML never changes what gets persisted). See [fire_detection_runtime_ml.md](fire_detection_runtime_ml.md) for the runtime architecture, decision modes, and the escalation-threshold analysis. This document's own V1/V2/V3 experiment results above are unchanged and still describe the offline comparison, not the runtime behavior.

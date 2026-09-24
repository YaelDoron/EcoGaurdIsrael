# Fire Detection ML — V5 Synthetic Dataset (Task 6)

`data/fire_detection/training_v5.csv` — 10,000 rows, seed 42. **Nothing is trained on it in this task.**
Schema and extractor: [fire_detection_feature_schema_v5.md](fire_detection_feature_schema_v5.md).

```
python -m scripts.generate_fire_detection_training_data_v5      # validates BEFORE writing
python -m scripts.validate_fire_detection_dataset_v5            # prints + writes training_v5_validation_report.json
```

SHA-256 (seed 42, 10,000 rows): `37e056793fb4dda72f517a20153b21094b284e1579a065140483b8a6e1d78035`.
Regenerating gives a byte-identical file (fixed `\n` line endings; a test compares against a fresh generation).

## What a row is

```
current FireDetectionCandidate  (validated by the real class: <= 5 km / 60 min, chained)
+ FireDetectionEventHistory     (the real Task 5B class, None before a FireEvent exists)
+ ground-truth label
        │
        ▼  FireDetectionFeatureExtractorV5      ← the ONLY place a feature is computed
   training_v5 row  ==  what runtime will compute
```

The generator builds **evidence only**; it imports no feature code, `FireDetectionCalculator`, hybrid policy, ML model,
Fire Danger or repository (tests enforce this). Everything is offline and deterministic (per-row `random.Random` seeded
from `(seed, unit index)`, environments from `(seed, environment id)`).

## Regimes are evidence situations, not labels

A **regime** says what kind of evidence exists. It never encodes fire / no-fire: every regime holds both labels,
exactly 50/50 (fire share = 50.0 % in each; regime-only AUC = 0.500). The label comes from a **latent process** chosen
before any evidence is drawn:

| Latent subtype | Label | Behaviour (overlapping by design) |
|---|---|---|
| `early_wildfire` | fire | small, weakly detected; may have weak or no news; short history |
| `established_wildfire` | fire | mid FRP, growing pixel scatter, stronger news |
| `large_wildfire` | fire | high FRP, many pixels, mostly strong news |
| `sensor_noise` | no fire | weak, erratic pixels; sometimes bright, sometimes repeats |
| `industrial_heat_source` | no fire | steady, spatially stable, persistent for hours, can be high-FRP and seen at night |
| `controlled_or_agricultural_burn` | no fire | fire-like, short-lived, decaying, real smoke reports |
| `false_or_rumour_report` | no fire | news only (or news with coincidental clutter); can be STRONG and repeated |

| Regime (rows) | Evidence shape | Fire generation | No-fire generation | Expected overlap |
|---|---|---|---|---|
| `satellite_only_single_pass` (1,600) | exactly 1 hotspot, no news, no history | early / established / large | noise / industrial / controlled | single-pixel FRP, confidence and brightness overlap heavily (best per-regime AUC ≈ 0.64) |
| `multi_pixel_single_pass` (1,600) | ≥ 2 hotspots of one pass, no news | larger, wider scatter | industrial cluster, controlled burn, noise | scatter is the main signal (radius AUC ≈ 0.69) |
| `news_led` (1,400) | news first; hotspot follows (58 %) or never appears (42 %) | real reports, mixed strength | false / repeated reports (STRONG possible), controlled, industrial | strong-news AUC ≈ 0.63 |
| `satellite_news` (1,800) | hotspots + news, news before or after | real fire + real news | thermal source + false / unrelated news | high-confidence count, news and lag each ≈ 0.60–0.62 |
| `persistent_thermal` (2,600) | ≥ 1 earlier pass (1–5 earlier looks, gaps 2–11 h) | growing / declining, wandering centroid | persistent industrial heat, burns, recurring noise | centroid stability ≈ 0.66; **pass count AUC ≈ 0.5** |
| `sparse_early_evidence` (1,000) | one weak item: a weak hotspot (≈ 79 %) or a weak report (≈ 21 %) | early fire, weak by rejection sampling | noise / false report / controlled | intentionally ambiguous: best AUC ≈ 0.57 |

Weak items in the sparse regime are rejection-sampled from the *same* latent process, never forced by label.

## Environments and pairs

* **Environment** (`environment_id`, 250 environments, 32–51 rows each): shared nuisance — location, pixel scale, FRP
  calibration offset, background temperature, FRP / brightness missing rates, revisit period, coverage (clouds), night
  share, confidence bias, day/night-unknown rate. 100 % of environments contain both labels (fire share 36–66 %). Rows of
  one environment must stay in one fold.
* **Pair** (`pair_id`, 2,500 pairs = 50 % of rows): one fire + one no-fire row sharing environment, regime, observation
  opportunities (timing grid and coverage of earlier passes), anchor time (±10 min) and day/night regime, with **independent**
  latent draws. `pair_type` names the construction: weak early fire vs weak satellite noise; multi-pixel fire vs industrial
  cluster; news-led fire vs repeated false reports; satellite + real news fire vs thermal source + false news; persistent
  wildfire vs persistent industrial heat; sparse early fire vs sparse false signal. 1.1 % of pairs have identical feature
  vectors (lone weak reports); pair-level win rates of the key features are 0.48–0.71 (neither side wins on one feature).

## Metadata columns (analysis only — never features)

`sample_id, seed, environment_id, regime, latent_subtype, pair_id, pair_type, as_of_utc`, then the 25 features, then `label`.
`forbidden_feature_names()` and tests guarantee no metadata name is, or looks like, a feature.

## Anti-shortcut design and results

Nothing is derived from the label: no `fire → FRP > x`, `fire → ≥ 2 passes` or `night → fire`. Validation (all 0 violations):

| Check | Result |
|---|---|
| Regime fire share | 50.0 % in all six; regime-only AUC 0.500 |
| Fire share by day / night | 49.9 % / 49.9 % (unknown 50.8 %); by 4-hour UTC bucket 49.5–50.8 % |
| Single hotspot (3,582 rows) | 1,731 fire / 1,851 no-fire |
| Multiple passes (2,600) | 1,300 / 1,300 — and inside `persistent_thermal` pass-count AUC ≈ 0.5 |
| High-FRP top quartile | 1,360 fire / 939 no-fire (59 % fire) |
| Low-FRP bottom quartile | 725 / 993 (42 % fire) |
| STRONG news | 1,028 / 553 (65 % fire) — false reports can be STRONG |
| No news | 2,847 / 2,854 |
| Night hotspot rows | 2,204 / 2,216 |

Highest univariate AUCs overall: centroid stability 0.664, low-confidence count 0.595, FRP trend 0.582, high-confidence
count 0.581, brightness trend 0.572 — **no feature reaches the 0.75 warning or 0.85 shortcut threshold**, overall or inside
any regime. This is a hard dataset by design: Task 7 must not expect near-perfect accuracy.

## Duplicates

* Exact duplicate feature rows: 7.6 % overall — all but 38 rows are **news-only** rows (805 rows, 89.8 % duplicated: a lone
  report has 5 possible feature states). Rows with hotspot measurements: 0.4 %.
* Near-duplicates (nearest neighbour < 0.05 std): 21 % of satellite rows, ~9.6 % with the opposite label. They are
  concentrated in the single-item regimes (`satellite_only_single_pass` 74 %, `sparse_early_evidence` 96 %; ~34–44 % of them
  cross-label) — one pixel has little feature variety, so the same low-information evidence occurring under both labels is the
  intended ambiguity, not a generator bug. The other regimes are ≈ 0 %. Because folds are grouped by environment this is not
  leakage.

## Grouped evaluation layout (for Task 7, nothing trained here)

* **Primary**: `grouped_environment_folds` — whole environments held out; 5 folds of ≈ 2,000 rows / 50 environments each,
  fire share 49.5–50.4 %, every regime holds both labels in every fold, pair members always share a fold.
* **Stress**: `leave_one_regime_out` (persistent_thermal is the only regime with history, so holding it out also tests
  extrapolation of the history features); `leave_one_no_fire_subtype_out` (validation = the held-out no-fire rows, i.e.
  specificity on an unseen false-alarm process; fires stay in training).
* **Paired**: `pair_indices` → (fire, no-fire) row per `pair_id`.

## Known limitations

* **Synthetic, no real labelled data.** Parameters are plausible, not measured; conclusions transfer only as far as the
  generator's assumptions do.
* **Sparse early evidence is ambiguous on purpose** (one weak item under either label); the goal is calibrated
  uncertainty, not accuracy.
* **No site recurrence** over days (needs a repository query; not V5 core) and **no multi-platform diversity** (one
  platform, `NOAA-20` / `VIIRS`).
* **No news source count** (the evidence model does not carry `source_feed`).
* History exists only in `persistent_thermal` (a leave-one-regime-out on it removes all history examples), and the
  `FRP` / `brightness` trends exist for only ~10 % of rows (they need ≥ 3 passes with a measurement).
* News-only rows are duplicate by nature (see above).

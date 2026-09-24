# Fire Detection - V4 Feature Pipeline

## Status

Task 3: one stable V4 feature contract shared by offline training and future runtime
inference. **No model is trained, no preprocessing is added, and nothing in runtime
Fire Detection changed** (`FireDetectionAgent`, `FireDetectionHybridPolicy`,
`FIRE_DETECTION_DECISION_MODE`, the V3 classifier and artifacts, `.env`, FireEvent
persistence). The frozen Task 2 dataset (`training_v4.csv`) is byte-identical after this
refactor.

## Pipeline

```text
Satellite / News Evidence
          |
          v
FireDetectionCandidate  ----------------+
          |                              |
          |                    FireDetectionContext
          |                    (Fire Danger / FFWI context,
          |                     built by FireDetectionContextService)
          |                              |
          +--------------+---------------+
                         v
        FireDetectionFeatureExtractorV4
                         v
              19 ordered features
                         v
        [Task 4: preprocessing (missing-value strategy, scaling)]
                         v
        [Task 4: ML model]
```

The same extractor produces the rows of `training_v4.csv` (from a synthetic candidate plus
its synthetic `FireDetectionContext`) and, later, the model input at runtime (from the real
candidate plus the context `FireDetectionContextService` returns). There is exactly one
implementation, so the two cannot drift apart (no train/serving skew).

## Components

| Piece | File |
| --- | --- |
| Canonical ordered names, missing marker, validated vector `FireDetectionFeaturesV4` | `src/ml/fire_detection/fire_detection_features_v4.py` |
| Extractor | `src/ml/fire_detection/fire_detection_feature_extractor_v4.py` |
| Evidence features (unchanged) | `src/ml/fire_detection/fire_detection_feature_extractor_v3.py` |
| Training rows built through the extractor | `src/ml/fire_detection/fire_detection_dataset_v4.py` |

## The 19 features (canonical order)

```text
 1 satellite_low_count               11 news_moderate_count
 2 satellite_nominal_count           12 news_strong_count
 3 satellite_high_count              13 news_unknown_count
 4 satellite_frp_available_ratio     14 time_span_minutes
 5 satellite_frp_mean                15 max_pairwise_distance_km
 6 satellite_frp_max                 16 fire_danger_available
 7 satellite_brightness_available_ratio  17 fire_danger_score
 8 satellite_brightness_mean         18 fire_danger_age_minutes
 9 satellite_brightness_max          (16-18 = Fire Danger context)
10 news_none_count
```

`FIRE_DETECTION_FEATURE_NAMES_V4` is the only ordered list. It is built from the V3 names
plus the three context names and is never re-typed; a test fails if any other module spells
out (a copy of) it. `validate_feature_names_v4()` checks any claimed schema (extractor
output, dataset header, and in Task 4 the saved model's metadata) against it.

Never features: `fire_danger_level` (a deterministic function of the score), scenario /
family / archetype / `ScenarioType`, ground truth, timestamps, rule confidence/status, any
ML probability, FireEvent status.

## How the extractor works

* **Reuse, not duplication.** Features 1-15 come straight from the unchanged
  `FireDetectionFeatureExtractorV3` (composition; injectable for tests). FRP, brightness,
  news, time-span and distance logic is not re-implemented. The V4 extractor imports no
  `math`, and V3 validation of these 16 values is reused by delegating to
  `FireDetectionFeaturesV3`.
* **Context features 16-18** come from an already-built `FireDetectionContext`
  (`available`, `score`, `age`). Its level, assessment id and timestamps are ignored.
* **Input is a candidate and a context, nothing else** (`extract(candidate, context)`; a raw
  evidence tuple is also accepted for merged evidence). It never queries a repository, never
  calls `FireDetectionContextService`, and imports no repository, service, database, agent,
  rule-calculator, decision-policy, FireEvent or ground-truth module (AST-tested).
* Candidate validity (one connected component, no duplicates) is enforced exactly as at
  runtime by the V3 extractor through `FireDetectionCandidate`.

## Missing Fire Danger vs a real FFWI of 0

| Situation | `fire_danger_available` | `fire_danger_score` | `fire_danger_age_minutes` |
| --- | --- | --- | --- |
| Assessment available, FFWI 0.0 (very low danger), assessed just now | `1` | `0.0` | `0.0` |
| No usable assessment (none, stale, no area, INSUFFICIENT_DATA, lookup failure) | `0` | `NaN` | `NaN` |

The ML-facing missing marker is `NaN` (`MISSING_VALUE_V4`). Unavailable is never `0.0`,
because 0.0 would claim "the weather is at its calmest / assessed this minute". The storage
forms (`to_nullable_tuple/dict`, CSV) use `None` / an empty cell for the same thing, and
`FireDetectionFeaturesV4.from_optional_values()` converts back. The vector validates and
fails loudly on any of: wrong feature count, non-numeric values, non-binary availability,
available Fire Danger without score/age, a zero-filled (non-NaN) unavailable score/age,
out-of-range values, or NaN anywhere except the two unavailable Fire Danger values.

**Task 4 will need a missing-value strategy** for `fire_danger_score` and
`fire_danger_age_minutes` (e.g. an imputer used together with the `fire_danger_available`
indicator). None is added here.

## Train / runtime parity

The mechanism is structural, not just tested: `fire_detection_dataset_v4.py` computes no
feature itself - `row_from_sample` calls `FireDetectionFeatureExtractorV4.extract(candidate,
context)`, exactly as runtime will. Tests additionally prove:

* the refactored dataset features equal Task 2's hand-built computation for all 9,000 samples;
* an independent runtime-style extraction (reversed evidence order; context rebuilt with a
  wrong level, id and timestamp) equals the dataset row for all 9,000 samples and equals the
  committed CSV values;
* representative families (strong real fire, weak real fire, hard negative, extreme-danger
  no-fire, missing-danger fire, missing-danger no-fire) match feature-for-feature;
* the dataset really goes through the canonical extractor (patching it breaks dataset
  generation), and a zero-filled-missing skew is caught.

`training_v4.csv` regenerated with seed 42 is byte-identical to the Task 2 file (SHA-256
`6e14edc5ff536f3e55e680dfe5e89862b9e712ea1e5a6639be2a1e10767d812f`), so the Task 2 benchmark
is unchanged.

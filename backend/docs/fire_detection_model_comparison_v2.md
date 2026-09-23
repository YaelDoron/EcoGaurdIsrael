# Fire Detection ML - V2 Dataset Robustness Evaluation

## Status

Offline analysis and comparison only. `training_v1.csv`, its models, and `scripts/compare_fire_detection_models.py` are untouched. `FireDetectionAgent` and `FireDetectionCalculator` remain unchanged. No runtime classifier is selected here - this document reports the robustness results the next task will use to decide.

## Why V2 was created

Task 2's Logistic Regression coefficient inspection and Random Forest feature-importance analysis both surfaced the same concerning pattern in `training_v1.csv`: `satellite_low_count` was the **strongest** Random Forest feature, and it was a perfect (though unintended) negative indicator - `satellite_low_count == 0` for every single positive (label=1) row. No V1 positive scenario family ever generated low-confidence satellite evidence. This is a generator artifact, not a realistic wildfire property: a real active-fire candidate can plausibly include a weak/low-confidence satellite pass alongside stronger evidence (e.g. one HIGH detection and one LOW detection of the same fire, or a NOMINAL detection plus a LOW detection plus a news report). V1's models may therefore have partly learned "LOW satellite evidence present -> no fire" as a shortcut rather than a more genuine evidence-combination pattern.

V2 adds new scenario families (both positive and negative) with mixed-confidence satellite evidence - including LOW alongside NOMINAL/HIGH - so `satellite_low_count > 0` is no longer a trivial shortcut for either class.

## Random split vs. grouped split

Two different, complementary questions are being asked:

- **Random (stratified) split/CV** answers: *can the model generalize to new samples drawn from scenario-family distributions it has already seen during training?* Both `training_v1.csv`'s Task 2 evaluation and V2's standard evaluation below use this.
- **Grouped (by `scenario_family`) CV** answers the harder question: *can the model generalize to an entire scenario family it has never seen at all during training?* Every sample from a given family is placed entirely in either the training or the validation side of a fold - never split across both. This is new in this task.

Grouped evaluation is expected to be harder and is not evidence of a broken model - it measures a fundamentally different, stronger notion of generalization. `scenario_family` is only ever used as the CV *group*; it is never a model feature (verified programmatically - see `fire_detection_dataset_validation` and tests).

## V2 sanity checks

`scripts.generate_fire_detection_training_data --dataset-version v2` runs `validate_training_samples` before writing the CSV and fails loudly on violation. All of the following passed on the generated 3000-row V2 dataset:

- both labels present, reasonably balanced (exactly 1500/1500 by construction)
- `satellite_low_count`, `satellite_nominal_count`, `satellite_high_count`, `news_count` each `> 0` for at least one row of **both** labels
- `satellite_low_count > 0`, `satellite_high_count > 0`, and `news_count > 0` each occur under **both** labels (not just "at least one row" - the full feature-presence set)
- satellite-only rows exist under both labels; multi-source (satellite+news) rows exist under both labels
- no single feature is an exact one-feature label separator (checked for all 7 features)
- every generated row is confirmed a valid `FireDetectionCandidate` via `FireDetectionFeatureExtractor`

Running the same validator against a freshly regenerated V1 dataset **fails** it (`satellite_low_count > 0` occurs under label 0 only) - confirming the validator actually detects the known V1 artifact rather than rubber-stamping any input.

### Low-confidence satellite evidence under both labels (V1 vs V2)

| | V1 | V2 |
| --- | --- | --- |
| `satellite_low_count` mean, label=0 | 0.645 | 0.718 |
| `satellite_low_count` mean, label=1 | **0.000** (exact, every row) | **0.467** |
| % of positive rows with `satellite_low_count > 0` | 0% | 30.3% |

30.3% is substantial without being the majority - `P(low satellite evidence | fire) > 0` and meaningful, but not every positive example carries it, matching the task's explicit target.

## V2 dataset

- 3000 samples, seed 42, deterministic (verified: same seed -> identical output)
- 1500 positive / 1500 negative (exact 50/50, by construction)
- 20 scenario families: the original 13 V1 families (unchanged, still part of the V2 pool) plus 7 new ones:
  - Positive: `mixed_confidence_satellite_real_fire`, `high_low_satellite_with_news_real_fire`, `nominal_low_satellite_with_news_real_fire`, `multi_confidence_persistent_real_fire`
  - Negative: `mixed_confidence_non_fire_heat_source`, `high_low_satellite_false_alarm`, `mixed_satellite_with_false_news`
- `training_v1.csv` is untouched (byte-identical row count/content before and after this task; regeneration with `dataset_version="v1"` - the default - is verified equivalent to the saved file by a dedicated test)

## V1 vs V2 feature statistics (label=0 mean -> label=1 mean)

| Feature | V1 | V2 |
| --- | --- | --- |
| `satellite_count` | 1.302 -> 1.813 | 1.648 -> 2.318 |
| `news_count` | 0.794 -> 1.128 | 0.777 -> 1.226 |
| `satellite_low_count` | 0.645 -> **0.000** | 0.718 -> **0.467** |
| `satellite_nominal_count` | 0.513 -> 0.947 | 0.561 -> 0.937 |
| `satellite_high_count` | 0.144 -> 0.866 | 0.369 -> 0.913 |
| `time_span_minutes` | 10.27 -> 15.93 | 12.43 -> 20.01 |
| `max_pairwise_distance_km` | 0.964 -> 1.589 | 1.195 -> 1.965 |

The label=0/label=1 gap widens slightly for most features in V2 (the new families tend to involve more evidence items and larger spread), but the qualitative direction is preserved for every feature except the one that mattered: `satellite_low_count` now genuinely overlaps between classes instead of perfectly separating them.

## Standard (random split) evaluation - V2

Held-out test (80/20 stratified, `random_state=42`, same split for both models):

| Metric | Logistic Regression | Random Forest |
| --- | --- | --- |
| Accuracy | 0.6783 | 0.7783 |
| Precision | 0.6801 | 0.7685 |
| Recall | 0.6733 | 0.7967 |
| F1 | 0.6767 | 0.7823 |
| ROC-AUC | 0.7671 | 0.8525 |
| False Positives | 95 | 72 |
| False Negatives | 98 | 61 |

5-fold stratified CV (`n_splits=5, shuffle=True, random_state=42`):

| Metric | Logistic Regression | Random Forest |
| --- | --- | --- |
| Accuracy | 0.699 ± 0.007 | 0.768 ± 0.009 |
| Precision | 0.711 ± 0.007 | 0.773 ± 0.016 |
| Recall | 0.671 ± 0.016 | 0.760 ± 0.029 |
| F1 | 0.690 ± 0.010 | 0.766 ± 0.011 |
| ROC-AUC | 0.779 ± 0.009 | 0.849 ± 0.010 |

**Both models score lower on V2 than on V1** (LR held-out accuracy 0.715 -> 0.678, ROC-AUC 0.851 -> 0.767; RF held-out accuracy 0.835 -> 0.778, ROC-AUC 0.922 -> 0.853). This drop is expected and is a *good* sign, not a regression: it shows the V1 models were partly relying on the `satellite_low_count` shortcut that V2 removes, and the harder, more realistic overlapping dataset gives a more honest (lower, but more trustworthy) performance estimate. Random Forest still clearly outperforms Logistic Regression on every standard metric.

## Grouped (unseen scenario-family) evaluation - V2

### Fold count

`choose_grouped_fold_count` searches from 10 down to 3 for the largest `n_splits` where every family-label group has at least `n_splits` distinct families, `StratifiedGroupKFold` does not raise, and every fold's train and validation sets each contain both labels with zero family overlap. With 10 positive and 10 negative scenario families, `n_splits=10` down to `n_splits=8` all failed this safety check (the exact-and-near-exact family/fold ratio does not leave `StratifiedGroupKFold`'s balancing heuristic enough slack to guarantee both labels in every fold); **`n_splits=7`** is the largest value that passes, and was used. `verify_grouped_folds` independently re-confirms zero family overlap and full label coverage for the chosen count.

### Grouped 7-fold CV (mean ± std)

| Metric | Logistic Regression | Random Forest |
| --- | --- | --- |
| Accuracy | 0.543 ± 0.278 | 0.382 ± 0.158 |
| Precision | 0.539 ± 0.365 | 0.382 ± 0.223 |
| Recall | 0.515 ± 0.345 | 0.428 ± 0.225 |
| F1 | 0.511 ± 0.329 | 0.396 ± 0.214 |
| ROC-AUC | 0.581 ± 0.362 | **0.223** ± 0.233 |

Both models degrade drastically compared to standard evaluation, and the fold-to-fold standard deviations are enormous (e.g. LR accuracy 0.543 ± 0.278 spans nearly the entire [0,1] range across folds). This is expected per this task's framing - it is a substantially harder generalization test - but the magnitude and direction of the result are still worth stating plainly: **Random Forest's mean grouped ROC-AUC (0.223) is worse than chance (0.5)**, and its mean grouped accuracy (0.382) is also below chance. Logistic Regression's grouped accuracy (0.543) and ROC-AUC (0.581) are only marginally above chance. Neither model demonstrates reliable generalization to a scenario family it has never seen during training on this dataset.

### Per-fold detail (Random Forest)

| Fold | Held-out families | Accuracy | F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| 0 | high_low_satellite_false_alarm, low_confidence_satellite_noise, multi_satellite_real_fire | 0.625 | 0.447 | 0.630 |
| 1 | mixed_confidence_non_fire_heat_source, mixed_satellite_with_false_news, news_only_before_satellite | 0.263 | 0.000 | 0.002 |
| 2 | correlated_satellite_news_false_alarm, high_confidence_satellite_false_positive, high_confidence_satellite_with_news | 0.326 | 0.383 | 0.228 |
| 3 | high_low_satellite_with_news_real_fire, multi_satellite_plus_multi_news, nominal_satellite_false_heat_source | 0.459 | 0.621 | 0.195 |
| 4 | multi_confidence_persistent_real_fire, news_rumor_no_fire, nominal_low_satellite_with_news_real_fire | 0.420 | 0.592 | **0.000** |
| 5 | mixed_confidence_satellite_real_fire, nominal_satellite_with_news, persistent_non_wildfire_heat_source | 0.486 | 0.551 | 0.493 |
| 6 | multiple_news_same_false_alarm, satellite_only_before_news | 0.097 | 0.177 | 0.016 |

Fold 4 is the most striking: held out `news_rumor_no_fire` (label 0) with two positive families, and Random Forest predicted **every single** `news_rumor_no_fire` row as fire (0 true negatives, ROC-AUC exactly 0.000 for that fold).

## Per-family generalization observations (Part 16)

The clearest, most consistent pattern across both models: **families representing one isolated, near-zero-spread evidence item generalize worst**, and specific "twin" family pairs get inverted:

| Family | Label | RF predicted-positive rate | RF accuracy | LR accuracy |
| --- | --- | --- | --- | --- |
| `news_rumor_no_fire` | 0 | 1.000 | 0.000 | **1.000** |
| `news_only_before_satellite` | 1 | 0.000 | 0.000 | 0.000 |
| `high_confidence_satellite_false_positive` | 0 | 1.000 | 0.000 | **1.000** |
| `satellite_only_before_news` | 1 | 0.195 | 0.195 | 0.000 |
| `multiple_news_same_false_alarm` | 0 | 1.000 | 0.000 | 0.027 |
| `low_confidence_satellite_noise` | 0 | 0.007 | 0.993 | 1.000 |

**Why this happens:** `news_rumor_no_fire` (a single lone news report, zero time span, zero spread - by design, since it is one evidence item) and `news_only_before_satellite` (a single lone news report describing a real fire, also necessarily zero time span/spread for one item) occupy **overlapping feature-space regions** - the `time_span_minutes`/`max_pairwise_distance_km` geometry features are identically `0.0` for any single-item candidate regardless of what actually happened, and the extractor has no way to distinguish "a report describing a real fire" from "a report describing a rumor" beyond that. When a model has never seen either family during training, it falls back on the broader statistical tendency learned from other training families (in this dataset, `news_count > 0` correlates positively with fire on average - 74% of V2 positive rows vs 47% of V2 negative rows have `news_count > 0`), so it predicts "fire" for both, correctly on `news_only_before_satellite` sample-composition-wise but *always* wrong on `news_rumor_no_fire`. The same logic explains the `high_confidence_satellite_false_positive` / `satellite_only_before_news` pair on the satellite side. Logistic Regression happened to get `news_rumor_no_fire` and `high_confidence_satellite_false_positive` exactly right (1.000 accuracy) in the specific fold draw where they were held out, while Random Forest got them exactly wrong (0.000) - a genuine, non-random difference in how the two model families extrapolate outside their training distribution on this specific challenge, not just noise (both used the identical fold assignment, from the same seeded `StratifiedGroupKFold`).

This is a **feature-representation limitation**, not merely a "need more training data" problem: a single-evidence-item candidate is close to informationally indistinguishable under `FIRE_DETECTION_FEATURE_NAMES` alone (geometry features collapse to constants). Any future feature-set revision aimed at improving unseen-family generalization should look at this specifically.

## Feature ablation: with vs. without `satellite_count`

`satellite_count == satellite_low_count + satellite_nominal_count + satellite_high_count` holds exactly for every row (this holds for both V1 and V2, by construction of `FireDetectionFeatures`). This experiment compares the full 7-feature set against a 6-feature set with `satellite_count` removed, for both models, under both standard and grouped CV (F1, ROC-AUC only, per the task's minimum requirement):

| Model | Feature set | Standard CV F1 | Standard CV ROC-AUC | Grouped CV F1 | Grouped CV ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | full (7) | 0.6904 ± 0.0097 | 0.7789 ± 0.0088 | 0.5113 ± 0.3288 | 0.5815 ± 0.3625 |
| Logistic Regression | without `satellite_count` (6) | 0.6915 ± 0.0098 | 0.7788 ± 0.0087 | 0.5107 ± 0.3286 | 0.5815 ± 0.3624 |
| Random Forest | full (7) | 0.7657 ± 0.0115 | 0.8485 ± 0.0096 | 0.3958 ± 0.2139 | 0.2233 ± 0.2330 |
| Random Forest | without `satellite_count` (6) | 0.7613 ± 0.0076 | 0.8474 ± 0.0095 | 0.3994 ± 0.2145 | 0.2289 ± 0.2364 |

**Interpretation:** every difference is a few thousandths to ~0.004 - well within one CV fold's own standard deviation (0.008-0.012 for standard CV, 0.21-0.36 for grouped CV) - so **removing `satellite_count` has no meaningful effect on either model, under either evaluation regime**. This is exactly what the redundancy analysis predicted: for Random Forest, the information in `satellite_count` is fully reconstructable from the three confidence-specific counts already present, so a tree ensemble loses nothing by not having it directly; for Logistic Regression, removing an exactly-collinear feature does not change what function of the inputs the model can express, only how its weight is distributed among the (now six, still correlated) remaining features - consistent with Task 2's finding that `satellite_count`'s own coefficient sign was an artifact of multicollinearity rather than a real independent effect. **`satellite_count` appears redundant, not harmful and not essential** - no production feature-set change is made in this task.

## Random Forest overfitting: V1 vs V2

| | Train Accuracy | Train F1 | Test Accuracy | Test F1 | Gap (Acc / F1) |
| --- | --- | --- | --- | --- | --- |
| V1 | 0.9156 | 0.9078 | 0.8350 | 0.8226 | ~8 / ~9 pts |
| V2 | 0.9575 | 0.9556 | 0.7783 | 0.7823 | ~18 / ~17 pts |

**The train/test gap roughly doubled on V2.** Random Forest fits the V2 training data even more closely (train accuracy 0.958 vs 0.916) while held-out performance drops (0.778 vs 0.835) - consistent with V2 being a genuinely harder, noisier, more overlapping task: with the easy `satellite_low_count` shortcut removed and more scenario families to memorize idiosyncrasies of, the default (unconstrained `max_depth=None`, `min_samples_leaf=1`) Random Forest has more opportunity to fit noise specific to the training split. No hyperparameter changes were applied to close this gap, per this task's instructions not to tune.

## Does Random Forest still outperform Logistic Regression after V2?

- **Standard split/CV: yes, clearly** - Random Forest leads Logistic Regression on every standard metric (accuracy, precision, F1, ROC-AUC), by a margin comparable to V1.
- **Grouped CV (unseen families): no** - Logistic Regression's grouped accuracy (0.543) and ROC-AUC (0.581) are both higher than Random Forest's (0.382 and 0.223 respectively), and Random Forest's grouped ROC-AUC is below chance. On this harder generalization test, Random Forest's standard-evaluation advantage does not carry over - if anything it reverses.

This is the central, most important finding of this task: **which model "wins" depends entirely on which kind of generalization is being measured.** A runtime-selection decision based only on standard held-out/CV metrics (as Task 2 was) would have picked Random Forest without ever seeing this reversal.

## Suspicious remaining patterns

- The single-evidence-item "twin family" confusion described above is real and structural, not a bug in this task's code - it reflects a genuine limitation of the current 7-feature geometric/count representation for degenerate (single-item) candidates.
- Grouped-CV standard deviations are very large (up to ±0.36) with only 7 folds and 2-3 held-out families per fold - individual fold results should be read as illustrative of *which* families are hard, not as a tightly estimated population statistic. A larger number of scenario families (more than this task's 20) would be needed to shrink this variance meaningfully.
- No new `satellite_count`-style structural shortcut was found in V2 by the sanity checks (Part 9) or the correlation/statistics analysis - the checks that would have caught a repeat of the V1 artifact all passed.

## Important scientific limitation (repeated)

**All results in this document, including the grouped-CV "unseen scenario family" results, are based entirely on synthetic scenarios.** Grouped evaluation is a stronger internal robustness check - it does **not** convert this into real-world validation. A synthetic scenario family standing in for "a real fire reported only by a single early news article" is still a designed proxy, not observed real-world data. These results should inform, but not substitute for, eventual validation against real satellite/news history before any operational deployment decision.

## No runtime selection

This document reports factual, sometimes conflicting, results across evaluation regimes. It does not declare a winner and does not select a runtime classifier. `FireDetectionAgent` and `FireDetectionCalculator` remain unchanged.

# Wildfire Spread Prediction — User Story 4.2

Status: methodology draft, scientifically verified against primary sources (documentation only — no implementation yet).

Primary sources used for verification (Task 4A):
- Trucchia, A. et al. (2020). "PROPAGATOR: An Operational Cellular-Automata Based Wildfire Simulator." *Fire*, 3(3), 26. DOI: [10.3390/fire3030026](https://doi.org/10.3390/fire3030026). **The publisher page (mdpi.com) returned HTTP 403 to automated fetch during this task, so the paper's prose/figures (including the Figure 7 graphical presentation of the wind factor) could not be independently re-read in this session.** Formula references to "the paper" below rely on the equations as re-published, cited, and implemented in the official source code (next item), which explicitly attributes them to this paper.
- CIMAFoundation/propagator_sim, `legacy` branch, official reference implementation: https://github.com/CIMAFoundation/propagator_sim/tree/legacy — specifically `propagator/propagator.py`, `propagator/constants.py`, `prob_table.txt`, `v0_table.txt`, `p_vegetation.txt`, fetched and read verbatim (not AI-summarized) during this task.
- Israel Meteorological Service (IMS), "Application Programming Interface (API) for meteorological data" (official PDF), `https://ims.gov.il/sites/default/files/2023-01/API_Explanation_en.pdf`, read verbatim (Appendix C, channels list).

Every formula, constant, and table cell below is labeled with its verification status. Nothing is presented as a verified PROPAGATOR value unless it was read directly from one of the sources above.

## 1. Purpose

This document defines the agreed methodology for User Story 4.2, which answers:

> "Given an already active wildfire, which nearby geographic locations are likely to be affected as the wildfire develops?"

The feature predicts spread **around an existing, already-detected `FireEvent`**. It is a forward-looking projection over a bounded geographic area and a bounded time horizon, built on inputs EcoGuard has already collected and persisted for that fire.

## 2. Scope and Responsibilities

User Story 4.2 is responsible for:
- Producing a deterministic spread prediction for an active `FireEvent`, expressed as a set of geographic cells with a spread probability and risk score.

User Story 4.2 is explicitly **not** responsible for, and must not perform:
- Wildfire detection (creating or updating a `FireEvent`) — owned by User Story 2.2.
- Fire Danger calculation — owned by User Story 2.1.
- Fire Severity calculation — owned by User Story 2.3.
- Routing or shortest-path computation (e.g. Dijkstra).
- Firefighting resource allocation.
- Genetic-algorithm-based optimization.

Those responsibilities belong to other EcoGuard user stories and must not be duplicated or reimplemented here.

## 3. Selected Methodology

EcoGuard uses a **deterministic Cellular Automata wildfire-spread methodology based on established PROPAGATOR-style fire propagation principles**, adapted to the environmental data already available in EcoGuard.

This is **not** presented as a complete reproduction of the original PROPAGATOR model. Three categories must be kept separate throughout design, implementation, and review — every formula and constant elsewhere in this document is labeled with one of these:

- **[A] Published PROPAGATOR methodology** — verified either from the cited paper or, where the paper's own presentation is graphical/non-closed-form, from the official CIMAFoundation reference implementation (permitted explicitly per the Task 4A brief: "If the official source code provides the exact implementation, use that as the implementation reference and document it").
- **[B] Existing EcoGuard data reused** — data EcoGuard already collects, calculates, or persists (FireEvent, FireSeverityAssessment, WeatherObservation, the FFWI equilibrium-moisture helper, the vegetation snapshot).
- **[C] EcoGuard V1 adaptation/limitation** — a deliberate EcoGuard implementation decision (deterministic threshold, terrain-neutral handling, fixed time step, event-level vegetation homogeneity, etc.) that is not itself a published PROPAGATOR value.

### 3.1 Verified transition equation [A]

The published PROPAGATOR fire-spread probability, confirmed against the official reference implementation (`propagator/propagator.py`, function `p_probability`):

```
p_ij = (1 - (1 - p_n) ** alpha_wh) * e_m
```

where:
- `p_ij` — probability that a burning cell `i` propagates to neighboring cell `j`
- `p_n` — nominal fire-spread probability determined by the (target, source) vegetation-class pair — see §15.2
- `alpha_wh` — combined wind + topographic influence on `p_n` — see §15.3
- `e_m` — fine-fuel-moisture influence factor — see §15.4

Source (`propagator.py`):
```python
def p_probability(self, dem_from, dem_to, veg_from, veg_to, angle_to, dist_to, moist, w_dir, w_speed):
    dh = (dem_to - dem_from)
    alpha_wh = w_h_effect_on_p(angle_to, w_speed, w_dir, dh, dist_to)
    p_moist = self.p_moist(moist)
    p_m = np.clip(p_moist, 0, 1.0)
    p_veg = prob_table[veg_to - 1, veg_from - 1]
    p = 1 - (1 - p_veg) ** alpha_wh
    p_clip = np.clip(p * p_m, 0, 1.0)
    return p_clip
```

This confirms `p_veg` above is exactly the task's `p_n`, and `p_m` (from `self.p_moist(moist)`) is exactly `e_m`. The code computes `alpha_wh` as one combined function rather than as a literal `alpha_w * alpha_h` product at the call site — see §15.3 for the exact internal decomposition (it *is* a product of a wind-only term and a slope-only term before a final probability-specific rescaling step, so the paper's `alpha_wh = alpha_w * alpha_h` description is consistent with, not contradicted by, the code).

### 3.2 Structural note: the official model is stochastic, EcoGuard V1 is not [A] vs [C]

Directly confirmed from the official implementation (`propagator.py`, `__apply_updates`):
```python
p_prob = p_probability(self, dem_from, dem_to, veg_from, veg_to, angle_to, dist_to, moisture_r, w_dir_r, w_speed_r)
p = p_prob > rand(p_prob.shape[0])
```
Cell ignition in the official model is decided by comparing `p_ij` against an independent uniform random draw per cell, per step — **not** by a fixed threshold. Separately, the code also perturbs wind speed and direction randomly per step before computing `p_ij` itself:
```python
w_dir_r = (w_dir + (pi/16)*(0.5 - rand(from_num))).repeat(nb_num)
w_speed_r = (w_speed * (1.2 - 0.4 * rand(from_num))).repeat(nb_num)
```
Both of these are genuinely stochastic mechanics of the published/official model. EcoGuard V1's deterministic `PROPAGATION_THRESHOLD = 0.5` rule (§8) replaces the random-draw ignition test, and EcoGuard V1 does not perturb wind — both are explicit, labeled EcoGuard V1 adaptations, not published behavior. This is stated with higher confidence than in the previous draft of this document, now that the exact stochastic mechanism has been read directly from source.

The code separately confirms that probability contour values of `0`, `0.5`, `0.75`, `0.9` are used only for output **isochrone visualization**, not for the per-cell transition rule:
```python
thresholds=[0, 0.5, 0.75, 0.9]
```
(`__update_isochrones`). This directly substantiates §8's existing statement that PROPAGATOR's published `0.5`/`0.75`/`0.9` contours are a visualization convention, not the stochastic transition threshold.

## 4. Existing EcoGuard Inputs Reused

A core design rule for this feature: **reuse existing calculated and persisted data instead of recalculating or refetching it.**

### 4.1 FireEvent

The existing persisted `FireEvent` (`backend/src/models/fire_event.py`, table `fire_events`) is reused as-is; no new detection logic is introduced.

- Active statuses: `SUSPECTED`, `CONFIRMED`.
- Inactive statuses: `RESOLVED`, `DISMISSED`.
- A spread prediction must **not** be produced for a `RESOLVED` or `DISMISSED` event.

`FireEvent` already provides everything this feature needs to identify the fire: location (`latitude`, `longitude`), `detected_at`, `updated_at`, `status`, `detection_confidence`, and detection `methodology`/`methodology_version`. Wildfire detection itself is not repeated.

### 4.2 Fire Severity

The latest stored `FireSeverityAssessment` is reused via the existing repository method:

```
FireSeverityAssessmentRepository.get_latest_for_event(fire_event_id)
```

Fire Severity is **not** recalculated, and no new severity formula is introduced inside User Story 4.2. The persisted assessment already contains: `fire_event_id`, `assessed_at`, `status`, `score`, `level`, `methodology`, `methodology_version`, a vegetation snapshot, traceability to the weather observations used, and traceability to satellite inputs. This record is preserved as context/traceability for the spread prediction, not recomputed.

### 4.3 Vegetation

The vegetation snapshot already persisted on the latest `FireSeverityAssessment` is reused. Available persisted fields: `vegetation_source`, `vegetation_dataset_year`, `vegetation_radius_km`, `vegetation_dominant_land_cover`, `vegetation_fuel_score`.

Copernicus must **not** be called again when this information already exists on the stored severity assessment. The full Copernicus `land_cover_distribution` (per-category fractions) is not currently persisted anywhere in EcoGuard, and for V1 the full distribution is **not required** for the mapping policy below (see §4.3.2 for why `vegetation_fuel_score` alone is *not* used as a substitute for `p_n`).

#### 4.3.1 EcoGuard's exact `dominant_land_cover` labels [B]

Read directly from `backend/src/mappers/vegetation_mapper.py` (`COPERNICUS_FUEL_MAPPINGS`), these are the only strings `dominant_land_cover` can ever hold:

`"Tree cover"`, `"Shrub cover"`, `"Grass cover"`, `"Crop cover"`, `"Bare cover"`, `"Built-up cover"`, `"Permanent water cover"`, `"Seasonal water cover"`, `"Moss and lichen cover"`, `"Snow cover"`.

#### 4.3.2 Vegetation-to-PROPAGATOR mapping analysis [A]+[B]→[C]

**`vegetation_fuel_score` must not be used as `p_n`.** EcoGuard's 0–1 `fuel_score` is an EcoGuard application-level weighting (`backend/src/mappers/vegetation_mapper.py`, module docstring: "EcoGuard fuel scores are application-level context weights, not official Copernicus fire-risk values"), computed for the unrelated Fire Severity formula. There is no scientific source establishing that this value is mathematically equivalent to PROPAGATOR's calibrated `p_n` matrix (§15.2). Presenting `fuel_score` as `p_n` would be an invented mapping, which this task must not do.

Instead, `dominant_land_cover` is mapped to the closest official PROPAGATOR vegetation class (§15.2's verified 7-class table, which includes an explicit "bare soil / non-burnable" class with `p_n ≈ 0.005` — i.e. effectively non-burnable):

| EcoGuard `dominant_land_cover` | Closest PROPAGATOR class | Mapping confidence | Reason |
|---|---|---|---|
| `Shrub cover` | Shrubs (`cespugli`) | High | Direct semantic and structural match. |
| `Grass cover` | Grassland (`erba`) | High | Direct semantic and structural match. |
| `Crop cover` | Agro-forestry areas (`coltivi`) | High | Direct semantic match; PROPAGATOR's class is explicitly cultivated/agro-forestry land. |
| `Bare cover` | Bare soil / non-burnable (`aree_nude`) | High | Direct match; PROPAGATOR's own bare-soil row/column carries `p_n ≈ 0.005` for every pairing, i.e. effectively non-burnable. |
| `Built-up cover` | Non-vegetated/non-burnable | High | No PROPAGATOR vegetation class represents built environment; treating as non-burnable is the only defensible choice, not a guess among fuel classes. |
| `Permanent water cover` | Non-vegetated/non-burnable | High | Water does not burn; no PROPAGATOR class applies. |
| `Seasonal water cover` | Non-vegetated/non-burnable | High | Same as above. |
| `Snow cover` | Non-vegetated/non-burnable | High | Snow-covered ground is not burnable fuel. |
| `Moss and lichen cover` | **Ambiguous — no PROPAGATOR class corresponds** | Low | Moss/lichen is not equivalent to grassland, bare soil, or shrubland in fuel-structure terms; PROPAGATOR's published class set (§15.2) has no analogous category, and treating it as any specific class would be an invented equivalence. |
| `Tree cover` | **Ambiguous — 3 possible PROPAGATOR classes** | Low | Copernicus's generic tree-cover fraction does not distinguish broadleaves vs. conifers, nor fire-proneness. The verified official class set (§15.2) separately defines "fire-prone broadleaves" (`p_n` diagonal `0.300`), "fire-prone conifers" (`p_n` diagonal `0.350`), and "non-fire-prone broadleaves" (`p_n` diagonal `0.075`) — a >4x spread in self-transition probability depending on which is chosen. Silently picking one would materially bias every prediction touching forested terrain, which is the most fire-relevant land cover in this domain. |

**Recommended V1 policy: Approach A — map only unambiguous categories; treat ambiguous vegetation as insufficient/unknown.** Concretely: when a `FireSeverityAssessment`'s `vegetation_dominant_land_cover` is `"Tree cover"` or `"Moss and lichen cover"` (or `None`), the spread prediction cannot honestly compute a PROPAGATOR-class-based `p_n` and must produce a `FireSpreadPredictionStatus.INSUFFICIENT_DATA` result rather than an invented value — this is exactly the failure mode `FireSpreadPredictionStatus.INSUFFICIENT_DATA` (already defined in the Task 3 domain model) exists for. Approaches B (a documented conservative fallback class) and C (a literature-supported interpolation) were considered and rejected for V1: no literature source was found in this task that supports either a specific fallback conifer/broadleaf choice or a fuel_score-to-p_n interpolation for forest sub-types, so adopting either now would mean inventing the missing science rather than reporting it.

This mapping policy is implemented exactly as decided here — see §15.7 item 1 (RESOLVED, Task 5).

#### 4.3.3 Spatial vegetation limitation [C]

EcoGuard currently has **one** persisted vegetation snapshot per `FireSeverityAssessment`, anchored at the `FireEvent` location with radius `vegetation_radius_km` — it does not have per-cell vegetation across the full 5 km CA grid. The official PROPAGATOR model, by contrast, assigns a vegetation class to every grid cell independently and uses both the source cell's and the target cell's class to look up `p_n[target][source]` (§3.1, §15.2).

Given the architectural rule to reuse already-persisted data rather than issue new Copernicus requests per cell, **EcoGuard V1's policy is to treat the mapped vegetation class as spatially homogeneous across the entire prediction grid**: the same mapped PROPAGATOR class is used as both the source-cell and target-cell vegetation for every `p_n` lookup in a given prediction run. This means:
- `p_n` becomes `prob_table[mapped_class][mapped_class]` (the table's own diagonal) for every cell pair in the grid, rather than varying by the actual, unknown, per-cell vegetation.
- This is an **EcoGuard V1 approximation**, not equivalent to full PROPAGATOR spatial fuel modeling.
- It is expected to under-represent real heterogeneity (e.g. a fire-adjacent grassland patch inside an otherwise forested prediction radius would incorrectly use the forest's `p_n` diagonal value instead of the true grassland-to-forest or forest-to-grassland transition probability).

This limitation is carried into §14 and is not resolved in this task, consistent with the instruction not to introduce a new Copernicus per-cell fetch.

### 4.4 Weather

Weather observations are already persisted in Neon/PostgreSQL (`WeatherObservation` / `weather_observations`), with fields: `temperature`, `relative_humidity`, `wind_speed`, `wind_direction`, `wind_gust`, `rainfall`, `timestamp`. `wind_speed` and `wind_gust` are stored in **km/h** (canonical unit for every source; IMS `WS`/`WSmax`, reported in m/s, are converted by `WeatherMapper` at ingestion).

Fire Severity already persists the exact `weather_observation_id`s used in its calculation. The preferred V1 approach is to **reuse those same persisted weather observations** where possible, so User Story 4.2 stays traceable to the same environmental state that produced the latest severity assessment, rather than querying a different, potentially inconsistent set of observations.

Known implementation gap (documented, not solved here): `WeatherRepository` does not currently expose a batch `get_observations_by_ids()` method to turn a list of observation IDs back into full `WeatherObservation` records. This is a small gap to be addressed during implementation, not in this documentation task.

### 4.5 Moisture

No second **equilibrium-moisture-content estimate** is introduced. EcoGuard already contains a Fosberg FFWI equilibrium-moisture-content calculation in `backend/src/calculators/fire_danger/ffwi_calculator.py`:

```
_equilibrium_moisture_content(relative_humidity_pct, temperature_f)
```

This value is currently computed internally within the FFWI calculation and is **not persisted**. This *estimate* (an equilibrium moisture content derived from humidity and temperature) is reused as the `M_f` input to the verified PROPAGATOR moisture-damping polynomial — see §15.4 for the exact, verified `e_m` formula.

**Important correction confirmed during this task:** `_equilibrium_moisture_content`'s output must feed the *verified* PROPAGATOR `e_m` polynomial (§15.4), which is a different, independently-sourced formula from EcoGuard's own existing `_moisture_damping_coefficient` (the Fosberg damping coefficient used internally by FFWI, `1 - 2x + 1.5x² - 0.5x³`). The two polynomials are structurally similar (both cubic-shaped damping curves normalized by a moisture-of-extinction-like divisor) but are **not the same formula** and must not be conflated. Only the equilibrium-moisture *estimate* is reused; PROPAGATOR's own moisture-damping polynomial (§15.4) must be applied to it, not FFWI's.

The helper is currently a private, module-level function. A small reuse-oriented refactor (e.g. making it importable/public, or exposing it through a shared calculation surface) may later be required so the spread calculator can call it cleanly. That refactor is out of scope for this documentation task.

## 5. Cellular Automata Configuration

Agreed V1 configuration:

| Parameter | Value |
|---|---|
| Cell size | 250 m × 250 m |
| Prediction radius around the FireEvent | 5 km |
| Neighborhood | Moore neighborhood (8 neighbors) |
| CA time step | 5 minutes |
| Prediction horizons | 30 minutes, 60 minutes |
| 30-minute horizon in steps | 6 CA steps |
| 60-minute horizon in steps | 12 CA steps |
| Terrain/slope | not used in V1 |
| Randomness | none in V1 (fully deterministic) |

A Moore neighborhood allows propagation from a cell to its 8 surrounding cells: north, northeast, east, southeast, south, southwest, west, and northwest.

These configuration values must later be centralized in a dedicated configuration module (following the existing `*_config.py` convention used by other calculators, e.g. `ffwi_config.py`, `fire_severity_config.py`) rather than scattered throughout implementation code. That configuration file is not created as part of this task.

**Terrain-neutral justification [A]→[C]:** the verified official slope term (`h_effect`, §15.3) is `2 ** tanh((slope * 3)² * sign(slope))`, where `slope = dh / (cellsize * dist)`. When the elevation difference `dh` between two cells is held at `0` (no terrain data available), `slope = 0`, `tanh(0) = 0`, and `h_effect = 2⁰ = 1` — a mathematically exact neutral (no-op) value, not an approximation. EcoGuard V1's "`alpha_h = 1.0`" policy is therefore not an arbitrary simplification; it is the verified formula's own well-defined behavior at `dh = 0`. It is still labeled **"EcoGuard V1 neutral-terrain adaptation"**, not original PROPAGATOR behavior, because the *decision* to always supply `dh = 0` (rather than real elevation data) is EcoGuard's, even though the resulting arithmetic is exact.

**Fixed-step vs. rate-of-spread timing [C]:** the official implementation models cell-to-cell transition *time* using a rate-of-spread value (`v_wh`, in m/min, from the vegetation's nominal spread velocity `v0` adjusted by wind/slope/moisture — see `p_time_standard`/`p_time_rothermel`/`p_time_wang` in `propagator.py`), so different cells ignite at different simulated times depending on local conditions. EcoGuard V1 instead evaluates the same fixed 5-minute CA step for every cell, uniformly. This is a deliberate EcoGuard V1 simplification: it is internally consistent with the User Story 4.2 acceptance criteria as written (deterministic 30-/60-minute horizon snapshots, not a continuous rate-of-spread timeline), but it means EcoGuard V1 does **not** reproduce PROPAGATOR's actual arrival-time modeling — a cell that would ignite in official PROPAGATOR after (say) 3 simulated minutes and one that would take 22 minutes are both only ever evaluated at 5-minute-step boundaries in EcoGuard V1. Rate-of-spread-based timing is not implemented in this task, per scope.

## 6. Geographic Processing

The spread engine will need to calculate:
- cell coordinates (deriving a grid of 250 m cells covering the 5 km prediction radius around the `FireEvent`),
- distance between cells,
- bearing/direction between cells,
- the relationship between a cell's direction (relative to the fire) and the current wind direction.

Findings from the existing-code audit relevant to this need:
- A haversine distance formula already exists **twice** in the codebase, both as private, feature-coupled implementations (`backend/src/calculators/fire_detection/fire_detection_calculator.py` and `backend/src/repositories/fire_event_repository.py`).
- There is currently **no shared geographic utility module** in EcoGuard.
- **Bearing calculation does not yet exist** anywhere in the codebase.
- **Point-at-distance-and-bearing calculation does not yet exist** anywhere in the codebase.

Implementation should avoid creating a third duplicate haversine formula. A shared geo utility module may be introduced later if the team decides to consolidate, but it is not created in this documentation task.

**Wind-direction convention — flagged, not yet confirmed (see §15.6).** Comparing a cell's bearing (relative to the fire) against `WeatherObservation.wind_direction` is only correct if both angles use the same "direction wind blows toward" convention. IMS's own API documentation (verified in §15.6) labels the `WD` channel only as "Wind direction, deg" without stating whether it is the meteorological "from" convention or another one. §15.6 documents what was found and the required conversion if the standard meteorological convention applies.

## 7. Spread Probability

The CA computes a transition/spread probability, `p_ij`, for propagation from cell `i` to a neighboring cell `j`, using the **verified** equation from §3.1:

```
p_ij = (1 - (1 - p_n) ** alpha_wh) * e_m
```

Relevant EcoGuard inputs feeding this methodology, and what they feed:
- `wind_direction`, `wind_speed` → `alpha_wh` (§15.3)
- `relative_humidity`, `temperature` → the existing equilibrium-moisture estimate (§4.5) → `e_m` (§15.4)
- the mapped PROPAGATOR vegetation class from `vegetation_dominant_land_cover` (§4.3.2, **not** `vegetation_fuel_score`) → `p_n` (§15.2)
- `FireEvent` location → grid origin for cell coordinates (§6)

Terrain is excluded from V1 (see §5, §15.3, §14): `alpha_wh` is computed with the slope input held at zero, which the verified source formula reduces to a wind-only term (§15.3).

This equation, `p_n` table, `alpha_wh` formula, and `e_m` formula are all **verified [A]** against the official reference implementation (§15). The remaining open items before implementation are narrower than "the equation is unknown": they are (1) the vegetation-mapping policy sign-off (§4.3.2), (2) the wind-speed unit expected by the `alpha_wh` formula (§15.3), (3) two numeric discrepancies between values quoted in an earlier draft of this task and the independently re-verified official source tables (§15.2), and (4) confirmation of the IMS wind-direction convention (§15.6). See §15.7 for the full list.

## 8. Deterministic Propagation Policy

The original CA/PROPAGATOR approach may use stochastic (random) behavior. User Story 4.2 explicitly requires **identical inputs → identical prediction outputs**, so EcoGuard V1 uses deterministic propagation instead.

Agreed V1 rule:

```
PROPAGATION_THRESHOLD = 0.5

if p_ij >= 0.5:
    the target cell may continue propagation during the next CA step
if p_ij < 0.5:
    the risk result may still be recorded,
    but that cell does not propagate further
```

**Implemented behavior (methodology version 1.1).** `PROPAGATION_THRESHOLD = 0.5` is a **propagation** threshold only — it decides whether a cell can act as a source in the next CA step, not whether the cell appears in the prediction. Each CA step evaluates every not-yet-spreading cell that has at least one spreading Moore neighbor (the origin counts as spreading), using the maximum `p_ij` over those spreading neighbors (multi-parent rule below):

- `p_ij >= 0.5` → **spreading cell**: emitted with that `p_ij` and the step it was reached, and it acts as a propagation source from the next step on.
- `0 < p_ij < 0.5` → **risk-only cell**: emitted, but it never acts as a propagation source. It keeps the highest `p_ij` it received from any spreading neighbor within the horizon, with `reached_step` set to the earliest step at which that highest value was recorded (i.e. the step of its recorded risk exposure, not a fire arrival).
- A risk-only cell becomes a spreading cell if a later step gives it `p_ij >= 0.5` (e.g. from a newly spreading neighbor); it is then emitted once, as a spreading cell with that later step.
- `p_ij == 0` → not emitted.

Every emitted cell appears exactly once, and the origin is never emitted. Invariant: `spread_probability >= 0.5` ⇔ the cell is propagation-capable; `spread_probability < 0.5` ⇔ risk-only. No new probability formula is introduced — risk-only cells carry the same verified `p_ij` (§3.1) that is compared against the threshold. A `VALID` prediction is therefore empty only when every evaluated `p_ij` is 0 (e.g. fuel moisture at/above the moisture of extinction, §15.4). Methodology version 1.0 emitted spreading cells only, so it produced `VALID` predictions with 0 cells whenever no neighbor reached 0.5.

`0.5` is an **EcoGuard deterministic implementation threshold**. It is not claimed to be an original PROPAGATOR scientific constant. This is now confirmed with direct source evidence, not inference: the official model's actual per-cell ignition test is `p_prob > rand(...)` — a stochastic comparison against an independent random draw, not any fixed threshold — and PROPAGATOR's published `0.5`/`0.75`/`0.9` values are isochrone-visualization contour levels in a separate output step, not the transition rule itself (§3.2).

No random number generation is used in V1. The official model additionally perturbs wind speed and direction randomly at every step before computing `p_ij` (§3.2) — EcoGuard V1 explicitly does not do this either; it uses the persisted, unperturbed `WeatherObservation` values as-is.

**EcoGuard deterministic multi-parent aggregation rule [C]** (implemented in Task 4B). A target cell may be adjacent to more than one already-reached Moore neighbor in the same CA step. EcoGuard V1's rule: compute `p_ij` from every already-reached neighbor to that target for the step, take the **maximum** of those values, and compare only that maximum against `PROPAGATION_THRESHOLD`. If the target becomes reached, the stored `spread_probability` for that cell is this maximum value. This is a deterministic conflict-resolution/aggregation rule for adapting the CA to multiple simultaneous candidate sources — it is **not** a new scientific spread formula, and it does not combine probabilities via any invented probabilistic-fusion formula (e.g. no averaging, no independent-event union). It has no analog to verify against the official stochastic model, since that model evaluates each source-target pair as an independent random trial rather than aggregating candidates deterministically.

## 9. Required and Optional Inputs

**Required inputs:**
- an active `FireEvent` (`SUSPECTED` or `CONFIRMED`)
- the latest valid `FireSeverityAssessment` for that event
- relevant persisted `WeatherObservation`s, specifically `wind_direction`, `wind_speed`, `relative_humidity`, and `temperature` (the latter needed for the existing equilibrium-moisture calculation, §4.5)

**Vegetation policy:**
- reuse the persisted `vegetation_fuel_score` when available (§4.3)
- do not perform a new Copernicus fetch merely because spread prediction is running

**Terrain policy:**
- not used in V1
- do not fabricate slope/elevation values

**Missing-data policy:** if a mandatory input is missing, the system must not generate a misleading "valid" prediction. An explicit `INSUFFICIENT_DATA` state must be used instead. This mirrors the existing status-based pattern used by Fire Danger and Fire Severity assessments.

**Insufficient-data reason (implemented).** Every newly produced `INSUFFICIENT_DATA` prediction also records a machine-readable `insufficient_data_reason` (`FireSpreadInsufficientDataReason`, stored as its lowercase value in the nullable `fire_spread_predictions.insufficient_data_reason` column and exposed on the EventDetails spread prediction). It is set by `FireSpreadInputService` at the exact branch that fails:

| Reason | Cause |
|---|---|
| `event_unavailable` | FireEvent not found, or not in a usable status |
| `missing_severity` | no severity assessment for this event at or before `as_of` |
| `severity_not_valid` | the latest severity assessment is not `VALID` |
| `missing_weather` | no linked or retrievable weather observation |
| `incomplete_weather` | observations exist but none has temperature, humidity, wind speed and wind direction |
| `stale_weather` | complete observations exist but are older than the freshness window |
| `future_weather` | complete observations exist but are all timestamped after `as_of` |
| `missing_vegetation` | the severity snapshot has no `vegetation_dominant_land_cover` (Copernicus unavailable **or** no usable Copernicus data — the two are indistinguishable at this layer) |
| `unsupported_vegetation` | "Tree cover", "Moss and lichen cover", or any unmapped label (§4.3.2); the exact label remains on the linked severity assessment |

Weather reasons follow a fixed precedence: `missing_weather` → `incomplete_weather` → `stale_weather` (if any complete observation is too old) → `future_weather`. `VALID` and `INACTIVE_EVENT` predictions never carry a reason. Rows stored before the reason existed have `NULL` (reason unknown) and are never backfilled. Operational refresh treats an insufficient-data horizon as a no-op only when both the status **and** the reason are unchanged, so a changed reason (including a historical `NULL` → known reason) persists a new append-only row and the latest row always states the current reason. The reason does not affect the spread calculation, so it does not change the methodology version.

A `RESOLVED` or `DISMISSED` `FireEvent` must not produce an active spread prediction.

## 10. Prediction Horizons

The same deterministic CA methodology is evaluated across discrete 5-minute steps:
- 30-minute prediction → 6 CA steps
- 60-minute prediction → 12 CA steps

The prediction output must preserve the horizon explicitly. The 30-minute and 60-minute predictions are **not** the same record — they must be represented as distinct, separately identifiable prediction results, not conflated or overwritten by one another.

Relationship between the two horizons (methodology 1.1, §8): the 60-minute run repeats the 30-minute run's first 6 steps exactly, so every spreading cell of the 30-minute prediction appears in the 60-minute prediction with the same step and probability, and the 60-minute prediction may extend further. Risk-only cells may differ: steps 7–12 can re-expose a risk-only cell with a higher `p_ij` or upgrade it to spreading. When no cell reaches the propagation threshold, nothing spreads beyond the origin, and both horizons legitimately contain the same first ring of risk-only cells (see §14).

## 11. Persistence and Traceability

Prediction results will later be persisted in the existing Neon/PostgreSQL database. This document does not design or implement the final schema.

A future persisted `FireSpreadPrediction` should preserve at minimum:
- `FireEvent` identity
- prediction timestamp
- prediction horizon
- methodology name
- methodology version
- prediction status
- for `INSUFFICIENT_DATA`, the insufficient-data reason (§9; `NULL` for historical rows)
- traceability to important reused input data (e.g. the `FireSeverityAssessment` id used, and the weather observation IDs used)

Predicted cells should preserve at minimum:
- latitude
- longitude
- spread probability `p_ij` (or equivalent final cell probability)
- `spread_risk_score`
- predicted reach step and/or predicted reach minutes

As implemented (methodology 1.1), spreading and risk-only cells share the same persisted cell fields; no separate flag column exists. Whether a cell is propagation-capable is determined by `spread_probability >= PROPAGATION_THRESHOLD` (§8). For a risk-only cell, `reached_step`/`reached_minutes` record the step at which its stored risk exposure was recorded, not a predicted fire arrival.

Historical predictions must be preserved rather than overwritten, following the existing EcoGuard append-only assessment/history pattern (as already used by `fire_danger_assessments` and `fire_severity_assessments`, where a new row is inserted per assessment run and the "latest" is retrieved via an ordered query) where appropriate.

### 11.1 Simulation integration (Task 10) [B]

`SimulationFireSpreadCoordinator` (`backend/src/simulation/analysis/simulation_fire_spread_coordinator.py`) triggers spread prediction for simulated `FireEvent`s the same way `SimulationFireSeverityCoordinator` triggers severity assessment: as a thin trigger-policy layer with no scientific or persistence logic of its own. After a simulated severity assessment succeeds for one or more `FireEvent`s, the coordinator calls the **exact same production** `FireSpreadPredictionAgent` — backed by the exact same `FireSpreadInputService`, `FireSpreadCalculator`, and `FireSpreadPredictionRepository` used by the live pipeline — once per `FireEvent` at each supported horizon (30 and 60 minutes). There is no separate simulation spread methodology, no simulation-specific PROPAGATOR variant, and no simulation-only persistence path. Simulation-specific identifiers (incident id, scenario metadata) stay entirely inside the simulation layer and are never passed into `FireSpreadInput`, `FireSpreadPrediction`, or any US 4.2 production class, keeping that code simulation-agnostic.

## 12. Methodology Name and Version

Planned EcoGuard methodology identity:

```
methodology = "ECOGUARD_PROPAGATOR_CA"
methodology_version = "1.1"
```

Version history: `1.0` emitted spreading cells only; `1.1` additionally emits risk-only cells below the propagation threshold (§8). Because the methodology version is part of the spread effective-state fingerprint (`FireSpreadEffectiveState`), 1.0 predictions are never reused as a no-op for 1.1 — the next refresh recalculates them, while historical 1.0 rows keep their own version for traceability.

This version string identifies EcoGuard's own deterministic adaptation. It does not claim to be, and must not be presented as, the official version identifier of the original PROPAGATOR research model.

## 13. Relationship to Response Targets (User Story 4.3)

Downstream flow:

```
FireEvent
  → FireSpreadPrediction
    → predicted geographic cells
      → relevant/high-risk predicted cells
        → User Story 4.3
          → PREDICTED_RISK ResponseTargets
```

The spread-prediction output must contain geographic coordinates suitable for later response-target creation.

User Story 4.2 must **not** include:
- Dijkstra graph-node IDs
- routing
- ETA calculation
- resource allocation
- genetic-algorithm logic

Those belong to later features (User Story 4.3 and beyond).

## 14. Limitations

- Terrain/slope is not used in V1; the neutral value is mathematically exact at `dh = 0` in the verified formula, but the decision to withhold real elevation data is EcoGuard's own (§5).
- The full Copernicus land-cover distribution is not persisted and is not used by V1. `vegetation_fuel_score` is **not** used as the spread-probability vegetation input (§4.3.2) — only the mapped `vegetation_dominant_land_cover` class is, and only when unambiguous.
- `"Tree cover"` and `"Moss and lichen cover"` cannot currently be mapped to a specific PROPAGATOR vegetation class without inventing an equivalence; predictions for fire events whose vegetation snapshot resolves to one of these produce `INSUFFICIENT_DATA` in V1 (§4.3.2).
- Vegetation is treated as spatially homogeneous across the entire 5 km prediction grid, using one event-level snapshot instead of true per-cell vegetation (§4.3.3) — a real EcoGuard V1 simplification relative to PROPAGATOR's per-cell fuel modeling.
- The model is a deterministic adaptation rather than a fully stochastic PROPAGATOR execution: the official model draws a per-cell random ignition test and randomly perturbs wind every step; EcoGuard V1 does neither (§3.2, §8).
- `PROPAGATION_THRESHOLD = 0.5` is an EcoGuard implementation decision, not a verified PROPAGATOR scientific constant; the official model's `0.5`/`0.75`/`0.9` values are isochrone visualization contours, not a transition threshold (§3.2).
- Because the threshold is applied to a single neighbor-to-neighbor `p_ij`, realistic weather/fuel combinations often produce no spreading cell (e.g. shrubs at 35 °C, 25% RH, 30 km/h peak at `p_ij` ≈ 0.33). In that case the prediction contains only the first ring of risk-only cells around the origin, and the 30- and 60-minute predictions are identical (§8, §10). Risk-only cells score below 50, so they never pass User Story 4.3's own `MIN_PREDICTED_TARGET_RISK_SCORE` (60.0) filter.
- EcoGuard V1 evaluates every cell at fixed 5-minute step boundaries rather than modeling rate-of-spread-based arrival time per cell, unlike the official model (§5).
- The wind-speed unit expected by the verified `alpha_wh` wind formula could not be confirmed with certainty from the available source material (§15.3); this stays an internal-to-the-calculator assumption. Separately, the EcoGuard-boundary unit is resolved: `WeatherObservation.wind_speed` is canonically km/h (IMS m/sec is converted by `WeatherMapper` at ingestion), matching the calculator's `wind_speed_kmh` contract — see §15.7 item 2.
- The IMS `WD` wind-direction convention ("from" vs. another convention) is not explicitly stated in IMS's own API documentation (§15.6); the standard meteorological "from" convention is assumed with high confidence but is not yet explicitly confirmed.
- Two numeric cells in an earlier draft's `p_n`/`v0` tables did not match the values independently re-verified in this task directly from the official source's default data files; this must be reconciled before implementation (§15.2).
- Prediction quality depends on the freshness and quality of the persisted weather/severity data being reused.
- This feature predicts potential spread and does not guarantee that a real fire will reach a predicted cell.

## 15. Scientific Verification Details (Task 4A)

This section is the full technical backing for the `[A]`-labeled claims made above. All content here was read directly from primary sources listed at the top of this document, not inferred or reconstructed.

### 15.1 Source access notes

The MDPI publisher page for the paper (`mdpi.com/2571-6255/3/3/26`) returned HTTP 403 to automated fetch in this task, so the paper's own prose and figures (notably Figure 7's graphical presentation of the wind factor) were not directly re-read here. Instead, per this task's explicit instruction to prefer the official implementation when the paper presents a formula graphically, the exact formulas were read from the official CIMAFoundation reference implementation, `legacy` branch, which explicitly cites Trucchia et al., *Fire* 2020 in its own docstrings for the moisture formula (§15.4). A follow-up task should still obtain the peer-reviewed PDF directly (e.g. via institutional access) to cross-check the implementation against the paper's prose, since a reference implementation can, in principle, drift from its own publication over time.

### 15.2 Verified `p_n` (nominal vegetation spread probability) table

Fetched verbatim from `https://raw.githubusercontent.com/CIMAFoundation/propagator_sim/legacy/prob_table.txt`, cross-referenced against the vegetation-class comment in `propagator.py` (`# [latifoglie cespugli aree_nude erba conifere coltivi faggete]`) and against `p_vegetation.txt`'s per-class comments:

| target ＼ source | Broadleaves (fire-prone) | Shrubs | Bare soil | Grassland | Conifers (fire-prone) | Agro-forestry | Broadleaves (non-fire-prone) |
|---|---|---|---|---|---|---|---|
| Broadleaves (fire-prone) | 0.300 | 0.375 | 0.005 | 0.250 | 0.275 | 0.250 | 0.250 |
| Shrubs | 0.375 | 0.375 | 0.005 | 0.350 | 0.400 | 0.300 | 0.375 |
| Bare soil | 0.005 | 0.005 | 0.005 | 0.005 | 0.005 | 0.005 | 0.005 |
| Grassland | 0.450 | 0.475 | 0.005 | 0.475 | 0.475 | 0.375 | 0.475 |
| Conifers (fire-prone) | 0.225 | 0.325 | 0.005 | **0.100** | 0.350 | 0.200 | 0.350 |
| Agro-forestry | 0.250 | 0.250 | 0.005 | 0.300 | 0.475 | 0.350 | 0.250 |
| Broadleaves (non-fire-prone) | 0.075 | 0.100 | 0.005 | 0.075 | 0.275 | 0.075 | 0.075 |

The official implementation therefore models **7** vegetation classes (including an explicit near-zero "bare soil" class), not 6. Class semantics are per `p_vegetation.txt`'s own inline comments: class 1 is specifically "fire-prone broadleaves" (not generic broadleaves), and class 7 is specifically "non fire-prone broadleaves" (a commented-out alternate line in the same file shows "non fire-prone conifers" was considered but is not the active class 7). `veg_from`/`veg_to` are 1-indexed into this table in the source (`prob_table[veg_to - 1, veg_from - 1]`).

Verified `v0` (nominal spread velocity, m/min) table, `https://raw.githubusercontent.com/CIMAFoundation/propagator_sim/legacy/v0_table.txt`, same class order: Broadleaves(fire-prone)=**140**, Shrubs=140, Bare soil=20, Grassland=120, Conifers(fire-prone)=200, Agro-forestry=120, Broadleaves(non-fire-prone)=60. (`v0` feeds rate-of-spread timing, not `p_ij` directly — see §5's fixed-step note — but is recorded here for completeness and traceability.)

**Discrepancy vs. an earlier draft of this task:** cross-checking cell-by-cell against a 6×6 table (with the bare-soil row/column already dropped) that had circulated earlier in this task's material, every cell matches this verified table **except two**:
1. `p_n[Conifers(fire-prone)][Grassland]`: earlier draft stated `0.250`; the independently re-fetched official file shows `0.100`.
2. `v0[Broadleaves(fire-prone)]`: earlier draft stated `100`; the independently re-fetched official file shows `140`.

Both are reported here rather than silently resolved. Given the second re-verification was a byte-for-byte fetch of the live official repository file (not a transcription), it is recommended as the value to adopt, but this is flagged as a decision for sign-off, not silently applied (§15.7).

### 15.3 Verified `alpha_wh` (combined wind + topography) formula

Source: `propagator/propagator.py`, `w_h_effect` and `w_h_effect_on_p`; constants from `propagator/constants.py`.

```python
# constants.py
D1, D2, D3, D4, D5 = 0.5, 1.4, 8.2, 2.0, 50.0
A = 1 - ((D1 * (D2 * np.tanh((0 / D3) - D4))) + (0 / D5))   # A is w_effect_module evaluated at w_speed = 0

# propagator.py
def w_h_effect(angle_to, w_speed, w_dir, dh, dist):
    w_effect_module = A + (D1 * (D2 * np.tanh((w_speed / D3) - D4))) + (w_speed / D5)
    a = (w_effect_module - 1) / 4
    w_effect_on_direction = (a + 1) * (1 - a ** 2) / (1 - a * np.cos(normalize(w_dir - angle_to)))
    slope = dh / (cellsize * dist)
    h_effect = 2 ** (np.tanh((slope * 3) ** 2 * np.sign(slope)))
    w_h = h_effect * w_effect_on_direction
    return w_h

def w_h_effect_on_p(angle_to, w_speed, w_dir, dh, dist_to):
    """scales the wh factor for using it on the probability modulation"""
    w_speed_norm = np.clip(w_speed, 0, 60)
    wh_orig = w_h_effect(angle_to, w_speed_norm, w_dir, dh, dist_to)
    wh = wh_orig - 1.0
    wh[wh > 0] = wh[wh > 0] / 2.13
    wh[wh < 0] = wh[wh < 0] / 1.12
    wh += 1.0
    return wh
```

`alpha_wh` (as used in §3.1's `p_probability`) is the return value of `w_h_effect_on_p`. Structurally, this **does** decompose as a product of a wind-only term (`w_effect_on_direction`, an anisotropic ellipse-shaped function of wind speed and the angle between wind direction and the propagation direction `angle_to`) and a slope-only term (`h_effect`, a function of `dh`/`dist` alone) — consistent with the paper's `alpha_wh = alpha_w * alpha_h` description — followed by a final rescaling/recentering step (`w_h_effect_on_p`) that is applied to the *combined* product specifically for use in the probability equation (a separate, non-rescaled version of `w_h` is used for rate-of-spread timing elsewhere in the same file).

At `dh = 0` (EcoGuard V1's terrain-neutral policy, §5), `h_effect = 1` exactly, so `alpha_wh` reduces to the wind-only term through the same rescaling step — this is the exact, sourced basis for treating `alpha_wh` as wind-only in EcoGuard V1.

**Open item — wind-speed unit not confirmed.** The source does not state, in a comment or docstring, whether `w_speed` entering `w_h_effect`/`w_h_effect_on_p` is expected in km/h or m/s. Evidence is mixed: the sibling Rothermel/Wang rate-of-spread functions (`p_time_rothermel`, `p_time_wang`) explicitly divide their own `w_speed` input by `3.6` (km/h → m/s) before use, but `p_time_standard` — which calls the *same* `w_h_effect` function used by `w_h_effect_on_p` — does **not** apply this conversion, passing `w_speed` straight through. The `w_speed_norm = np.clip(w_speed, 0, 60)` clamp in `w_h_effect_on_p` is also ambiguous as a unit signal: 60 m/s (≈216 km/h) is a physically plausible upper bound for extreme fire-weather gusts, while 60 km/h would be an unusually low cap for a wildfire wind model. This was not resolved by reading the equations notebook in the same repository (`notebooks/propagator - equations.ipynb`), which contains embedded binary/rendered content that could not be reliably parsed as text in this task. **This must be confirmed before calibrating EcoGuard's `wind_speed_kmh` input into `alpha_wh`** — see §15.7.

### 15.4 Verified `e_m` (fine-fuel moisture) formula

Source: `propagator/propagator.py`, `moist_proba_correction_1` — confirmed as the **default** moisture model (`get_p_moist_fn` maps both `DEFAULT_TAG` and `NEW_FORMULATION_TAG` to this function, and it is also the fallback for any unrecognized code):

```python
def moist_proba_correction_1(moist):
    """
    e_m is the moisture correction to the transition probability p_{i,j}.
    e_m = f(m), with m the Fine Fuel Moisture Content
    e_m = -11,507x5 + 22,963x4 - 17,331x3 + 6,598x2 - 1,7211x + 1,0003, where x is moisture / moisture of extintion (Mx).
    Mx = 0.3
    (reference: Trucchia et al, Fire 2020 )
    """
    Mx = 0.3
    x = np.clip(moist, 0.0, 1.0) / Mx
    p_moist = (-11.507 * x**5) + (22.963 * x**4) + (-17.331 * x**3) + (6.598 * x**2) + (-1.7211 * x) + 1.0003
    p_moist = np.clip(p_moist, 0.0, 1.0)
    return p_moist
```

I.e., the verified, quintic (5th-degree) polynomial, with the docstring itself directly attributing it to Trucchia et al., *Fire* 2020:

```
Mx = 0.30            (moisture of extinction, as a fraction — 30%)
r  = M_f / Mx         where M_f is the fine-fuel moisture content, also as a fraction
e_m = -11.507*r⁵ + 22.963*r⁴ - 17.331*r³ + 6.598*r² - 1.7211*r + 1.0003
e_m clipped to [0, 1]
```

**Discrepancy vs. an earlier draft of this task: not a cubic polynomial.** An earlier draft of this task's material proposed `e_m = 1 - 2.59r + 5.11r² - 3.52r³` (a cubic) as the standard Rothermel/Burgan moisture-damping relation. The verified default official implementation uses the **quintic** polynomial above instead, explicitly cited to the same paper. Neither this quintic's coefficients nor the earlier draft's cubic coefficients should be conflated with EcoGuard's own, separately-existing, unrelated `_moisture_damping_coefficient` cubic (`1 - 2x + 1.5x² - 0.5x³`, §4.5) — three distinct polynomials exist across these three sources and must not be mixed.

For completeness: the same file also defines `moist_proba_correction_2` ("Old formulation by Baghino, adopted in Trucchia et al, Fire 2020"), a *different* cubic — `p_moist = M1·moist³ + M2·moist² + M3·moist + M4` with `M1=-3.5995, M2=5.2389, M3=-2.6355, M4=1.019` (from `constants.py`) — but this is explicitly the *non-default*, legacy alternative (selected only via `ROTHERMEL_TAG`/`STD_FORMULATION_TAG`), not the model's default behavior, and its coefficients also do not match the earlier draft's proposed cubic.

**Unit reasoning confirmed:** EcoGuard's `_equilibrium_moisture_content` returns a percentage (e.g. `10`–`30`). Since the verified formula's `Mx = 0.30` is a fraction (30%) and `x = moist/Mx` expects `moist` as a fraction in `[0, 1]`, the correct normalization when reusing EcoGuard's percent-valued estimate is:
```
x = (equilibrium_moisture_pct / 100) / 0.30  =  equilibrium_moisture_pct / 30
```
This confirms the "divide by 30" normalization anticipated for this task is arithmetically correct, even though — as established above — the polynomial it feeds into is the verified quintic, not the previously-assumed cubic.

### 15.5 Wind-direction conversion evidence in the official implementation

Source: `propagator/propagator.py`, `run()`:
```python
w_dir_deg = float(bc.get(W_DIR_TAG, 0))
wdir = normalize((180 - w_dir_deg + 90) * np.pi / 180.0)
```
This confirms that the official implementation does **not** use its raw input wind-direction degrees directly — it applies an explicit conversion (`270 - w_dir_deg`, in degrees, before converting to radians and normalizing) before using the value internally as `w_dir` in the propagation-angle comparisons of §15.3. No inline comment explains the conversion's physical meaning at that exact line, but its form (a reflection plus a quadrant shift) is consistent with converting a compass bearing (clockwise from north) into a standard mathematical angle (counterclockwise from east) while also flipping between a "from" and a "toward" sense — i.e., exactly the kind of conversion this task was asked to check for. This is presented as evidence that such a conversion is a real, non-trivial requirement of the official model, not as a fully independent confirmation of IMS's own convention (§15.6).

### 15.6 IMS wind-direction convention

Source: IMS "Application Programming Interface (API) for meteorological data" PDF, Appendix C (channels list), read verbatim:

> `WD | deg | Wind direction`
> `WDmax | deg | Gust wind direction`

This is the entirety of what IMS's own API documentation states about the `WD` channel — it gives units (`deg`) but does **not** state whether this is the direction the wind blows *from* or *toward*. No other EcoGuard file (`backend/src/mappers/weather_mapper.py`, `backend/src/external/ims/ims_client.py`, `backend/docs/*`) documents this convention either — `weather_mapper.py` simply maps the raw `WD` channel value straight through to `WeatherObservation.wind_direction` with no transformation or comment.

**Assessment:** the international meteorological convention (WMO Manual on Codes), used essentially universally by national meteorological services, defines wind direction as the direction *from which* the wind blows, in degrees clockwise from true north. IMS is Israel's national meteorological service and its API is not documented as deviating from this norm. This makes the "from" convention highly likely, but it is **not explicitly confirmed** in IMS's own documentation, and per this task's instruction not to assume the convention without checking, it is reported here as **unconfirmed with high confidence** rather than verified. If the standard "from" convention applies, comparing `wind_direction` against a cell-to-cell propagation bearing requires converting it to a "toward" direction (a 180° rotation) before use, consistent with what §15.5 shows the official implementation itself doing to its own input.

### 15.7 Open items before Task 4B (see also §14)

1. **Vegetation-mapping policy sign-off** — adopt Approach A (§4.3.2): map only unambiguous `dominant_land_cover` values to a PROPAGATOR class; produce `INSUFFICIENT_DATA` for `"Tree cover"` and `"Moss and lichen cover"`. **RESOLVED (Task 5):** implemented exactly as decided, in `src/services/fire_spread/fire_spread_input_service.py` (`_DOMINANT_LAND_COVER_TO_FUEL_CLASS`, `_map_dominant_land_cover`) — the 8 unambiguous EcoGuard labels map to their PROPAGATOR class, `"Tree cover"`/`"Moss and lichen cover"`/any unrecognized or missing label return `None`, which the service turns into `INSUFFICIENT_DATA`.
2. **Wind-speed unit for `alpha_wh`** — confirm whether `w_h_effect`/`w_h_effect_on_p` expects m/s or km/h (§15.3) before converting EcoGuard's `wind_speed_kmh` into it. **RESOLVED for the EcoGuard boundary (Task 5):** `FireSpreadCalculator`'s own contract (Task 4B) is `wind_speed_kmh`; the internal m/s reasoning from §15.3 is unchanged and stays inside the calculator. Separately, IMS's own API documentation states `WS` and `WSmax` are in `m/sec` (Appendix C). **Wind-unit consistency fix:** `WeatherObservation.wind_speed`/`wind_gust` are canonically **km/h** for every source — `WeatherMapper` converts IMS `WS`/`WSmax` from m/s to km/h (× 3.6) at ingestion, and simulated weather is already generated in km/h. `FireSpreadInputService` therefore passes the stored value through unchanged as `wind_speed_kmh` (no conversion), exactly as Fire Danger and Fire Severity already consume it; the calculator's own internal km/h → m/s conversion for `alpha_wh` (§15.3) is unchanged. (Previously `FireSpreadInputService` multiplied the stored value by 3.6, assuming m/s, which over-stated simulated km/h wind 3.6× in spread predictions.)
3. **Two numeric-table discrepancies** — reconcile `p_n[Conifers(fire-prone)][Grassland]` (0.250 vs. verified 0.100) and `v0[Broadleaves(fire-prone)]` (100 vs. verified 140) against the independently re-fetched official source (§15.2); recommend adopting the freshly re-fetched official values. **RESOLVED (Task 4B):** the verified `p_n` table (including the corrected `0.100` entry) is what `fire_spread_config.py`'s `NOMINAL_SPREAD_PROBABILITY` implements; `v0` is not used by V1's fixed-step CA (§5) and was not implemented.
4. **IMS wind-direction convention** — obtain explicit confirmation (e.g. from IMS directly, or by cross-checking a live sample against a known wind event) that `WD` is the standard meteorological "from" convention, or determine otherwise (§15.6). **Adopted as the operational contract (Task 5), not independently re-verified beyond Task 4A's research:** `FireSpreadInputService` passes `WeatherObservation.wind_direction` straight through to `FireSpreadInput.wind_direction_deg` with no transformation, treating it as the standard meteorological "from" convention per §15.6's high-confidence assessment. If IMS is later confirmed to use a different convention, only this pass-through assumption (and possibly the calculator's internal `_wind_math_angle_rad` conversion, Task 4B) would need revisiting — no other layer depends on this choice.
5. **Spatial vegetation homogeneity** — confirm the team accepts the single event-level vegetation snapshot applied uniformly across the 5 km grid (§4.3.3) as the V1 policy, given it deviates from PROPAGATOR's per-cell fuel modeling. **RESOLVED (Task 5):** implemented exactly as decided — `FireSpreadInputService` resolves one `FireSpreadFuelClass` from the severity assessment's vegetation snapshot and `FireSpreadCalculator` (Task 4B) already applies it uniformly as both source and target class for every grid cell.

None of these require inventing a formula, constant, or mapping — each has either a verified answer already documented above, or a narrow, concrete follow-up action.

## 16. User Story 4.2 Consistency Check

| Requirement | Covered in |
|---|---|
| Active FireEvent retrieval/use | §4.1, §9 |
| Latest FireSeverityAssessment reuse | §4.2, §9 |
| Current/relevant weather reuse | §4.4, §9 |
| Wind direction | §4.4, §7, §9, §6, §15.5, §15.6 |
| Wind speed | §4.4, §7, §9, §15.3 |
| Humidity / moisture | §4.4, §4.5, §7, §9, §15.4 |
| Distance / geographic cells | §5, §6 |
| Vegetation / fuel handling | §4.3, §4.3.1, §4.3.2, §4.3.3, §7, §9, §15.2 |
| Terrain policy | §5, §9, §14, §15.3 |
| Deterministic CA calculation | §3, §8 |
| 30-minute horizon | §5, §10 |
| 60-minute horizon | §5, §10 |
| Geographic coordinates in prediction output | §11, §13 |
| Prediction timestamp/horizon | §10, §11 |
| Methodology name/version | §12 |
| Traceability | §4.2, §4.4, §11 |
| Persistence expectations | §11 |
| Insufficient mandatory data | §9 |
| RESOLVED/DISMISSED behavior | §4.1, §9 |
| Compatibility with future ResponseTargets | §13 |

All User Story 4.2 requirements listed above are addressed by this methodology document. As of Task 4A, the core `p_ij` transition equation, the `p_n` table, the `alpha_wh` formula, and the `e_m` formula are all verified against the official PROPAGATOR reference implementation (§15). As of Task 5, all five items in §15.7 are resolved or have an adopted operational contract: vegetation-mapping policy (Approach A), the EcoGuard-boundary wind-speed unit conversion, the corrected `p_n` table's adoption into the calculator, the wind-direction pass-through contract, and spatial vegetation homogeneity. None required inventing a formula, constant, or mapping. The only item not independently re-verified beyond Task 4A's own research is IMS's exact wind-direction convention (§15.7 item 4) — it is an adopted, documented assumption, not a confirmed fact.

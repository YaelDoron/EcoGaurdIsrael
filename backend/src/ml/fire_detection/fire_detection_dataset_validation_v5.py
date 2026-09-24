"""Validation and reporting for the Fire Detection ML V5 dataset.

Works on FireDetectionDatasetRowV5 rows (from the CSV via load_training_dataset_rows_v5, or from in-memory
samples via row_from_sample), so the exact same checks run on the file that will be trained on. Nothing
here trains a model and the diagnostics make no causal claim: they exist to spot generator shortcuts
(a single feature, a regime, a history shape, a time of day, a day/night flag or a paired construction that
alone determines the label) and to document how much the classes overlap.

The validator REPORTS shortcuts; it never edits the generator.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from src.ml.fire_detection.fire_detection_dataset_v5 import (
    FireDetectionDatasetRowV5,
    environment_fold_assignments,
    leave_one_no_fire_subtype_out,
    leave_one_regime_out,
    pair_indices,
)
from src.ml.fire_detection.fire_detection_features_v5 import (
    FIRE_DETECTION_FEATURE_NAMES_V5,
    NULLABLE_FEATURE_NAMES_V5,
    REMOVED_V4_FEATURE_NAMES,
    forbidden_feature_names,
)
from src.ml.fire_detection.fire_detection_training_data_generator_v5 import (
    NO_FIRE_SUBTYPES_V5,
    FIRE_SUBTYPES_V5,
    RegimeV5,
)

SUSPICIOUS_UNIVARIATE_AUC = 0.85  # any single feature this predictive (overall or inside a regime) is a shortcut
WARNING_UNIVARIATE_AUC = 0.75  # reported, not fatal
MAX_PERSISTENT_PASS_COUNT_AUC = 0.65  # persistence alone must not solve persistent_thermal
MAX_SPARSE_FEATURE_AUC = 0.70  # sparse early evidence must stay ambiguous
MAX_REGIME_FIRE_SHARE_DEVIATION = 0.05  # |fire share of a regime - 0.5|
MAX_REGIME_ONLY_AUC = 0.55  # predicting the label from the regime alone must be near chance
MAX_DUPLICATE_RATE = 0.12  # exact duplicate feature vectors overall (news-only rows are duplicate by nature)
MAX_SATELLITE_DUPLICATE_RATE = 0.01  # ... but rows with hotspot measurements must be essentially unique
MAX_DAY_NIGHT_FIRE_SHARE_GAP = 0.06
MAX_HOURLY_LABEL_RATE_DEVIATION = 0.08
MIN_SLICE_ROWS_PER_LABEL = 30
SLICE_FIRE_SHARE_RANGE = (0.25, 0.75)
MIN_ENVIRONMENTS = 20
MIN_ENVIRONMENT_BOTH_LABEL_FRACTION = 0.95
MAX_FOLD_FIRE_SHARE_DEVIATION = 0.05  # each environment fold keeps the overall fire share
MAX_FOLD_SIZE_RATIO = 1.5  # largest / smallest environment fold
MAX_PAIR_IDENTICAL_VECTOR_FRACTION = 0.30
PAIR_WIN_RATE_RANGE = (0.2, 0.8)
MIN_PAIRS_FOR_WIN_RATE = 100
NEAR_DUPLICATE_DISTANCE = 0.05  # in per-feature standard deviations
_HOUR_BUCKETS = 6

_SATELLITE_COUNT_FEATURES = ("satellite_low_count", "satellite_nominal_count", "satellite_high_count")
_NEWS_COUNT_FEATURES = (
    "news_none_count",
    "news_weak_count",
    "news_moderate_count",
    "news_strong_count",
    "news_unknown_count",
)
PAIR_KEY_FEATURES = (
    "satellite_frp_max",
    "satellite_high_count",
    "satellite_cluster_radius_km",
    "satellite_pass_count",
    "news_strong_count",
    "satellite_centroid_stability_km",
)


class DatasetValidationReportV5:
    """Pass/fail result of validate_dataset_rows_v5 with human-readable violations."""

    def __init__(self, violations: tuple[str, ...]) -> None:
        self.violations = tuple(violations)

    @property
    def is_valid(self) -> bool:
        return not self.violations


def _f(row: FireDetectionDatasetRowV5, name: str) -> float:
    value = row.feature(name)
    return math.nan if value is None else value


def _z(row: FireDetectionDatasetRowV5, name: str) -> float:
    value = row.feature(name)
    return 0.0 if value is None else value


def _satellite_total(row) -> float:
    return sum(_z(row, name) for name in _SATELLITE_COUNT_FEATURES)


def _news_total(row) -> float:
    return sum(_z(row, name) for name in _NEWS_COUNT_FEATURES)


def _auc(labels: list[int], values: list[float]) -> float | None:
    if len(set(labels)) < 2 or len(set(values)) < 2:
        return None
    return float(roc_auc_score(labels, values))


def _best_orientation(auc: float | None) -> float:
    return 0.5 if auc is None else max(auc, 1.0 - auc)


def _fire_share(rows) -> float | None:
    rows = list(rows)
    return (sum(row.label for row in rows) / len(rows)) if rows else None


# --- tables -------------------------------------------------------------------------------------


def univariate_diagnostics(rows: tuple[FireDetectionDatasetRowV5, ...]) -> list[dict]:
    """Per-feature label relationship, on the rows where the feature is computable."""
    diagnostics = []
    for index, name in enumerate(FIRE_DETECTION_FEATURE_NAMES_V5):
        present = [(row.features[index], row.label) for row in rows if row.features[index] is not None]
        values = [value for value, _ in present]
        labels = [label for _, label in present]
        negatives = [value for value, label in present if label == 0]
        positives = [value for value, label in present if label == 1]
        auc = _auc(labels, values)
        diagnostics.append(
            {
                "feature": name,
                "rows_present": len(present),
                "rows_missing": len(rows) - len(present),
                "mean_no_fire": float(np.mean(negatives)) if negatives else None,
                "mean_fire": float(np.mean(positives)) if positives else None,
                "signed_auc": auc,
                "univariate_auc": _best_orientation(auc),
                "suspicious_shortcut": _best_orientation(auc) >= SUSPICIOUS_UNIVARIATE_AUC,
                "warning": _best_orientation(auc) >= WARNING_UNIVARIATE_AUC,
            }
        )
    return diagnostics


def per_regime_univariate_auc(rows: tuple[FireDetectionDatasetRowV5, ...]) -> dict[str, dict[str, float]]:
    """regime -> feature -> best-orientation AUC inside that regime (features with >= 100 rows and both labels)."""
    result: dict[str, dict[str, float]] = {}
    for regime in sorted({row.regime for row in rows}):
        subset = [row for row in rows if row.regime == regime]
        table = {}
        for index, name in enumerate(FIRE_DETECTION_FEATURE_NAMES_V5):
            present = [(row.features[index], row.label) for row in subset if row.features[index] is not None]
            if len(present) < 100:
                continue
            auc = _auc([label for _, label in present], [value for value, _ in present])
            if auc is not None:
                table[name] = _best_orientation(auc)
        result[regime] = dict(sorted(table.items(), key=lambda item: item[1], reverse=True))
    return result


def regime_label_table(rows: tuple[FireDetectionDatasetRowV5, ...]) -> dict[str, dict]:
    counts = Counter((row.regime, row.label) for row in rows)
    table = {}
    for regime in sorted({row.regime for row in rows}):
        fire, no_fire = counts[(regime, 1)], counts[(regime, 0)]
        table[regime] = {"fire": fire, "no_fire": no_fire, "total": fire + no_fire, "fire_share": fire / (fire + no_fire)}
    return table


def regime_only_auc(rows: tuple[FireDetectionDatasetRowV5, ...]) -> float:
    """AUC of predicting the label from the regime's fire share alone (in-sample; ~0.5 when regimes are balanced)."""
    share = {regime: entry["fire_share"] for regime, entry in regime_label_table(rows).items()}
    auc = _auc([row.label for row in rows], [share[row.regime] for row in rows])
    return 0.5 if auc is None else auc


def subtype_tables(rows: tuple[FireDetectionDatasetRowV5, ...]) -> dict:
    by_subtype: dict[str, dict] = {}
    by_regime: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        entry = by_subtype.setdefault(row.latent_subtype, {"rows": 0, "label": row.label})
        entry["rows"] += 1
        by_regime[row.regime][row.latent_subtype] += 1
    return {
        "rows_per_subtype": dict(sorted(by_subtype.items())),
        "subtypes_per_regime": {regime: dict(sorted(counts.items())) for regime, counts in sorted(by_regime.items())},
    }


def environment_summary(rows: tuple[FireDetectionDatasetRowV5, ...]) -> dict:
    by_environment: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        by_environment[row.environment_id].append(row.label)
    sizes = sorted(len(labels) for labels in by_environment.values())
    both = sum(1 for labels in by_environment.values() if set(labels) == {0, 1})
    shares = [sum(labels) / len(labels) for labels in by_environment.values()]
    return {
        "environments": len(by_environment),
        "rows_per_environment_min": sizes[0],
        "rows_per_environment_median": sizes[len(sizes) // 2],
        "rows_per_environment_max": sizes[-1],
        "environments_with_both_labels": both,
        "both_label_fraction": both / len(by_environment),
        "environment_fire_share_min": min(shares),
        "environment_fire_share_max": max(shares),
    }


def _pair_rows(rows) -> dict[int, tuple[FireDetectionDatasetRowV5, FireDetectionDatasetRowV5]]:
    return {pair_id: (rows[fire], rows[no_fire]) for pair_id, (fire, no_fire) in pair_indices(rows).items()}


def pair_analysis(rows: tuple[FireDetectionDatasetRowV5, ...]) -> dict:
    """How much fire / no-fire members of one pair overlap in feature space."""
    pairs = _pair_rows(rows)
    if not pairs:
        return {"pairs": 0}
    by_type = Counter(fire.pair_type for fire, _ in pairs.values())
    same_environment = sum(1 for fire, no_fire in pairs.values() if fire.environment_id == no_fire.environment_id)
    same_regime = sum(1 for fire, no_fire in pairs.values() if fire.regime == no_fire.regime)
    identical = sum(1 for fire, no_fire in pairs.values() if fire.features == no_fire.features)

    regime_std: dict[tuple[str, str], float] = {}
    for regime in {row.regime for row in rows}:
        for name in PAIR_KEY_FEATURES:
            values = [v for row in rows if row.regime == regime and (v := row.feature(name)) is not None]
            regime_std[(regime, name)] = float(np.std(values)) if len(values) > 1 else 0.0

    features = {}
    for name in PAIR_KEY_FEATURES:
        wins = ties = comparable = close = 0
        for fire, no_fire in pairs.values():
            a, b = fire.feature(name), no_fire.feature(name)
            if a is None or b is None:
                continue
            comparable += 1
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
            spread = regime_std[(fire.regime, name)]
            if spread > 0 and abs(a - b) / spread < 0.5:
                close += 1
        features[name] = {
            "comparable_pairs": comparable,
            "fire_greater_win_rate": ((wins + 0.5 * ties) / comparable) if comparable else None,
            "tie_fraction": (ties / comparable) if comparable else None,
            "within_half_std_fraction": (close / comparable) if comparable else None,
        }
    return {
        "pairs": len(pairs),
        "pairs_per_type": dict(sorted(by_type.items())),
        "same_environment": same_environment,
        "same_regime": same_regime,
        "identical_feature_vector_fraction": identical / len(pairs),
        "key_feature_overlap": features,
    }


def history_distribution(rows: tuple[FireDetectionDatasetRowV5, ...]) -> dict:
    pass_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"fire": 0, "no_fire": 0})
    span_buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"fire": 0, "no_fire": 0})
    availability: dict[str, dict[str, int]] = defaultdict(lambda: {"fire": 0, "no_fire": 0})

    def span_bucket(span: float | None) -> str:
        if span is None:
            return "no_satellite_pass"
        if span == 0:
            return "0 (single pass)"
        if span < 180:
            return "(0, 3h)"
        if span < 360:
            return "[3h, 6h)"
        if span < 720:
            return "[6h, 12h)"
        return ">=12h"

    key = lambda row: "fire" if row.label else "no_fire"  # noqa: E731
    for row in rows:
        pass_counts[str(int(_z(row, "satellite_pass_count")))][key(row)] += 1
        span_buckets[span_bucket(row.feature("satellite_history_span_minutes"))][key(row)] += 1
        for name in ("satellite_centroid_stability_km", "satellite_frp_trend_per_hour", "satellite_brightness_trend_per_hour"):
            state = "available" if row.feature(name) is not None else "NaN"
            availability[f"{name}:{state}"][key(row)] += 1

    def with_share(table):
        return {
            label: {**cell, "total": cell["fire"] + cell["no_fire"], "fire_share": cell["fire"] / (cell["fire"] + cell["no_fire"])}
            for label, cell in sorted(table.items())
        }

    return {
        "pass_count_by_label": with_share(pass_counts),
        "history_span_by_label": with_share(span_buckets),
        "history_feature_availability_by_label": with_share(availability),
    }


def day_night_distribution(rows: tuple[FireDetectionDatasetRowV5, ...]) -> dict:
    def state(row) -> str:
        night = row.feature("satellite_night_fraction")
        if night is None:
            return "unknown_or_no_satellite"
        return "day" if night == 0 else "night" if night == 1 else "mixed"

    table: dict[str, dict[str, int]] = defaultdict(lambda: {"fire": 0, "no_fire": 0})
    for row in rows:
        table[state(row)]["fire" if row.label else "no_fire"] += 1
    return {
        name: {**cell, "total": cell["fire"] + cell["no_fire"], "fire_share": cell["fire"] / (cell["fire"] + cell["no_fire"])}
        for name, cell in sorted(table.items())
    }


def news_distribution(rows: tuple[FireDetectionDatasetRowV5, ...]) -> dict:
    def state(row) -> str:
        if _news_total(row) == 0:
            return "no_news"
        strongest = next(
            (name for name in ("strong", "moderate", "weak", "none") if _z(row, f"news_{name}_count") > 0),
            "unknown_only",
        )
        return f"strongest_{strongest}"

    table: dict[str, dict[str, int]] = defaultdict(lambda: {"fire": 0, "no_fire": 0})
    for row in rows:
        table[state(row)]["fire" if row.label else "no_fire"] += 1
    return {
        name: {**cell, "total": cell["fire"] + cell["no_fire"], "fire_share": cell["fire"] / (cell["fire"] + cell["no_fire"])}
        for name, cell in sorted(table.items())
    }


def hourly_label_rates(rows: tuple[FireDetectionDatasetRowV5, ...]) -> dict[str, dict]:
    width = 24 // _HOUR_BUCKETS
    buckets: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        buckets[row.as_of_utc.hour // width].append(row.label)
    return {
        f"{bucket * width:02d}-{bucket * width + width - 1:02d}h": {"rows": len(labels), "fire_share": sum(labels) / len(labels)}
        for bucket, labels in sorted(buckets.items())
    }


def coverage_slices(rows: tuple[FireDetectionDatasetRowV5, ...]) -> dict[str, dict]:
    """Evidence situations that must appear under BOTH labels (anti-shortcut coverage)."""
    frp = np.array([v for row in rows if (v := row.feature("satellite_frp_max")) is not None and _satellite_total(row) > 0])
    high_frp = float(np.percentile(frp, 75)) if len(frp) else math.inf
    low_frp = float(np.percentile(frp, 25)) if len(frp) else -math.inf
    definitions = {
        "single_satellite_hotspot": lambda r: _satellite_total(r) == 1 and _z(r, "satellite_pass_count") == 1,
        "multiple_hotspots": lambda r: _satellite_total(r) >= 2,
        "multiple_passes": lambda r: _z(r, "satellite_pass_count") >= 2,
        "high_frp_top_quartile": lambda r: _satellite_total(r) > 0 and (r.feature("satellite_frp_max") or 0.0) >= high_frp and _z(r, "satellite_frp_available_ratio") > 0,
        "low_frp_bottom_quartile": lambda r: _satellite_total(r) > 0 and _z(r, "satellite_frp_available_ratio") > 0 and (r.feature("satellite_frp_max") or 0.0) <= low_frp,
        "strong_news": lambda r: _z(r, "news_strong_count") > 0,
        "missing_news": lambda r: _news_total(r) == 0,
        "news_only": lambda r: _satellite_total(r) == 0 and _news_total(r) > 0,
        "night_observation": lambda r: r.feature("satellite_night_fraction") == 1.0,
        "day_observation": lambda r: r.feature("satellite_night_fraction") == 0.0,
        "missing_frp": lambda r: _satellite_total(r) > 0 and _z(r, "satellite_frp_available_ratio") == 0,
        "news_before_satellite": lambda r: (r.feature("news_satellite_lag_minutes") or 0.0) < 0,
    }
    table = {}
    for name, predicate in definitions.items():
        subset = [row for row in rows if predicate(row)]
        fire = sum(row.label for row in subset)
        table[name] = {
            "fire": fire,
            "no_fire": len(subset) - fire,
            "total": len(subset),
            "fire_share": (fire / len(subset)) if subset else None,
        }
    return table


def duplicate_analysis(rows: tuple[FireDetectionDatasetRowV5, ...]) -> dict:
    def summarize(subset) -> dict:
        vectors = Counter(row.features for row in subset)
        labels_by_vector: dict[tuple, set[int]] = defaultdict(set)
        for row in subset:
            labels_by_vector[row.features].add(row.label)
        unique = len(vectors)
        return {
            "rows": len(subset),
            "unique_feature_rows": unique,
            "duplicate_rows": len(subset) - unique,
            "duplicate_rate": ((len(subset) - unique) / len(subset)) if subset else 0.0,
            "vectors_with_both_labels": sum(1 for labels in labels_by_vector.values() if len(labels) > 1),
        }

    satellite_rows = tuple(row for row in rows if _satellite_total(row) > 0)
    news_only_rows = tuple(row for row in rows if _satellite_total(row) == 0)
    return {
        "overall": summarize(rows),
        "rows_with_satellite": summarize(satellite_rows),
        "news_only_rows": summarize(news_only_rows),
        "near_duplicates": _near_duplicates(satellite_rows),
    }


def _near_duplicates(rows: tuple[FireDetectionDatasetRowV5, ...]) -> dict:
    """Nearest-neighbour distance (per-feature-std units, NaN -> 0 plus a missing-indicator) among satellite rows."""
    if len(rows) < 3:
        return {"rows": len(rows), "near_duplicate_rate": 0.0, "cross_label_near_duplicate_rate": 0.0}
    matrix = np.array([[math.nan if v is None else v for v in row.features] for row in rows], dtype=float)
    missing = np.isnan(matrix[:, [FIRE_DETECTION_FEATURE_NAMES_V5.index(n) for n in NULLABLE_FEATURE_NAMES_V5]])
    std = np.nanstd(matrix, axis=0)
    std[std == 0] = 1.0
    scaled = np.nan_to_num((matrix - np.nanmean(matrix, axis=0)) / std, nan=0.0)
    scaled = np.hstack([scaled, missing.astype(float)])
    distances, neighbours = NearestNeighbors(n_neighbors=2).fit(scaled).kneighbors(scaled)
    labels = np.array([row.label for row in rows])
    near = distances[:, 1] < NEAR_DUPLICATE_DISTANCE
    cross = near & (labels != labels[neighbours[:, 1]])
    return {
        "rows": len(rows),
        "near_duplicate_distance_std": NEAR_DUPLICATE_DISTANCE,
        "near_duplicate_rate": float(near.mean()),
        "cross_label_near_duplicate_rate": float(cross.mean()),
    }


def grouped_layout(rows: tuple[FireDetectionDatasetRowV5, ...]) -> dict:
    assignments = environment_fold_assignments(rows, 5)
    folds = {}
    for fold in range(5):
        members = [row for row in rows if assignments[row.environment_id] == fold]
        folds[str(fold)] = {
            "rows": len(members),
            "environments": sum(1 for a in assignments.values() if a == fold),
            "fire_rows": sum(row.label for row in members),
            "regime_fire_no_fire": {
                regime: [sum(1 for r in members if r.regime == regime and r.label == 1), sum(1 for r in members if r.regime == regime and r.label == 0)]
                for regime in sorted({row.regime for row in rows})
            },
        }
    return {
        "environment_folds_5": folds,
        "leave_one_regime_out": {
            regime: {"train_rows": len(train), "validation_rows": len(val), "validation_fire_rows": sum(rows[i].label for i in val)}
            for regime, (train, val) in leave_one_regime_out(rows)
        },
        "leave_one_no_fire_subtype_out": {
            subtype: {"train_rows": len(train), "validation_rows": len(val)}
            for subtype, (train, val) in leave_one_no_fire_subtype_out(rows)
        },
    }


def build_dataset_report_v5(rows: tuple[FireDetectionDatasetRowV5, ...]) -> dict:
    """A JSON-serializable statistics report over a V5 dataset."""
    positives = sum(row.label for row in rows)
    return {
        "basic": {
            "rows": len(rows),
            "positive_rows": positives,
            "negative_rows": len(rows) - positives,
            "positive_percent": round(100.0 * positives / len(rows), 2) if rows else None,
            "seeds": sorted({row.seed for row in rows}),
        },
        "regimes": regime_label_table(rows),
        "regime_only_auc": regime_only_auc(rows),
        "latent_subtypes": subtype_tables(rows),
        "environments": environment_summary(rows),
        "pairs": pair_analysis(rows),
        "univariate_feature_diagnostics": sorted(
            univariate_diagnostics(rows), key=lambda item: item["univariate_auc"], reverse=True
        ),
        "per_regime_univariate_auc": per_regime_univariate_auc(rows),
        "history": history_distribution(rows),
        "day_night": day_night_distribution(rows),
        "time_of_day_fire_share": hourly_label_rates(rows),
        "news": news_distribution(rows),
        "coverage_slices": coverage_slices(rows),
        "duplicates": duplicate_analysis(rows),
        "grouped_evaluation": grouped_layout(rows),
    }


# --- validation -----------------------------------------------------------------------------------


def validate_dataset_rows_v5(
    rows: tuple[FireDetectionDatasetRowV5, ...],
    min_rows: int = 0,
) -> DatasetValidationReportV5:
    """Run every V5 sanity/shortcut check and return the combined violation list."""
    violations: list[str] = []
    if not rows:
        return DatasetValidationReportV5(("no rows.",))
    if len(rows) < min_rows:
        violations.append(f"dataset has {len(rows)} rows; at least {min_rows} required.")

    _check_schema(violations)
    if {row.label for row in rows} != {0, 1}:
        violations.append(f"both labels must be present; found {sorted({row.label for row in rows})}.")
        return DatasetValidationReportV5(tuple(violations))

    _check_row_contract(rows, violations)
    _check_regimes(rows, violations)
    _check_subtypes(rows, violations)
    _check_environments(rows, violations)
    _check_pairs(rows, violations)
    _check_shortcuts(rows, violations)
    _check_slices(rows, violations)
    _check_duplicates(rows, violations)
    _check_grouped_layout(rows, violations)
    return DatasetValidationReportV5(tuple(violations))


def _check_schema(violations: list[str]) -> None:
    leaking = forbidden_feature_names()
    if leaking:
        violations.append(f"leakage-looking ML feature names: {leaking}")
    present_removed = [name for name in REMOVED_V4_FEATURE_NAMES if name in FIRE_DETECTION_FEATURE_NAMES_V5]
    if present_removed:
        violations.append(f"features removed from V4 are present in V5: {present_removed}")


def _check_row_contract(rows, violations: list[str]) -> None:
    for row in rows:
        try:
            row.to_features_v5()
        except ValueError as exc:
            violations.append(f"sample {row.sample_id}: features violate the V5 contract: {exc}")
            return  # one example is enough to fail; avoid flooding the report


def _check_regimes(rows, violations: list[str]) -> None:
    table = regime_label_table(rows)
    known = {regime.value for regime in RegimeV5}
    if set(table) != known:
        violations.append(f"regimes present {sorted(table)} differ from the defined regimes {sorted(known)}.")
    for regime, entry in table.items():
        if entry["fire"] == 0 or entry["no_fire"] == 0:
            violations.append(f"regime {regime!r} does not contain both labels.")
        elif abs(entry["fire_share"] - 0.5) > MAX_REGIME_FIRE_SHARE_DEVIATION:
            violations.append(f"regime {regime!r} is label-skewed: fire share {entry['fire_share']:.3f}.")
    auc = regime_only_auc(rows)
    if auc > MAX_REGIME_ONLY_AUC:
        violations.append(f"the regime alone predicts the label (AUC {auc:.3f}): regimes leak the target.")


def _check_subtypes(rows, violations: list[str]) -> None:
    present = {row.latent_subtype for row in rows}
    for subtype in (*FIRE_SUBTYPES_V5, *NO_FIRE_SUBTYPES_V5):
        if subtype.value not in present:
            violations.append(f"latent subtype {subtype.value!r} is missing.")
    fire_values = {s.value for s in FIRE_SUBTYPES_V5}
    for row in rows:
        if (row.latent_subtype in fire_values) != (row.label == 1):
            violations.append(f"sample {row.sample_id}: latent_subtype {row.latent_subtype!r} contradicts label {row.label}.")
            break


def _check_environments(rows, violations: list[str]) -> None:
    summary = environment_summary(rows)
    if summary["environments"] < MIN_ENVIRONMENTS and len(rows) >= 1000:
        violations.append(f"only {summary['environments']} environments; at least {MIN_ENVIRONMENTS} required.")
    if summary["both_label_fraction"] < MIN_ENVIRONMENT_BOTH_LABEL_FRACTION:
        violations.append(f"only {summary['both_label_fraction']:.1%} of environments contain both labels.")


def _check_pairs(rows, violations: list[str]) -> None:
    try:
        pairs = _pair_rows(rows)
    except ValueError as exc:
        violations.append(str(exc))
        return
    for pair_id, (fire, no_fire) in pairs.items():
        if fire.environment_id != no_fire.environment_id:
            violations.append(f"pair {pair_id} members live in different environments.")
            break
        if fire.regime != no_fire.regime:
            violations.append(f"pair {pair_id} members belong to different regimes.")
            break
    if not pairs:
        violations.append("no paired fire / no-fire examples.")
        return
    analysis = pair_analysis(rows)
    if analysis["identical_feature_vector_fraction"] > MAX_PAIR_IDENTICAL_VECTOR_FRACTION:
        violations.append(
            f"{analysis['identical_feature_vector_fraction']:.1%} of pairs have identical feature vectors "
            "(pair members must differ through plausible draws)."
        )
    for name, entry in analysis["key_feature_overlap"].items():
        rate = entry["fire_greater_win_rate"]
        if entry["comparable_pairs"] >= MIN_PAIRS_FOR_WIN_RATE and rate is not None:
            if not PAIR_WIN_RATE_RANGE[0] <= rate <= PAIR_WIN_RATE_RANGE[1]:
                violations.append(f"pair feature {name!r} separates fire from no-fire inside pairs (win rate {rate:.2f}).")


def _check_shortcuts(rows, violations: list[str]) -> None:
    for item in univariate_diagnostics(rows):
        if item["suspicious_shortcut"]:
            violations.append(f"feature {item['feature']!r} alone separates the label (AUC {item['univariate_auc']:.3f}).")
    per_regime = per_regime_univariate_auc(rows)
    for regime, table in per_regime.items():
        for name, auc in table.items():
            if auc >= SUSPICIOUS_UNIVARIATE_AUC:
                violations.append(f"feature {name!r} alone separates the label inside regime {regime!r} (AUC {auc:.3f}).")
    persistent = per_regime.get(RegimeV5.PERSISTENT_THERMAL.value, {})
    if persistent.get("satellite_pass_count", 0.5) > MAX_PERSISTENT_PASS_COUNT_AUC:
        violations.append(
            f"persistence alone solves persistent_thermal (pass_count AUC {persistent['satellite_pass_count']:.3f})."
        )
    for name, auc in per_regime.get(RegimeV5.SPARSE_EARLY_EVIDENCE.value, {}).items():
        if auc > MAX_SPARSE_FEATURE_AUC:
            violations.append(f"sparse_early_evidence is not ambiguous: {name!r} has AUC {auc:.3f}.")

    overall = sum(row.label for row in rows) / len(rows)
    for bucket, entry in hourly_label_rates(rows).items():
        if entry["rows"] >= 200 and abs(entry["fire_share"] - overall) > MAX_HOURLY_LABEL_RATE_DEVIATION:
            violations.append(f"time of day {bucket} has fire share {entry['fire_share']:.3f} vs overall {overall:.3f}.")
    day_night = day_night_distribution(rows)
    if "day" in day_night and "night" in day_night:
        gap = abs(day_night["day"]["fire_share"] - day_night["night"]["fire_share"])
        if gap > MAX_DAY_NIGHT_FIRE_SHARE_GAP:
            violations.append(f"day/night predicts the label (fire share gap {gap:.3f}).")
    else:
        violations.append("both day and night observations must exist.")


def _check_slices(rows, violations: list[str]) -> None:
    for name, entry in coverage_slices(rows).items():
        if entry["fire"] < MIN_SLICE_ROWS_PER_LABEL or entry["no_fire"] < MIN_SLICE_ROWS_PER_LABEL:
            violations.append(f"slice {name!r} lacks both labels (fire {entry['fire']}, no-fire {entry['no_fire']}).")
        elif not SLICE_FIRE_SHARE_RANGE[0] <= entry["fire_share"] <= SLICE_FIRE_SHARE_RANGE[1]:
            violations.append(f"slice {name!r} is dominated by one label (fire share {entry['fire_share']:.2f}).")


def _check_duplicates(rows, violations: list[str]) -> None:
    duplicates = duplicate_analysis(rows)
    if duplicates["overall"]["duplicate_rate"] > MAX_DUPLICATE_RATE:
        violations.append(f"duplicate feature-row rate {duplicates['overall']['duplicate_rate']:.1%} exceeds {MAX_DUPLICATE_RATE:.0%}.")
    if duplicates["rows_with_satellite"]["duplicate_rate"] > MAX_SATELLITE_DUPLICATE_RATE:
        violations.append(
            f"rows with hotspot measurements duplicate at {duplicates['rows_with_satellite']['duplicate_rate']:.1%} "
            f"(limit {MAX_SATELLITE_DUPLICATE_RATE:.0%})."
        )


def _check_grouped_layout(rows, violations: list[str]) -> None:
    """The primary (environment) folds must be non-empty, similar in size, label-balanced and regime-complete."""
    if len({row.environment_id for row in rows}) < 5:
        return  # too few environments for 5 folds (tiny test datasets)
    folds = grouped_layout(rows)["environment_folds_5"]
    sizes = [entry["rows"] for entry in folds.values()]
    if min(sizes) == 0:
        violations.append(f"an environment fold is empty (fold sizes {sizes}).")
        return
    if max(sizes) / min(sizes) > MAX_FOLD_SIZE_RATIO:
        violations.append(f"environment folds are unbalanced (fold sizes {sizes}).")
    overall = sum(row.label for row in rows) / len(rows)
    for fold, entry in folds.items():
        share = entry["fire_rows"] / entry["rows"]
        if abs(share - overall) > MAX_FOLD_FIRE_SHARE_DEVIATION:
            violations.append(f"environment fold {fold} has fire share {share:.3f} vs overall {overall:.3f}.")
        for regime, (fire, no_fire) in entry["regime_fire_no_fire"].items():
            if fire == 0 or no_fire == 0:
                violations.append(f"environment fold {fold} lacks a label in regime {regime!r}.")

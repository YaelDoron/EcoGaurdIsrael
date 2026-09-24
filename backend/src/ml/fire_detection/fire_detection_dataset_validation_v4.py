"""Validation and reporting for the Fire Detection ML V4 dataset.

Works on FireDetectionDatasetRowV4 rows (from the CSV via
load_training_dataset_rows_v4, or from in-memory samples via row_from_sample),
so the exact same checks run on the file that will be trained on. Nothing here
trains a model, and the diagnostics make no causal claim: they exist to spot
generator shortcuts (a single feature, evidence shape, geometry, time of day,
missingness, or Fire Danger that alone nearly determines the label).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from src.calculators.fire_danger.ffwi_config import (
    EXTREME_THRESHOLD,
    FFWI_MAX_SCORE,
    FFWI_MIN_SCORE,
    HIGH_THRESHOLD,
    MODERATE_THRESHOLD,
    VERY_HIGH_THRESHOLD,
)
from src.ml.fire_detection.fire_detection_dataset_v4 import (
    FireDetectionDatasetRowV4,
    family_fold_assignments,
    leave_one_archetype_out,
)
from src.ml.fire_detection.fire_detection_features_v4 import (
    FIRE_DANGER_BAND_MISSING,
    FIRE_DANGER_BANDS,
    FIRE_DETECTION_FEATURE_NAMES_V4,
    forbidden_feature_names,
)
from src.services.fire_detection.fire_detection_context_config import MAX_FIRE_DANGER_ASSESSMENT_AGE_MINUTES

SUSPICIOUS_UNIVARIATE_AUC = 0.85  # any single feature this predictive is treated as a shortcut
MAX_GEOMETRY_AUC = 0.65  # distance / time-span should not decide the label (V3's main shortcut)
MAX_FIRE_DANGER_AUC = 0.65  # FFWI is weak context, not a label proxy
MAX_DUPLICATE_RATE = 0.05
MAX_MISSINGNESS_LABEL_GAP = 0.05
MAX_HOURLY_LABEL_RATE_DEVIATION = 0.08
MIN_MINORITY_LABEL_FRACTION = 0.40
MIN_BAND_MINORITY_FRACTION = 0.15
MIN_BAND_ROWS_FOR_FRACTION_CHECK = 30
MIN_HARD_NEGATIVE_FRACTION = 0.50
MIN_WEAK_POSITIVE_FRACTION = 0.05
MIN_FAMILIES = 10
_HOUR_BUCKETS = 6

_SATELLITE_COUNT_FEATURES = ("satellite_low_count", "satellite_nominal_count", "satellite_high_count")
_NEWS_COUNT_FEATURES = (
    "news_none_count",
    "news_weak_count",
    "news_moderate_count",
    "news_strong_count",
    "news_unknown_count",
)
_COUNT_FEATURES = _SATELLITE_COUNT_FEATURES + _NEWS_COUNT_FEATURES
_RATIO_FEATURES = ("satellite_frp_available_ratio", "satellite_brightness_available_ratio", "fire_danger_available")


@dataclass(frozen=True)
class DatasetValidationReportV4:
    """Pass/fail result of validate_dataset_rows_v4 with human-readable violations."""

    violations: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.violations


def band_for_score(score: float) -> str:
    """FFWI band name for a score, using the project's own FFWI thresholds."""
    if score < MODERATE_THRESHOLD:
        return "low"
    if score < HIGH_THRESHOLD:
        return "moderate"
    if score < VERY_HIGH_THRESHOLD:
        return "high"
    if score < EXTREME_THRESHOLD:
        return "very_high"
    return "extreme"


def _f(row: FireDetectionDatasetRowV4, name: str) -> float:
    value = row.feature(name)
    return 0.0 if value is None else value


def _satellite_total(row) -> float:
    return sum(_f(row, name) for name in _SATELLITE_COUNT_FEATURES)


def _news_total(row) -> float:
    return sum(_f(row, name) for name in _NEWS_COUNT_FEATURES)


def is_hard_negative(row: FireDetectionDatasetRowV4) -> bool:
    """A no-fire candidate that carries genuinely fire-looking evidence (not noise or empty)."""
    return row.label == 0 and (
        _f(row, "satellite_nominal_count") + _f(row, "satellite_high_count") > 0
        or _f(row, "news_moderate_count") + _f(row, "news_strong_count") > 0
    )


def is_weak_positive(row: FireDetectionDatasetRowV4) -> bool:
    """A real fire whose evidence is thin: one item, or hotspots that are all low/nominal with no news."""
    if row.label != 1:
        return False
    only_weak_satellites = _news_total(row) == 0 and _f(row, "satellite_high_count") == 0
    return _satellite_total(row) + _news_total(row) == 1 or only_weak_satellites


def _best_orientation_auc(labels: list[int], values: list[float]) -> float:
    if len(set(labels)) < 2 or len(set(values)) < 2:
        return 0.5
    auc = float(roc_auc_score(labels, values))
    return max(auc, 1.0 - auc)


def univariate_diagnostics(rows: tuple[FireDetectionDatasetRowV4, ...]) -> list[dict]:
    """Per-feature label relationship, on the rows where the feature is present."""
    diagnostics = []
    for index, name in enumerate(FIRE_DETECTION_FEATURE_NAMES_V4):
        present = [(row.features[index], row.label) for row in rows if row.features[index] is not None]
        values = [value for value, _ in present]
        labels = [label for _, label in present]
        negatives = [value for value, label in present if label == 0]
        positives = [value for value, label in present if label == 1]
        constant = len(set(values)) < 2
        diagnostics.append(
            {
                "feature": name,
                "rows_present": len(present),
                "mean_no_fire": float(np.mean(negatives)) if negatives else None,
                "mean_fire": float(np.mean(positives)) if positives else None,
                "pearson_r": None if constant else float(np.corrcoef(values, labels)[0, 1]),
                "univariate_auc": _best_orientation_auc(labels, values),
                "suspicious_shortcut": _best_orientation_auc(labels, values) >= SUSPICIOUS_UNIVARIATE_AUC,
            }
        )
    return diagnostics


def fire_danger_band_crosstab(rows: tuple[FireDetectionDatasetRowV4, ...]) -> dict[str, dict]:
    counts = Counter((row.fire_danger_band, row.label) for row in rows)
    table = {}
    for band in FIRE_DANGER_BANDS:
        no_fire, fire = counts[(band, 0)], counts[(band, 1)]
        total = no_fire + fire
        table[band] = {
            "no_fire": no_fire,
            "fire": fire,
            "total": total,
            "fire_share": (fire / total) if total else None,
        }
    return table


def hourly_label_rates(rows: tuple[FireDetectionDatasetRowV4, ...]) -> dict[str, dict]:
    width = 24 // _HOUR_BUCKETS
    buckets: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        buckets[row.as_of_utc.hour // width].append(row.label)
    return {
        f"{bucket * width:02d}-{bucket * width + width - 1:02d}h": {
            "rows": len(labels),
            "fire_share": sum(labels) / len(labels),
        }
        for bucket, labels in sorted(buckets.items())
    }


def duplicate_analysis(rows: tuple[FireDetectionDatasetRowV4, ...]) -> dict:
    vectors = Counter(row.features for row in rows)
    labels_by_vector: dict[tuple, set[int]] = defaultdict(set)
    for row in rows:
        labels_by_vector[row.features].add(row.label)
    unique = len(vectors)
    return {
        "rows": len(rows),
        "unique_feature_rows": unique,
        "duplicate_rows": len(rows) - unique,
        "duplicate_rate": (len(rows) - unique) / len(rows) if rows else 0.0,
        "vectors_with_both_labels": sum(1 for labels in labels_by_vector.values() if len(labels) > 1),
    }


def build_dataset_report_v4(rows: tuple[FireDetectionDatasetRowV4, ...]) -> dict:
    """A JSON-serializable statistics report over a V4 dataset."""
    total = len(rows)
    positives = sum(row.label for row in rows)
    family_counts: dict[str, dict] = {}
    for row in rows:
        entry = family_counts.setdefault(
            row.scenario_family, {"rows": 0, "label": row.label, "archetype": row.scenario_archetype}
        )
        entry["rows"] += 1
    archetype_counts: dict[str, dict] = defaultdict(lambda: {"fire": 0, "no_fire": 0})
    for row in rows:
        archetype_counts[row.scenario_archetype]["fire" if row.label else "no_fire"] += 1

    satellite_rows = [row for row in rows if _satellite_total(row) > 0]
    news_rows = [row for row in rows if _news_total(row) > 0]
    by_label = {label: [row for row in rows if row.label == label] for label in (0, 1)}

    def rate(subset, predicate) -> float | None:
        return (sum(1 for row in subset if predicate(row)) / len(subset)) if subset else None

    fire_danger_unavailable = lambda row: _f(row, "fire_danger_available") == 0  # noqa: E731
    hard_negatives = [row for row in by_label[0] if is_hard_negative(row)]
    weak_positives = [row for row in by_label[1] if is_weak_positive(row)]

    try:
        assignments = family_fold_assignments(rows, 5)
        family_folds = {
            fold: {
                "families": sorted(family for family, assigned in assignments.items() if assigned == fold),
                "fire_rows": sum(row.label for row in rows if assignments[row.scenario_family] == fold),
                "no_fire_rows": sum(1 - row.label for row in rows if assignments[row.scenario_family] == fold),
            }
            for fold in range(5)
        }
    except ValueError as exc:  # e.g. a family with both labels: reported by validation, not fatal here
        family_folds = {"error": str(exc)}

    return {
        "basic": {
            "rows": total,
            "positive_rows": positives,
            "negative_rows": total - positives,
            "positive_percent": round(100.0 * positives / total, 2) if total else None,
            "scenario_families": len(family_counts),
            "rows_per_family": {name: entry for name, entry in sorted(family_counts.items())},
            "archetype_label_counts": dict(sorted(archetype_counts.items())),
        },
        "missingness": {
            "fire_danger_unavailable_rate": rate(rows, fire_danger_unavailable),
            "fire_danger_unavailable_rate_by_label": {
                "no_fire": rate(by_label[0], fire_danger_unavailable),
                "fire": rate(by_label[1], fire_danger_unavailable),
            },
            "rows_with_satellite": len(satellite_rows),
            "frp_missing_rate_among_satellite_rows": rate(satellite_rows, lambda row: _f(row, "satellite_frp_available_ratio") == 0),
            "brightness_missing_rate_among_satellite_rows": rate(
                satellite_rows, lambda row: _f(row, "satellite_brightness_available_ratio") == 0
            ),
            "rows_with_news": len(news_rows),
            "news_analysis_unavailable_rate_among_news_rows": rate(
                news_rows, lambda row: _f(row, "news_unknown_count") == _news_total(row)
            ),
        },
        "fire_danger_band_by_label": fire_danger_band_crosstab(rows),
        "univariate_feature_diagnostics": sorted(
            univariate_diagnostics(rows), key=lambda item: item["univariate_auc"], reverse=True
        ),
        "shape_and_difficulty": {
            "hard_negative_fraction_of_negatives": (len(hard_negatives) / len(by_label[0])) if by_label[0] else None,
            "weak_positive_fraction_of_positives": (len(weak_positives) / len(by_label[1])) if by_label[1] else None,
            "single_item_candidate_fire_share": rate(
                [row for row in rows if _satellite_total(row) + _news_total(row) == 1], lambda row: row.label == 1
            ),
            "candidates_over_3km_fire_share": rate(
                [row for row in rows if _f(row, "max_pairwise_distance_km") > 3.0], lambda row: row.label == 1
            ),
        },
        "time_of_day_fire_share": hourly_label_rates(rows),
        "duplicates": duplicate_analysis(rows),
        "grouped_evaluation": {
            "family_folds_5": family_folds,
            "leave_one_archetype_out": {
                archetype: {
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "validation_fire_rows": sum(rows[i].label for i in validation),
                }
                for archetype, (train, validation) in leave_one_archetype_out(rows)
            },
        },
    }


def validate_dataset_rows_v4(
    rows: tuple[FireDetectionDatasetRowV4, ...],
    min_rows: int = 0,
) -> DatasetValidationReportV4:
    """Run every V4 sanity/shortcut check and return the combined violation list."""
    violations: list[str] = []
    if not rows:
        return DatasetValidationReportV4(("no rows.",))
    if len(rows) < min_rows:
        violations.append(f"dataset has {len(rows)} rows; at least {min_rows} required.")

    leaking = forbidden_feature_names()
    if leaking:
        violations.append(f"leakage-looking ML feature names: {leaking}")

    labels_present = {row.label for row in rows}
    if labels_present != {0, 1}:
        violations.append(f"both labels must be present; found {sorted(labels_present)}.")
        return DatasetValidationReportV4(tuple(violations))

    positive_count = sum(row.label for row in rows)
    if min(positive_count, len(rows) - positive_count) / len(rows) < MIN_MINORITY_LABEL_FRACTION:
        violations.append(f"labels are not reasonably balanced: {positive_count} fire of {len(rows)}.")

    _check_families(rows, violations)
    _check_value_ranges_and_missingness(rows, violations)
    _check_fire_danger(rows, violations)
    _check_difficulty(rows, violations)
    _check_shortcuts(rows, violations)
    return DatasetValidationReportV4(tuple(violations))


def _check_families(rows, violations: list[str]) -> None:
    labels_by_family: dict[str, set[int]] = defaultdict(set)
    labels_by_archetype: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        labels_by_family[row.scenario_family].add(row.label)
        labels_by_archetype[row.scenario_archetype].add(row.label)
    if len(labels_by_family) < MIN_FAMILIES:
        violations.append(f"only {len(labels_by_family)} scenario families; at least {MIN_FAMILIES} required.")
    for family, labels in labels_by_family.items():
        if len(labels) != 1:
            violations.append(f"scenario_family {family!r} mixes labels.")
    for archetype, labels in labels_by_archetype.items():
        if labels != {0, 1}:
            violations.append(f"scenario_archetype {archetype!r} does not contain both labels.")
    if any(not row.scenario_family or not row.scenario_archetype for row in rows):
        violations.append("group metadata (scenario_family / scenario_archetype) must be populated on every row.")


def _check_value_ranges_and_missingness(rows, violations: list[str]) -> None:
    for row in rows:
        problems = []
        for name in _COUNT_FEATURES:
            value = _f(row, name)
            if value < 0 or value != int(value):
                problems.append(f"{name}={value}")
        for name in _RATIO_FEATURES:
            if not 0.0 <= _f(row, name) <= 1.0:
                problems.append(f"{name}={_f(row, name)}")
        for name in ("satellite_frp_mean", "satellite_frp_max", "satellite_brightness_mean", "satellite_brightness_max", "time_span_minutes", "max_pairwise_distance_km"):
            if _f(row, name) < 0:
                problems.append(f"{name}<0")
        available = row.feature("fire_danger_available")
        score = row.feature("fire_danger_score")
        age = row.feature("fire_danger_age_minutes")
        if available == 1:
            if score is None or age is None:
                problems.append("available Fire Danger without score/age")
            else:
                if not FFWI_MIN_SCORE <= score <= FFWI_MAX_SCORE:
                    problems.append(f"fire_danger_score={score}")
                if not 0.0 <= age <= MAX_FIRE_DANGER_ASSESSMENT_AGE_MINUTES:
                    problems.append(f"fire_danger_age_minutes={age}")
                if row.fire_danger_band != band_for_score(score):
                    problems.append(f"fire_danger_band={row.fire_danger_band!r} does not match score {score}")
        elif available == 0:
            if score is not None or age is not None:
                problems.append("unavailable Fire Danger carries a score/age (missing must not become a value)")
            if row.fire_danger_band != FIRE_DANGER_BAND_MISSING:
                problems.append(f"unavailable Fire Danger has band {row.fire_danger_band!r}")
        else:
            problems.append(f"fire_danger_available={available}")
        if _satellite_total(row) + _news_total(row) < 1:
            problems.append("candidate without evidence")
        if problems:
            violations.append(f"sample {row.sample_id}: invalid values: {', '.join(problems)}")
            return  # one example is enough to fail; avoid flooding the report

    rates = {
        label: sum(1 for row in rows if row.label == label and _f(row, "fire_danger_available") == 0)
        / sum(1 for row in rows if row.label == label)
        for label in (0, 1)
    }
    if abs(rates[0] - rates[1]) > MAX_MISSINGNESS_LABEL_GAP:
        violations.append(f"Fire Danger missingness differs by label ({rates[0]:.3f} vs {rates[1]:.3f}); it would leak the label.")


def _check_fire_danger(rows, violations: list[str]) -> None:
    table = fire_danger_band_crosstab(rows)
    for band in FIRE_DANGER_BANDS:
        cell = table[band]
        if len(rows) >= 1000 and (cell["fire"] == 0 or cell["no_fire"] == 0):
            violations.append(f"Fire Danger band {band!r} does not contain both labels.")
        elif cell["total"] >= MIN_BAND_ROWS_FOR_FRACTION_CHECK:
            minority = min(cell["fire"], cell["no_fire"]) / cell["total"]
            if minority < MIN_BAND_MINORITY_FRACTION:
                violations.append(
                    f"Fire Danger band {band!r} is dominated by one label ({cell['fire']} fire / {cell['no_fire']} no-fire): "
                    "Fire Danger would act as a label proxy."
                )


def _check_difficulty(rows, violations: list[str]) -> None:
    negatives = [row for row in rows if row.label == 0]
    positives = [row for row in rows if row.label == 1]
    hard = sum(1 for row in negatives if is_hard_negative(row)) / len(negatives)
    if hard < MIN_HARD_NEGATIVE_FRACTION:
        violations.append(f"only {hard:.1%} of negatives are hard negatives (need >= {MIN_HARD_NEGATIVE_FRACTION:.0%}).")
    if not any(row.label == 0 and _satellite_total(row) > 0 and _news_total(row) > 0 for row in rows):
        violations.append("no no-fire candidate combines satellite and news evidence.")
    if not any(row.label == 0 and _f(row, "news_strong_count") > 0 for row in rows):
        violations.append("no no-fire candidate contains a STRONG news signal.")
    if not any(row.label == 0 and _f(row, "satellite_high_count") > 0 for row in rows):
        violations.append("no no-fire candidate contains a high-confidence hotspot.")
    weak = sum(1 for row in positives if is_weak_positive(row)) / len(positives)
    if weak < MIN_WEAK_POSITIVE_FRACTION:
        violations.append(f"only {weak:.1%} of fire candidates have weak/incomplete evidence (need >= {MIN_WEAK_POSITIVE_FRACTION:.0%}).")
    if not any(row.label == 1 and _satellite_total(row) + _news_total(row) == 1 for row in rows):
        violations.append("no fire candidate consists of a single evidence item.")
    if not any(row.label == 1 and _news_total(row) > 0 and _satellite_total(row) == 0 for row in rows):
        violations.append("no news-only fire candidate exists.")


def _check_shortcuts(rows, violations: list[str]) -> None:
    for item in univariate_diagnostics(rows):
        if item["suspicious_shortcut"]:
            violations.append(f"feature {item['feature']!r} alone separates the label (AUC {item['univariate_auc']:.3f}).")
        if item["feature"] in ("max_pairwise_distance_km", "time_span_minutes") and item["univariate_auc"] > MAX_GEOMETRY_AUC:
            violations.append(f"geometry feature {item['feature']!r} is a label shortcut (AUC {item['univariate_auc']:.3f}).")
        if item["feature"] == "fire_danger_score" and item["univariate_auc"] > MAX_FIRE_DANGER_AUC:
            violations.append(f"Fire Danger score is a label proxy (AUC {item['univariate_auc']:.3f}).")

    duplicates = duplicate_analysis(rows)
    if duplicates["duplicate_rate"] > MAX_DUPLICATE_RATE:
        violations.append(f"duplicate feature-row rate {duplicates['duplicate_rate']:.1%} exceeds {MAX_DUPLICATE_RATE:.0%}.")

    overall = sum(row.label for row in rows) / len(rows)
    for bucket, entry in hourly_label_rates(rows).items():
        if entry["rows"] >= 200 and abs(entry["fire_share"] - overall) > MAX_HOURLY_LABEL_RATE_DEVIATION:
            violations.append(f"time of day {bucket} has fire share {entry['fire_share']:.3f} vs overall {overall:.3f}: timestamps leak the label.")

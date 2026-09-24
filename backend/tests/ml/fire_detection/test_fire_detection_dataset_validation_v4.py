"""Tests for V4 dataset validation/reporting: it passes the real dataset and catches broken ones."""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path

import pytest

from src.ml.fire_detection.fire_detection_dataset_v4 import load_training_dataset_rows_v4
from src.ml.fire_detection.fire_detection_dataset_validation_v4 import (
    band_for_score,
    build_dataset_report_v4,
    duplicate_analysis,
    fire_danger_band_crosstab,
    is_hard_negative,
    is_weak_positive,
    validate_dataset_rows_v4,
)
from src.ml.fire_detection.fire_detection_features_v4 import FIRE_DANGER_BANDS, FIRE_DETECTION_FEATURE_NAMES_V4

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "fire_detection"


def with_feature(row, name, value):
    features = list(row.features)
    features[FIRE_DETECTION_FEATURE_NAMES_V4.index(name)] = value
    return replace(row, features=tuple(features))


def violations_for(rows, **kwargs):
    return validate_dataset_rows_v4(tuple(rows), **kwargs).violations


def assert_violation(violations, fragment):
    assert any(fragment in violation for violation in violations), (fragment, violations)


# --- the real dataset passes ---


def test_generated_dataset_passes_every_check(v4_rows):
    report = validate_dataset_rows_v4(v4_rows, min_rows=5000)

    assert report.is_valid, report.violations


def test_empty_dataset_is_invalid():
    assert_violation(violations_for([]), "no rows")


def test_too_few_rows_is_reported(v4_rows):
    assert_violation(violations_for(v4_rows[:100], min_rows=5000), "at least 5000")


# --- the validator catches broken datasets ---


def test_single_label_dataset_is_invalid(v4_rows):
    only_fire = [row for row in v4_rows if row.label == 1]

    assert_violation(violations_for(only_fire), "both labels")


def test_imbalanced_labels_are_reported(v4_rows):
    skewed = [row for row in v4_rows if row.label == 1 or row.sample_id % 5 == 0]

    assert_violation(violations_for(skewed), "balanced")


def test_a_family_that_mixes_labels_is_reported(v4_rows):
    rows = list(v4_rows)
    rows[0] = replace(rows[0], label=1 - rows[0].label)

    assert_violation(violations_for(rows), "mixes labels")


def test_archetype_missing_a_label_is_reported(v4_rows):
    rows = [row for row in v4_rows if not (row.scenario_archetype == "satellite_only" and row.label == 0)]

    assert_violation(violations_for(rows), "does not contain both labels")


def test_too_few_families_is_reported(v4_rows):
    kept = {"P1_strong_satellite_strong_news_high_danger", "N1_extreme_danger_no_fire"}
    rows = [row for row in v4_rows if row.scenario_family in kept]

    assert_violation(violations_for(rows), "scenario families")


def test_fire_danger_band_dominated_by_one_label_is_reported(v4_rows):
    # Keep only ~5% of the EXTREME no-fire rows: the band still has both labels but is nearly all fire.
    rows = [
        row for row in v4_rows
        if not (row.fire_danger_band == "extreme" and row.label == 0 and row.sample_id % 20 != 0)
    ]

    assert_violation(violations_for(rows), "'extreme' is dominated by one label")


def test_fire_danger_band_missing_a_label_is_reported(v4_rows):
    rows = [row for row in v4_rows if not (row.fire_danger_band == "low" and row.label == 1)]

    assert_violation(violations_for(rows), "'low' does not contain both labels")


def test_fire_danger_score_acting_as_a_label_proxy_is_reported(v4_rows):
    rows = []
    for row in v4_rows:
        if row.feature("fire_danger_available") == 1:
            score = 60.0 if row.label else 5.0
            row = replace(with_feature(row, "fire_danger_score", score), fire_danger_band=band_for_score(score))
        rows.append(row)

    assert_violation(violations_for(rows), "Fire Danger score is a label proxy")


def test_missing_fire_danger_becoming_a_value_is_reported(v4_rows):
    rows = list(v4_rows)
    index = next(i for i, row in enumerate(rows) if row.feature("fire_danger_available") == 0)
    rows[index] = with_feature(rows[index], "fire_danger_score", 0.0)

    assert_violation(violations_for(rows), "missing must not become a value")


def test_fire_danger_band_that_disagrees_with_the_score_is_reported(v4_rows):
    rows = list(v4_rows)
    index = next(i for i, row in enumerate(rows) if row.fire_danger_band == "low")
    rows[index] = replace(rows[index], fire_danger_band="extreme")

    assert_violation(violations_for(rows), "does not match score")


def test_missingness_that_depends_on_the_label_is_reported(v4_rows):
    rows = []
    for row in v4_rows:
        if row.label == 1:
            row = replace(
                with_feature(with_feature(with_feature(row, "fire_danger_available", 0), "fire_danger_score", None),
                             "fire_danger_age_minutes", None),
                fire_danger_band="missing",
            )
        rows.append(row)

    assert_violation(violations_for(rows), "missingness differs by label")


def test_a_single_feature_that_separates_the_label_is_reported(v4_rows):
    rows = [with_feature(row, "satellite_frp_max", 90.0 if row.label else 5.0) for row in v4_rows]

    assert_violation(violations_for(rows), "'satellite_frp_max' alone separates the label")


def test_geometry_that_predicts_the_label_is_reported(v4_rows):
    rows = [with_feature(row, "max_pairwise_distance_km", 4.0 if row.label else 0.5) for row in v4_rows]

    assert_violation(violations_for(rows), "geometry feature 'max_pairwise_distance_km'")


def test_time_of_day_that_predicts_the_label_is_reported(v4_rows):
    rows = [
        replace(row, as_of_utc=row.as_of_utc.replace(hour=2 if row.label else 14)) for row in v4_rows
    ]

    assert_violation(violations_for(rows), "timestamps leak the label")


def test_dataset_without_hard_negatives_is_reported(v4_rows):
    rows = [row for row in v4_rows if not is_hard_negative(row)]

    assert_violation(violations_for(rows), "hard negatives")


def test_dataset_without_weak_positives_is_reported(v4_rows):
    rows = [row for row in v4_rows if not is_weak_positive(row)]

    assert_violation(violations_for(rows), "weak/incomplete evidence")


def test_heavily_duplicated_rows_are_reported(v4_rows):
    rows = list(v4_rows[:1500]) * 4

    assert_violation(violations_for(rows), "duplicate feature-row rate")


def test_candidate_without_evidence_is_reported(v4_rows):
    empty = v4_rows[0]
    for name in FIRE_DETECTION_FEATURE_NAMES_V4[:14]:
        empty = with_feature(empty, name, 0.0)
    rows = [empty, *v4_rows[1:]]

    assert_violation(violations_for(rows), "candidate without evidence")


def test_out_of_range_values_are_reported(v4_rows):
    rows = list(v4_rows)
    rows[0] = with_feature(rows[0], "satellite_frp_available_ratio", 1.5)

    assert_violation(violations_for(rows), "satellite_frp_available_ratio")


def test_a_stale_fire_danger_age_is_reported(v4_rows):
    rows = list(v4_rows)
    index = next(i for i, row in enumerate(rows) if row.feature("fire_danger_available") == 1)
    rows[index] = with_feature(rows[index], "fire_danger_age_minutes", 500.0)

    assert_violation(violations_for(rows), "fire_danger_age_minutes")


def test_leakage_looking_feature_names_are_reported(v4_rows, monkeypatch):
    import src.ml.fire_detection.fire_detection_dataset_validation_v4 as module

    monkeypatch.setattr(module, "forbidden_feature_names", lambda: ("scenario_family",))

    assert_violation(violations_for(v4_rows), "leakage-looking ML feature names")


# --- report contents ---


def test_band_for_score_uses_the_ffwi_thresholds():
    assert [band_for_score(score) for score in (0.0, 14.99, 15.0, 24.99, 25.0, 39.99, 40.0, 59.99, 60.0, 100.0)] == [
        "low", "low", "moderate", "moderate", "high", "high", "very_high", "very_high", "extreme", "extreme",
    ]


def test_crosstab_covers_every_band_and_all_rows(v4_rows):
    table = fire_danger_band_crosstab(v4_rows)

    assert tuple(table) == FIRE_DANGER_BANDS
    assert sum(cell["total"] for cell in table.values()) == len(v4_rows)
    assert sum(cell["fire"] for cell in table.values()) == sum(row.label for row in v4_rows)


def test_duplicate_analysis_counts_unique_rows(v4_rows):
    result = duplicate_analysis(v4_rows)

    assert result["rows"] == len(v4_rows)
    assert result["unique_feature_rows"] + result["duplicate_rows"] == len(v4_rows)
    assert result["duplicate_rate"] == pytest.approx(result["duplicate_rows"] / len(v4_rows))


def test_report_has_every_required_section_and_is_json_serializable(v4_rows):
    report = build_dataset_report_v4(v4_rows)

    for section in (
        "basic",
        "missingness",
        "fire_danger_band_by_label",
        "univariate_feature_diagnostics",
        "shape_and_difficulty",
        "time_of_day_fire_share",
        "duplicates",
        "grouped_evaluation",
    ):
        assert section in report
    assert report["basic"]["rows"] == len(v4_rows)
    assert report["basic"]["scenario_families"] == 20
    assert {item["feature"] for item in report["univariate_feature_diagnostics"]} == set(FIRE_DETECTION_FEATURE_NAMES_V4)
    json.dumps(report)


def test_report_shows_every_feature_below_the_shortcut_threshold(v4_rows):
    report = build_dataset_report_v4(v4_rows)

    assert not any(item["suspicious_shortcut"] for item in report["univariate_feature_diagnostics"])


def test_committed_validation_report_matches_the_committed_dataset():
    rows = load_training_dataset_rows_v4(DATA_DIR / "training_v4.csv")
    committed = json.loads((DATA_DIR / "training_v4_validation_report.json").read_text(encoding="utf-8"))
    fresh = json.loads(json.dumps(build_dataset_report_v4(rows)))

    assert committed["violations"] == []
    for section, value in fresh.items():
        assert committed[section] == value, section

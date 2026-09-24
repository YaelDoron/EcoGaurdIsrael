"""Task 6: the V5 dataset validator / report - it must pass the real dataset and catch each kind of shortcut."""
from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from src.ml.fire_detection.fire_detection_dataset_v5 import load_training_dataset_rows_v5
from src.ml.fire_detection.fire_detection_dataset_validation_v5 import (
    PAIR_KEY_FEATURES,
    build_dataset_report_v5,
    coverage_slices,
    day_night_distribution,
    duplicate_analysis,
    history_distribution,
    news_distribution,
    pair_analysis,
    per_regime_univariate_auc,
    regime_label_table,
    regime_only_auc,
    univariate_diagnostics,
    validate_dataset_rows_v5,
)
from src.ml.fire_detection.fire_detection_features_v5 import FIRE_DETECTION_FEATURE_NAMES_V5 as NAMES
from src.ml.fire_detection.fire_detection_training_data_generator_v5 import RegimeV5

INDEX = {name: i for i, name in enumerate(NAMES)}


def with_feature(row, name, value):
    features = list(row.features)
    features[INDEX[name]] = value
    return replace(row, features=tuple(features))


def violations_of(rows):
    return validate_dataset_rows_v5(tuple(rows)).violations


def mentions(violations, fragment):
    return [v for v in violations if fragment in v]


# --- the real dataset ---


def test_the_default_dataset_passes_every_check(v5_rows):
    report = validate_dataset_rows_v5(v5_rows, min_rows=5000)
    assert report.is_valid, report.violations


def test_validation_also_works_on_the_loaded_csv_file_form(tmp_path, v5_samples):
    from src.ml.fire_detection.fire_detection_dataset_v5 import write_training_dataset_v5

    path = write_training_dataset_v5(v5_samples, tmp_path / "v5.csv")
    assert validate_dataset_rows_v5(load_training_dataset_rows_v5(path)).is_valid


def test_min_rows_and_empty_input_are_reported():
    assert validate_dataset_rows_v5(()).violations == ("no rows.",)
    assert not validate_dataset_rows_v5((), min_rows=10).is_valid


def test_a_single_label_dataset_is_rejected(v5_rows):
    only_fire = [r for r in v5_rows if r.label == 1]
    assert mentions(violations_of(only_fire), "both labels must be present")


# --- report content ---


def test_the_report_is_json_serializable_and_complete(v5_rows):
    report = build_dataset_report_v5(v5_rows)
    json.dumps(report)  # no NaN objects, sets or enums
    assert set(report) >= {
        "basic", "regimes", "regime_only_auc", "latent_subtypes", "environments", "pairs", "univariate_feature_diagnostics",
        "per_regime_univariate_auc", "history", "day_night", "time_of_day_fire_share", "news", "coverage_slices",
        "duplicates", "grouped_evaluation",
    }
    assert report["basic"]["rows"] == 10000 and report["basic"]["positive_percent"] == 50.0
    assert len(report["univariate_feature_diagnostics"]) == 25


def test_regime_table_and_regime_only_auc_on_the_real_dataset(v5_rows):
    table = regime_label_table(v5_rows)
    assert set(table) == {r.value for r in RegimeV5}
    assert all(entry["fire_share"] == pytest.approx(0.5, abs=0.001) for entry in table.values())
    assert regime_only_auc(v5_rows) == pytest.approx(0.5, abs=0.01)


def test_univariate_auc_is_reported_per_feature_and_stays_below_the_shortcut_threshold(v5_rows):
    diagnostics = {d["feature"]: d for d in univariate_diagnostics(v5_rows)}
    assert set(diagnostics) == set(NAMES)
    assert all(not d["suspicious_shortcut"] and not d["warning"] for d in diagnostics.values())
    assert diagnostics["satellite_pass_count"]["univariate_auc"] < 0.55  # persistence alone is not the label
    assert diagnostics["satellite_night_fraction"]["univariate_auc"] < 0.55
    per_regime = per_regime_univariate_auc(v5_rows)
    assert per_regime[RegimeV5.PERSISTENT_THERMAL.value]["satellite_pass_count"] < 0.60
    assert max(per_regime[RegimeV5.SPARSE_EARLY_EVIDENCE.value].values()) < 0.62


def test_history_distribution_shows_multi_pass_rows_under_both_labels(v5_rows):
    history = history_distribution(v5_rows)
    for count, cell in history["pass_count_by_label"].items():
        if count in {"1", "2", "3"}:
            assert cell["fire"] > 100 and cell["no_fire"] > 100, count
    availability = history["history_feature_availability_by_label"]
    assert availability["satellite_frp_trend_per_hour:available"]["fire"] > 100
    assert availability["satellite_frp_trend_per_hour:available"]["no_fire"] > 100


def test_day_night_and_news_tables_keep_both_labels(v5_rows):
    day_night = day_night_distribution(v5_rows)
    assert abs(day_night["day"]["fire_share"] - day_night["night"]["fire_share"]) < 0.03
    news = news_distribution(v5_rows)
    assert news["no_news"]["fire"] > 1000 and news["no_news"]["no_fire"] > 1000
    assert news["strongest_strong"]["fire"] > 300 and news["strongest_strong"]["no_fire"] > 300


def test_coverage_slices_contain_every_required_situation_under_both_labels(v5_rows):
    slices = coverage_slices(v5_rows)
    assert {"single_satellite_hotspot", "multiple_passes", "high_frp_top_quartile", "low_frp_bottom_quartile", "strong_news",
            "missing_news", "night_observation", "day_observation"} <= set(slices)
    for name, entry in slices.items():
        assert entry["fire"] >= 30 and entry["no_fire"] >= 30, name


def test_duplicates_are_reported_separately_for_satellite_and_news_only_rows(v5_rows):
    duplicates = duplicate_analysis(v5_rows)
    assert duplicates["rows_with_satellite"]["duplicate_rate"] < 0.01
    assert duplicates["news_only_rows"]["duplicate_rate"] > 0.5  # a lone report has almost no feature variety
    assert duplicates["overall"]["duplicate_rate"] < 0.12
    assert 0 <= duplicates["near_duplicates"]["near_duplicate_rate"] <= 1


def test_pair_analysis_shows_overlap_not_separation(v5_rows):
    analysis = pair_analysis(v5_rows)
    assert analysis["pairs"] == 2500 and analysis["same_environment"] == analysis["same_regime"] == 2500
    assert analysis["identical_feature_vector_fraction"] < 0.05
    for name in PAIR_KEY_FEATURES:
        entry = analysis["key_feature_overlap"][name]
        assert 0.3 <= entry["fire_greater_win_rate"] <= 0.75, name  # neither side wins the pair on one feature


# --- the validator catches each kind of shortcut ---


def test_a_regime_missing_a_label_or_skewed_is_a_violation(v5_rows):
    no_fire_in_news_led = [r for r in v5_rows if not (r.regime == RegimeV5.NEWS_LED.value and r.label == 1)]
    found = violations_of(no_fire_in_news_led)
    assert mentions(found, "does not contain both labels")
    assert mentions(found, "regime alone predicts the label")

    thinned = []
    dropped = 0
    for row in v5_rows:
        if row.regime == RegimeV5.SATELLITE_NEWS.value and row.label == 1 and dropped < 300:
            dropped += 1
            continue
        thinned.append(row)
    assert mentions(violations_of(thinned), "is label-skewed")


def test_a_single_feature_that_equals_the_label_is_flagged(v5_rows):
    leaky = [with_feature(r, "satellite_frp_max", 10.0 + 50.0 * r.label) if r.feature("satellite_frp_max") is not None else r for r in v5_rows]
    found = violations_of(leaky)
    assert mentions(found, "'satellite_frp_max' alone separates the label")


def test_night_that_predicts_fire_is_flagged(v5_rows):
    leaky = [
        with_feature(r, "satellite_night_fraction", float(r.label)) if r.feature("satellite_night_fraction") is not None else r
        for r in v5_rows
    ]
    found = violations_of(leaky)
    assert mentions(found, "day/night predicts the label")
    assert mentions(found, "'satellite_night_fraction' alone separates the label")


def test_persistence_alone_solving_the_persistent_regime_is_flagged(v5_rows):
    def shape(row):
        if row.regime != RegimeV5.PERSISTENT_THERMAL.value or row.feature("satellite_frp_trend_per_hour") is not None:
            return row
        return with_feature(row, "satellite_pass_count", 2 + row.label)  # fire: 3 passes, no fire: 2 passes

    found = violations_of([shape(r) for r in v5_rows])
    assert mentions(found, "persistence alone solves persistent_thermal")


def test_a_sparse_regime_that_becomes_easy_is_flagged(v5_rows):
    def shape(row):
        if row.regime != RegimeV5.SPARSE_EARLY_EVIDENCE.value or row.feature("satellite_frp_max") is None:
            return row
        return with_feature(row, "satellite_frp_max", 1.0 + 4.0 * row.label)

    assert mentions(violations_of([shape(r) for r in v5_rows]), "sparse_early_evidence is not ambiguous")


def test_pair_members_in_different_environments_or_regimes_are_flagged(v5_rows):
    rows = list(v5_rows)
    index = next(i for i, r in enumerate(rows) if r.pair_id is not None)
    rows[index] = replace(rows[index], environment_id=rows[index].environment_id + 1000)
    assert mentions(violations_of(rows), "different environments")

    rows = list(v5_rows)
    rows[index] = replace(rows[index], regime=RegimeV5.NEWS_LED.value if rows[index].regime != RegimeV5.NEWS_LED.value else RegimeV5.SATELLITE_NEWS.value)
    assert mentions(violations_of(rows), "different regimes")


def test_a_broken_pair_is_flagged(v5_rows):
    rows = [r for r in v5_rows]
    victim = next(i for i, r in enumerate(rows) if r.pair_id is not None)
    del rows[victim]
    assert mentions(violations_of(rows), "must hold exactly one fire and one no-fire row")


def test_paired_rows_that_are_identical_are_flagged(v5_rows):
    partner = {}
    for row in v5_rows:
        if row.pair_id is not None:
            partner.setdefault(row.pair_id, []).append(row)
    clone = {}
    for pair in partner.values():
        fire = next(r for r in pair if r.label == 1)
        clone[next(r for r in pair if r.label == 0).sample_id] = fire.features
    cloned = [replace(r, features=clone[r.sample_id]) if r.sample_id in clone else r for r in v5_rows]
    assert mentions(violations_of(cloned), "identical feature vectors")


def test_a_slice_that_loses_a_label_is_flagged(v5_rows):
    without_night_fires = [
        r for r in v5_rows if not (r.label == 1 and r.feature("satellite_night_fraction") == 1.0)
    ]
    found = violations_of(without_night_fires)
    assert mentions(found, "night_observation")


def test_heavy_duplication_of_measured_rows_is_flagged(v5_rows):
    duplicated = list(v5_rows) + [r for r in v5_rows if r.feature("satellite_frp_max") not in (None, 0.0)][:2000]
    found = violations_of(duplicated)
    assert mentions(found, "duplicate")


def test_rows_that_break_the_feature_contract_are_flagged(v5_rows):
    rows = list(v5_rows)
    index = next(i for i, r in enumerate(rows) if r.feature("satellite_pass_count") == 1)
    rows[index] = with_feature(rows[index], "satellite_frp_trend_per_hour", 0.0)  # a fake trend from one pass
    assert mentions(violations_of(rows), "violate the V5 contract")


def test_a_latent_subtype_that_contradicts_the_label_is_flagged(v5_rows):
    rows = list(v5_rows)
    rows[0] = replace(rows[0], latent_subtype="early_wildfire" if rows[0].label == 0 else "sensor_noise")
    assert mentions(violations_of(rows), "contradicts label")


def test_a_dataset_with_a_missing_subtype_or_regime_is_flagged(v5_rows):
    without_large = [r for r in v5_rows if r.latent_subtype != "large_wildfire"]
    assert mentions(violations_of(without_large), "large_wildfire")


def test_environment_check_flags_environments_holding_a_single_label(v5_rows):
    one_sided = [replace(r, environment_id=r.environment_id * 1000 + r.label) for r in v5_rows]
    assert mentions(violations_of(one_sided), "environments contain both labels")


def test_the_validator_never_modifies_its_input(v5_rows):
    before = tuple(v5_rows)
    validate_dataset_rows_v5(v5_rows)
    build_dataset_report_v5(v5_rows)
    assert tuple(v5_rows) == before


def test_no_diagnostic_produces_nan_that_would_break_json(v5_rows):
    def scan(value):
        if isinstance(value, float):
            assert not math.isnan(value)
        elif isinstance(value, dict):
            for v in value.values():
                scan(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                scan(v)

    scan(build_dataset_report_v5(v5_rows))

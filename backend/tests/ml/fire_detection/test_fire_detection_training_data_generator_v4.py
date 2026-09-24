"""Tests for FireDetectionTrainingDataGeneratorV4 (labels, determinism, families, ranges, leakage)."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from datetime import timezone
from pathlib import Path
import statistics

import pytest
from sklearn.metrics import roc_auc_score

from src.calculators.fire_danger.ffwi_config import (
    EXTREME_THRESHOLD,
    FFWI_MAX_SCORE,
    FFWI_MIN_SCORE,
    HIGH_THRESHOLD,
    MODERATE_THRESHOLD,
    VERY_HIGH_THRESHOLD,
)
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_detection.fire_detection_config import (
    MAX_EVIDENCE_DISTANCE_KM,
    MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES,
)
from src.ml.fire_detection.fire_detection_dataset_v4 import feature_values_for_sample, row_from_sample
from src.ml.fire_detection.fire_detection_dataset_validation_v4 import is_hard_negative
from src.ml.fire_detection.fire_detection_features_v4 import FIRE_DETECTION_FEATURE_NAMES_V4
from src.ml.fire_detection.fire_detection_training_data_generator_v4 import (
    FAMILIES_V4,
    FireDetectionTrainingDataGeneratorV4,
    FireDetectionTrainingSampleV4,
    ScenarioArchetypeV4,
    ground_truth_label,
)
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_evidence_type import FireEvidenceType
from src.services.fire_detection.fire_detection_context_config import MAX_FIRE_DANGER_ASSESSMENT_AGE_MINUTES
from src.simulation.scenario_type import ScenarioType

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "ml" / "fire_detection"
SMALL = 600


def generate(seed: int = 42, num_samples: int = SMALL):
    return FireDetectionTrainingDataGeneratorV4(seed=seed).generate(num_samples)


def fingerprint(samples):
    return tuple(
        (
            sample.sample_id,
            sample.scenario_family,
            sample.label,
            sample.as_of,
            tuple(
                (
                    item.evidence_id,
                    item.evidence_type,
                    item.latitude,
                    item.longitude,
                    item.observed_at,
                    item.satellite_confidence,
                    item.satellite_frp,
                    item.satellite_brightness,
                    item.news_wildfire_signal_strength,
                )
                for item in sample.evidence
            ),
            sample.context,
        )
        for sample in samples
    )


# --- labels come from ground truth, never from the rule-based detector ---


def test_label_is_derived_from_the_ground_truth_scenario_type(v4_samples):
    for sample in v4_samples:
        expected = 1 if sample.ground_truth_scenario_type is ScenarioType.ACTIVE_FIRE else 0
        assert sample.label == expected == ground_truth_label(sample.ground_truth_scenario_type)


def test_every_family_has_a_single_ground_truth_and_matching_label(v4_samples):
    fire_exists = {family.name: family.fire_exists for family in FAMILIES_V4}
    for sample in v4_samples:
        assert sample.label == (1 if fire_exists[sample.scenario_family] else 0)
        if sample.label == 1:
            assert sample.ground_truth_scenario_type is ScenarioType.ACTIVE_FIRE
        else:
            assert sample.ground_truth_scenario_type is not ScenarioType.ACTIVE_FIRE


def test_sample_rejects_a_label_that_contradicts_ground_truth(v4_samples):
    sample = v4_samples[0]
    with pytest.raises(ValueError):
        FireDetectionTrainingSampleV4(
            sample_id=sample.sample_id,
            seed=sample.seed,
            scenario_family=sample.scenario_family,
            scenario_archetype=sample.scenario_archetype,
            ground_truth_scenario_type=sample.ground_truth_scenario_type,
            label=1 - sample.label,
            evidence=sample.evidence,
            context=sample.context,
            as_of=sample.as_of,
        )


def test_labels_are_not_the_rule_calculators_decisions(v4_samples):
    """Rules disagree with ground truth in both directions, so the label cannot be derived from them."""
    calculator = FireDetectionCalculator()  # used here only to PROVE independence, never by the generator
    rule_says_no_event_but_fire = 0
    rule_confirms_but_no_fire = 0
    for sample in v4_samples:
        status = calculator.evaluate(sample.evidence).status
        if sample.label == 1 and status is FireDetectionStatus.NO_EVENT:
            rule_says_no_event_but_fire += 1
        if sample.label == 0 and status is FireDetectionStatus.CONFIRMED:
            rule_confirms_but_no_fire += 1

    assert rule_says_no_event_but_fire > 0
    assert rule_confirms_but_no_fire > 0


def test_weak_evidence_fire_is_still_labeled_fire_and_misleading_evidence_no_fire(v4_samples):
    single_weak_fire = [
        sample
        for sample in v4_samples
        if sample.scenario_family.startswith("P7")
        and len(sample.evidence) == 1
        and sample.evidence[0].satellite_confidence == "low"
    ]
    strong_looking_no_fire = [
        sample
        for sample in v4_samples
        if sample.scenario_family.startswith("N8")
        and any(item.satellite_confidence == "high" for item in sample.evidence)
        and any(item.evidence_type is FireEvidenceType.NEWS for item in sample.evidence)
    ]

    assert single_weak_fire and all(sample.label == 1 for sample in single_weak_fire)
    assert strong_looking_no_fire and all(sample.label == 0 for sample in strong_looking_no_fire)


def test_generation_does_not_call_the_rule_calculator(monkeypatch):
    def forbidden(self, evidence):  # noqa: ANN001
        raise AssertionError("dataset generation must not use FireDetectionCalculator")

    monkeypatch.setattr(FireDetectionCalculator, "evaluate", forbidden)

    assert len(generate(num_samples=200)) == 200


@pytest.mark.parametrize(
    "module_file",
    [
        "fire_detection_training_data_generator_v4.py",
        "fire_detection_dataset_v4.py",
        "fire_detection_dataset_validation_v4.py",
        "fire_detection_features_v4.py",
    ],
)
def test_v4_modules_do_not_import_rule_policy_or_ml_runtime_modules(module_file):
    tree = ast.parse((SRC_ROOT / module_file).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    forbidden_modules = {
        "src.calculators.fire_detection.fire_detection_calculator",
        "src.calculators.fire_detection.fire_detection_decision_policy",
        "src.calculators.fire_detection.fire_detection_ml_classifier",
        "src.agents.analysis.fire_detection_agent",
        "src.models.hybrid_fire_detection_decision",
        "src.models.fire_detection_decision",
        "src.models.fire_event",
    }

    assert not (imported & forbidden_modules)
    # Only the correlation limits may come from fire_detection_config, never a decision threshold.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.calculators.fire_detection.fire_detection_config":
            assert {alias.name for alias in node.names} <= {
                "MAX_EVIDENCE_DISTANCE_KM",
                "MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES",
            }


# --- determinism and independence ---


def test_same_seed_produces_identical_samples():
    assert fingerprint(generate(seed=42)) == fingerprint(generate(seed=42))


def test_same_seed_produces_identical_feature_rows():
    first = [row_from_sample(sample) for sample in generate(seed=7)]
    second = [row_from_sample(sample) for sample in generate(seed=7)]

    assert first == second


def test_different_seeds_produce_different_samples():
    first, second = generate(seed=42), generate(seed=43)

    assert fingerprint(first) != fingerprint(second)
    assert [row_from_sample(s).features for s in first] != [row_from_sample(s).features for s in second]


def test_seed_is_recorded_on_every_sample():
    assert {sample.seed for sample in generate(seed=11)} == {11}


def test_each_family_is_generated_at_any_dataset_size():
    assert len({sample.scenario_family for sample in generate(seed=5, num_samples=400)}) == len(FAMILIES_V4)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "10"])
def test_invalid_sample_count_is_rejected(bad):
    with pytest.raises(ValueError):
        FireDetectionTrainingDataGeneratorV4().generate(bad)


@pytest.mark.parametrize("bad", [True, 1.5, "42", None])
def test_invalid_seed_is_rejected(bad):
    with pytest.raises(ValueError):
        FireDetectionTrainingDataGeneratorV4(seed=bad)


def test_generated_rows_are_not_near_duplicates(v4_rows):
    unique = len({row.features for row in v4_rows})

    assert unique / len(v4_rows) > 0.95


# --- scenario families ---


def test_dataset_has_many_scenario_families_with_group_metadata(v4_samples):
    families = {sample.scenario_family for sample in v4_samples}
    archetypes = {sample.scenario_archetype for sample in v4_samples}

    assert len(families) == len(FAMILIES_V4) == 20
    assert all(sample.scenario_family for sample in v4_samples)
    assert all(isinstance(sample.scenario_archetype, ScenarioArchetypeV4) for sample in v4_samples)
    assert len(archetypes) == 4


def test_family_names_are_unique_and_cover_the_required_scenarios():
    names = [family.name for family in FAMILIES_V4]
    assert len(names) == len(set(names))
    for prefix in ("P1_", "P2_", "P3_", "P4_", "P5_", "P6_", "P7_", "P8_",
                   "N1_", "N2_", "N3_", "N4_", "N5_", "N6_", "N7_", "N8_"):
        assert any(name.startswith(prefix) for name in names), prefix


def test_labels_are_balanced_and_families_evenly_sized(v4_samples):
    labels = Counter(sample.label for sample in v4_samples)
    per_family = Counter(sample.scenario_family for sample in v4_samples)

    assert labels[0] == labels[1] == len(v4_samples) // 2
    assert max(per_family.values()) - min(per_family.values()) <= 1


def test_every_archetype_contains_both_labels(v4_samples):
    labels_by_archetype = defaultdict(set)
    for sample in v4_samples:
        labels_by_archetype[sample.scenario_archetype].add(sample.label)

    assert all(labels == {0, 1} for labels in labels_by_archetype.values())


def test_sample_ids_are_unique_and_sequential(v4_samples):
    assert [sample.sample_id for sample in v4_samples] == list(range(1, len(v4_samples) + 1))


# --- Fire Danger: breaks the FFWI <-> label shortcut, keeps missingness explicit ---


def _band_label_counts(rows):
    return Counter((row.fire_danger_band, row.label) for row in rows)


@pytest.mark.parametrize("band", ["low", "moderate", "high", "very_high", "extreme", "missing"])
def test_both_labels_exist_in_every_fire_danger_band(v4_rows, band):
    counts = _band_label_counts(v4_rows)

    assert counts[(band, 0)] > 100
    assert counts[(band, 1)] > 100


def test_high_and_extreme_danger_no_fire_examples_exist(v4_rows):
    counts = _band_label_counts(v4_rows)

    assert counts[("high", 0)] > 0
    assert counts[("very_high", 0)] > 0
    assert counts[("extreme", 0)] > 0


def test_low_and_moderate_danger_fire_examples_exist(v4_rows):
    counts = _band_label_counts(v4_rows)

    assert counts[("low", 1)] > 0
    assert counts[("moderate", 1)] > 0


def test_missing_fire_danger_exists_for_both_labels_at_similar_rates(v4_rows):
    def missing_rate(label):
        subset = [row for row in v4_rows if row.label == label]
        return sum(1 for row in subset if row.feature("fire_danger_available") == 0) / len(subset)

    assert missing_rate(0) > 0.05
    assert missing_rate(1) > 0.05
    assert abs(missing_rate(0) - missing_rate(1)) < 0.05


def test_missing_fire_danger_is_never_a_fake_value(v4_samples):
    unavailable = [sample for sample in v4_samples if not sample.context.fire_danger_available]

    assert unavailable
    for sample in unavailable:
        values = feature_values_for_sample(sample)
        assert values["fire_danger_available"] == 0
        assert values["fire_danger_score"] is None
        assert values["fire_danger_age_minutes"] is None


def test_available_fire_danger_is_valid_fresh_and_consistent(v4_samples):
    available = [sample for sample in v4_samples if sample.context.fire_danger_available]

    assert available
    for sample in available:
        context = sample.context
        assert FFWI_MIN_SCORE <= context.fire_danger_score <= FFWI_MAX_SCORE
        assert 0.0 <= context.fire_danger_age_minutes <= MAX_FIRE_DANGER_ASSESSMENT_AGE_MINUTES
        assert context.fire_danger_assessed_at <= sample.as_of
        elapsed = (sample.as_of - context.fire_danger_assessed_at).total_seconds() / 60.0
        assert elapsed == pytest.approx(context.fire_danger_age_minutes, abs=1e-6)


def test_fire_danger_level_matches_the_real_ffwi_thresholds(v4_samples):
    def expected(score):
        if score < MODERATE_THRESHOLD:
            return "low"
        if score < HIGH_THRESHOLD:
            return "moderate"
        if score < VERY_HIGH_THRESHOLD:
            return "high"
        return "very_high" if score < EXTREME_THRESHOLD else "extreme"

    for sample in v4_samples:
        if sample.context.fire_danger_available:
            assert sample.context.fire_danger_level.value == expected(sample.context.fire_danger_score)


def test_fire_danger_alone_does_not_predict_the_label(v4_rows):
    available = [row for row in v4_rows if row.feature("fire_danger_available") == 1]

    auc = roc_auc_score([row.label for row in available], [row.feature("fire_danger_score") for row in available])

    assert 0.40 < auc < 0.65


# --- hard negatives and weak positives ---


def test_hard_negative_families_exist(v4_samples):
    negatives = [sample for sample in v4_samples if sample.label == 0]

    def has(sample, predicate):
        return any(predicate(item) for item in sample.evidence)

    high_hotspot = [s for s in negatives if has(s, lambda i: i.satellite_confidence == "high")]
    strong_news = [
        s
        for s in negatives
        if has(s, lambda i: i.news_wildfire_signal_strength is not None and i.news_wildfire_signal_strength.value == "strong")
    ]
    satellite_and_news = [
        s
        for s in negatives
        if has(s, lambda i: i.evidence_type is FireEvidenceType.SATELLITE)
        and has(s, lambda i: i.evidence_type is FireEvidenceType.NEWS)
    ]
    multi_weak = [s for s in negatives if s.scenario_family.startswith("N6") and len(s.evidence) >= 3]

    assert high_hotspot and strong_news and satellite_and_news and multi_weak
    assert all(len(sample.evidence) >= 1 for sample in negatives)  # never trivially empty


def test_most_negatives_carry_fire_looking_evidence(v4_rows):
    negatives = [row for row in v4_rows if row.label == 0]

    assert sum(1 for row in negatives if is_hard_negative(row)) / len(negatives) > 0.5


def test_positive_candidates_with_weak_or_incomplete_evidence_exist(v4_samples):
    positives = [s for s in v4_samples if s.label == 1]
    single_item = [s for s in positives if len(s.evidence) == 1]
    only_low_satellites = [
        s
        for s in positives
        if all(i.evidence_type is FireEvidenceType.SATELLITE and i.satellite_confidence == "low" for i in s.evidence)
    ]
    unmeasured = [
        s
        for s in positives
        if any(i.satellite_frp is None for i in s.evidence if i.evidence_type is FireEvidenceType.SATELLITE)
    ]
    news_only = [s for s in positives if all(i.evidence_type is FireEvidenceType.NEWS for i in s.evidence)]

    assert single_item and only_low_satellites and unmeasured and news_only


def test_geometry_is_not_a_label_shortcut(v4_rows):
    labels = [row.label for row in v4_rows]

    for name in ("max_pairwise_distance_km", "time_span_minutes"):
        auc = roc_auc_score(labels, [row.feature(name) for row in v4_rows])
        assert 0.35 < auc < 0.65, name


# --- candidate semantics and value ranges ---


def test_every_candidate_is_a_valid_correlated_runtime_candidate(v4_samples):
    for sample in v4_samples:
        assert len(sample.evidence) >= 1
        FireDetectionCandidate(sample.evidence)  # raises if not one connected component


def test_pairwise_extent_stays_inside_the_runtime_correlation_limits(v4_rows):
    for row in v4_rows:
        assert row.feature("max_pairwise_distance_km") <= MAX_EVIDENCE_DISTANCE_KM
        assert row.feature("time_span_minutes") <= MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES


def test_numeric_feature_values_are_in_valid_ranges(v4_rows):
    count_names = [name for name in FIRE_DETECTION_FEATURE_NAMES_V4 if name.endswith("_count")]
    for row in v4_rows:
        for name in count_names:
            value = row.feature(name)
            assert value >= 0 and value == int(value)
        for name in ("satellite_frp_available_ratio", "satellite_brightness_available_ratio", "fire_danger_available"):
            assert 0.0 <= row.feature(name) <= 1.0
        for name in (
            "satellite_frp_mean",
            "satellite_frp_max",
            "satellite_brightness_mean",
            "satellite_brightness_max",
            "time_span_minutes",
            "max_pairwise_distance_km",
        ):
            assert row.feature(name) >= 0.0
        assert row.label in (0, 1)


def test_feature_schema_of_generated_rows_matches_the_declared_names(v4_samples):
    for sample in v4_samples[:200]:
        assert tuple(feature_values_for_sample(sample)) == FIRE_DETECTION_FEATURE_NAMES_V4


# --- timestamp leakage ---


def test_time_of_day_and_season_do_not_correlate_with_the_label(v4_samples):
    overall = sum(s.label for s in v4_samples) / len(v4_samples)
    by_hour_bucket = defaultdict(list)
    for sample in v4_samples:
        by_hour_bucket[sample.as_of.astimezone(timezone.utc).hour // 4].append(sample.label)
    for labels in by_hour_bucket.values():
        assert abs(sum(labels) / len(labels) - overall) < 0.06

    mean_day = {
        label: statistics.mean(s.as_of.timetuple().tm_yday for s in v4_samples if s.label == label) for label in (0, 1)
    }
    assert abs(mean_day[0] - mean_day[1]) < 15


def test_no_timestamp_is_an_ml_feature(v4_samples):
    values = feature_values_for_sample(v4_samples[0])

    assert all(not hasattr(value, "isoformat") for value in values.values())

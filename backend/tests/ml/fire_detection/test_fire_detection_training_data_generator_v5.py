"""Task 6: the V5 synthetic generator - regimes, latent processes, pairs, environments, history, independence."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import replace
import inspect

import pytest

from src.ml.fire_detection import fire_detection_training_data_generator_v5 as generator_module
from src.ml.fire_detection.fire_detection_training_data_generator_v5 import (
    DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V5,
    FIRE_SUBTYPES_V5,
    NO_FIRE_SUBTYPES_V5,
    PAIR_TYPE_BY_REGIME,
    FireDetectionTrainingDataGeneratorV5,
    FireDetectionTrainingSampleV5,
    LatentSubtypeV5,
    RegimeV5,
)
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_evidence_type import FireEvidenceType
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength

SATELLITE = FireEvidenceType.SATELLITE
NEWS = FireEvidenceType.NEWS


def items(sample, kind):
    return [e for e in sample.candidate.evidence if e.evidence_type is kind]


def by_regime(samples):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.regime].append(sample)
    return groups


# --- determinism ---


def test_same_seed_gives_identical_samples_and_a_different_seed_differs():
    a = FireDetectionTrainingDataGeneratorV5(seed=7).generate(300)
    b = FireDetectionTrainingDataGeneratorV5(seed=7).generate(300)
    c = FireDetectionTrainingDataGeneratorV5(seed=8).generate(300)
    assert a == b
    assert [s.candidate.evidence for s in a] != [s.candidate.evidence for s in c]


def test_the_number_of_rows_is_exact_and_ids_are_sequential(v5_samples):
    assert len(v5_samples) == DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V5 == 10000
    assert [s.sample_id for s in v5_samples] == list(range(1, 10001))


def test_invalid_arguments_are_rejected():
    for bad in (True, "42", None, 1.5):
        with pytest.raises(ValueError):
            FireDetectionTrainingDataGeneratorV5(seed=bad)
    for bad in (0, -5, 3, True, None):
        with pytest.raises(ValueError):
            FireDetectionTrainingDataGeneratorV5().generate(bad)


# --- regimes are evidence situations, not labels ---


def test_every_regime_holds_both_labels_and_is_balanced(v5_samples):
    groups = by_regime(v5_samples)
    assert set(groups) == set(RegimeV5)
    for regime, samples in groups.items():
        fire = sum(s.label for s in samples)
        assert 0 < fire < len(samples)
        assert abs(fire - (len(samples) - fire)) <= 1, regime  # ~50/50 inside every regime


@pytest.mark.parametrize("size", [60, 300, 1000])
def test_balance_also_holds_for_small_datasets(size):
    for regime, samples in by_regime(FireDetectionTrainingDataGeneratorV5(seed=3).generate(size)).items():
        fire = sum(s.label for s in samples)
        assert abs(fire - (len(samples) - fire)) <= 1, (size, regime)


def test_overall_labels_are_balanced(v5_samples):
    assert sum(s.label for s in v5_samples) == 5000


def test_a_regime_alone_does_not_reveal_the_label(v5_samples):
    for regime, samples in by_regime(v5_samples).items():
        assert abs(sum(s.label for s in samples) / len(samples) - 0.5) < 0.01, regime


def test_the_defined_regimes_and_pair_types():
    assert {r.value for r in RegimeV5} == {
        "satellite_only_single_pass",
        "multi_pixel_single_pass",
        "news_led",
        "satellite_news",
        "persistent_thermal",
        "sparse_early_evidence",
    }
    assert set(PAIR_TYPE_BY_REGIME) == set(RegimeV5)


def test_satellite_only_single_pass_is_one_hotspot_without_news_or_history(v5_samples):
    for sample in by_regime(v5_samples)[RegimeV5.SATELLITE_ONLY_SINGLE_PASS]:
        assert len(items(sample, SATELLITE)) == 1 and not items(sample, NEWS) and sample.history is None


def test_multi_pixel_single_pass_has_several_pixels_no_news_no_history(v5_samples):
    samples = by_regime(v5_samples)[RegimeV5.MULTI_PIXEL_SINGLE_PASS]
    assert all(len(items(s, SATELLITE)) >= 2 and not items(s, NEWS) and s.history is None for s in samples)
    assert {len(items(s, SATELLITE)) for s in samples} >= {2, 3, 4}


def test_news_led_always_has_news_and_sometimes_no_hotspot_at_all(v5_samples):
    samples = by_regime(v5_samples)[RegimeV5.NEWS_LED]
    assert all(items(s, NEWS) and s.history is None for s in samples)
    news_only = [s for s in samples if not items(s, SATELLITE)]
    with_satellite = [s for s in samples if items(s, SATELLITE)]
    assert news_only and with_satellite
    for sample in with_satellite:  # the report came FIRST
        assert min(e.observed_at for e in items(sample, NEWS)) < min(e.observed_at for e in items(sample, SATELLITE))


def test_satellite_news_has_both_families_with_news_before_and_after(v5_samples):
    samples = by_regime(v5_samples)[RegimeV5.SATELLITE_NEWS]
    assert all(items(s, SATELLITE) and items(s, NEWS) and s.history is None for s in samples)
    lags = [min(e.observed_at for e in items(s, NEWS)) - min(e.observed_at for e in items(s, SATELLITE)) for s in samples]
    assert any(lag.total_seconds() < 0 for lag in lags) and any(lag.total_seconds() > 0 for lag in lags)


def test_persistent_thermal_always_has_an_earlier_pass_and_a_current_hotspot(v5_samples):
    samples = by_regime(v5_samples)[RegimeV5.PERSISTENT_THERMAL]
    for sample in samples:
        assert sample.history is not None and items(sample, SATELLITE)
        current_start = min(c.observed_at for c in items(sample, SATELLITE))
        earlier = [e for e in sample.history.satellite_evidence if e.observed_at < current_start]
        assert earlier, "a persistent row needs at least one earlier detection"
    assert any(s.history.distinct_satellite_pass_count >= 4 for s in samples)


def test_only_persistent_thermal_carries_history(v5_samples):
    assert all((s.history is not None) == (s.regime is RegimeV5.PERSISTENT_THERMAL) for s in v5_samples)


def test_sparse_early_evidence_is_exactly_one_weak_item(v5_samples):
    samples = by_regime(v5_samples)[RegimeV5.SPARSE_EARLY_EVIDENCE]
    assert all(len(s.candidate.evidence) == 1 and s.history is None for s in samples)
    satellite = [s for s in samples if items(s, SATELLITE)]
    news = [s for s in samples if items(s, NEWS)]
    assert satellite and news
    for sample in satellite:
        pixel = items(sample, SATELLITE)[0]
        assert pixel.satellite_confidence in ("low", "nominal")
        assert pixel.satellite_frp is None or pixel.satellite_frp < 7.0
    weak = {NewsWildfireSignalStrength.NONE, NewsWildfireSignalStrength.WEAK, None}
    assert all(items(s, NEWS)[0].news_wildfire_signal_strength in weak for s in news)


def test_sparse_evidence_carries_both_labels_in_both_item_kinds(v5_samples):
    samples = by_regime(v5_samples)[RegimeV5.SPARSE_EARLY_EVIDENCE]
    for kind in (SATELLITE, NEWS):
        assert {s.label for s in samples if items(s, kind)} == {0, 1}, kind


# --- latent processes ---


def test_all_seven_latent_subtypes_appear_and_define_the_label(v5_samples):
    counts = Counter(s.latent_subtype for s in v5_samples)
    assert set(counts) == set(LatentSubtypeV5)
    assert {s.latent_subtype for s in v5_samples if s.label == 1} == set(FIRE_SUBTYPES_V5)
    assert {s.latent_subtype for s in v5_samples if s.label == 0} == set(NO_FIRE_SUBTYPES_V5)
    assert all(s.label == int(s.latent_subtype.is_fire) for s in v5_samples)


def test_label_must_agree_with_the_latent_subtype():
    sample = FireDetectionTrainingDataGeneratorV5(seed=1).generate(60)[0]
    with pytest.raises(ValueError, match="ground truth"):
        replace(sample, label=1 - sample.label)


def test_a_false_report_needs_news_and_noise_needs_a_hotspot(v5_samples):
    for sample in v5_samples:
        if sample.latent_subtype is LatentSubtypeV5.FALSE_OR_RUMOUR_REPORT:
            assert items(sample, NEWS)
        if sample.latent_subtype is LatentSubtypeV5.SENSOR_NOISE:
            assert items(sample, SATELLITE)


def test_every_regime_offers_more_than_one_subtype_on_each_side(v5_samples):
    for regime, samples in by_regime(v5_samples).items():
        fire_subtypes = {s.latent_subtype for s in samples if s.label == 1}
        no_fire_subtypes = {s.latent_subtype for s in samples if s.label == 0}
        assert len(fire_subtypes) >= 2 and len(no_fire_subtypes) >= 2, regime


def test_industrial_heat_persists_and_can_be_observed_at_night_like_a_fire(v5_samples):
    industrial = [s for s in v5_samples if s.latent_subtype is LatentSubtypeV5.INDUSTRIAL_HEAT_SOURCE and s.history is not None]
    assert industrial
    assert any(s.history.distinct_satellite_pass_count >= 3 for s in industrial)
    assert any(e.satellite_day_night == "N" for s in industrial for e in items(s, SATELLITE))


# --- candidate / history validity ---


def test_every_candidate_is_a_valid_runtime_candidate(v5_samples):
    for sample in v5_samples[:2000]:
        FireDetectionCandidate(sample.candidate.evidence)  # re-validated: <= 5 km / 60 min, chained
        assert all(sample.as_of >= e.observed_at for e in sample.candidate.evidence)
        assert 0 <= (sample.as_of - max(e.observed_at for e in sample.candidate.evidence)).total_seconds() <= 600


def test_history_is_a_real_history_inside_its_window_and_may_span_hours(v5_samples):
    spans = []
    for sample in (s for s in v5_samples if s.history is not None):
        history = sample.history
        assert history.as_of == sample.as_of and history.fire_event_id == sample.sample_id
        assert all(history.window_start <= e.observed_at <= history.as_of for e in history.evidence)
        assert history.satellite_pass_gap_minutes == 30.0
        spans.append(history.satellite_observation_span_minutes)
    assert max(spans) > 360  # hours, far beyond the 60-minute candidate rule


def test_some_histories_already_contain_the_current_hotspots_so_dedup_is_exercised(v5_samples):
    overlapping = 0
    for sample in (s for s in v5_samples if s.history is not None):
        current = {(e.evidence_type, e.evidence_id) for e in sample.candidate.evidence}
        if current & {(e.evidence_type, e.evidence_id) for e in sample.history.evidence}:
            overlapping += 1
    assert overlapping > 100


def test_the_generator_uses_one_satellite_platform_like_the_runtime_feed(v5_samples):
    platforms = {
        (e.satellite_name, e.satellite_instrument)
        for s in v5_samples[:500]
        for e in s.candidate.evidence
        if e.evidence_type is SATELLITE
    }
    assert platforms == {("NOAA-20", "VIIRS")}


# --- news uses the real runtime enum semantics ---


def test_news_signals_cover_the_real_enum_including_unavailable(v5_samples):
    seen = Counter(e.news_wildfire_signal_strength for s in v5_samples for e in items(s, NEWS))
    assert set(seen) == {*NewsWildfireSignalStrength, None}
    for label in (0, 1):  # STRONG and unknown exist under BOTH labels
        strengths = Counter(e.news_wildfire_signal_strength for s in v5_samples if s.label == label for e in items(s, NEWS))
        assert strengths[NewsWildfireSignalStrength.STRONG] > 100 and strengths[None] > 20


def test_real_fires_can_have_no_or_weak_news_and_false_reports_can_be_strong(v5_samples):
    fire_news = {e.news_wildfire_signal_strength for s in v5_samples if s.label == 1 for e in items(s, NEWS)}
    false_news = {e.news_wildfire_signal_strength for s in v5_samples if s.label == 0 for e in items(s, NEWS)}
    assert {NewsWildfireSignalStrength.NONE, NewsWildfireSignalStrength.WEAK} <= fire_news
    assert NewsWildfireSignalStrength.STRONG in false_news


# --- day / night and measurements ---


def test_day_night_is_sometimes_unknown_and_split_evenly_between_labels(v5_samples):
    tags = Counter()
    for sample in v5_samples:
        for e in items(sample, SATELLITE):
            tags[(e.satellite_day_night, sample.label)] += 1
    for code in ("D", "N", None):
        assert tags[(code, 0)] > 200 and tags[(code, 1)] > 200, code
    for code in ("D", "N"):
        share = tags[(code, 1)] / (tags[(code, 0)] + tags[(code, 1)])
        assert 0.42 < share < 0.58, code  # night is not evidence of fire


def test_frp_and_brightness_are_sometimes_missing_and_span_high_and_low_under_both_labels(v5_samples):
    for label in (0, 1):
        pixels = [e for s in v5_samples if s.label == label for e in items(s, SATELLITE)]
        assert any(e.satellite_frp is None for e in pixels) and any(e.satellite_brightness is None for e in pixels)
        assert any(e.satellite_frp is not None and e.satellite_frp > 60 for e in pixels)  # high FRP under both labels
        assert any(e.satellite_frp is not None and e.satellite_frp < 2 for e in pixels)  # low FRP under both labels


# --- pairs ---


def _pairs(samples):
    members = defaultdict(list)
    for sample in samples:
        if sample.pair_id is not None:
            members[sample.pair_id].append(sample)
    return members


def test_every_pair_is_one_fire_and_one_no_fire_in_the_same_environment_and_regime(v5_samples):
    members = _pairs(v5_samples)
    assert len(members) == 2500
    for pair in members.values():
        assert len(pair) == 2 and sorted(s.label for s in pair) == [0, 1]
        assert pair[0].environment_id == pair[1].environment_id
        assert pair[0].regime == pair[1].regime
        assert pair[0].pair_type == pair[1].pair_type == PAIR_TYPE_BY_REGIME[pair[0].regime]
        assert pair[0].candidate.evidence != pair[1].candidate.evidence  # they differ through plausible draws


def test_pair_members_share_timing_and_structure_but_not_their_latent_draws(v5_samples):
    members = _pairs(v5_samples)
    different_frp = 0
    for a, b in members.values():
        assert abs((a.as_of - b.as_of).total_seconds()) < 3 * 3600  # rough timing shared
        if a.regime is RegimeV5.PERSISTENT_THERMAL:
            assert a.history is not None and b.history is not None
        frp_a = [e.satellite_frp for e in items(a, SATELLITE) if e.satellite_frp is not None]
        frp_b = [e.satellite_frp for e in items(b, SATELLITE) if e.satellite_frp is not None]
        different_frp += frp_a != frp_b
    assert different_frp > 0.9 * len(members)


def test_pair_members_share_the_day_night_regime(v5_samples):
    agreeing = total = 0
    for pair in _pairs(v5_samples).values():
        a = {e.satellite_day_night for e in items(pair[0], SATELLITE)} - {None}
        b = {e.satellite_day_night for e in items(pair[1], SATELLITE)} - {None}
        if a and b:
            total += 1
            agreeing += a == b
    assert total > 500 and agreeing == total


def test_the_documented_pair_constructions_exist(v5_samples):
    assert len({s.pair_type for s in v5_samples if s.pair_id is not None}) == 6
    persistent = [s for s in v5_samples if s.pair_id is not None and s.regime is RegimeV5.PERSISTENT_THERMAL and s.label == 0]
    assert Counter(s.latent_subtype for s in persistent)[LatentSubtypeV5.INDUSTRIAL_HEAT_SOURCE] > 0.6 * len(persistent)
    news_led = [s for s in v5_samples if s.pair_id is not None and s.regime is RegimeV5.NEWS_LED and s.label == 0]
    assert Counter(s.latent_subtype for s in news_led)[LatentSubtypeV5.FALSE_OR_RUMOUR_REPORT] > 0.5 * len(news_led)
    assert any(len(items(s, NEWS)) >= 2 for s in news_led)  # repeated reports


# --- environments ---


def test_environments_are_many_shared_and_contain_both_labels(v5_samples):
    groups = defaultdict(list)
    for sample in v5_samples:
        groups[sample.environment_id].append(sample)
    assert len(groups) == 250
    assert all(30 <= len(group) <= 55 for group in groups.values())
    assert all({s.label for s in group} == {0, 1} for group in groups.values())
    assert all(len({s.regime for s in group}) >= 4 for group in groups.values())  # an environment is not a regime


def test_environments_differ_in_their_nuisance_parameters(v5_samples):
    missing = defaultdict(list)
    for sample in v5_samples:
        for e in items(sample, SATELLITE):
            missing[sample.environment_id].append(e.satellite_frp is None)
    rates = sorted(sum(v) / len(v) for v in missing.values())
    assert rates[-1] - rates[0] > 0.15  # the per-environment FRP-missing rate is a real nuisance variable


# --- independence from detectors, models and the feature schema ---


def test_the_generator_does_not_import_detectors_models_features_or_fire_danger():
    tree = ast.parse(inspect.getsource(generator_module))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    forbidden = (
        "fire_detection_calculator", "decision_policy", "ml_classifier", "feature_extractor", "features_v", "fire_danger",
        "ffwi", "fire_event", "repositor", "agents", "sklearn", "joblib", "context", "simulation", "requests", "sqlalchemy",
    )
    assert not [m for m in imported if any(f in m for f in forbidden)], imported
    # the ONLY calculators import is the correlation config (5 km / 60 min), used to check candidate validity
    assert [m for m in imported if "calculators" in m] == ["src.calculators.fire_detection.fire_detection_config"]


def test_the_generator_computes_no_ml_feature():
    tree = ast.parse(inspect.getsource(generator_module))
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    identifiers |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    identifiers |= {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for feature in ("frp_sum", "cluster_radius", "centroid_stability", "frp_trend", "night_fraction", "news_satellite_lag", "FeaturesV5", "ExtractorV5"):
        assert not [name for name in identifiers if feature in name], feature


def test_the_sample_is_metadata_plus_evidence_only():
    fields = set(FireDetectionTrainingSampleV5.__dataclass_fields__)
    assert fields == {
        "sample_id", "seed", "environment_id", "regime", "latent_subtype", "pair_id", "pair_type",
        "label", "candidate", "history", "as_of",
    }
    assert not any("danger" in f or "ffwi" in f for f in fields)

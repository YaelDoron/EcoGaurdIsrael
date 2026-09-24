"""Task 6: FireDetectionFeatureExtractorV5 - current-evidence and history features."""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import inspect
import math
import random

import pytest

from src.ml.fire_detection import fire_detection_feature_extractor_v5 as extractor_module
from src.ml.fire_detection.fire_detection_feature_extractor_v3 import FireDetectionFeatureExtractorV3
from src.ml.fire_detection.fire_detection_feature_extractor_v5 import FireDetectionFeatureExtractorV5
from src.ml.fire_detection.fire_detection_features_v5 import FIRE_DETECTION_FEATURE_NAMES_V5, RETAINED_FEATURE_NAMES_V5
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_event_history import FireDetectionEventHistory
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_detection_satellite_pass import group_satellite_passes
from src.models.fire_evidence_type import FireEvidenceType
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength

T0 = datetime(2026, 9, 14, 6, 0, tzinfo=timezone.utc)
LAT, LON = 32.0, 35.0
EXTRACTOR = FireDetectionFeatureExtractorV5()


def hotspot(evidence_id, minutes=0.0, *, lat=LAT, lon=LON, frp=None, brightness=None, confidence="nominal", day_night=None,
            satellite="NOAA-20", instrument="VIIRS"):
    return FireDetectionEvidence(
        evidence_id=evidence_id,
        evidence_type=FireEvidenceType.SATELLITE,
        latitude=lat,
        longitude=lon,
        observed_at=T0 + timedelta(minutes=minutes),
        satellite_confidence=confidence,
        satellite_frp=frp,
        satellite_brightness=brightness,
        satellite_day_night=day_night,
        satellite_name=satellite,
        satellite_instrument=instrument,
    )


def report(evidence_id, minutes=0.0, *, strength=NewsWildfireSignalStrength.MODERATE, lat=LAT, lon=LON):
    return FireDetectionEvidence(
        evidence_id=evidence_id,
        evidence_type=FireEvidenceType.NEWS,
        latitude=lat,
        longitude=lon,
        observed_at=T0 + timedelta(minutes=minutes),
        news_wildfire_signal_strength=strength,
    )


def history_of(evidence, as_of_minutes=None):
    last = max(e.observed_at for e in evidence)
    as_of = last + timedelta(minutes=5) if as_of_minutes is None else T0 + timedelta(minutes=as_of_minutes)
    return FireDetectionEventHistory(
        fire_event_id=1,
        as_of=as_of,
        window_start=as_of - timedelta(hours=24),
        evidence=tuple(evidence),
        satellite_pass_gap_minutes=30.0,
    )


def extract(evidence, history=None):
    return EXTRACTOR.extract(tuple(evidence), history).as_dict()


# --- shape ---


def test_output_follows_the_canonical_order_and_accepts_candidate_or_raw_evidence():
    evidence = (hotspot(1, frp=5.0, brightness=320.0),)
    from_raw = EXTRACTOR.extract(evidence)
    from_candidate = EXTRACTOR.extract(FireDetectionCandidate(evidence))
    assert from_raw.names == FIRE_DETECTION_FEATURE_NAMES_V5
    assert from_raw.as_tuple() == from_candidate.as_tuple()


def test_a_candidate_must_still_be_a_valid_candidate():
    with pytest.raises(ValueError):
        EXTRACTOR.extract((hotspot(1, 0), hotspot(2, 180)))  # 3 h apart: the CURRENT candidate keeps the 60-min rule
    with pytest.raises(ValueError):
        EXTRACTOR.extract(())


def test_history_must_be_a_history_object():
    with pytest.raises(ValueError, match="FireDetectionEventHistory"):
        EXTRACTOR.extract((hotspot(1),), history=(hotspot(2, -180),))


# --- the 14 retained features keep their V3 meaning ---


def test_retained_features_equal_the_unchanged_v3_extractor():
    rng = random.Random(3)
    v3 = FireDetectionFeatureExtractorV3()
    for _ in range(40):
        items = [
            hotspot(
                i + 1, rng.uniform(0, 5), lat=LAT + rng.uniform(-0.01, 0.01), lon=LON + rng.uniform(-0.01, 0.01),
                frp=rng.choice([None, rng.uniform(0, 80)]), brightness=rng.choice([None, rng.uniform(300, 367)]),
                confidence=rng.choice(["low", "nominal", "high"]),
            )
            for i in range(rng.randint(1, 6))
        ]
        items += [report(j + 1, rng.uniform(0, 20), strength=rng.choice([None, *NewsWildfireSignalStrength])) for j in range(rng.randint(0, 3))]
        v5 = EXTRACTOR.extract(tuple(items)).as_dict()
        expected = v3.extract(tuple(items)).as_dict()
        for name in RETAINED_FEATURE_NAMES_V5:
            assert v5[name] == expected[name], name


def test_v5_composes_the_v3_extractor_instead_of_reimplementing_it():
    assert extractor_module.FireDetectionFeatureExtractorV3 is FireDetectionFeatureExtractorV3
    source = inspect.getsource(extractor_module)
    for reimplemented in ("_count_confidence", "_availability_stats", "_news_signal_counts"):
        assert reimplemented not in source


# --- current-evidence features ---


def test_frp_sum_std_and_brightness_std_are_population_statistics():
    features = extract([hotspot(1, frp=2.0, brightness=300.0), hotspot(2, 1, frp=4.0, brightness=310.0), hotspot(3, 2, frp=9.0, brightness=350.0)])
    assert features["satellite_frp_sum"] == pytest.approx(15.0)
    mean = 5.0
    assert features["satellite_frp_std"] == pytest.approx(math.sqrt(((2 - mean) ** 2 + (4 - mean) ** 2 + (9 - mean) ** 2) / 3))  # ddof = 0
    assert features["satellite_brightness_std"] == pytest.approx(math.sqrt(((300 - 320) ** 2 + (310 - 320) ** 2 + (350 - 320) ** 2) / 3))


def test_frp_sum_and_stds_are_zero_when_nothing_can_be_measured():
    none_available = extract([hotspot(1), hotspot(2, 1)])
    assert none_available["satellite_frp_sum"] == 0.0 and none_available["satellite_frp_std"] == 0.0
    assert none_available["satellite_brightness_std"] == 0.0
    assert none_available["satellite_frp_available_ratio"] == 0.0  # the ratio preserves "unmeasured" vs "measured as zero"
    one_value = extract([hotspot(1, frp=7.0, brightness=330.0), hotspot(2, 1)])
    assert one_value["satellite_frp_std"] == 0.0 and one_value["satellite_brightness_std"] == 0.0  # < 2 observations
    assert one_value["satellite_frp_sum"] == 7.0
    news_only = extract([report(1)])
    assert news_only["satellite_frp_sum"] == 0.0


def test_measured_zero_frp_is_distinguishable_from_missing_frp():
    zero = extract([hotspot(1, frp=0.0)])
    missing = extract([hotspot(1)])
    assert zero["satellite_frp_sum"] == missing["satellite_frp_sum"] == 0.0
    assert zero["satellite_frp_available_ratio"] == 1.0 and missing["satellite_frp_available_ratio"] == 0.0


def test_night_fraction_counts_only_known_day_night():
    assert extract([hotspot(1, day_night="N"), hotspot(2, 1, day_night="N")])["satellite_night_fraction"] == 1.0
    assert extract([hotspot(1, day_night="D")])["satellite_night_fraction"] == 0.0
    assert extract([hotspot(1, day_night="N"), hotspot(2, 1, day_night="D"), hotspot(3, 2, day_night="D"), hotspot(4, 3, day_night="D")])["satellite_night_fraction"] == pytest.approx(0.25)
    # unknown hotspots are excluded from the denominator, not counted as day
    assert extract([hotspot(1, day_night="N"), hotspot(2, 1)])["satellite_night_fraction"] == 1.0
    assert extract([hotspot(1, day_night=" n ")])["satellite_night_fraction"] == 1.0  # tolerant of case/whitespace


def test_unknown_day_night_is_nan_never_zero():
    assert math.isnan(extract([hotspot(1)])["satellite_night_fraction"])
    assert math.isnan(extract([hotspot(1, day_night="?")])["satellite_night_fraction"])
    assert math.isnan(extract([report(1)])["satellite_night_fraction"])  # no satellite at all


def test_cluster_radius_is_zero_for_one_hotspot_and_rms_from_the_satellite_centroid():
    assert extract([hotspot(1)])["satellite_cluster_radius_km"] == 0.0
    d = 0.009  # ~1 km of latitude each way
    two = extract([hotspot(1, lat=LAT + d), hotspot(2, 1, lat=LAT - d)])
    assert two["satellite_cluster_radius_km"] == pytest.approx(1.0, abs=0.02)  # both are ~1 km from the centroid
    three = extract([hotspot(1, lat=LAT), hotspot(2, 1, lat=LAT), hotspot(3, 2, lat=LAT + 3 * d)])
    # centroid is d north of the origin: distances d, d, 2d (in km ~1, 1, 2)
    assert three["satellite_cluster_radius_km"] == pytest.approx(math.sqrt((1 + 1 + 4) / 3) * 1.0, abs=0.05)


def test_cluster_radius_ignores_news_geocoding_and_is_nan_without_hotspots():
    hotspots_only = extract([hotspot(1), hotspot(2, 1, lat=LAT + 0.005)])
    with_far_news = extract([hotspot(1), hotspot(2, 1, lat=LAT + 0.005), report(1, 5, lat=LAT + 0.03)])
    assert with_far_news["satellite_cluster_radius_km"] == pytest.approx(hotspots_only["satellite_cluster_radius_km"])
    assert math.isnan(extract([report(1), report(2, 10)])["satellite_cluster_radius_km"])


def test_news_satellite_lag_is_signed_first_news_minus_first_satellite():
    assert extract([hotspot(1, 0), report(1, 25)])["news_satellite_lag_minutes"] == pytest.approx(25.0)  # news after the satellite
    assert extract([hotspot(1, 30), report(1, 0)])["news_satellite_lag_minutes"] == pytest.approx(-30.0)  # news first
    assert extract([hotspot(1, 0), report(1, 0)])["news_satellite_lag_minutes"] == 0.0  # a REAL simultaneous 0
    multi = extract([hotspot(1, 10), hotspot(2, 12), report(1, 40), report(2, 20)])
    assert multi["news_satellite_lag_minutes"] == pytest.approx(10.0)  # earliest news (20) - earliest hotspot (10)


def test_news_satellite_lag_is_nan_when_a_family_is_missing():
    assert math.isnan(extract([hotspot(1)])["news_satellite_lag_minutes"])
    assert math.isnan(extract([report(1)])["news_satellite_lag_minutes"])


def test_lag_uses_utc_instants_so_timezone_offsets_do_not_matter():
    israel = timezone(timedelta(hours=3))
    news = FireDetectionEvidence(
        evidence_id=1, evidence_type=FireEvidenceType.NEWS, latitude=LAT, longitude=LON,
        observed_at=(T0 + timedelta(minutes=10)).astimezone(israel),
        news_wildfire_signal_strength=NewsWildfireSignalStrength.WEAK,
    )
    assert extract([hotspot(1, 0), news])["news_satellite_lag_minutes"] == pytest.approx(10.0)


# --- history features ---


def test_no_history_means_the_candidate_is_the_only_pass():
    features = extract([hotspot(1, frp=6.0), hotspot(2, 2, frp=7.0)])
    assert features["satellite_pass_count"] == 1  # one pass however many pixels
    assert features["satellite_history_span_minutes"] == 0.0
    for name in ("satellite_centroid_stability_km", "satellite_frp_trend_per_hour", "satellite_brightness_trend_per_hour"):
        assert math.isnan(features[name]), name  # NaN, not a fake "perfectly stable" zero


def test_news_only_candidate_without_history_has_no_pass_and_nan_history_features():
    features = extract([report(1)])
    assert features["satellite_pass_count"] == 0
    assert math.isnan(features["satellite_history_span_minutes"])
    assert math.isnan(features["satellite_centroid_stability_km"])


def test_news_only_candidate_inherits_the_events_satellite_history():
    history = history_of([hotspot(1, -360, frp=5.0), hotspot(2, -180, frp=6.0)], as_of_minutes=10)
    features = extract([report(1, 0)], history)
    assert features["satellite_pass_count"] == 2
    assert features["satellite_history_span_minutes"] == pytest.approx(180.0)
    assert math.isnan(features["satellite_cluster_radius_km"])  # nothing CURRENT to measure
    assert features["satellite_low_count"] + features["satellite_nominal_count"] + features["satellite_high_count"] == 0  # current features ignore history


def test_three_overpasses_t0_t3h_t6h_give_three_passes_360_minutes_and_the_deterministic_trend():
    previous = [
        hotspot(1, 0, frp=5.0, brightness=320.0, lat=LAT),
        hotspot(2, 180, frp=10.0, brightness=330.0, lat=LAT),
    ]
    current = [hotspot(3, 360, frp=15.0, brightness=340.0, lat=LAT, day_night="N")]
    features = extract(current, history_of(previous))

    assert features["satellite_pass_count"] == 3
    assert features["satellite_history_span_minutes"] == pytest.approx(360.0)
    assert features["satellite_centroid_stability_km"] == pytest.approx(0.0, abs=1e-9)  # same place every pass
    assert features["satellite_frp_trend_per_hour"] == pytest.approx(5.0 / 3.0)  # 5 -> 10 -> 15 MW over 0/3/6 h
    assert features["satellite_brightness_trend_per_hour"] == pytest.approx(10.0 / 3.0)
    # ...while the CURRENT features still describe only the current pass
    assert features["satellite_frp_sum"] == 15.0 and features["satellite_nominal_count"] == 1
    assert features["satellite_night_fraction"] == 1.0


def test_two_previous_passes_alone_or_one_previous_pass_give_no_trend():
    one_previous = extract([hotspot(2, 180, frp=10.0)], history_of([hotspot(1, 0, frp=5.0)]))
    assert one_previous["satellite_pass_count"] == 2 and one_previous["satellite_history_span_minutes"] == 180.0
    assert not math.isnan(one_previous["satellite_centroid_stability_km"])
    assert math.isnan(one_previous["satellite_frp_trend_per_hour"])  # < 3 passes


def test_centroid_stability_is_the_rms_distance_of_pass_centroids_from_their_mean():
    d = 0.009  # ~1 km latitude
    previous = [hotspot(1, 0, lat=LAT - d), hotspot(2, 180, lat=LAT + d)]
    features = extract([hotspot(3, 360, lat=LAT)], history_of(previous))
    # centroids: -1, +1, 0 km around a mean of 0 -> RMS = sqrt((1 + 1 + 0) / 3)
    assert features["satellite_centroid_stability_km"] == pytest.approx(math.sqrt(2 / 3), abs=0.02)


def test_a_pass_with_many_pixels_counts_once_in_stability():
    d = 0.009
    crowded = [hotspot(1, 0, lat=LAT - d), hotspot(2, 1, lat=LAT - d), hotspot(3, 2, lat=LAT - d), hotspot(4, 3, lat=LAT - d)]
    features = extract([hotspot(5, 180, lat=LAT + d)], history_of(crowded, as_of_minutes=185))
    assert features["satellite_pass_count"] == 2
    assert features["satellite_centroid_stability_km"] == pytest.approx(1.0, abs=0.02)  # 2 centroids, each 1 km from the mean


def test_trend_needs_three_passes_with_a_measurement_and_never_fabricates_missing_frp():
    previous = [hotspot(1, 0, frp=5.0), hotspot(2, 180, frp=None)]  # the middle pass has NO FRP
    features = extract([hotspot(3, 360, frp=15.0)], history_of(previous))
    assert features["satellite_pass_count"] == 3
    assert math.isnan(features["satellite_frp_trend_per_hour"])  # only 2 measured passes: no trend, no zero-fill
    with_third = extract([hotspot(3, 360, frp=15.0)], history_of([hotspot(1, 0, frp=5.0), hotspot(2, 180, frp=None), hotspot(4, 90, frp=8.0)]))
    assert not math.isnan(with_third["satellite_frp_trend_per_hour"])  # 4 passes, 3 measured


def test_pass_level_trend_uses_the_passes_maximum_not_individual_pixel_order():
    previous = [hotspot(1, 0, frp=2.0), hotspot(2, 1, frp=5.0), hotspot(3, 180, frp=10.0), hotspot(4, 181, frp=1.0)]
    features = extract([hotspot(5, 360, frp=15.0), hotspot(6, 361, frp=3.0)], history_of(previous))
    assert features["satellite_frp_trend_per_hour"] == pytest.approx(5.0 / 3.0)  # maxima 5, 10, 15


def test_history_that_already_contains_the_current_candidate_is_deduplicated():
    previous = [hotspot(1, 0, frp=5.0), hotspot(2, 180, frp=10.0)]
    current = [hotspot(3, 360, frp=15.0)]
    without = extract(current, history_of(previous))
    with_current_attached = extract(current, history_of(previous + current))
    assert without == pytest.approx(with_current_attached, nan_ok=True)
    assert with_current_attached["satellite_pass_count"] == 3  # not 4


def test_source_aware_identity_satellite_and_news_ids_may_overlap():
    history = history_of([hotspot(1, 0, frp=5.0), report(1, 10)], as_of_minutes=200)  # satellite 1 and news 1
    features = extract([hotspot(2, 180, frp=6.0)], history)
    assert features["satellite_pass_count"] == 2  # the news item is history but not a satellite pass


def test_history_news_does_not_change_current_news_features():
    history = history_of([hotspot(1, 0, frp=5.0), report(7, 5, strength=NewsWildfireSignalStrength.STRONG)], as_of_minutes=200)
    features = extract([hotspot(2, 180, frp=6.0)], history)
    assert features["news_strong_count"] == 0 and math.isnan(features["news_satellite_lag_minutes"])


def test_different_platforms_are_separate_passes_and_yield_no_trend():
    previous = [hotspot(1, 0, frp=5.0), hotspot(2, 180, frp=9.0, satellite="Suomi-NPP")]
    features = extract([hotspot(3, 360, frp=15.0)], history_of(previous))
    assert features["satellite_pass_count"] == 3
    assert math.isnan(features["satellite_frp_trend_per_hour"])  # FRP of different platforms is not comparable


# --- one canonical pass implementation ---


def test_pass_features_agree_with_the_task_5b_history_object():
    """The extractor and FireDetectionEventHistory must never disagree about the same evidence."""
    items = [hotspot(1, 0, frp=5.0, brightness=310.0, lat=LAT - 0.004), hotspot(2, 2, frp=6.0, brightness=312.0, lat=LAT - 0.004),
             hotspot(3, 185, frp=11.0, brightness=325.0, lat=LAT), hotspot(4, 366, frp=13.0, brightness=331.0, lat=LAT + 0.004)]
    history = history_of(items)
    features = extract([items[-1]], history_of(items[:-1]))
    assert features["satellite_pass_count"] == history.distinct_satellite_pass_count
    assert features["satellite_frp_trend_per_hour"] == pytest.approx(history.frp_trend().slope_per_hour)
    assert features["satellite_brightness_trend_per_hour"] == pytest.approx(history.brightness_trend().slope_per_hour)
    passes = group_satellite_passes(items, 30.0)
    assert [p.pixel_count for p in passes] == [2, 1, 1]


def test_the_extractor_has_no_second_pass_algorithm():
    source = inspect.getsource(extractor_module)
    assert "group_satellite_passes" in source and "satellite_pass_trend" in source
    assert "least_squares" not in source and "timedelta" not in source  # no local trend / gap logic


# --- purity ---


def test_extraction_is_deterministic_and_order_independent():
    items = [hotspot(1, 0, frp=5.0, lat=LAT), hotspot(2, 2, frp=7.0, lat=LAT + 0.004), report(1, 30)]
    previous = [hotspot(3, -200, frp=4.0), hotspot(4, -400, frp=3.0)]
    history = history_of(previous, as_of_minutes=35)
    def same(a, b):
        return len(a) == len(b) and all((x == y) or (math.isnan(x) and math.isnan(y)) for x, y in zip(a, b))

    first = EXTRACTOR.extract(tuple(items), history).as_tuple()
    shuffled = EXTRACTOR.extract(tuple(reversed(items)), history_of(list(reversed(previous)), 35)).as_tuple()
    assert same(first, shuffled)  # input order does not matter
    assert same(first, EXTRACTOR.extract(tuple(items), history).as_tuple())  # repeatable


def test_extraction_does_not_mutate_the_history_or_the_candidate():
    previous = [hotspot(1, 0, frp=5.0), hotspot(2, 180, frp=10.0)]
    history = history_of(previous)
    candidate = FireDetectionCandidate((hotspot(3, 360, frp=15.0),))
    before = (history.evidence, history.satellite_passes, candidate.evidence)
    EXTRACTOR.extract(candidate, history)
    assert (history.evidence, history.satellite_passes, candidate.evidence) == before
    with pytest.raises(FrozenInstanceError):
        history.fire_event_id = 5


def test_the_extractor_is_pure_it_imports_no_repository_service_calculator_ml_or_label_code():
    tree = ast.parse(inspect.getsource(extractor_module))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    forbidden = ("repositor", "calculators", "agents", "fire_danger", "context", "generator", "hybrid", "classifier", "sqlalchemy", "requests")
    assert not [m for m in imported if any(f in m for f in forbidden)], imported
    # the only service import is the pass-gap CONFIG constant, not a service that queries anything
    assert [m for m in imported if ".services." in m] == ["src.services.fire_detection.fire_detection_evidence_config"]
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    identifiers |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    identifiers |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    for word in ("scenario", "regime", "latent", "environment", "ground_truth", "label", "rule_status", "FireEvent", "ml_probability"):
        assert not [name for name in identifiers if word in name], word


def test_extractor_signature_takes_only_evidence_and_history():
    parameters = list(inspect.signature(FireDetectionFeatureExtractorV5.extract).parameters)
    assert parameters == ["self", "candidate", "history"]

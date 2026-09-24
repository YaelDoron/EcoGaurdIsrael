"""Task 5B: FireDetectionEventHistory, satellite-pass grouping and the conservative summaries."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_event_history import (
    FireDetectionEventHistory,
    correlated_evidence,
    satellite_pass_centroid_rms_km,
    satellite_pass_time_span_minutes,
    satellite_pass_trend,
)
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_detection_satellite_pass import SatellitePass, group_satellite_passes
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength

T0 = datetime(2026, 9, 14, 6, 0, tzinfo=timezone.utc)
GAP = 30.0
AS_OF = T0 + timedelta(hours=24)


def hotspot(evidence_id, minutes=0.0, *, lat=32.0, lon=35.0, frp=None, brightness=None, satellite="NOAA-20", instrument="VIIRS"):
    return FireDetectionEvidence(
        evidence_id=evidence_id,
        evidence_type=FireEvidenceType.SATELLITE,
        latitude=lat,
        longitude=lon,
        observed_at=T0 + timedelta(minutes=minutes),
        satellite_confidence="nominal",
        satellite_frp=frp,
        satellite_brightness=brightness,
        satellite_name=satellite,
        satellite_instrument=instrument,
    )


def news(evidence_id, minutes=0.0):
    return FireDetectionEvidence(
        evidence_id=evidence_id,
        evidence_type=FireEvidenceType.NEWS,
        latitude=32.0,
        longitude=35.0,
        observed_at=T0 + timedelta(minutes=minutes),
        news_wildfire_signal_strength=NewsWildfireSignalStrength.STRONG,
    )


def history(evidence, **kwargs):
    kwargs.setdefault("fire_event_id", 1)
    kwargs.setdefault("as_of", AS_OF)
    kwargs.setdefault("window_start", T0 - timedelta(hours=1))
    kwargs.setdefault("satellite_pass_gap_minutes", GAP)
    return FireDetectionEventHistory(evidence=tuple(evidence), **kwargs)


# --- the history is NOT a candidate ---


def test_history_accepts_evidence_hours_apart_which_a_candidate_refuses():
    items = (hotspot(1, 0), hotspot(2, 180), hotspot(3, 360))

    with pytest.raises(ValueError):
        FireDetectionCandidate(items)  # the 60-minute candidate invariant is unchanged
    assert len(history(items).evidence) == 3  # the history has no such rule


def test_history_is_immutable():
    h = history([hotspot(1)])
    with pytest.raises(FrozenInstanceError):
        h.fire_event_id = 2
    assert isinstance(h.evidence, tuple) and isinstance(h.satellite_passes, tuple)


def test_history_orders_evidence_chronologically_and_source_aware():
    h = history([hotspot(2, 120), news(1, 0), hotspot(1, 0)])
    assert [(e.evidence_type, e.evidence_id) for e in h.evidence] == [
        (FireEvidenceType.SATELLITE, 1),
        (FireEvidenceType.NEWS, 1),
        (FireEvidenceType.SATELLITE, 2),
    ]


def test_same_numeric_id_for_satellite_and_news_is_not_a_duplicate():
    assert len(history([hotspot(5, 0), news(5, 0)]).evidence) == 2


def test_duplicate_identity_is_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        history([hotspot(1, 0), hotspot(1, 5)])


def test_evidence_outside_the_window_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        history([hotspot(1, -180)])  # before window_start (T0 - 1h)
    with pytest.raises(ValueError, match="outside"):
        history([hotspot(1, 24 * 60 + 5)])  # after as_of


@pytest.mark.parametrize("bad", [0, -1, True, "1", None])
def test_fire_event_id_must_be_a_positive_int(bad):
    with pytest.raises(ValueError):
        history([hotspot(1)], fire_event_id=bad)


def test_timestamps_must_be_timezone_aware_and_ordered():
    with pytest.raises(ValueError):
        history([hotspot(1)], as_of=datetime(2026, 9, 15))
    with pytest.raises(ValueError):
        history([hotspot(1)], window_start=AS_OF + timedelta(minutes=1))


def test_empty_history_is_valid_and_conservative():
    h = history([])
    assert h.distinct_satellite_pass_count == 0
    assert h.satellite_observation_span_minutes is None
    assert h.satellite_centroid_max_shift_km is None
    assert h.frp_trend() is None


# --- satellite pass grouping ---


def test_pixels_of_one_acquisition_form_one_pass():
    passes = group_satellite_passes([hotspot(1, 0), hotspot(2, 1), hotspot(3, 2, lat=32.01)], GAP)
    assert len(passes) == 1 and passes[0].pixel_count == 3


def test_a_gap_larger_than_the_tolerance_starts_a_new_pass_and_equal_does_not():
    assert len(group_satellite_passes([hotspot(1, 0), hotspot(2, 30)], GAP)) == 1  # gap == tolerance
    assert len(group_satellite_passes([hotspot(1, 0), hotspot(2, 30.5)], GAP)) == 2


def test_grouping_is_single_linkage_in_time_not_fixed_buckets():
    # each hotspot is within 25 min of the previous, so the whole chain is one pass
    passes = group_satellite_passes([hotspot(i, i * 25) for i in range(1, 5)], GAP)
    assert len(passes) == 1


def test_different_platforms_at_the_same_minute_are_separate_passes():
    passes = group_satellite_passes(
        [hotspot(1, 0, satellite="NOAA-20"), hotspot(2, 0, satellite="Suomi-NPP", instrument="VIIRS")], GAP
    )
    assert len(passes) == 2
    assert {p.platform for p in passes} == {("NOAA-20", "VIIRS"), ("Suomi-NPP", "VIIRS")}


def test_platform_names_are_never_hardcoded_and_unknown_platforms_group_together():
    passes = group_satellite_passes(
        [hotspot(1, 0, satellite=None, instrument=None), hotspot(2, 3, satellite=None, instrument=None)], GAP
    )
    assert len(passes) == 1 and passes[0].platform == (None, None)


def test_grouping_is_independent_of_input_order():
    items = [hotspot(1, 0), hotspot(2, 200), hotspot(3, 1), hotspot(4, 201)]
    forward = group_satellite_passes(items, GAP)
    backward = group_satellite_passes(list(reversed(items)), GAP)
    assert [[e.evidence_id for e in p.evidence] for p in forward] == [[e.evidence_id for e in p.evidence] for p in backward]
    assert [[e.evidence_id for e in p.evidence] for p in forward] == [[1, 3], [2, 4]]


def test_news_is_ignored_by_pass_grouping():
    assert group_satellite_passes([news(1, 0)], GAP) == ()


def test_invalid_gap_is_rejected():
    for bad in (0, -5, True, None, "30"):
        with pytest.raises(ValueError):
            group_satellite_passes([hotspot(1)], bad)


def test_pass_statistics_keep_missing_values_missing():
    p = group_satellite_passes([hotspot(1, 0, frp=None), hotspot(2, 1, frp=None)], GAP)[0]
    assert p.frp_statistic("max") is None and p.brightness_statistic("mean") is None

    q = group_satellite_passes([hotspot(1, 0, frp=4.0), hotspot(2, 1, frp=None), hotspot(3, 2, frp=8.0)], GAP)[0]
    assert q.frp_statistic("max") == 8.0
    assert q.frp_statistic("mean") == 6.0  # the missing value is not counted as 0
    assert q.frp_statistic("sum") == 12.0
    with pytest.raises(ValueError):
        q.frp_statistic("median")


def test_satellite_pass_rejects_empty_and_news():
    with pytest.raises(ValueError):
        SatellitePass("NOAA-20", "VIIRS", ())
    with pytest.raises(ValueError):
        SatellitePass("NOAA-20", "VIIRS", (news(1),))


# --- summaries ---


def test_pass_count_counts_waves_not_pixels():
    h = history([hotspot(1, 0), hotspot(2, 1), hotspot(3, 2), hotspot(4, 180), hotspot(5, 181)])
    assert h.distinct_satellite_pass_count == 2
    assert len(h.satellite_evidence) == 5


def test_observation_span_covers_first_to_last_hotspot():
    h = history([hotspot(1, 0), hotspot(2, 200)])
    assert h.satellite_observation_span_minutes == pytest.approx(200.0)


def test_single_pass_has_no_centroid_shift_and_two_passes_do():
    assert history([hotspot(1, 0), hotspot(2, 1)]).satellite_centroid_max_shift_km is None
    h = history([hotspot(1, 0, lat=32.0), hotspot(2, 180, lat=32.009)])  # ~1 km north
    assert h.satellite_centroid_max_shift_km == pytest.approx(1.0, abs=0.05)


def test_frp_trend_needs_three_passes():
    two = history([hotspot(1, 0, frp=5.0), hotspot(2, 180, frp=10.0)])
    assert two.frp_trend() is None

    three = history([hotspot(1, 0, frp=5.0), hotspot(2, 180, frp=10.0), hotspot(3, 360, frp=15.0)])
    trend = three.frp_trend()
    assert trend.pass_count == 3
    assert trend.slope_per_hour == pytest.approx(5.0 / 3.0)


def test_falling_frp_gives_a_negative_slope_and_flat_gives_zero():
    falling = history([hotspot(i + 1, i * 180, frp=30.0 - 10 * i) for i in range(3)])
    assert falling.frp_trend().slope_per_hour < 0
    flat = history([hotspot(i + 1, i * 180, frp=7.0) for i in range(3)])
    assert flat.frp_trend().slope_per_hour == pytest.approx(0.0)


def test_trend_ignores_passes_without_measurements_and_needs_three_measured_passes():
    h = history([hotspot(1, 0, frp=5.0), hotspot(2, 180, frp=None), hotspot(3, 360, frp=15.0)])
    assert h.frp_trend() is None  # only two passes carry an FRP - missing stays missing


def test_trend_is_none_across_platforms_unless_one_is_requested():
    items = [
        hotspot(1, 0, frp=5.0, satellite="NOAA-20"),
        hotspot(2, 180, frp=9.0, satellite="Suomi-NPP"),
        hotspot(3, 360, frp=12.0, satellite="NOAA-20"),
        hotspot(4, 540, frp=20.0, satellite="NOAA-20"),
    ]
    h = history(items)
    assert h.frp_trend() is None
    per_platform = h.frp_trend(platform=("NOAA-20", "VIIRS"))
    assert per_platform is not None and per_platform.pass_count == 3


def test_brightness_trend_mirrors_frp_trend():
    h = history([hotspot(i + 1, i * 180, brightness=300.0 + 10 * i) for i in range(3)])
    assert h.brightness_trend().slope_per_hour == pytest.approx(10.0 / 3.0)


def test_news_evidence_does_not_change_satellite_summaries():
    with_news = history([hotspot(1, 0), hotspot(2, 180), news(1, 90)])
    without = history([hotspot(1, 0), hotspot(2, 180)])
    assert with_news.distinct_satellite_pass_count == without.distinct_satellite_pass_count == 2
    assert with_news.satellite_observation_span_minutes == without.satellite_observation_span_minutes
    assert len(with_news.news_evidence) == 1


def test_event_persistence_is_not_site_recurrence():
    """History is only what is attached to ONE event inside its window; multi-day recurrence is not modelled here."""
    h = history([hotspot(1, 0)])
    assert not any("recurr" in name for name in dir(h))


# --- correlated_evidence (the current-candidate component) ---


def ref(item):
    return FireEvidenceRef(evidence_type=item.evidence_type, evidence_id=item.evidence_id)


def test_correlated_evidence_keeps_only_the_seed_component():
    early, late_a, late_b = hotspot(1, 0), hotspot(2, 180), hotspot(3, 200)
    assert correlated_evidence((early, late_a, late_b), (ref(late_a),)) == (late_a, late_b)


def test_correlated_evidence_follows_chains_like_a_candidate_does():
    a, b, c = hotspot(1, 0), hotspot(2, 50), hotspot(3, 100)  # a-c is 100 min apart but a-b and b-c correlate
    assert correlated_evidence((a, b, c), (ref(a),)) == (a, b, c)
    FireDetectionCandidate((a, b, c))  # same definition: this is a valid candidate


def test_correlated_evidence_excludes_far_away_evidence_at_the_same_time():
    near, far = hotspot(1, 0), hotspot(2, 0, lat=32.5)
    assert correlated_evidence((near, far), (ref(near),)) == (near,)


def test_correlated_evidence_without_a_matching_seed_is_empty():
    assert correlated_evidence((hotspot(1),), (FireEvidenceRef(FireEvidenceType.NEWS, 99),)) == ()


def test_are_correlated_boundaries_are_unchanged():
    base = hotspot(1, 0)
    assert FireDetectionCandidate.are_correlated(base, hotspot(2, 60))
    assert not FireDetectionCandidate.are_correlated(base, hotspot(2, 61))  # candidate window still 60 minutes
    assert FireDetectionCandidate.are_correlated(base, hotspot(3, 0, lat=32.04))  # ~4.4 km
    assert not FireDetectionCandidate.are_correlated(base, hotspot(4, 0, lat=32.05))  # ~5.6 km


# --- shared pure pass helpers (Task 5B primitives reused by the V5 feature extractor) ---


def test_pass_time_span_is_between_pass_times_and_zero_for_one_pass():
    assert satellite_pass_time_span_minutes(()) is None
    one_pass = group_satellite_passes([hotspot(1, 0), hotspot(2, 20)], GAP)  # pixels 20 min apart, still one pass
    assert satellite_pass_time_span_minutes(one_pass) == 0.0
    three = group_satellite_passes([hotspot(1, 0), hotspot(2, 180), hotspot(3, 360)], GAP)
    assert satellite_pass_time_span_minutes(three) == pytest.approx(360.0)


def test_pass_centroid_rms_needs_two_passes_and_counts_each_pass_once():
    assert satellite_pass_centroid_rms_km(()) is None
    assert satellite_pass_centroid_rms_km(group_satellite_passes([hotspot(1, 0)], GAP)) is None  # not 0.0
    still = group_satellite_passes([hotspot(1, 0), hotspot(2, 180)], GAP)
    assert satellite_pass_centroid_rms_km(still) == pytest.approx(0.0, abs=1e-9)
    moved = group_satellite_passes([hotspot(1, 0, lat=32.0), hotspot(2, 180, lat=32.009)], GAP)
    assert satellite_pass_centroid_rms_km(moved) == pytest.approx(0.5, abs=0.03)  # each centroid ~0.5 km from the mean


def test_pass_trend_helper_matches_the_history_properties():
    items = [hotspot(1, 0, frp=5.0), hotspot(2, 180, frp=10.0), hotspot(3, 360, frp=15.0)]
    passes = group_satellite_passes(items, GAP)
    helper = satellite_pass_trend(passes, lambda p: p.frp_statistic("max"))
    assert helper.slope_per_hour == pytest.approx(history(items).frp_trend().slope_per_hour)
    assert satellite_pass_trend(passes[:2], lambda p: p.frp_statistic("max")) is None  # < 3 passes
    assert satellite_pass_trend((), lambda p: 1.0) is None

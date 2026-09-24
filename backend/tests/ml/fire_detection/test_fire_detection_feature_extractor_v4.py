"""Tests for FireDetectionFeatureExtractorV4: V3 parity, Fire Danger context features, isolation."""
from __future__ import annotations

import ast
import inspect
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.ml.fire_detection.fire_detection_feature_extractor_v3 import FireDetectionFeatureExtractorV3
from src.ml.fire_detection.fire_detection_feature_extractor_v4 import FireDetectionFeatureExtractorV4
from src.ml.fire_detection.fire_detection_features_v3 import FIRE_DETECTION_FEATURE_NAMES_V3
from src.ml.fire_detection.fire_detection_features_v4 import (
    FIRE_DETECTION_FEATURE_NAMES_V4,
    FireDetectionFeaturesV4,
)
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_context import FireDetectionContext
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_type import FireEvidenceType
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength

EXTRACTOR_FILE = Path(__file__).resolve().parents[3] / "src" / "ml" / "fire_detection" / "fire_detection_feature_extractor_v4.py"
T0 = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
LAT, LON = 31.7, 35.0  # an arbitrary synthetic point


def sat(evidence_id, confidence="nominal", frp=40.0, brightness=330.0, minutes=0.0, dlat=0.0, dlon=0.0):
    return FireDetectionEvidence(
        evidence_id=evidence_id,
        evidence_type=FireEvidenceType.SATELLITE,
        latitude=LAT + dlat,
        longitude=LON + dlon,
        observed_at=T0 + timedelta(minutes=minutes),
        satellite_confidence=confidence,
        satellite_frp=frp,
        satellite_brightness=brightness,
    )


def news(evidence_id, strength=NewsWildfireSignalStrength.MODERATE, minutes=0.0, dlat=0.0, dlon=0.0):
    return FireDetectionEvidence(
        evidence_id=evidence_id,
        evidence_type=FireEvidenceType.NEWS,
        latitude=LAT + dlat,
        longitude=LON + dlon,
        observed_at=T0 + timedelta(minutes=minutes),
        news_wildfire_signal_strength=strength,
    )


def available_context(score=42.5, age=20.0, level=FireDangerLevel.VERY_HIGH, assessment_id=7, assessed_at=None):
    return FireDetectionContext(
        fire_danger_available=True,
        fire_danger_score=score,
        fire_danger_age_minutes=age,
        fire_danger_level=level,
        fire_danger_assessment_id=assessment_id,
        fire_danger_assessed_at=assessed_at or (T0 - timedelta(minutes=age)),
    )


UNAVAILABLE = FireDetectionContext.unavailable()

REPRESENTATIVE_CANDIDATES = {
    "satellite_only": (sat(1),),
    "news_only": (news(1, NewsWildfireSignalStrength.STRONG),),
    "mixed_satellite_news": (sat(1, "high"), news(2, NewsWildfireSignalStrength.WEAK, minutes=5)),
    "multiple_confidence_levels": (sat(1, "low"), sat(2, "nominal", minutes=3), sat(3, "high", minutes=6)),
    "missing_frp": (sat(1, frp=None), sat(2, frp=25.0, minutes=2)),
    "missing_brightness": (sat(1, brightness=None), sat(2, brightness=310.0, minutes=2)),
    "all_frp_and_brightness_missing": (sat(1, frp=None, brightness=None),),
    "all_news_strengths": (
        news(1, NewsWildfireSignalStrength.NONE),
        news(2, NewsWildfireSignalStrength.WEAK, minutes=1),
        news(3, NewsWildfireSignalStrength.MODERATE, minutes=2),
        news(4, NewsWildfireSignalStrength.STRONG, minutes=3),
        news(5, None, minutes=4),  # unknown: no reliable analysis
    ),
    "multiple_timestamps": (sat(1, minutes=0), sat(2, minutes=17.5), news(3, minutes=41.25)),
    "multiple_geographic_points": (sat(1), sat(2, dlat=0.01, dlon=0.012, minutes=4), news(3, dlat=-0.015, dlon=0.005, minutes=8)),
}


def same_values(first, second):
    """Exact element-wise equality where NaN equals NaN."""
    return len(first) == len(second) and all(
        (a == b) or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
        for a, b in zip(first, second)
    )


# --- shape of the output ---


def test_extractor_returns_exactly_19_ordered_named_features():
    features = FireDetectionFeatureExtractorV4().extract(FireDetectionCandidate((sat(1),)), available_context())

    assert isinstance(features, FireDetectionFeaturesV4)
    assert len(features.as_tuple()) == 19
    assert features.names == FIRE_DETECTION_FEATURE_NAMES_V4
    assert list(features.as_dict()) == list(FIRE_DETECTION_FEATURE_NAMES_V4)


def test_canonical_v4_feature_order_is_frozen():
    assert FIRE_DETECTION_FEATURE_NAMES_V4 == (
        "satellite_low_count",
        "satellite_nominal_count",
        "satellite_high_count",
        "satellite_frp_available_ratio",
        "satellite_frp_mean",
        "satellite_frp_max",
        "satellite_brightness_available_ratio",
        "satellite_brightness_mean",
        "satellite_brightness_max",
        "news_none_count",
        "news_weak_count",
        "news_moderate_count",
        "news_strong_count",
        "news_unknown_count",
        "time_span_minutes",
        "max_pairwise_distance_km",
        "fire_danger_available",
        "fire_danger_score",
        "fire_danger_age_minutes",
    )


def test_named_access_matches_positional_order():
    features = FireDetectionFeatureExtractorV4().extract(
        FireDetectionCandidate((sat(1, "high"), news(2, NewsWildfireSignalStrength.STRONG, minutes=3))),
        available_context(score=31.0, age=9.5),
    )

    assert features["satellite_high_count"] == 1
    assert features["news_strong_count"] == 1
    assert features["fire_danger_score"] == 31.0
    assert features.as_tuple()[FIRE_DETECTION_FEATURE_NAMES_V4.index("fire_danger_age_minutes")] == 9.5


# --- the first 16 features are exactly the V3 features ---


@pytest.mark.parametrize("case", sorted(REPRESENTATIVE_CANDIDATES))
@pytest.mark.parametrize("context", [available_context(), UNAVAILABLE], ids=["danger", "no-danger"])
def test_first_16_features_are_identical_to_v3(case, context):
    evidence = REPRESENTATIVE_CANDIDATES[case]
    v3 = FireDetectionFeatureExtractorV3().extract(evidence)

    v4 = FireDetectionFeatureExtractorV4().extract(FireDetectionCandidate(evidence), context)

    assert v4.as_tuple()[:16] == v3.as_tuple()
    assert tuple(v4.as_dict())[:16] == FIRE_DETECTION_FEATURE_NAMES_V3
    for name in FIRE_DETECTION_FEATURE_NAMES_V3:
        assert v4[name] == getattr(v3, name)


def test_candidate_and_raw_evidence_tuple_give_identical_features():
    evidence = REPRESENTATIVE_CANDIDATES["multiple_geographic_points"]
    extractor = FireDetectionFeatureExtractorV4()

    from_candidate = extractor.extract(FireDetectionCandidate(evidence), available_context())
    from_tuple = extractor.extract(evidence, available_context())

    assert same_values(from_candidate.as_tuple(), from_tuple.as_tuple())


def test_evidence_order_does_not_change_the_features():
    evidence = REPRESENTATIVE_CANDIDATES["all_news_strengths"] + (sat(9, "high", minutes=2),)
    extractor = FireDetectionFeatureExtractorV4()

    forward = extractor.extract(evidence, available_context())
    backward = extractor.extract(tuple(reversed(evidence)), available_context())

    assert same_values(forward.as_tuple(), backward.as_tuple())


def test_the_evidence_features_come_from_the_injected_v3_extractor_not_a_copy():
    class FakeEvidenceExtractor:
        calls = 0

        def extract(self, evidence):
            FakeEvidenceExtractor.calls += 1
            return FireDetectionFeatureExtractorV3().extract(evidence)

    features = FireDetectionFeatureExtractorV4(evidence_extractor=FakeEvidenceExtractor()).extract(
        (sat(1),), available_context()
    )

    assert FakeEvidenceExtractor.calls == 1
    assert features["satellite_nominal_count"] == 1


def test_the_extractor_module_does_not_reimplement_evidence_calculations():
    tree = ast.parse(EXTRACTOR_FILE.read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    source = EXTRACTOR_FILE.read_text(encoding="utf-8")

    assert "math" not in imported
    for calculation in ("haversine", "radians", "satellite_frp", "brightness_mean", "total_seconds"):
        assert calculation not in source.replace("FireDetectionEvidence", "")  # no FRP/geometry/time maths here


# --- Fire Danger context features ---


def test_available_context_features_are_included_correctly():
    features = FireDetectionFeatureExtractorV4().extract((sat(1),), available_context(score=42.5, age=20.0))

    assert features["fire_danger_available"] == 1
    assert features["fire_danger_score"] == 42.5
    assert features["fire_danger_age_minutes"] == 20.0
    assert features.fire_danger_available is True


def test_zero_ffwi_is_preserved_and_distinguishable_from_unavailable():
    zero = FireDetectionFeatureExtractorV4().extract(
        (sat(1),), available_context(score=0.0, age=12.5, level=FireDangerLevel.LOW)
    )
    missing = FireDetectionFeatureExtractorV4().extract((sat(1),), UNAVAILABLE)

    assert zero["fire_danger_available"] == 1
    assert zero["fire_danger_score"] == 0.0
    assert not math.isnan(zero["fire_danger_score"])
    assert missing["fire_danger_available"] == 0
    assert math.isnan(missing["fire_danger_score"])
    assert not same_values(zero.as_tuple(), missing.as_tuple())
    assert zero.to_nullable_tuple()[-2] == 0.0  # storage form keeps the real zero
    assert missing.to_nullable_tuple()[-2] is None  # ... and turns only "missing" into None


def test_unavailable_context_is_nan_never_zero():
    features = FireDetectionFeatureExtractorV4().extract((sat(1),), UNAVAILABLE)

    assert features["fire_danger_available"] == 0
    assert math.isnan(features["fire_danger_score"])
    assert math.isnan(features["fire_danger_age_minutes"])
    assert features["fire_danger_score"] != 0.0
    assert features["fire_danger_age_minutes"] != 0.0


def test_zero_age_is_a_real_value_distinct_from_unavailable():
    fresh = FireDetectionFeatureExtractorV4().extract((sat(1),), available_context(score=30.0, age=0.0))

    assert fresh["fire_danger_age_minutes"] == 0.0
    assert fresh["fire_danger_available"] == 1


def test_fire_danger_age_is_preserved():
    for age in (0.0, 0.5, 33.33, 60.0):
        features = FireDetectionFeatureExtractorV4().extract((sat(1),), available_context(age=age))
        assert features["fire_danger_age_minutes"] == age


def test_fire_danger_level_is_ignored():
    extractor = FireDetectionFeatureExtractorV4()
    evidence = REPRESENTATIVE_CANDIDATES["mixed_satellite_news"]

    base = extractor.extract(evidence, available_context(score=42.5, age=20.0, level=FireDangerLevel.VERY_HIGH))
    other_levels = [
        extractor.extract(evidence, available_context(score=42.5, age=20.0, level=level))
        for level in FireDangerLevel
    ]

    assert all(same_values(base.as_tuple(), other.as_tuple()) for other in other_levels)
    assert "fire_danger_level" not in FIRE_DETECTION_FEATURE_NAMES_V4


def test_assessment_id_and_assessed_at_are_ignored():
    extractor = FireDetectionFeatureExtractorV4()

    first = extractor.extract((sat(1),), available_context(assessment_id=1, assessed_at=T0 - timedelta(minutes=20)))
    second = extractor.extract((sat(1),), available_context(assessment_id=999, assessed_at=T0 - timedelta(days=3)))

    assert same_values(first.as_tuple(), second.as_tuple())


def test_context_never_changes_the_evidence_features():
    extractor = FireDetectionFeatureExtractorV4()
    evidence = REPRESENTATIVE_CANDIDATES["multiple_confidence_levels"]

    with_danger = extractor.extract(evidence, available_context(score=95.0, age=1.0))
    without = extractor.extract(evidence, UNAVAILABLE)

    assert with_danger.as_tuple()[:16] == without.as_tuple()[:16]


def test_high_fire_danger_alone_has_no_evidence_features():
    """Fire Danger is context: it cannot appear as satellite/news evidence in the feature vector."""
    features = FireDetectionFeatureExtractorV4().extract((news(1, NewsWildfireSignalStrength.NONE),), available_context(score=99.0, age=1.0))

    assert features["satellite_low_count"] + features["satellite_nominal_count"] + features["satellite_high_count"] == 0
    assert features["news_none_count"] == 1


# --- determinism ---


def test_extraction_is_deterministic():
    extractor = FireDetectionFeatureExtractorV4()
    evidence = REPRESENTATIVE_CANDIDATES["multiple_timestamps"]

    first = extractor.extract(evidence, available_context())
    second = FireDetectionFeatureExtractorV4().extract(evidence, available_context())

    assert same_values(first.as_tuple(), second.as_tuple())


def test_extraction_does_not_mutate_its_inputs():
    candidate = FireDetectionCandidate(REPRESENTATIVE_CANDIDATES["mixed_satellite_news"])
    context = available_context()
    evidence_before, context_before = candidate.evidence, context

    FireDetectionFeatureExtractorV4().extract(candidate, context)

    assert candidate.evidence == evidence_before
    assert context == context_before


# --- input validation ---


def test_a_non_context_is_rejected():
    with pytest.raises(ValueError):
        FireDetectionFeatureExtractorV4().extract((sat(1),), None)
    with pytest.raises(ValueError):
        FireDetectionFeatureExtractorV4().extract((sat(1),), {"fire_danger_available": True})


def test_disconnected_or_duplicate_evidence_is_rejected_like_runtime():
    far_apart = (sat(1), sat(2, dlat=1.0))  # ~111 km apart: not one candidate
    duplicate = (sat(1), sat(1))

    with pytest.raises(ValueError):
        FireDetectionFeatureExtractorV4().extract(far_apart, UNAVAILABLE)
    with pytest.raises(ValueError):
        FireDetectionFeatureExtractorV4().extract(duplicate, UNAVAILABLE)
    with pytest.raises(ValueError):
        FireDetectionFeatureExtractorV4().extract((), UNAVAILABLE)


# --- isolation: no repositories, no services, no labels / rules ---


def test_extract_takes_only_a_candidate_and_a_context():
    parameters = list(inspect.signature(FireDetectionFeatureExtractorV4.extract).parameters)

    assert parameters == ["self", "candidate", "context"]


def test_the_extractor_imports_no_repository_service_database_agent_rule_or_ground_truth_module():
    tree = ast.parse(EXTRACTOR_FILE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    forbidden_prefixes = (
        "src.repositories",
        "src.services",
        "src.database",
        "src.agents",
        "src.api",
        "src.simulation",
        "src.calculators.fire_detection",  # rule calculator, decision policy, V3 runtime classifier
        "src.models.hybrid_fire_detection_decision",
        "src.models.fire_detection_decision",
        "src.models.fire_event",
        "src.models.fire_detection_ml",
        "src.ml.fire_detection.fire_detection_training_data_generator",
        "src.ml.fire_detection.fire_detection_dataset",
        "sqlalchemy",
        "psycopg",
        "requests",
        "joblib",
        "sklearn",
    )
    offenders = sorted(name for name in imported if name.startswith(forbidden_prefixes))
    assert not offenders, offenders
    # ... and no code (as opposed to documentation) refers to the context service or a repository.
    referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not {name for name in referenced if "Service" in name or "Repository" in name}


def test_the_extractor_works_without_any_database_or_environment(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    features = FireDetectionFeatureExtractorV4().extract((sat(1), news(2)), available_context())

    assert len(features.as_tuple()) == 19

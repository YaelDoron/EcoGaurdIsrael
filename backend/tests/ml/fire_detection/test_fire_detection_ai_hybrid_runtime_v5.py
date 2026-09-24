"""Task 9B: candidate (+ history) -> V5 features -> HGB -> locked AI Hybrid Policy v5.0.

Boundary / guardrail tests use a scripted model runtime (the probability is the input under test); the real-artifact tests only
assert validity - never hard-coded probabilities.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from types import SimpleNamespace

import pytest

from src.config.settings import settings
from src.ml.fire_detection.fire_detection_ai_hybrid_runtime_v5 import FireDetectionAIHybridClassifierV5
from src.ml.fire_detection.fire_detection_feature_extractor_v5 import FireDetectionFeatureExtractorV5
from src.ml.fire_detection.fire_detection_features_v5 import FIRE_DETECTION_FEATURE_NAMES_V5
from src.ml.fire_detection.fire_detection_model_runtime_v5 import (
    FireDetectionModelV5InferenceError,
    FireDetectionModelV5Runtime,
    FireDetectionModelV5UnavailableError,
    clear_model_cache_v5,
)
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_event_history import FireDetectionEventHistory
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_evidence_type import FireEvidenceType as Kind
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength as Strength

T0 = datetime(2026, 9, 14, 6, 0, tzinfo=timezone.utc)
NO_EVENT, SUSPECTED, CONFIRMED = FireDetectionStatus.NO_EVENT, FireDetectionStatus.SUSPECTED, FireDetectionStatus.CONFIRMED


def sat(evidence_id, minutes=0, lat=32.7, lon=35.0, confidence="nominal", frp=10.0, brightness=330.0):
    return FireDetectionEvidence(
        evidence_id, Kind.SATELLITE, lat, lon, T0 + timedelta(minutes=minutes), satellite_confidence=confidence,
        satellite_frp=frp, satellite_brightness=brightness, satellite_day_night="D", satellite_name="NOAA-20",
        satellite_instrument="VIIRS")


def news(evidence_id, strength=Strength.STRONG, minutes=5):
    return FireDetectionEvidence(evidence_id, Kind.NEWS, 32.7, 35.0, T0 + timedelta(minutes=minutes),
                                 news_wildfire_signal_strength=strength)


class ScriptedRuntime:
    """Stands in for FireDetectionModelV5Runtime: returns the scripted probability and records the features it was given."""

    def __init__(self, probability):
        self.probability = probability
        self.features = []
        self.loads = 0

    def ensure_loaded(self):
        self.loads += 1
        return SimpleNamespace(model_name="fire_detection_hgb_v5", model_version="5.0", feature_schema_version="v5",
                               policy_version="ai_hybrid_policy_v5.0")

    def predict_probability(self, features):
        self.features.append(features.as_dict())
        return self.probability


def classify(probability, evidence, history=None):
    runtime = ScriptedRuntime(probability)
    return FireDetectionAIHybridClassifierV5(runtime).assess(FireDetectionCandidate(tuple(evidence)), history), runtime


def history_of(evidence):
    return FireDetectionEventHistory(
        fire_event_id=1, as_of=T0 + timedelta(minutes=10), window_start=T0 - timedelta(hours=30), evidence=tuple(evidence),
        satellite_pass_gap_minutes=30)


ONE_PIXEL = (sat(1),)
TWO_PIXELS = (sat(1), sat(2, lat=32.701))


# --- the locked policy boundaries (Task 8 semantics, exactly) ---------------------------------------------------------------


@pytest.mark.parametrize(
    "probability,evidence,expected",
    [
        (0.399999, TWO_PIXELS, NO_EVENT),
        (0.400000, ONE_PIXEL, SUSPECTED),
        (0.400000, TWO_PIXELS, SUSPECTED),
        (0.799999, TWO_PIXELS, SUSPECTED),  # just under the confirm threshold, multi-pixel
        (0.800000, ONE_PIXEL, SUSPECTED),  # confirm threshold but a single pixel
        (0.800000, TWO_PIXELS, CONFIRMED),
        (1.0, TWO_PIXELS, CONFIRMED),
        (0.0, ONE_PIXEL, NO_EVENT),
    ],
)
def test_threshold_boundaries(probability, evidence, expected):
    assessment, _ = classify(probability, evidence)
    assert assessment.policy_status is expected
    assert assessment.probability == probability


def test_high_probability_with_news_only_is_suspected_not_confirmed():
    """The one confirmation branch needs CURRENT multi-pixel satellite corroboration; news cannot supply it."""
    assessment, _ = classify(0.95, (news(1), news(2, minutes=6)))
    assert assessment.policy_status is SUSPECTED and assessment.current_satellite_pixel_count == 0


def test_satellite_plus_news_with_a_single_pixel_is_still_only_suspected():
    assessment, _ = classify(0.95, (sat(1), news(2)))
    assert assessment.policy_status is SUSPECTED and assessment.current_satellite_pixel_count == 1


def test_historical_pixels_never_satisfy_the_current_multi_pixel_guardrail():
    """Three old pixels in the event history + ONE current pixel: the guardrail counts the current candidate only."""
    history_passes = tuple(sat(10 + i, minutes=-180 * (i + 1), lat=32.7 + 0.0001 * i) for i in range(3))
    history = history_of(history_passes)

    assessment, runtime = classify(0.95, ONE_PIXEL, history)

    assert assessment.current_satellite_pixel_count == 1
    assert assessment.satellite_pass_count == 4  # the history IS used by the features...
    assert assessment.policy_status is SUSPECTED  # ...but cannot corroborate the current candidate


def test_the_classifier_reports_model_policy_and_history_context_but_not_the_feature_vector():
    assessment, _ = classify(0.55, ONE_PIXEL)
    assert (assessment.model_name, assessment.model_version, assessment.feature_schema_version, assessment.policy_version) == (
        "fire_detection_hgb_v5", "5.0", "v5", "ai_hybrid_policy_v5.0")
    assert assessment.history_available is False and assessment.satellite_pass_count == 1
    assert not hasattr(assessment, "features")


# --- new candidate (no history) and existing event (history) ---------------------------------------------------------------


def test_a_brand_new_candidate_produces_a_valid_first_pass_vector():
    _, runtime = classify(0.5, ONE_PIXEL)
    (features,) = runtime.features
    assert list(features) == list(FIRE_DETECTION_FEATURE_NAMES_V5) and len(features) == 25
    assert features["satellite_pass_count"] == 1 and features["satellite_history_span_minutes"] == 0.0
    for name in ("satellite_frp_trend_per_hour", "satellite_brightness_trend_per_hour", "satellite_centroid_stability_km"):
        assert math.isnan(features[name])  # not computable from one pass: NaN exactly as defined by Task 6


def test_news_only_candidate_has_zero_pass_count_and_nan_satellite_features():
    assessment, runtime = classify(0.5, (news(1),))
    (features,) = runtime.features
    assert features["satellite_pass_count"] == 0 and math.isnan(features["satellite_cluster_radius_km"])
    assert assessment.satellite_pass_count == 0 and assessment.current_satellite_pixel_count == 0


def test_history_is_combined_with_the_current_candidate_through_the_extractor_not_one_giant_candidate():
    earlier = tuple(sat(10 + i, minutes=-180 * (i + 1)) for i in range(2))  # T-3h, T-6h
    history = history_of(earlier)

    assessment, runtime = classify(0.6, ONE_PIXEL, history)

    (features,) = runtime.features
    assert assessment.history_available is True
    assert features["satellite_pass_count"] == 3 and features["satellite_history_span_minutes"] == pytest.approx(360.0)
    assert not math.isnan(features["satellite_frp_trend_per_hour"])
    # the candidate itself is still ONE 60-minute candidate (a giant 6 h candidate would be invalid)
    with pytest.raises(ValueError):
        FireDetectionCandidate(ONE_PIXEL + earlier)


def test_evidence_already_attached_to_the_event_is_not_counted_twice():
    history = history_of(ONE_PIXEL)
    assessment, runtime = classify(0.5, ONE_PIXEL, history)  # the same hotspot is in both
    assert runtime.features[0]["satellite_pass_count"] == 1 and assessment.satellite_pass_count == 1


def test_the_extractor_is_the_shared_task_6_implementation():
    classifier = FireDetectionAIHybridClassifierV5(ScriptedRuntime(0.5))
    assert isinstance(classifier._extractor, FireDetectionFeatureExtractorV5)


# --- failures never become a decision ----------------------------------------------------------------------------------------


def test_an_unavailable_artifact_raises_before_any_feature_work(tmp_path):
    classifier = FireDetectionAIHybridClassifierV5(FireDetectionModelV5Runtime(tmp_path / "a.joblib", tmp_path / "a.json"))
    with pytest.raises(FireDetectionModelV5UnavailableError):
        classifier.assess(FireDetectionCandidate(ONE_PIXEL))


def test_an_extraction_failure_is_an_explicit_inference_error(monkeypatch):
    classifier = FireDetectionAIHybridClassifierV5(ScriptedRuntime(0.5))
    monkeypatch.setattr(classifier._extractor, "extract", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad")))
    with pytest.raises(FireDetectionModelV5InferenceError, match="feature extraction failed"):
        classifier.assess(FireDetectionCandidate(ONE_PIXEL))


def test_the_classifier_is_stateless_between_predictions():
    classifier = FireDetectionAIHybridClassifierV5(ScriptedRuntime(0.9))
    first = classifier.assess(FireDetectionCandidate(TWO_PIXELS))
    classifier.assess(FireDetectionCandidate(ONE_PIXEL))
    again = classifier.assess(FireDetectionCandidate(TWO_PIXELS))
    assert first == again


# --- the REAL approved artifact: validity only ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_classifier():
    clear_model_cache_v5()
    yield FireDetectionAIHybridClassifierV5(
        FireDetectionModelV5Runtime(settings.FIRE_DETECTION_AI_V5_MODEL_PATH, settings.FIRE_DETECTION_AI_V5_METADATA_PATH))
    clear_model_cache_v5()


REPRESENTATIVE = {
    "news_only_strong": (news(1),),
    "news_only_weak": (news(1, Strength.WEAK),),
    "single_satellite_pixel": (sat(1),),
    "multi_pixel_satellite": (sat(1, frp=30), sat(2, lat=32.702, frp=40), sat(3, lat=32.703, frp=35)),
    "satellite_plus_news": (sat(1), news(2)),
    "sparse_ambiguous": (sat(1, confidence="low", frp=1.5, brightness=305.0),),
}


@pytest.mark.parametrize("name", sorted(REPRESENTATIVE))
def test_real_artifact_scores_every_representative_candidate_validly(real_classifier, name):
    assessment = real_classifier.assess(FireDetectionCandidate(REPRESENTATIVE[name]))

    assert math.isfinite(assessment.probability) and 0.0 <= assessment.probability <= 1.0
    assert assessment.policy_status in (NO_EVENT, SUSPECTED, CONFIRMED)
    assert assessment.model_version == "5.0" and assessment.history_available is False


def test_real_artifact_scores_a_candidate_with_history(real_classifier):
    earlier = tuple(sat(10 + i, minutes=-180 * (i + 1)) for i in range(2))
    history = history_of(earlier)

    assessment = real_classifier.assess(FireDetectionCandidate(TWO_PIXELS), history)

    assert 0.0 <= assessment.probability <= 1.0 and assessment.satellite_pass_count == 3 and assessment.history_available

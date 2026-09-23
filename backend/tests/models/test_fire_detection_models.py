"""Tests for active wildfire detection domain models."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import math

import pytest

from src.models import (
    FireDetectionCandidate,
    FireDetectionDecision,
    FireDetectionEvidence,
    FireDetectionStatus,
    FireEvidenceRef,
    FireEvidenceType,
)
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength

OBSERVED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def make_evidence(**overrides) -> FireDetectionEvidence:
    defaults = dict(
        evidence_id=1,
        evidence_type=FireEvidenceType.SATELLITE,
        latitude=32.731,
        longitude=35.046,
        observed_at=OBSERVED_AT,
        satellite_confidence="nominal",
    )
    defaults.update(overrides)
    return FireDetectionEvidence(**defaults)


def test_fire_evidence_type_values():
    assert [evidence_type.value for evidence_type in FireEvidenceType] == ["satellite", "news"]


def test_fire_detection_status_values():
    assert [status.value for status in FireDetectionStatus] == ["no_event", "suspected", "confirmed"]


def ref(evidence_type=FireEvidenceType.SATELLITE, evidence_id=1) -> FireEvidenceRef:
    return FireEvidenceRef(evidence_type=evidence_type, evidence_id=evidence_id)


def test_valid_fire_evidence_ref_construction():
    evidence_ref = ref(FireEvidenceType.NEWS, 5)

    assert evidence_ref.evidence_type is FireEvidenceType.NEWS
    assert evidence_ref.evidence_id == 5


def test_fire_evidence_ref_rejects_invalid_type():
    with pytest.raises(ValueError):
        FireEvidenceRef(evidence_type="satellite", evidence_id=5)


@pytest.mark.parametrize("evidence_id", [0, -1, None, "5", True])
def test_fire_evidence_ref_rejects_invalid_id(evidence_id):
    with pytest.raises(ValueError):
        FireEvidenceRef(evidence_type=FireEvidenceType.SATELLITE, evidence_id=evidence_id)


def test_fire_evidence_ref_is_immutable():
    evidence_ref = ref()

    with pytest.raises(FrozenInstanceError):
        evidence_ref.evidence_id = 2


def test_valid_satellite_evidence_construction():
    evidence = make_evidence()

    assert evidence.evidence_id == 1
    assert evidence.evidence_type is FireEvidenceType.SATELLITE
    assert evidence.satellite_confidence == "nominal"


def test_valid_news_evidence_construction():
    evidence = make_evidence(
        evidence_type=FireEvidenceType.NEWS,
        satellite_confidence=None,
    )

    assert evidence.evidence_type is FireEvidenceType.NEWS
    assert evidence.satellite_confidence is None


@pytest.mark.parametrize("latitude", [-90.1, 90.1, math.nan, math.inf, -math.inf])
def test_invalid_evidence_latitude_rejected(latitude):
    with pytest.raises(ValueError):
        make_evidence(latitude=latitude)


@pytest.mark.parametrize("longitude", [-180.1, 180.1, math.nan, math.inf, -math.inf])
def test_invalid_evidence_longitude_rejected(longitude):
    with pytest.raises(ValueError):
        make_evidence(longitude=longitude)


def test_naive_observed_at_rejected():
    with pytest.raises(ValueError):
        make_evidence(observed_at=datetime(2026, 9, 14, 12, 0))


def test_invalid_satellite_confidence_rejected():
    with pytest.raises(ValueError):
        make_evidence(satellite_confidence="medium")


def test_news_with_satellite_confidence_rejected():
    with pytest.raises(ValueError):
        make_evidence(evidence_type=FireEvidenceType.NEWS, satellite_confidence="high")


def test_satellite_without_satellite_confidence_rejected():
    with pytest.raises(ValueError):
        make_evidence(satellite_confidence=None)


def test_evidence_is_immutable():
    evidence = make_evidence()

    with pytest.raises(FrozenInstanceError):
        evidence.latitude = 33.0


# --- V3 evidence enrichment (Task 4) ---


def test_satellite_evidence_carries_frp_and_brightness():
    evidence = make_evidence(satellite_frp=42.5, satellite_brightness=310.0, satellite_day_night="D")

    assert evidence.satellite_frp == 42.5
    assert evidence.satellite_brightness == 310.0
    assert evidence.satellite_day_night == "D"


def test_satellite_evidence_physical_measurements_default_to_none():
    evidence = make_evidence()

    assert evidence.satellite_frp is None
    assert evidence.satellite_brightness is None
    assert evidence.satellite_day_night is None


@pytest.mark.parametrize("frp", [-1.0, math.nan, math.inf])
def test_invalid_satellite_frp_rejected(frp):
    with pytest.raises(ValueError):
        make_evidence(satellite_frp=frp)


@pytest.mark.parametrize("brightness", [-1.0, math.nan, math.inf])
def test_invalid_satellite_brightness_rejected(brightness):
    with pytest.raises(ValueError):
        make_evidence(satellite_brightness=brightness)


def test_satellite_evidence_rejects_news_signal_field():
    with pytest.raises(ValueError):
        make_evidence(news_wildfire_signal_strength=NewsWildfireSignalStrength.STRONG)


def test_satellite_evidence_rejects_string_day_night_type_violation():
    with pytest.raises(ValueError):
        make_evidence(satellite_day_night=123)


def test_news_evidence_carries_wildfire_signal_strength():
    evidence = make_evidence(
        evidence_type=FireEvidenceType.NEWS,
        satellite_confidence=None,
        news_wildfire_signal_strength=NewsWildfireSignalStrength.MODERATE,
    )

    assert evidence.news_wildfire_signal_strength is NewsWildfireSignalStrength.MODERATE


def test_news_evidence_signal_defaults_to_none_meaning_unknown():
    evidence = make_evidence(evidence_type=FireEvidenceType.NEWS, satellite_confidence=None)

    assert evidence.news_wildfire_signal_strength is None


def test_news_evidence_rejects_satellite_frp():
    with pytest.raises(ValueError):
        make_evidence(evidence_type=FireEvidenceType.NEWS, satellite_confidence=None, satellite_frp=10.0)


def test_news_evidence_rejects_satellite_brightness():
    with pytest.raises(ValueError):
        make_evidence(evidence_type=FireEvidenceType.NEWS, satellite_confidence=None, satellite_brightness=300.0)


def test_news_evidence_rejects_satellite_day_night():
    with pytest.raises(ValueError):
        make_evidence(evidence_type=FireEvidenceType.NEWS, satellite_confidence=None, satellite_day_night="D")


def test_valid_candidate_construction_allows_overlapping_numeric_ids_across_sources():
    satellite = make_evidence(evidence_id=5)
    news = make_evidence(
        evidence_id=5,
        evidence_type=FireEvidenceType.NEWS,
        satellite_confidence=None,
    )

    candidate = FireDetectionCandidate((news, satellite))

    assert candidate.evidence == (satellite, news)


def test_candidate_duplicate_source_identity_rejected():
    first = make_evidence(evidence_id=5)
    second = make_evidence(evidence_id=5)

    with pytest.raises(ValueError):
        FireDetectionCandidate((first, second))


def test_candidate_disconnected_evidence_rejected():
    first = make_evidence(evidence_id=1)
    second = make_evidence(evidence_id=2, latitude=33.5, longitude=35.5)

    with pytest.raises(ValueError):
        FireDetectionCandidate((first, second))


def test_valid_confirmed_decision_construction_sorts_supporting_evidence_deterministically():
    decision = FireDetectionDecision(
        confidence=0.8,
        status=FireDetectionStatus.CONFIRMED,
        latitude=32.731,
        longitude=35.046,
        supporting_evidence=(ref(FireEvidenceType.NEWS, 2), ref(FireEvidenceType.SATELLITE, 1)),
    )

    assert decision.confidence == 0.8
    assert decision.status is FireDetectionStatus.CONFIRMED
    assert decision.supporting_evidence == (ref(FireEvidenceType.NEWS, 2), ref(FireEvidenceType.SATELLITE, 1))


def test_decision_preserves_overlapping_numeric_ids_across_sources():
    decision = FireDetectionDecision(
        confidence=0.8,
        status=FireDetectionStatus.CONFIRMED,
        latitude=32.731,
        longitude=35.046,
        supporting_evidence=(ref(FireEvidenceType.SATELLITE, 5), ref(FireEvidenceType.NEWS, 5)),
    )

    assert decision.supporting_evidence == (
        ref(FireEvidenceType.NEWS, 5),
        ref(FireEvidenceType.SATELLITE, 5),
    )


def test_no_event_decision_allows_empty_support_and_no_location():
    decision = FireDetectionDecision(
        confidence=0.4,
        status=FireDetectionStatus.NO_EVENT,
        latitude=None,
        longitude=None,
        supporting_evidence=(),
    )

    assert decision.status is FireDetectionStatus.NO_EVENT


def test_suspected_decision_requires_location():
    with pytest.raises(ValueError):
        FireDetectionDecision(
            confidence=0.5,
            status=FireDetectionStatus.SUSPECTED,
            latitude=None,
            longitude=35.046,
            supporting_evidence=(ref(),),
        )


def test_suspected_decision_requires_supporting_ids():
    with pytest.raises(ValueError):
        FireDetectionDecision(
            confidence=0.5,
            status=FireDetectionStatus.SUSPECTED,
            latitude=32.731,
            longitude=35.046,
            supporting_evidence=(),
        )


def test_decision_duplicate_supporting_evidence_rejected():
    with pytest.raises(ValueError):
        FireDetectionDecision(
            confidence=0.8,
            status=FireDetectionStatus.CONFIRMED,
            latitude=32.731,
            longitude=35.046,
            supporting_evidence=(ref(), ref()),
        )


def test_decision_same_numeric_id_from_different_sources_is_not_duplicate():
    decision = FireDetectionDecision(
        confidence=0.8,
        status=FireDetectionStatus.CONFIRMED,
        latitude=32.731,
        longitude=35.046,
        supporting_evidence=(ref(FireEvidenceType.SATELLITE, 5), ref(FireEvidenceType.NEWS, 5)),
    )

    assert len(decision.supporting_evidence) == 2


def test_decision_rejects_non_ref_supporting_evidence():
    with pytest.raises(ValueError):
        FireDetectionDecision(
            confidence=0.8,
            status=FireDetectionStatus.CONFIRMED,
            latitude=32.731,
            longitude=35.046,
            supporting_evidence=(1,),
        )


@pytest.mark.parametrize("confidence", [-0.1, 1.1, math.nan, math.inf, -math.inf])
def test_invalid_decision_confidence_rejected(confidence):
    with pytest.raises(ValueError):
        FireDetectionDecision(
            confidence=confidence,
            status=FireDetectionStatus.NO_EVENT,
            latitude=None,
            longitude=None,
            supporting_evidence=(),
        )


def test_decision_is_immutable():
    decision = FireDetectionDecision(
        confidence=0.4,
        status=FireDetectionStatus.NO_EVENT,
        latitude=None,
        longitude=None,
        supporting_evidence=(),
    )

    with pytest.raises(FrozenInstanceError):
        decision.confidence = 0.5

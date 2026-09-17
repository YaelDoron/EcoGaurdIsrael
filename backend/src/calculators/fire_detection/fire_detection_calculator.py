"""Pure active-wildfire evidence correlation and confidence calculation.

FireDetectionCalculator evaluates one already-prepared candidate group of
direct evidence and returns a deterministic FireDetectionDecision. Satellite
and news are direct evidence sources. Fire danger/FFWI is intentionally not an
input: high fire danger alone must not create an active wildfire detection.

This first version does not cluster a broad evidence stream into multiple
incidents. If supplied evidence does not form one connected component by
location and time (the same definition FireDetectionCandidate enforces),
evaluation fails clearly instead of fusing unrelated signals. Evaluate() may
be called with a raw evidence tuple rather than a constructed
FireDetectionCandidate, so it independently validates connectivity using
FireDetectionCandidate.is_connected rather than duplicating the definition.
"""
from __future__ import annotations

from src.calculators.fire_detection.fire_detection_config import (
    CONFIRMED_THRESHOLD,
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    NEWS_CONFIDENCE,
    SATELLITE_HIGH_CONFIDENCE,
    SATELLITE_LOW_CONFIDENCE,
    SATELLITE_NOMINAL_CONFIDENCE,
    SUSPECTED_THRESHOLD,
)
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_decision import FireDetectionDecision
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType

_SATELLITE_CONFIDENCE_WEIGHTS = {
    "low": SATELLITE_LOW_CONFIDENCE,
    "nominal": SATELLITE_NOMINAL_CONFIDENCE,
    "high": SATELLITE_HIGH_CONFIDENCE,
}


class FireDetectionCalculator:
    """Evaluate confidence that direct evidence belongs to one wildfire candidate.

    The returned statuses may later be mapped by persistence/orchestration
    tasks. This calculator does not create persisted event records and does not
    model RESOLVED/DISMISSED lifecycle states.
    """

    def evaluate(self, evidence: tuple[FireDetectionEvidence, ...]) -> FireDetectionDecision:
        """Return a deterministic detection decision for one candidate evidence group."""
        evidence_items = tuple(evidence)
        self._validate_evidence_items(evidence_items)

        if not evidence_items:
            return FireDetectionDecision(
                confidence=MIN_CONFIDENCE,
                status=FireDetectionStatus.NO_EVENT,
                latitude=None,
                longitude=None,
                supporting_evidence=(),
            )

        self._validate_connected_candidate(evidence_items)
        confidence = _combine_source_family_confidence(evidence_items)
        status = _classify_confidence(confidence)
        latitude, longitude = _estimate_location(evidence_items, status)
        return FireDetectionDecision(
            confidence=confidence,
            status=status,
            latitude=latitude,
            longitude=longitude,
            supporting_evidence=_supporting_evidence_refs(evidence_items),
        )

    @staticmethod
    def _validate_evidence_items(evidence_items: tuple[FireDetectionEvidence, ...]) -> None:
        evidence_refs: set[FireEvidenceRef] = set()
        for item in evidence_items:
            if not isinstance(item, FireDetectionEvidence):
                raise ValueError(f"evidence must contain FireDetectionEvidence items, got {item!r}")
            evidence_ref = _evidence_ref(item)
            if evidence_ref in evidence_refs:
                raise ValueError(f"duplicate evidence identity: {evidence_ref!r}")
            evidence_refs.add(evidence_ref)

    @staticmethod
    def _validate_connected_candidate(evidence_items: tuple[FireDetectionEvidence, ...]) -> None:
        if not FireDetectionCandidate.is_connected(evidence_items):
            raise ValueError(
                "Evidence does not form one connected fire-detection candidate by location and time: "
                f"{_supporting_evidence_refs(evidence_items)!r}."
            )


def _combine_source_family_confidence(evidence_items: tuple[FireDetectionEvidence, ...]) -> float:
    satellite_contribution = _strongest_satellite_contribution(evidence_items)
    news_contribution = NEWS_CONFIDENCE if any(item.evidence_type is FireEvidenceType.NEWS for item in evidence_items) else None
    contributions = tuple(
        contribution
        for contribution in (satellite_contribution, news_contribution)
        if contribution is not None
    )
    confidence = 1.0
    for contribution in contributions:
        confidence *= 1 - contribution
    return min(max(1 - confidence, MIN_CONFIDENCE), MAX_CONFIDENCE)


def _supporting_evidence_refs(evidence_items: tuple[FireDetectionEvidence, ...]) -> tuple[FireEvidenceRef, ...]:
    refs = {_evidence_ref(item) for item in evidence_items}
    return tuple(sorted(refs, key=lambda ref: (ref.evidence_type.value, ref.evidence_id)))


def _evidence_ref(evidence: FireDetectionEvidence) -> FireEvidenceRef:
    return FireEvidenceRef(evidence_type=evidence.evidence_type, evidence_id=evidence.evidence_id)


def _strongest_satellite_contribution(evidence_items: tuple[FireDetectionEvidence, ...]) -> float | None:
    satellite_weights = [
        _SATELLITE_CONFIDENCE_WEIGHTS[item.satellite_confidence]
        for item in evidence_items
        if item.evidence_type is FireEvidenceType.SATELLITE
    ]
    return max(satellite_weights) if satellite_weights else None


def _classify_confidence(confidence: float) -> FireDetectionStatus:
    if confidence < SUSPECTED_THRESHOLD:
        return FireDetectionStatus.NO_EVENT
    if confidence < CONFIRMED_THRESHOLD:
        return FireDetectionStatus.SUSPECTED
    return FireDetectionStatus.CONFIRMED


def _estimate_location(
    evidence_items: tuple[FireDetectionEvidence, ...],
    status: FireDetectionStatus,
) -> tuple[float | None, float | None]:
    if status is FireDetectionStatus.NO_EVENT:
        return None, None
    satellite_items = tuple(item for item in evidence_items if item.evidence_type is FireEvidenceType.SATELLITE)
    if satellite_items:
        return _centroid(satellite_items)
    news_items = tuple(item for item in evidence_items if item.evidence_type is FireEvidenceType.NEWS)
    return _centroid(news_items)


def _centroid(evidence_items: tuple[FireDetectionEvidence, ...]) -> tuple[float, float]:
    return (
        sum(item.latitude for item in evidence_items) / len(evidence_items),
        sum(item.longitude for item in evidence_items) / len(evidence_items),
    )

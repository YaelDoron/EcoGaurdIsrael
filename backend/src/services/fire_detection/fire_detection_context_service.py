"""Look up Fire Danger context for a Fire Detection candidate.

Fire Danger is contextual risk, not direct evidence of an active wildfire: this
service never creates, merges, or filters candidates, and nothing here changes
a detection decision. It only reads an already-persisted Fire Danger
assessment (written by FireDangerAssessmentAgent) and reports it as a
FireDetectionContext for later ML feature extraction.

Geographic relevance: the persistence model has no independent area registry -
each persisted FireDangerAssessment embeds its own area center and radius. An
assessment is relevant to a candidate when the candidate's location lies inside
that circle (great-circle distance <= area_radius_km).

Selection: among relevant assessments with
`as_of - MAX_FIRE_DANGER_ASSESSMENT_AGE_MINUTES <= assessed_at <= as_of`, the
newest wins (ties: nearest area center, then highest assessment id). If that
newest assessment is INSUFFICIENT_DATA, context is unavailable - an older VALID
assessment is deliberately not substituted, since a newer attempt found the
weather inputs unreliable.

Any missing/stale/insufficient/failed lookup yields
FireDetectionContext.unavailable(); this service never raises for data or
repository problems, so Fire Detection can never fail because of it.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import logging

from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_context import FireDetectionContext
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_type import FireEvidenceType
from src.repositories.fire_danger_assessment_repository import (
    FireDangerAssessmentRepository,
    StoredFireDangerAssessment,
)
from src.services.fire_detection.fire_detection_context_config import MAX_FIRE_DANGER_ASSESSMENT_AGE_MINUTES
from src.utils.geo import haversine_distance_km

logger = logging.getLogger(__name__)

_DISTANCE_TOLERANCE_KM = 1e-9


class FireDetectionContextService:
    """Resolve the latest fresh, geographically relevant Fire Danger assessment for a candidate."""

    def __init__(
        self,
        fire_danger_repository: FireDangerAssessmentRepository | None = None,
        max_fire_danger_age_minutes: float = MAX_FIRE_DANGER_ASSESSMENT_AGE_MINUTES,
    ) -> None:
        if isinstance(max_fire_danger_age_minutes, bool) or max_fire_danger_age_minutes <= 0:
            raise ValueError(
                f"max_fire_danger_age_minutes must be greater than 0, got {max_fire_danger_age_minutes!r}"
            )
        self._fire_danger_repository = fire_danger_repository or FireDangerAssessmentRepository()
        self._max_age = timedelta(minutes=max_fire_danger_age_minutes)

    def build_context(
        self,
        candidate: FireDetectionCandidate | tuple[FireDetectionEvidence, ...],
        as_of: datetime,
    ) -> FireDetectionContext:
        """Return Fire Danger context for a candidate (or its raw evidence tuple) at `as_of`."""
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}.")
        evidence = candidate.evidence if isinstance(candidate, FireDetectionCandidate) else tuple(candidate)
        if not evidence:
            raise ValueError("Fire Detection context requires at least one evidence item.")

        try:
            return self._resolve_context(evidence, as_of)
        except Exception:
            logger.warning("Fire Detection context lookup failed; continuing without Fire Danger context.", exc_info=True)
            return FireDetectionContext.unavailable()

    def _resolve_context(self, evidence: tuple[FireDetectionEvidence, ...], as_of: datetime) -> FireDetectionContext:
        window_start = as_of - self._max_age
        latitude, longitude = _candidate_location(evidence)

        matches: list[tuple[StoredFireDangerAssessment, float]] = []
        for stored in self._fire_danger_repository.get_assessed_between(window_start, as_of):
            assessment = stored.assessment
            # Enforced here as well as in the repository query, so the rules
            # hold for any repository implementation.
            if not window_start <= assessment.assessed_at <= as_of:
                continue
            distance_km = haversine_distance_km(
                latitude, longitude, assessment.area_latitude, assessment.area_longitude
            )
            if distance_km <= assessment.area_radius_km + _DISTANCE_TOLERANCE_KM:
                matches.append((stored, distance_km))

        if not matches:
            return FireDetectionContext.unavailable()

        selected, _ = min(
            matches,
            key=lambda match: (-match[0].assessment.assessed_at.timestamp(), match[1], -match[0].assessment_id),
        )
        assessment = selected.assessment
        if assessment.status is not FireDangerAssessmentStatus.VALID:
            return FireDetectionContext.unavailable()

        return FireDetectionContext(
            fire_danger_available=True,
            fire_danger_score=assessment.score,
            fire_danger_age_minutes=(as_of - assessment.assessed_at).total_seconds() / 60.0,
            fire_danger_level=assessment.level,
            fire_danger_assessment_id=selected.assessment_id,
            fire_danger_assessed_at=assessment.assessed_at,
        )


def _candidate_location(evidence: tuple[FireDetectionEvidence, ...]) -> tuple[float, float]:
    """Satellite-priority centroid - the same location rule FireDetectionCalculator uses for an event."""
    satellite_items = tuple(item for item in evidence if item.evidence_type is FireEvidenceType.SATELLITE)
    items = satellite_items or evidence
    return (
        sum(item.latitude for item in items) / len(items),
        sum(item.longitude for item in items) / len(items),
    )

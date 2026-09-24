"""Build the evidence history of an active FireEvent (Task 5B).

Reads what is already attached to the FireEvent, reloads it through the evidence service
(so satellite and news items are normalized exactly as for detection) and returns an immutable
FireDetectionEventHistory bounded by FIRE_EVENT_EVIDENCE_HISTORY_HOURS.

Deliberately NOT done here:
  * no 60-minute candidate correlation - the history is expected to span several hours;
  * no rule / ML decision and no ML feature extraction;
  * no writes.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import logging

from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_detection_event_history import FireDetectionEventHistory, chronological_key
from src.models.fire_evidence_ref import FireEvidenceRef
from src.repositories.fire_event_repository import FireEventRepository, StoredFireEvent
from src.services.fire_detection.fire_detection_evidence_config import (
    FIRE_EVENT_EVIDENCE_HISTORY_HOURS,
    SATELLITE_PASS_GAP_MINUTES,
)
from src.services.fire_detection.fire_detection_evidence_service import FireDetectionEvidenceService

logger = logging.getLogger(__name__)


def resolve_refs_tolerantly(
    evidence_service: FireDetectionEvidenceService,
    refs: tuple[FireEvidenceRef, ...],
) -> tuple[tuple[FireDetectionEvidence, ...], tuple[FireEvidenceRef, ...]]:
    """Resolve refs to evidence; refs that no longer resolve are reported, not fatal.

    The whole set is tried first (one lookup pass). Only if that raises ValueError - the
    evidence service's "ref not found / cannot be normalized" signal - is each ref resolved on
    its own so a single vanished row cannot hide the rest of an event's history. Any other
    exception (e.g. a database failure) propagates: that is an outage, not missing evidence.
    Returns (evidence in chronological order, unresolved refs).
    """
    refs = tuple(refs)
    if not refs:
        return (), ()
    try:
        return tuple(evidence_service.resolve_evidence_refs(refs)), ()
    except ValueError:
        pass

    resolved: list[FireDetectionEvidence] = []
    unresolved: list[FireEvidenceRef] = []
    for ref in refs:
        try:
            resolved.extend(evidence_service.resolve_evidence_refs((ref,)))
        except ValueError:
            logger.warning("Evidence %s %s of a FireEvent could not be resolved; skipping it.", ref.evidence_type.value, ref.evidence_id)
            unresolved.append(ref)
    return tuple(sorted(resolved, key=chronological_key)), tuple(unresolved)


class FireDetectionHistoryService:
    """FireEvent + as_of -> FireDetectionEventHistory."""

    def __init__(
        self,
        evidence_service: FireDetectionEvidenceService,
        fire_event_repository: FireEventRepository,
        history_hours: float = FIRE_EVENT_EVIDENCE_HISTORY_HOURS,
        satellite_pass_gap_minutes: float = SATELLITE_PASS_GAP_MINUTES,
    ) -> None:
        if isinstance(history_hours, bool) or not isinstance(history_hours, (int, float)) or history_hours <= 0:
            raise ValueError(f"history_hours must be a positive number, got {history_hours!r}")
        if (
            isinstance(satellite_pass_gap_minutes, bool)
            or not isinstance(satellite_pass_gap_minutes, (int, float))
            or satellite_pass_gap_minutes <= 0
        ):
            raise ValueError(f"satellite_pass_gap_minutes must be a positive number, got {satellite_pass_gap_minutes!r}")
        self._evidence_service = evidence_service
        self._fire_event_repository = fire_event_repository
        self._history_hours = history_hours
        self._satellite_pass_gap_minutes = satellite_pass_gap_minutes

    def build_history(self, fire_event: StoredFireEvent, as_of: datetime) -> FireDetectionEventHistory:
        """Evidence attached to `fire_event` observed within [as_of - history window, as_of]."""
        if not isinstance(fire_event, StoredFireEvent):
            raise ValueError(f"fire_event must be a StoredFireEvent, got {fire_event!r}")
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")

        window_start = as_of - timedelta(hours=self._history_hours)
        refs = self._fire_event_repository.get_evidence_refs(fire_event.id)
        evidence, unresolved = resolve_refs_tolerantly(self._evidence_service, tuple(dict.fromkeys(refs)))

        in_window: dict[tuple, FireDetectionEvidence] = {}
        for item in evidence:
            if window_start <= item.observed_at <= as_of:
                in_window.setdefault((item.evidence_type, item.evidence_id), item)

        return FireDetectionEventHistory(
            fire_event_id=fire_event.id,
            as_of=as_of,
            window_start=window_start,
            evidence=tuple(in_window.values()),
            satellite_pass_gap_minutes=self._satellite_pass_gap_minutes,
            unresolved_evidence=unresolved,
        )

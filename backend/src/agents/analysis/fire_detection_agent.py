"""Orchestrates active wildfire detection from evidence candidates to FireEvents."""
from __future__ import annotations

from datetime import datetime
import logging

from src.agents.analysis.fire_detection_result import FireDetectionResult
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.models.fire_detection_decision import FireDetectionDecision
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.repositories.fire_event_repository import FireEventRepository, StoredFireEvent
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.fire_detection.fire_detection_evidence_service import FireDetectionEvidenceService

logger = logging.getLogger(__name__)


class FireDetectionAgent:
    """Coordinates evidence retrieval, detection calculation, and FireEvent persistence."""

    def __init__(
        self,
        evidence_service: FireDetectionEvidenceService,
        calculator: FireDetectionCalculator,
        fire_event_repository: FireEventRepository,
        satellite_repository: SatelliteHotspotRepository | None = None,
        news_repository: NewsRepository | None = None,
    ) -> None:
        self._evidence_service = evidence_service
        self._calculator = calculator
        self._fire_event_repository = fire_event_repository
        self._satellite_repository = satellite_repository
        self._news_repository = news_repository

    def detect(self, as_of: datetime) -> FireDetectionResult:
        """Evaluate current evidence candidates at a timezone-aware instant."""
        self._validate_as_of(as_of)
        state = _DetectionRunState()

        try:
            candidates = self._evidence_service.build_candidates(as_of)
            for candidate in candidates:
                state.candidates_processed += 1
                decision = self._calculator.evaluate(candidate.evidence)
                logger.info("Evaluated fire-detection candidate with status %s", decision.status.value)

                if decision.status is FireDetectionStatus.NO_EVENT:
                    state.no_event_count += 1
                    logger.info("Ignored NO_EVENT fire-detection candidate")
                    continue

                event_id, created, updated = self._persist_detection_decision(decision, candidate.evidence)
                state.event_ids.add(event_id)
                if created:
                    state.events_created += 1
                if updated:
                    state.events_updated += 1

        except Exception:
            logger.exception("Fire detection orchestration failed")
            return FireDetectionResult(
                success=False,
                candidates_processed=state.candidates_processed,
                no_event_count=state.no_event_count,
                events_created=state.events_created,
                events_updated=state.events_updated,
                event_ids=tuple(state.event_ids),
                error_message="Fire detection orchestration failed.",
            )

        return FireDetectionResult(
            success=True,
            candidates_processed=state.candidates_processed,
            no_event_count=state.no_event_count,
            events_created=state.events_created,
            events_updated=state.events_updated,
            event_ids=tuple(state.event_ids),
            error_message=None,
        )

    def _persist_detection_decision(
        self,
        decision: FireDetectionDecision,
        candidate_evidence: tuple[FireDetectionEvidence, ...],
    ) -> tuple[int, bool, bool]:
        if decision.latitude is None or decision.longitude is None:
            raise ValueError("Event-producing detection decisions must include a location.")

        reference_time = self._latest_observed_at(candidate_evidence)
        existing_event = self._fire_event_repository.find_matching_active_event(
            latitude=decision.latitude,
            longitude=decision.longitude,
            observed_at=reference_time,
        )
        if existing_event is None:
            created = self._create_event(decision, candidate_evidence)
            logger.info("Created FireEvent %s", created.id)
            return created.id, True, False

        logger.info("Matched existing FireEvent %s", existing_event.id)
        updated = self._update_existing_event(existing_event, decision.supporting_evidence)
        return existing_event.id, False, updated

    def _create_event(
        self,
        decision: FireDetectionDecision,
        candidate_evidence: tuple[FireDetectionEvidence, ...],
    ) -> StoredFireEvent:
        event = FireEvent(
            latitude=decision.latitude,
            longitude=decision.longitude,
            detected_at=self._earliest_observed_at(candidate_evidence),
            updated_at=self._latest_observed_at(candidate_evidence),
            status=self._to_event_status(decision.status),
            detection_confidence=decision.confidence,
            methodology=FIRE_DETECTION_METHODOLOGY_NAME,
            methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
            location_name=self._resolve_location_name(candidate_evidence),
        )
        return self._fire_event_repository.create_event(event, supporting_evidence=decision.supporting_evidence)

    def _update_existing_event(
        self,
        existing_event: StoredFireEvent,
        new_refs: tuple[FireEvidenceRef, ...],
    ) -> bool:
        existing_refs = self._fire_event_repository.get_evidence_refs(existing_event.id)
        combined_refs = self._sort_refs(tuple(set(existing_refs).union(new_refs)))
        combined_evidence = self._evidence_service.resolve_evidence_refs(combined_refs)
        combined_decision = self._calculator.evaluate(combined_evidence)
        if combined_decision.status is FireDetectionStatus.NO_EVENT:
            raise ValueError("Combined active-event evidence unexpectedly evaluated as NO_EVENT.")

        next_status = self._monotonic_status(existing_event.event.status, self._to_event_status(combined_decision.status))
        updated_event = FireEvent(
            latitude=combined_decision.latitude,
            longitude=combined_decision.longitude,
            detected_at=existing_event.event.detected_at,
            updated_at=self._latest_observed_at(combined_evidence),
            status=next_status,
            detection_confidence=combined_decision.confidence,
            methodology=existing_event.event.methodology,
            methodology_version=existing_event.event.methodology_version,
            # location_name is set once at creation and never replaced by a
            # later update (Part H) - carried forward unchanged here so it
            # is never spuriously cleared/altered by re-evaluation.
            location_name=existing_event.event.location_name,
        )

        newly_added_refs = tuple(ref for ref in combined_refs if ref not in set(existing_refs))
        event_changed = updated_event != existing_event.event
        evidence_changed = bool(newly_added_refs)
        if not event_changed and not evidence_changed:
            return False

        if evidence_changed:
            self._fire_event_repository.attach_evidence(existing_event.id, newly_added_refs)
            logger.info("Attached %s evidence refs to FireEvent %s", len(newly_added_refs), existing_event.id)
        if event_changed:
            self._fire_event_repository.update_event(existing_event.id, updated_event)
            if existing_event.event.status is FireEventStatus.SUSPECTED and next_status is FireEventStatus.CONFIRMED:
                logger.info("Upgraded FireEvent %s from SUSPECTED to CONFIRMED", existing_event.id)
        return True

    @staticmethod
    def _to_event_status(status: FireDetectionStatus) -> FireEventStatus:
        if status is FireDetectionStatus.SUSPECTED:
            return FireEventStatus.SUSPECTED
        if status is FireDetectionStatus.CONFIRMED:
            return FireEventStatus.CONFIRMED
        raise ValueError(f"Detection status {status!r} does not map to a FireEvent status.")

    @staticmethod
    def _monotonic_status(existing_status: FireEventStatus, calculated_status: FireEventStatus) -> FireEventStatus:
        if existing_status is FireEventStatus.CONFIRMED and calculated_status is FireEventStatus.SUSPECTED:
            return FireEventStatus.CONFIRMED
        return calculated_status

    @staticmethod
    def _sort_refs(refs: tuple[FireEvidenceRef, ...]) -> tuple[FireEvidenceRef, ...]:
        return tuple(sorted(refs, key=lambda ref: (ref.evidence_type.value, ref.evidence_id)))

    @staticmethod
    def _resolve_location_name(evidence: tuple[FireDetectionEvidence, ...]) -> str | None:
        """Pick the candidate's trustworthy location, if any evidence item carries one.

        Only SATELLITE evidence is considered - defense-in-depth alongside
        FireDetectionEvidenceService._normalize_news, which already never
        copies WildfireReport.location_name onto NEWS evidence (that field
        is a best-effort, sometimes-wrong NLP guess for real ingestion, not
        verified provenance). All evidence in one FireDetectionCandidate is
        already correlated by distance/time (see FireDetectionCalculator) -
        i.e. it is understood to describe the SAME incident - so any
        non-null satellite `location_name` values present are expected to
        agree. Deterministic pick: sorted by evidence_id ascending, first
        non-null wins. `None` when no satellite evidence carries one (e.g.
        real, non-simulation evidence) - never guessed from coordinates here.
        """
        satellite_evidence = sorted(
            (item for item in evidence if item.evidence_type is FireEvidenceType.SATELLITE),
            key=lambda candidate: candidate.evidence_id,
        )
        for item in satellite_evidence:
            if item.location_name is not None:
                return item.location_name
        return None

    @staticmethod
    def _earliest_observed_at(evidence: tuple[FireDetectionEvidence, ...]) -> datetime:
        return min(item.observed_at for item in evidence)

    @staticmethod
    def _latest_observed_at(evidence: tuple[FireDetectionEvidence, ...]) -> datetime:
        return max(item.observed_at for item in evidence)

    @staticmethod
    def _validate_as_of(as_of: datetime) -> None:
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}.")


class _DetectionRunState:
    """Mutable counters for one detect() run."""

    def __init__(self) -> None:
        self.candidates_processed = 0
        self.no_event_count = 0
        self.events_created = 0
        self.events_updated = 0
        self.event_ids: set[int] = set()

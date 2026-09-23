"""Orchestrates active wildfire detection from evidence candidates to FireEvents.

Task 5: also runs the runtime Logistic Regression V3 ML assessment (when the
configured decision_mode enables it) and combines it with the deterministic
rule decision through FireDetectionHybridPolicy. This Agent never touches
sklearn/joblib directly, never builds a raw feature array, and never knows
about StandardScaler - all of that lives in MLFireDetectionClassifier. The
Agent only ever sees FireDetectionMLAssessment/HybridFireDetectionDecision.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging

from src.agents.analysis.fire_detection_candidate_assessment import FireDetectionCandidateAssessment
from src.agents.analysis.fire_detection_result import FireDetectionResult
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.calculators.fire_detection.fire_detection_decision_policy import FireDetectionHybridPolicy
from src.calculators.fire_detection.fire_detection_ml_classifier import MLFireDetectionClassifier
from src.config.settings import settings
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_detection_ml_assessment import FireDetectionMLAssessment
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_event import FireEvent
from src.models.fire_event_ml_assessment import FireEventMLAssessment
from src.models.fire_event_status import FireEventStatus
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.models.hybrid_fire_detection_decision import HybridFireDetectionDecision
from src.repositories.fire_event_repository import FireEventRepository, StoredFireEvent
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.fire_detection.fire_detection_evidence_service import FireDetectionEvidenceService

logger = logging.getLogger(__name__)

# Tolerance for deciding whether a re-evaluated ML assessment is "materially
# unchanged" (Part 28): avoids a DB write on every detect() cycle just
# because predict_proba's float output jittered in the last decimal place.
ML_PROBABILITY_CHANGE_TOLERANCE = 0.01
RULE_CONFIDENCE_CHANGE_TOLERANCE = 1e-9


class FireDetectionAgent:
    """Coordinates evidence retrieval, rule+ML detection, and FireEvent persistence."""

    def __init__(
        self,
        evidence_service: FireDetectionEvidenceService,
        calculator: FireDetectionCalculator,
        fire_event_repository: FireEventRepository,
        satellite_repository: SatelliteHotspotRepository | None = None,
        news_repository: NewsRepository | None = None,
        decision_policy: FireDetectionHybridPolicy | None = None,
        ml_classifier: MLFireDetectionClassifier | None = None,
    ) -> None:
        self._evidence_service = evidence_service
        self._calculator = calculator
        self._fire_event_repository = fire_event_repository
        self._satellite_repository = satellite_repository
        self._news_repository = news_repository
        self._decision_policy = decision_policy or _default_decision_policy()
        # RULE_ONLY never loads/calls ML, regardless of whether a classifier
        # instance was supplied - _run_ml_assessment() gates the call, and
        # here we also avoid even constructing a default one needlessly.
        if ml_classifier is not None:
            self._ml_classifier = ml_classifier
        elif self._decision_policy.mode is FireDetectionDecisionMode.RULE_ONLY:
            self._ml_classifier = None
        else:
            self._ml_classifier = MLFireDetectionClassifier(
                model_path=settings.FIRE_DETECTION_ML_MODEL_PATH,
                metadata_path=settings.FIRE_DETECTION_ML_METADATA_PATH,
            )

    def detect(self, as_of: datetime) -> FireDetectionResult:
        """Evaluate current evidence candidates at a timezone-aware instant."""
        self._validate_as_of(as_of)
        state = _DetectionRunState()

        try:
            candidates = self._evidence_service.build_candidates(as_of)
            for candidate in candidates:
                state.candidates_processed += 1
                rule_decision = self._calculator.evaluate(candidate.evidence)
                ml_assessment = self._run_ml_assessment(candidate.evidence)
                hybrid_decision = self._decision_policy.decide(rule_decision, ml_assessment, candidate.evidence)
                logger.info(
                    "Evaluated fire-detection candidate: rule=%s final=%s mode=%s agreement=%s",
                    rule_decision.status.value,
                    hybrid_decision.final_status.value,
                    hybrid_decision.decision_mode.value,
                    hybrid_decision.agreement.value,
                )

                if hybrid_decision.final_status is FireDetectionStatus.NO_EVENT:
                    state.no_event_count += 1
                    logger.info("Ignored NO_EVENT fire-detection candidate")
                    state.candidate_assessments.append(self._to_candidate_assessment(None, hybrid_decision))
                    continue

                event_id, created, updated, persisted_decision = self._persist_hybrid_decision(
                    hybrid_decision, candidate.evidence
                )
                state.event_ids.add(event_id)
                if created:
                    state.events_created += 1
                if updated:
                    state.events_updated += 1
                state.candidate_assessments.append(self._to_candidate_assessment(event_id, persisted_decision))

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
            candidate_assessments=tuple(state.candidate_assessments),
        )

    def _run_ml_assessment(self, evidence: tuple[FireDetectionEvidence, ...]) -> FireDetectionMLAssessment:
        if self._decision_policy.mode is FireDetectionDecisionMode.RULE_ONLY:
            return _unavailable_assessment(failure_reason=None)  # not a failure - ML intentionally not run
        if self._ml_classifier is None:
            return _unavailable_assessment(failure_reason="ML classifier not configured.")
        return self._ml_classifier.assess(evidence)

    def _persist_hybrid_decision(
        self,
        hybrid_decision: HybridFireDetectionDecision,
        candidate_evidence: tuple[FireDetectionEvidence, ...],
    ) -> tuple[int, bool, bool, HybridFireDetectionDecision]:
        if hybrid_decision.final_latitude is None or hybrid_decision.final_longitude is None:
            raise ValueError("Event-producing hybrid decisions must include a location.")

        reference_time = self._latest_observed_at(candidate_evidence)
        existing_event = self._fire_event_repository.find_matching_active_event(
            latitude=hybrid_decision.final_latitude,
            longitude=hybrid_decision.final_longitude,
            observed_at=reference_time,
        )
        if existing_event is None:
            created = self._create_event(hybrid_decision, candidate_evidence)
            logger.info("Created FireEvent %s", created.id)
            self._maybe_record_ml_assessment(created.id, hybrid_decision)
            return created.id, True, False, hybrid_decision

        logger.info("Matched existing FireEvent %s", existing_event.id)
        updated, combined_decision = self._update_existing_event(existing_event, hybrid_decision.final_supporting_evidence)
        self._maybe_record_ml_assessment(existing_event.id, combined_decision)
        return existing_event.id, False, updated, combined_decision

    def _create_event(
        self,
        hybrid_decision: HybridFireDetectionDecision,
        candidate_evidence: tuple[FireDetectionEvidence, ...],
    ) -> StoredFireEvent:
        event = FireEvent(
            latitude=hybrid_decision.final_latitude,
            longitude=hybrid_decision.final_longitude,
            detected_at=self._earliest_observed_at(candidate_evidence),
            updated_at=self._latest_observed_at(candidate_evidence),
            status=self._to_event_status(hybrid_decision.final_status),
            detection_confidence=hybrid_decision.final_confidence,
            methodology=FIRE_DETECTION_METHODOLOGY_NAME,
            methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
            location_name=self._resolve_location_name(candidate_evidence),
        )
        return self._fire_event_repository.create_event(
            event, supporting_evidence=hybrid_decision.final_supporting_evidence
        )

    def _update_existing_event(
        self,
        existing_event: StoredFireEvent,
        new_refs: tuple[FireEvidenceRef, ...],
    ) -> tuple[bool, HybridFireDetectionDecision]:
        existing_refs = self._fire_event_repository.get_evidence_refs(existing_event.id)
        combined_refs = self._sort_refs(tuple(set(existing_refs).union(new_refs)))
        combined_evidence = self._evidence_service.resolve_evidence_refs(combined_refs)

        combined_rule_decision = self._calculator.evaluate(combined_evidence)
        if combined_rule_decision.status is FireDetectionStatus.NO_EVENT:
            raise ValueError("Combined active-event evidence unexpectedly evaluated as NO_EVENT.")

        # Re-run BOTH rule and ML on the normalized union of evidence (Part
        # 27) - never keep a stale ML probability after new evidence arrives.
        combined_ml_assessment = self._run_ml_assessment(combined_evidence)
        combined_decision = self._decision_policy.decide(combined_rule_decision, combined_ml_assessment, combined_evidence)
        # An already-active event's rule decision can never be NO_EVENT
        # (checked above), so HYBRID's NO_EVENT-only escalation path cannot
        # apply here - combined_decision.final_status always equals
        # combined_rule_decision.status for a reevaluation.

        next_status = self._monotonic_status(existing_event.event.status, self._to_event_status(combined_decision.final_status))
        updated_event = FireEvent(
            latitude=combined_decision.final_latitude,
            longitude=combined_decision.final_longitude,
            detected_at=existing_event.event.detected_at,
            updated_at=self._latest_observed_at(combined_evidence),
            status=next_status,
            detection_confidence=combined_decision.final_confidence,
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

        if evidence_changed:
            self._fire_event_repository.attach_evidence(existing_event.id, newly_added_refs)
            logger.info("Attached %s evidence refs to FireEvent %s", len(newly_added_refs), existing_event.id)
        if event_changed:
            self._fire_event_repository.update_event(existing_event.id, updated_event)
            if existing_event.event.status is FireEventStatus.SUSPECTED and next_status is FireEventStatus.CONFIRMED:
                logger.info("Upgraded FireEvent %s from SUSPECTED to CONFIRMED", existing_event.id)

        return (event_changed or evidence_changed), combined_decision

    def _maybe_record_ml_assessment(self, fire_event_id: int, hybrid_decision: HybridFireDetectionDecision) -> None:
        """RULE_ONLY never writes an ML trace row at all - true backward compatibility
        (zero new writes for that mode). SHADOW/HYBRID always attempt a write, subject
        to the unchanged-skip in _record_ml_assessment."""
        if self._decision_policy.mode is FireDetectionDecisionMode.RULE_ONLY:
            return
        self._record_ml_assessment(fire_event_id, hybrid_decision)

    def _record_ml_assessment(self, fire_event_id: int, hybrid_decision: HybridFireDetectionDecision) -> None:
        """Upsert the FireEvent's ML/decision trace, but only when something meaningful changed (Part 28)."""
        new_assessment = FireEventMLAssessment(
            fire_event_id=fire_event_id,
            decision_mode=hybrid_decision.decision_mode,
            rule_status=hybrid_decision.rule_decision.status,
            rule_confidence=hybrid_decision.rule_decision.confidence,
            ml_available=hybrid_decision.ml_assessment.available,
            ml_probability=hybrid_decision.ml_assessment.probability,
            ml_model_name=hybrid_decision.ml_assessment.model_name,
            ml_model_version=hybrid_decision.ml_assessment.model_version,
            ml_feature_schema_version=hybrid_decision.ml_assessment.feature_schema_version,
            ml_failure_reason=hybrid_decision.ml_assessment.failure_reason,
            agreement=hybrid_decision.agreement,
            updated_at=datetime.now(timezone.utc),
        )

        existing = self._fire_event_repository.get_ml_assessment(fire_event_id)
        if existing is not None and self._ml_assessment_unchanged(existing, new_assessment):
            return
        self._fire_event_repository.upsert_ml_assessment(fire_event_id, new_assessment)

    @staticmethod
    def _ml_assessment_unchanged(existing: FireEventMLAssessment, new: FireEventMLAssessment) -> bool:
        if existing.decision_mode is not new.decision_mode:
            return False
        if existing.rule_status is not new.rule_status:
            return False
        if abs(existing.rule_confidence - new.rule_confidence) > RULE_CONFIDENCE_CHANGE_TOLERANCE:
            return False
        if existing.ml_available != new.ml_available:
            return False
        if existing.ml_model_version != new.ml_model_version:
            return False
        if existing.agreement is not new.agreement:
            return False
        if existing.ml_available:
            existing_probability = existing.ml_probability or 0.0
            new_probability = new.ml_probability or 0.0
            if abs(existing_probability - new_probability) > ML_PROBABILITY_CHANGE_TOLERANCE:
                return False
        return True

    @staticmethod
    def _to_candidate_assessment(
        event_id: int | None,
        hybrid_decision: HybridFireDetectionDecision,
    ) -> FireDetectionCandidateAssessment:
        return FireDetectionCandidateAssessment(
            event_id=event_id,
            rule_status=hybrid_decision.rule_decision.status,
            rule_confidence=hybrid_decision.rule_decision.confidence,
            final_status=hybrid_decision.final_status,
            decision_mode=hybrid_decision.decision_mode,
            ml_available=hybrid_decision.ml_assessment.available,
            ml_probability=hybrid_decision.ml_assessment.probability,
            ml_model_version=hybrid_decision.ml_assessment.model_version,
            agreement=hybrid_decision.agreement,
        )

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


def _default_decision_policy() -> FireDetectionHybridPolicy:
    mode = FireDetectionDecisionMode(settings.FIRE_DETECTION_DECISION_MODE)
    raw_suspect_threshold = settings.FIRE_DETECTION_ML_SUSPECT_THRESHOLD.strip()
    suspect_threshold = float(raw_suspect_threshold) if raw_suspect_threshold else None
    return FireDetectionHybridPolicy(
        mode=mode,
        ml_classification_threshold=settings.FIRE_DETECTION_ML_CLASSIFICATION_THRESHOLD,
        ml_suspect_threshold=suspect_threshold,
    )


def _unavailable_assessment(failure_reason: str | None) -> FireDetectionMLAssessment:
    return FireDetectionMLAssessment(
        available=False,
        probability=None,
        model_name=None,
        model_version=None,
        feature_schema_version=None,
        failure_reason=failure_reason,
    )


class _DetectionRunState:
    """Mutable counters for one detect() run."""

    def __init__(self) -> None:
        self.candidates_processed = 0
        self.no_event_count = 0
        self.events_created = 0
        self.events_updated = 0
        self.event_ids: set[int] = set()
        self.candidate_assessments: list[FireDetectionCandidateAssessment] = []

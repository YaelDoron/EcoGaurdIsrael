"""Orchestrates active wildfire detection from evidence candidates to FireEvents.

Task 5: also runs the runtime Logistic Regression V3 ML assessment (when the
configured decision_mode enables it) and combines it with the deterministic
rule decision through FireDetectionHybridPolicy. This Agent never touches
sklearn/joblib directly, never builds a raw feature array, and never knows
about StandardScaler - all of that lives in MLFireDetectionClassifier. The
Agent only ever sees FireDetectionMLAssessment/HybridFireDetectionDecision.

Task 9B adds ONE separate path, selected only by the explicit AI_HYBRID_V5 decision
mode: candidate -> matching active FireEvent -> its history -> V5 features -> HGB V5
-> locked AI Hybrid Policy -> status. In that mode the rule calculator runs for
diagnostics only and can never decide the status, and there is NO silent fallback to
rules (an unusable artifact / history fails the detection cycle explicitly). The
legacy modes (rule_only / shadow / hybrid) follow the unchanged code path below.
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
from src.calculators.fire_detection.fire_detection_decision_policy import (
    FireDetectionHybridPolicy,
    estimate_candidate_location,
)
from src.calculators.fire_detection.fire_detection_ml_classifier import MLFireDetectionClassifier
from src.config.settings import settings
from src.ml.fire_detection.fire_detection_ai_hybrid_runtime_v5 import FireDetectionAIHybridClassifierV5
from src.ml.fire_detection.fire_detection_model_runtime_v5 import (
    FireDetectionAIAssessmentPersistenceError,
    FireDetectionAIError,
    FireDetectionHistoryUnavailableError,
    FireDetectionModelV5Runtime,
)
from src.models.fire_detection_ai_assessment import FireDetectionAIAssessment
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_ml_rule_agreement import FireDetectionMLRuleAgreement
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_event_history import correlated_evidence
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_detection_ml_assessment import FireDetectionMLAssessment
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_event import FireEvent
from src.models.fire_event_ml_assessment import FireEventMLAssessment
from src.models.fire_event_status import FireEventStatus
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.models.hybrid_fire_detection_decision import HybridFireDetectionDecision
from src.repositories.exceptions import FireEventRepositoryError
from src.repositories.fire_event_repository import FireEventRepository, StoredFireEvent
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.fire_detection.fire_detection_evidence_service import FireDetectionEvidenceService
from src.services.fire_detection.fire_detection_history_service import (
    FireDetectionHistoryService,
    resolve_refs_tolerantly,
)

logger = logging.getLogger(__name__)

# Tolerance for deciding whether a re-evaluated ML assessment is "materially
# unchanged" (Part 28): avoids a DB write on every detect() cycle just
# because predict_proba's float output jittered in the last decimal place.
ML_PROBABILITY_CHANGE_TOLERANCE = 0.01
RULE_CONFIDENCE_CHANGE_TOLERANCE = 1e-9

# FireEvents created by the AI Hybrid V5 path carry their own provenance (the version is the locked policy version).
AI_HYBRID_V5_METHODOLOGY_NAME = "ECOGUARD_AI_HYBRID_DETECTION"


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
        decision_mode: FireDetectionDecisionMode | None = None,
        ai_classifier: FireDetectionAIHybridClassifierV5 | None = None,
        history_service: FireDetectionHistoryService | None = None,
    ) -> None:
        self._evidence_service = evidence_service
        self._calculator = calculator
        self._fire_event_repository = fire_event_repository
        self._satellite_repository = satellite_repository
        self._news_repository = news_repository

        mode = _resolve_decision_mode(decision_mode, decision_policy, ai_classifier)
        self._ai_classifier: FireDetectionAIHybridClassifierV5 | None = None
        self._history_service: FireDetectionHistoryService | None = None
        self._ml_classifier: MLFireDetectionClassifier | None = None
        if mode is FireDetectionDecisionMode.AI_HYBRID_V5:
            # The AI path has no rule/ML hybrid policy and no V3 classifier; see _detect_candidate_ai_v5.
            self._decision_policy = None
            self._ai_classifier = ai_classifier or _default_ai_classifier()
            self._history_service = history_service or FireDetectionHistoryService(evidence_service, fire_event_repository)
            return

        self._decision_policy = decision_policy or _default_decision_policy(mode)
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

    @property
    def decision_mode(self) -> FireDetectionDecisionMode:
        if self._ai_classifier is not None:
            return FireDetectionDecisionMode.AI_HYBRID_V5
        return self._decision_policy.mode

    def detect(self, as_of: datetime) -> FireDetectionResult:
        """Evaluate current evidence candidates at a timezone-aware instant."""
        self._validate_as_of(as_of)
        state = _DetectionRunState()

        try:
            candidates = self._evidence_service.build_candidates(as_of)
            for candidate in candidates:
                state.candidates_processed += 1
                if self._ai_classifier is not None:
                    self._detect_candidate_ai_v5(candidate, as_of, state)
                    continue
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

        except FireDetectionAIError as exc:
            # AI Hybrid V5 failed explicitly (artifact / feature / inference / history). No rule fallback, and the failing
            # candidate created no FireEvent. The message is path-free and says which mode failed.
            logger.error("Fire detection (ai_hybrid_v5) failed: %s", exc.public_message)
            return FireDetectionResult(
                success=False,
                candidates_processed=state.candidates_processed,
                no_event_count=state.no_event_count,
                events_created=state.events_created,
                events_updated=state.events_updated,
                event_ids=tuple(state.event_ids),
                error_message=f"Fire detection (ai_hybrid_v5) failed: {exc.public_message}",
                candidate_assessments=tuple(state.candidate_assessments),
            )
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

    # ------------------------------------------------------------------------------------------------------
    # AI Hybrid V5 path (Task 9B) - only reachable when the decision mode is AI_HYBRID_V5
    # ------------------------------------------------------------------------------------------------------

    def _detect_candidate_ai_v5(self, candidate: FireDetectionCandidate, as_of: datetime, state: "_DetectionRunState") -> None:
        """candidate -> matching active event -> its history -> V5 features -> HGB -> locked policy -> persistence.

        The event match comes BEFORE inference because the event's history is an input of the V5 features.
        Matching semantics are unchanged (5 km / 6 h via FireEventRepository.find_matching_active_event).
        """
        evidence = candidate.evidence
        # Diagnostics only: the rule result is recorded next to the AI result but can never change the status.
        rule_decision = self._calculator.evaluate(evidence)
        latitude, longitude = estimate_candidate_location(evidence)

        existing_event = self._fire_event_repository.find_matching_active_event(
            latitude=latitude,
            longitude=longitude,
            observed_at=self._latest_observed_at(evidence),
        )
        history = None
        if existing_event is not None:
            try:
                history = self._history_service.build_history(existing_event, as_of)
            except Exception as exc:  # noqa: BLE001 - a history outage must NOT become "no history"
                logger.error("Could not load the history of FireEvent %s: %s: %s", existing_event.id, type(exc).__name__, exc)
                raise FireDetectionHistoryUnavailableError(
                    "the matching FireEvent's evidence history could not be loaded."
                ) from exc

        ai_assessment = self._ai_classifier.assess(candidate, history)
        decision = self._to_ai_decision(evidence, rule_decision, ai_assessment, latitude, longitude)
        logger.info(
            "Evaluated fire-detection candidate: mode=ai_hybrid_v5 policy_status=%s probability=%.3f "
            "current_pixels=%s pass_count=%s rule_diagnostic=%s matched_event=%s",
            ai_assessment.policy_status.value,
            ai_assessment.probability,
            ai_assessment.current_satellite_pixel_count,
            ai_assessment.satellite_pass_count,
            rule_decision.status.value,
            existing_event.id if existing_event is not None else None,
        )

        if existing_event is None:
            if ai_assessment.policy_status is FireDetectionStatus.NO_EVENT:
                # Nothing to attach the diagnostics to: no fake FireEvent is created for a negative prediction. The
                # in-memory candidate assessment (and the log line above) is the only trace.
                state.no_event_count += 1
                state.candidate_assessments.append(self._to_candidate_assessment(None, decision))
                return
            created = self._create_ai_event(decision, evidence)
            logger.info("Created FireEvent %s (ai_hybrid_v5, %s)", created.id, decision.final_status.value)
            state.event_ids.add(created.id)
            state.events_created += 1
            state.candidate_assessments.append(self._to_candidate_assessment(created.id, decision))
            return

        updated = self._update_existing_event_ai(existing_event, decision, evidence)
        state.event_ids.add(existing_event.id)
        if updated:
            state.events_updated += 1
        state.candidate_assessments.append(self._to_candidate_assessment(existing_event.id, decision))

    @staticmethod
    def _persistence_error(exc: FireEventRepositoryError) -> FireDetectionAIAssessmentPersistenceError:
        logger.error("Could not store the AI detection result atomically: %s", exc)
        return FireDetectionAIAssessmentPersistenceError(
            "the AI detection result (FireEvent + assessment) could not be stored and was rolled back "
            "(if this persists, has scripts.migrate_add_fire_event_ml_assessment_ai_columns been applied?)."
        )

    @staticmethod
    def _to_ai_decision(
        evidence: tuple[FireDetectionEvidence, ...],
        rule_decision,
        ai: FireDetectionAIAssessment,
        latitude: float,
        longitude: float,
    ) -> HybridFireDetectionDecision:
        rule_says_fire = rule_decision.status is not FireDetectionStatus.NO_EVENT
        ai_says_fire = ai.policy_status is not FireDetectionStatus.NO_EVENT
        if rule_says_fire and ai_says_fire:
            agreement = FireDetectionMLRuleAgreement.AGREE_FIRE
        elif not rule_says_fire and not ai_says_fire:
            agreement = FireDetectionMLRuleAgreement.AGREE_NO_FIRE
        elif rule_says_fire:
            agreement = FireDetectionMLRuleAgreement.RULE_STRONGER
        else:
            agreement = FireDetectionMLRuleAgreement.ML_STRONGER
        return HybridFireDetectionDecision(
            decision_mode=FireDetectionDecisionMode.AI_HYBRID_V5,
            final_status=ai.policy_status,
            final_confidence=ai.probability,
            final_latitude=latitude,
            final_longitude=longitude,
            final_supporting_evidence=tuple(
                FireEvidenceRef(evidence_type=item.evidence_type, evidence_id=item.evidence_id) for item in evidence
            ),
            rule_decision=rule_decision,
            ml_assessment=FireDetectionMLAssessment(
                available=True,
                probability=ai.probability,
                model_name=ai.model_name,
                model_version=ai.model_version,
                feature_schema_version=ai.feature_schema_version,
                failure_reason=None,
            ),
            agreement=agreement,
            ai_assessment=ai,
        )

    def _create_ai_event(
        self,
        decision: HybridFireDetectionDecision,
        candidate_evidence: tuple[FireDetectionEvidence, ...],
    ) -> StoredFireEvent:
        event = FireEvent(
            latitude=decision.final_latitude,
            longitude=decision.final_longitude,
            detected_at=self._earliest_observed_at(candidate_evidence),
            updated_at=self._latest_observed_at(candidate_evidence),
            status=self._to_event_status(decision.final_status),
            detection_confidence=decision.ai_assessment.probability,
            methodology=AI_HYBRID_V5_METHODOLOGY_NAME,
            methodology_version=decision.ai_assessment.policy_version,
            location_name=self._resolve_location_name(candidate_evidence),
        )
        try:
            # ONE transaction: the FireEvent, its evidence refs and its AI assessment commit together or not at all.
            return self._fire_event_repository.create_event_with_ml_assessment(
                event,
                decision.final_supporting_evidence,
                lambda fire_event_id: self._build_ml_assessment(fire_event_id, decision),
            )
        except FireEventRepositoryError as exc:
            raise self._persistence_error(exc) from exc

    def _update_existing_event_ai(
        self,
        existing_event: StoredFireEvent,
        decision: HybridFireDetectionDecision,
        candidate_evidence: tuple[FireDetectionEvidence, ...],
    ) -> bool:
        """Apply one AI assessment to a matched active event. Returns whether anything was persisted.

        * status: CONFIRMED never downgrades; SUSPECTED is promoted only by a CONFIRMED assessment and is NEVER
          dismissed by a NO_EVENT one (expiry/dismissal is a separate lifecycle concern);
        * detection_confidence stays the event's monotonic PEAK; the latest probability lives in the ML assessment row;
        * the candidate's evidence is attached whatever the verdict - a weak pass is still a pass in the event history.
        """
        ai = decision.ai_assessment
        existing = existing_event.event
        if ai.policy_status is FireDetectionStatus.CONFIRMED:
            next_status = FireEventStatus.CONFIRMED
        else:
            next_status = existing.status  # SUSPECTED stays SUSPECTED, CONFIRMED stays CONFIRMED
        if ai.policy_status is FireDetectionStatus.NO_EVENT:
            latitude, longitude = existing.latitude, existing.longitude
        else:
            latitude, longitude = decision.final_latitude, decision.final_longitude

        candidate_refs = decision.final_supporting_evidence
        existing_refs = set(self._fire_event_repository.get_evidence_refs(existing_event.id))
        new_refs = tuple(ref for ref in candidate_refs if ref not in existing_refs)
        updated_event = FireEvent(
            latitude=latitude,
            longitude=longitude,
            detected_at=existing.detected_at,
            updated_at=max(existing.updated_at, self._latest_observed_at(candidate_evidence)),
            status=next_status,
            detection_confidence=max(existing.detection_confidence, ai.probability),
            methodology=existing.methodology,
            methodology_version=existing.methodology_version,
            location_name=existing.location_name,  # set once at creation, never replaced
        )
        event_changed = updated_event != existing

        new_assessment = self._build_ml_assessment(existing_event.id, decision)
        stored_assessment = self._fire_event_repository.get_ml_assessment(existing_event.id)
        assessment_changed = stored_assessment is None or not self._ml_assessment_unchanged(stored_assessment, new_assessment)

        if not (event_changed or new_refs or assessment_changed):
            return False
        try:
            # ONE transaction: status promotion + confidence + new evidence + latest assessment commit together or not at all.
            self._fire_event_repository.update_event_with_ml_assessment(
                existing_event.id,
                event=updated_event if event_changed else None,
                new_evidence=self._sort_refs(new_refs),
                assessment=new_assessment if assessment_changed else None,
            )
        except FireEventRepositoryError as exc:
            raise self._persistence_error(exc) from exc
        if new_refs:
            logger.info("Attached %s evidence refs to FireEvent %s", len(new_refs), existing_event.id)
        if existing.status is FireEventStatus.SUSPECTED and next_status is FireEventStatus.CONFIRMED:
            logger.info("Upgraded FireEvent %s from SUSPECTED to CONFIRMED (ai_hybrid_v5)", existing_event.id)
        return event_changed or bool(new_refs)

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
        # Refs that vanished since they were attached must not block a new observation (Task 5B).
        combined_evidence, unresolved = resolve_refs_tolerantly(self._evidence_service, combined_refs)
        if set(unresolved).intersection(new_refs):
            # Only OLD history may vanish; the evidence being evaluated right now must resolve.
            raise ValueError("Current fire-detection evidence could not be resolved.")

        # Task 5B: evaluate the CURRENT candidate, not everything ever attached to the event. The
        # decision unit is the evidence that correlates (5 km / 60 min, chained) with the newly
        # supplied evidence; earlier observation waves hours ago stay attached to the event as its
        # history (see FireDetectionHistoryService) and are never forced into one
        # FireDetectionCandidate. When ALL attached evidence is correlated this is exactly the
        # previous "re-evaluate the union" behaviour.
        decision_evidence = correlated_evidence(combined_evidence, new_refs) or combined_evidence
        older_history_excluded = len(decision_evidence) < len(combined_evidence)

        combined_rule_decision = self._calculator.evaluate(decision_evidence)

        # Re-run BOTH rule and ML on the current evidence (Part 27) - never keep a stale ML
        # probability after new evidence arrives.
        combined_ml_assessment = self._run_ml_assessment(decision_evidence)
        combined_decision = self._decision_policy.decide(combined_rule_decision, combined_ml_assessment, decision_evidence)

        latest_observed_at = self._latest_observed_at(combined_evidence)
        if combined_decision.final_status is FireDetectionStatus.NO_EVENT:
            # Only reachable when the rule says NO_EVENT: an event created earlier by HYBRID ML
            # escalation whose ML score has since dropped below the threshold. The re-evaluation
            # is recorded as such (no crash), but a persisted event is never deleted or downgraded
            # by it: location, status and confidence stay, only the observation time advances.
            updated_event = FireEvent(
                latitude=existing_event.event.latitude,
                longitude=existing_event.event.longitude,
                detected_at=existing_event.event.detected_at,
                updated_at=max(existing_event.event.updated_at, latest_observed_at),
                status=existing_event.event.status,
                detection_confidence=existing_event.event.detection_confidence,
                methodology=existing_event.event.methodology,
                methodology_version=existing_event.event.methodology_version,
                location_name=existing_event.event.location_name,
            )
        else:
            next_status = self._monotonic_status(
                existing_event.event.status, self._to_event_status(combined_decision.final_status)
            )
            confidence = combined_decision.final_confidence
            if older_history_excluded:
                # The current wave alone may be weaker than the fire's earlier observations (e.g. one
                # low-confidence hotspot after a confirmed wave); the event keeps its peak confidence.
                confidence = max(confidence, existing_event.event.detection_confidence)
            updated_event = FireEvent(
                latitude=combined_decision.final_latitude,
                longitude=combined_decision.final_longitude,
                detected_at=existing_event.event.detected_at,
                updated_at=latest_observed_at,
                status=next_status,
                detection_confidence=confidence,
                methodology=existing_event.event.methodology,
                methodology_version=existing_event.event.methodology_version,
                # location_name is set once at creation and never replaced by a
                # later update (Part H) - carried forward unchanged here so it
                # is never spuriously cleared/altered by re-evaluation.
                location_name=existing_event.event.location_name,
            )
        next_status = updated_event.status

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

    def _build_ml_assessment(self, fire_event_id: int, hybrid_decision: HybridFireDetectionDecision) -> FireEventMLAssessment:
        return FireEventMLAssessment(
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
            **self._ai_assessment_fields(hybrid_decision),
        )

    def _record_ml_assessment(self, fire_event_id: int, hybrid_decision: HybridFireDetectionDecision) -> None:
        """Upsert the FireEvent's ML/decision trace, but only when something meaningful changed (Part 28)."""
        new_assessment = self._build_ml_assessment(fire_event_id, hybrid_decision)
        existing = self._fire_event_repository.get_ml_assessment(fire_event_id)
        if existing is not None and self._ml_assessment_unchanged(existing, new_assessment):
            return
        self._fire_event_repository.upsert_ml_assessment(fire_event_id, new_assessment)

    @staticmethod
    def _ai_assessment_fields(hybrid_decision: HybridFireDetectionDecision) -> dict:
        ai = hybrid_decision.ai_assessment
        if ai is None:
            return {}
        return {
            "policy_version": ai.policy_version,
            "policy_status": ai.policy_status,
            "history_available": ai.history_available,
            "satellite_pass_count": ai.satellite_pass_count,
            "current_satellite_pixel_count": ai.current_satellite_pixel_count,
        }

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
        for name in (
            "policy_version", "policy_status", "history_available", "satellite_pass_count", "current_satellite_pixel_count",
        ):
            if getattr(existing, name) != getattr(new, name):
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
        ai = hybrid_decision.ai_assessment
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
            ai_policy_version=ai.policy_version if ai is not None else None,
            ai_current_satellite_pixel_count=ai.current_satellite_pixel_count if ai is not None else None,
            ai_satellite_pass_count=ai.satellite_pass_count if ai is not None else None,
            ai_history_available=ai.history_available if ai is not None else None,
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


def _resolve_decision_mode(
    decision_mode: FireDetectionDecisionMode | None,
    decision_policy: FireDetectionHybridPolicy | None,
    ai_classifier: FireDetectionAIHybridClassifierV5 | None,
) -> FireDetectionDecisionMode:
    """The one place the agent's mode is decided (explicit arguments first, then configuration)."""
    if decision_mode is not None and not isinstance(decision_mode, FireDetectionDecisionMode):
        raise ValueError(f"decision_mode must be a FireDetectionDecisionMode, got {decision_mode!r}")
    if ai_classifier is not None:
        if decision_policy is not None or decision_mode not in (None, FireDetectionDecisionMode.AI_HYBRID_V5):
            raise ValueError("an ai_classifier can only be combined with decision_mode=ai_hybrid_v5 and no decision_policy.")
        return FireDetectionDecisionMode.AI_HYBRID_V5
    if decision_mode is FireDetectionDecisionMode.AI_HYBRID_V5:
        if decision_policy is not None:
            raise ValueError("decision_mode=ai_hybrid_v5 cannot be combined with a rule/ML decision_policy.")
        return decision_mode
    if decision_policy is not None:
        if decision_mode is not None and decision_policy.mode is not decision_mode:
            raise ValueError("decision_mode conflicts with decision_policy.mode.")
        return decision_policy.mode
    return decision_mode or FireDetectionDecisionMode(settings.FIRE_DETECTION_DECISION_MODE)


def _default_ai_classifier() -> FireDetectionAIHybridClassifierV5:
    """The approved artifact from configuration. Not loaded here: loading is lazy and cached (and may fail explicitly)."""
    return FireDetectionAIHybridClassifierV5(
        FireDetectionModelV5Runtime(
            model_path=settings.FIRE_DETECTION_AI_V5_MODEL_PATH,
            metadata_path=settings.FIRE_DETECTION_AI_V5_METADATA_PATH,
        )
    )


def _default_decision_policy(mode: FireDetectionDecisionMode | None = None) -> FireDetectionHybridPolicy:
    mode = mode or FireDetectionDecisionMode(settings.FIRE_DETECTION_DECISION_MODE)
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

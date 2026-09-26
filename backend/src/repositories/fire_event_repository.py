"""Persistence layer for wildfire events and their evidence traceability."""
from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import weakref

from sqlalchemy import or_, select
from sqlalchemy import insert, inspect, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload, sessionmaker, undefer_group

from src.database.connection import get_session_factory
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_event_ml_assessment_db import AI_HYBRID_V5_COLUMN_GROUP, FireEventMLAssessmentDB
from src.database.models.fire_event_news_evidence_db import FireEventNewsEvidenceDB
from src.database.models.fire_event_satellite_evidence_db import FireEventSatelliteEvidenceDB
from src.database.models.satellite_hotspot_db import SatelliteHotspotDB
from src.database.models.wildfire_report_db import WildfireReportDB
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_ml_rule_agreement import FireDetectionMLRuleAgreement
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_event import FireEvent
from src.models.fire_event_ml_assessment import FireEventMLAssessment
from src.models.fire_event_status import FireEventStatus
from src.models.fire_event_response_eligibility import ACTIVE_FOR_MONITORING_STATUSES, RESPONSE_ELIGIBLE_STATUSES
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.repositories.exceptions import FireEventRepositoryError
from src.repositories.fire_event_config import (
    ACTIVE_EVENT_MATCH_DISTANCE_KM,
    ACTIVE_EVENT_MATCH_WINDOW_HOURS,
)

logger = logging.getLogger(__name__)

_EARTH_RADIUS_KM = 6371.0088
_DISTANCE_TOLERANCE_KM = 1e-9
_ACTIVE_STATUSES = set(ACTIVE_FOR_MONITORING_STATUSES)  # active for detection / monitoring (NOT response eligibility)


@dataclass(frozen=True)
class StoredFireEvent:
    """Persisted FireEvent plus database identity and source-aware evidence refs.

    `created_at` is the row's own DB-insert timestamp (server-assigned, set
    once via the ORM column default at creation, never updated afterward -
    see FireEventRepository.update_event, which never touches it). It
    represents when EcoGuard actually opened/persisted this FireEvent,
    distinct from `event.detected_at` (the earliest correlated evidence's
    own observation time, which may be earlier).
    """

    id: int
    event: FireEvent
    supporting_evidence: tuple[FireEvidenceRef, ...] = ()
    created_at: datetime | None = None


# Per database pool (weakly held): whether fire_event_ml_assessments already has
# the ai_hybrid_v5 audit columns. Checked once per engine; the migration only
# adds columns, so a positive answer never goes stale while the process runs.
_ML_AI_COLUMNS_BY_POOL: "weakref.WeakKeyDictionary[object, bool]" = weakref.WeakKeyDictionary()
_ML_AI_COLUMN_NAMES = frozenset(
    ("policy_version", "policy_status", "history_available", "satellite_pass_count", "current_satellite_pixel_count")
)


def _ml_assessment_ai_columns_exist(session: Session) -> bool:
    bind = session.get_bind()
    pool = getattr(bind, "pool", None)
    if pool is not None and pool in _ML_AI_COLUMNS_BY_POOL:
        return _ML_AI_COLUMNS_BY_POOL[pool]
    try:
        columns = {column["name"] for column in inspect(session.connection()).get_columns("fire_event_ml_assessments")}
    except SQLAlchemyError:
        columns = set()
    available = _ML_AI_COLUMN_NAMES <= columns
    if pool is not None:
        _ML_AI_COLUMNS_BY_POOL[pool] = available
    return available


class FireEventRepository:
    """Persists FireEvent domain objects and evidence associations."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        self._session_factory = session_factory or get_session_factory()

    @contextmanager
    def _session_scope(self) -> Iterator[Session]:
        """Run a block of work in a session, committing on success and rolling back on error."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_event(
        self,
        event: FireEvent,
        supporting_evidence: tuple[FireEvidenceRef, ...],
    ) -> StoredFireEvent:
        """Atomically persist a new active event and its direct evidence refs."""
        self._validate_event(event)
        if event.status not in _ACTIVE_STATUSES:
            raise FireEventRepositoryError("Detection-created FireEvents must be SUSPECTED or CONFIRMED.")
        evidence_refs = self._normalize_evidence_refs(supporting_evidence)
        if not evidence_refs:
            raise FireEventRepositoryError("FireEvent creation requires at least one supporting evidence ref.")

        with self._session_scope() as session:
            db_event = self._to_db_event(event)
            session.add(db_event)
            try:
                session.flush()
                self._insert_missing_evidence_refs(session, db_event.id, evidence_refs)
                session.flush()
            except (IntegrityError, SQLAlchemyError) as exc:
                raise FireEventRepositoryError("FireEvent persistence failed.") from exc

            logger.info("Stored FireEvent %s with %s evidence refs", db_event.id, len(evidence_refs))
            return self._to_stored_event(db_event, evidence_refs)

    def get_by_id(self, fire_event_id: int) -> StoredFireEvent | None:
        """Return a persisted FireEvent by id, or None when no row exists."""
        self._validate_fire_event_id(fire_event_id)
        with self._session_scope() as session:
            db_event = self._get_db_event(session, fire_event_id)
            if db_event is None:
                return None
            return self._to_stored_event(db_event, self._get_evidence_refs(session, fire_event_id))

    def get_evidence_refs(self, fire_event_id: int) -> tuple[FireEvidenceRef, ...]:
        """Return source-aware evidence refs for a FireEvent in deterministic order."""
        self._validate_fire_event_id(fire_event_id)
        with self._session_scope() as session:
            return self._get_evidence_refs(session, fire_event_id)

    def attach_evidence(
        self,
        fire_event_id: int,
        evidence: tuple[FireEvidenceRef, ...],
    ) -> tuple[FireEvidenceRef, ...]:
        """Attach new evidence refs idempotently and return all stored refs."""
        self._validate_fire_event_id(fire_event_id)
        evidence_refs = self._normalize_evidence_refs(evidence)

        with self._session_scope() as session:
            if self._get_db_event(session, fire_event_id) is None:
                raise FireEventRepositoryError(f"FireEvent {fire_event_id!r} was not found.")
            try:
                self._insert_missing_evidence_refs(session, fire_event_id, evidence_refs)
                session.flush()
            except (IntegrityError, SQLAlchemyError) as exc:
                raise FireEventRepositoryError("FireEvent evidence attachment failed.") from exc
            return self._get_evidence_refs(session, fire_event_id)

    def update_event(self, fire_event_id: int, event: FireEvent) -> StoredFireEvent:
        """Update caller-supplied event fields without recalculating confidence or status."""
        self._validate_fire_event_id(fire_event_id)
        self._validate_event(event)

        with self._session_scope() as session:
            db_event = self._get_db_event(session, fire_event_id)
            if db_event is None:
                raise FireEventRepositoryError(f"FireEvent {fire_event_id!r} was not found.")

            db_event.latitude = event.latitude
            db_event.longitude = event.longitude
            db_event.updated_at = event.updated_at
            db_event.status = event.status.value
            db_event.detection_confidence = event.detection_confidence
            try:
                session.flush()
            except SQLAlchemyError as exc:
                raise FireEventRepositoryError("FireEvent update failed.") from exc

            return self._to_stored_event(db_event, self._get_evidence_refs(session, fire_event_id))

    def upsert_ml_assessment(self, fire_event_id: int, assessment: FireEventMLAssessment) -> FireEventMLAssessment:
        """Insert or update the single ML/decision trace row for a FireEvent (Task 5).

        One row per FireEvent - reevaluation overwrites it in place, it is
        never a growing history log. Raises if the FireEvent does not exist.
        """
        self._validate_fire_event_id(fire_event_id)
        if not isinstance(assessment, FireEventMLAssessment):
            raise FireEventRepositoryError(f"assessment must be a FireEventMLAssessment, got {assessment!r}.")
        if assessment.fire_event_id != fire_event_id:
            raise FireEventRepositoryError("assessment.fire_event_id must match fire_event_id.")

        with self._session_scope() as session:
            return self._upsert_ml_assessment_in_session(session, fire_event_id, assessment)

    def _upsert_ml_assessment_in_session(
        self, session: Session, fire_event_id: int, assessment: FireEventMLAssessment
    ) -> FireEventMLAssessment:
        """The upsert itself, on the caller's session (no commit): shared by the standalone and the atomic paths."""
        if self._get_db_event(session, fire_event_id) is None:
            raise FireEventRepositoryError(f"FireEvent {fire_event_id!r} was not found.")

        db_assessment = self._get_db_ml_assessment(session, fire_event_id)
        is_ai_row = assessment.decision_mode is FireDetectionDecisionMode.AI_HYBRID_V5
        if db_assessment is None and not is_ai_row:
            # A legacy-mode row is inserted with an explicit column list that never names the Task 9B columns: an ORM
            # INSERT would list every mapped column (as NULL), which would break legacy modes on a database that
            # has not been migrated yet.
            try:
                session.execute(
                    insert(FireEventMLAssessmentDB.__table__).values(
                        fire_event_id=fire_event_id,
                        decision_mode=assessment.decision_mode.value,
                        rule_status=assessment.rule_status.value,
                        rule_confidence=assessment.rule_confidence,
                        ml_available=assessment.ml_available,
                        ml_probability=assessment.ml_probability,
                        ml_model_name=assessment.ml_model_name,
                        ml_model_version=assessment.ml_model_version,
                        ml_feature_schema_version=assessment.ml_feature_schema_version,
                        ml_failure_reason=assessment.ml_failure_reason,
                        agreement=assessment.agreement.value,
                        updated_at=assessment.updated_at,
                    )
                )
                session.flush()
            except SQLAlchemyError as exc:
                raise FireEventRepositoryError("FireEvent ML assessment upsert failed.") from exc
            return assessment
        if db_assessment is None:
            db_assessment = FireEventMLAssessmentDB(fire_event_id=fire_event_id)
            session.add(db_assessment)

        db_assessment.decision_mode = assessment.decision_mode.value
        db_assessment.rule_status = assessment.rule_status.value
        db_assessment.rule_confidence = assessment.rule_confidence
        db_assessment.ml_available = assessment.ml_available
        db_assessment.ml_probability = assessment.ml_probability
        db_assessment.ml_model_name = assessment.ml_model_name
        db_assessment.ml_model_version = assessment.ml_model_version
        db_assessment.ml_feature_schema_version = assessment.ml_feature_schema_version
        db_assessment.ml_failure_reason = assessment.ml_failure_reason
        db_assessment.agreement = assessment.agreement.value
        db_assessment.updated_at = assessment.updated_at
        if is_ai_row:
            # Only AI_HYBRID_V5 writes/reads the Task 9B audit columns (see FireEventMLAssessmentDB).
            db_assessment.policy_version = assessment.policy_version
            db_assessment.policy_status = assessment.policy_status.value if assessment.policy_status else None
            db_assessment.history_available = assessment.history_available
            db_assessment.satellite_pass_count = assessment.satellite_pass_count
            db_assessment.current_satellite_pixel_count = assessment.current_satellite_pixel_count

        try:
            session.flush()
        except SQLAlchemyError as exc:
            raise FireEventRepositoryError("FireEvent ML assessment upsert failed.") from exc

        return self._to_domain_ml_assessment(db_assessment)

    def create_event_with_ml_assessment(
        self,
        event: FireEvent,
        supporting_evidence: tuple[FireEvidenceRef, ...],
        assessment_factory,
    ) -> StoredFireEvent:
        """Atomically persist a new event, its evidence refs AND its ML assessment row (Task 9C).

        `assessment_factory(fire_event_id)` builds the assessment once the id exists. Everything happens in ONE
        transaction: if the assessment cannot be stored the FireEvent is not committed either.
        """
        self._validate_event(event)
        if event.status not in _ACTIVE_STATUSES:
            raise FireEventRepositoryError("Detection-created FireEvents must be SUSPECTED or CONFIRMED.")
        evidence_refs = self._normalize_evidence_refs(supporting_evidence)
        if not evidence_refs:
            raise FireEventRepositoryError("FireEvent creation requires at least one supporting evidence ref.")

        with self._session_scope() as session:
            db_event = self._to_db_event(event)
            session.add(db_event)
            try:
                session.flush()
                self._insert_missing_evidence_refs(session, db_event.id, evidence_refs)
                session.flush()
            except (IntegrityError, SQLAlchemyError) as exc:
                raise FireEventRepositoryError("FireEvent persistence failed.") from exc
            assessment = assessment_factory(db_event.id)
            self._validate_assessment(db_event.id, assessment)
            self._upsert_ml_assessment_in_session(session, db_event.id, assessment)
            logger.info("Stored FireEvent %s with %s evidence refs and its ML assessment", db_event.id, len(evidence_refs))
            return self._to_stored_event(db_event, evidence_refs)

    def update_event_with_ml_assessment(
        self,
        fire_event_id: int,
        *,
        event: FireEvent | None,
        new_evidence: tuple[FireEvidenceRef, ...],
        assessment: FireEventMLAssessment | None,
    ) -> None:
        """Atomically apply an existing event's status/confidence update, new evidence refs and ML assessment (Task 9C).

        Any of the three parts may be absent (nothing to change), but the parts that are present commit together or not at
        all - e.g. a SUSPECTED -> CONFIRMED promotion is never left behind by a failed assessment write.
        """
        self._validate_fire_event_id(fire_event_id)
        if event is not None:
            self._validate_event(event)
        evidence_refs = self._normalize_evidence_refs(new_evidence)
        if assessment is not None:
            self._validate_assessment(fire_event_id, assessment)

        with self._session_scope() as session:
            db_event = self._get_db_event(session, fire_event_id)
            if db_event is None:
                raise FireEventRepositoryError(f"FireEvent {fire_event_id!r} was not found.")
            try:
                if evidence_refs:
                    self._insert_missing_evidence_refs(session, fire_event_id, evidence_refs)
                if event is not None:
                    db_event.latitude = event.latitude
                    db_event.longitude = event.longitude
                    db_event.updated_at = event.updated_at
                    db_event.status = event.status.value
                    db_event.detection_confidence = event.detection_confidence
                session.flush()
            except (IntegrityError, SQLAlchemyError) as exc:
                raise FireEventRepositoryError("FireEvent update failed.") from exc
            if assessment is not None:
                self._upsert_ml_assessment_in_session(session, fire_event_id, assessment)

    @staticmethod
    def _validate_assessment(fire_event_id: int, assessment: FireEventMLAssessment) -> None:
        if not isinstance(assessment, FireEventMLAssessment):
            raise FireEventRepositoryError(f"assessment must be a FireEventMLAssessment, got {assessment!r}.")
        if assessment.fire_event_id != fire_event_id:
            raise FireEventRepositoryError("assessment.fire_event_id must match fire_event_id.")

    def get_ml_assessment(self, fire_event_id: int) -> FireEventMLAssessment | None:
        """Return the FireEvent's latest ML/decision trace, or None if it was never evaluated with ML."""
        self._validate_fire_event_id(fire_event_id)
        with self._session_scope() as session:
            db_assessment = self._get_db_ml_assessment(session, fire_event_id)
            return self._to_domain_ml_assessment(db_assessment) if db_assessment is not None else None

    def get_ml_assessments_for_events(self, fire_event_ids: Iterable[int]) -> dict[int, FireEventMLAssessment]:
        """Return each event's ML/decision trace, batched in one query.

        `fire_event_ml_assessments` already has one row per FireEvent (a
        unique FK, upserted in place - see FireEventMLAssessmentDB), so this
        is a plain `WHERE fire_event_id IN (...)`, not a "latest of many"
        window-function query like FireSeverityAssessmentRepository's
        get_latest_for_events - there is only ever one row per event to
        begin with. Added to avoid one get_ml_assessment() call per event
        when listing many active FireEvents (dashboard Active Fire cards).
        FireEvent ids with no persisted assessment are simply absent from
        the returned mapping - never a fabricated entry.
        """
        ids = self._normalize_fire_event_ids("fire_event_ids", tuple(fire_event_ids))
        if not ids:
            return {}
        with self._session_scope() as session:
            db_assessments = (
                session.execute(
                    self._select_ml_assessments(session).where(FireEventMLAssessmentDB.fire_event_id.in_(ids))
                )
                .scalars()
                .all()
            )
            return {
                db_assessment.fire_event_id: self._to_domain_ml_assessment(db_assessment)
                for db_assessment in db_assessments
            }

    def find_matching_active_event(
        self,
        latitude: float,
        longitude: float,
        observed_at: datetime,
    ) -> StoredFireEvent | None:
        """Find a nearby active event using updated_at as the event recency timestamp.

        Eligible matches must be active, within ACTIVE_EVENT_MATCH_DISTANCE_KM,
        and within ACTIVE_EVENT_MATCH_WINDOW_HOURS of updated_at. Multiple
        matches are ordered by distance, time difference, most recent update,
        then smallest event id.
        """
        self._validate_coordinate("latitude", latitude, -90, 90)
        self._validate_coordinate("longitude", longitude, -180, 180)
        self._validate_aware_datetime("observed_at", observed_at)
        window = timedelta(hours=ACTIVE_EVENT_MATCH_WINDOW_HOURS)
        start_time = observed_at - window
        end_time = observed_at + window

        with self._session_scope() as session:
            db_events = (
                session.execute(
                    select(FireEventDB)
                    .options(
                        selectinload(FireEventDB.satellite_evidence),
                        selectinload(FireEventDB.news_evidence),
                    )
                    .where(
                        FireEventDB.status.in_(status.value for status in _ACTIVE_STATUSES),
                        FireEventDB.updated_at >= start_time,
                        FireEventDB.updated_at <= end_time,
                    )
                )
                .scalars()
                .all()
            )

            matches = []
            for db_event in db_events:
                distance_km = self._haversine_distance_km(
                    latitude,
                    longitude,
                    db_event.latitude,
                    db_event.longitude,
                )
                if distance_km > ACTIVE_EVENT_MATCH_DISTANCE_KM + _DISTANCE_TOLERANCE_KM:
                    continue
                event_updated_at = self._ensure_aware_datetime(db_event.updated_at)
                time_difference_seconds = abs((observed_at - event_updated_at).total_seconds())
                matches.append(
                    (
                        distance_km,
                        time_difference_seconds,
                        -event_updated_at.timestamp(),
                        db_event.id,
                        db_event,
                    )
                )

            if not matches:
                return None
            db_event = sorted(matches, key=lambda match: match[:4])[0][4]
            return self._to_stored_event(db_event, self._get_evidence_refs(session, db_event.id))

    def get_active_events_near(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
        as_of: datetime,
    ) -> tuple[StoredFireEvent, ...]:
        """Return events ACTIVE FOR MONITORING (SUSPECTED/CONFIRMED) near a coordinate at a deterministic instant.

        This is NOT the set that may trigger emergency response - see get_response_eligible_events_near.
        """
        return self._events_near(latitude, longitude, radius_km, as_of, _ACTIVE_STATUSES)

    def get_response_eligible_events_near(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
        as_of: datetime,
    ) -> tuple[StoredFireEvent, ...]:
        """Return only RESPONSE-ELIGIBLE (CONFIRMED) events near a coordinate: the only ones that may drive
        severity / spread / targets / routing / allocation / planning (Task 9A)."""
        return self._events_near(latitude, longitude, radius_km, as_of, RESPONSE_ELIGIBLE_STATUSES)

    def _events_near(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
        as_of: datetime,
        statuses,
    ) -> tuple[StoredFireEvent, ...]:
        self._validate_coordinate("latitude", latitude, -90, 90)
        self._validate_coordinate("longitude", longitude, -180, 180)
        self._validate_radius_km(radius_km)
        self._validate_aware_datetime("as_of", as_of)

        with self._session_scope() as session:
            db_events = (
                session.execute(
                    select(FireEventDB)
                    .options(
                        selectinload(FireEventDB.satellite_evidence),
                        selectinload(FireEventDB.news_evidence),
                    )
                    .where(
                        FireEventDB.status.in_(status.value for status in statuses),
                        FireEventDB.detected_at <= as_of,
                    )
                )
                .scalars()
                .all()
            )

            matches = []
            for db_event in db_events:
                distance_km = self._haversine_distance_km(
                    latitude,
                    longitude,
                    db_event.latitude,
                    db_event.longitude,
                )
                if distance_km > radius_km + _DISTANCE_TOLERANCE_KM:
                    continue
                matches.append((distance_km, db_event.id, db_event))

            return tuple(
                self._to_stored_event(match[2], self._get_evidence_refs(session, match[2].id))
                for match in sorted(matches, key=lambda match: (match[0], match[1]))
            )

    def get_active_events(self) -> tuple[StoredFireEvent, ...]:
        """Return all currently active (SUSPECTED/CONFIRMED) FireEvents.

        Unscoped by location, like get_active_fire_event_ids, but returns
        full FireEvent data instead of bare ids. Ordered most-recently-
        updated first (updated_at desc), tie-broken by id desc, for
        deterministic dashboard-style listing (US 6.1). Evidence traces are
        not loaded here (supporting_evidence is left empty on each result)
        since this listing doesn't need them - use get_by_id/
        get_evidence_refs for a specific event's evidence.
        """
        with self._session_scope() as session:
            db_events = (
                session.execute(
                    select(FireEventDB)
                    .where(FireEventDB.status.in_(status.value for status in _ACTIVE_STATUSES))
                    .order_by(FireEventDB.updated_at.desc(), FireEventDB.id.desc())
                )
                .scalars()
                .all()
            )
            return tuple(self._to_stored_event(db_event) for db_event in db_events)

    def get_active_event_ids_with_only_marked_evidence_updated_before(
        self,
        *,
        updated_before: datetime,
        satellite_name: str,
        news_source_feed: str,
    ) -> tuple[int, ...]:
        """Return ids of active FireEvents last updated strictly before
        `updated_before` whose supporting evidence ALL carries the given
        source markers: every satellite hotspot has `satellite ==
        satellite_name` and every news report has `source_feed ==
        news_source_feed`.

        Conservative by construction: an event with no evidence, or with any
        hotspot/report whose marker differs or is NULL, is never returned.
        The markers are supplied by the caller, so this repository stays
        agnostic of what they identify (e.g. simulation-generated evidence).
        Ordered by id.
        """
        self._validate_aware_datetime("updated_before", updated_before)
        for field_name, value in (("satellite_name", satellite_name), ("news_source_feed", news_source_feed)):
            if not isinstance(value, str) or not value.strip():
                raise FireEventRepositoryError(f"{field_name} must be a non-empty string, got {value!r}.")

        has_satellite = (
            select(FireEventSatelliteEvidenceDB.id)
            .where(FireEventSatelliteEvidenceDB.fire_event_id == FireEventDB.id)
            .exists()
        )
        has_news = (
            select(FireEventNewsEvidenceDB.id)
            .where(FireEventNewsEvidenceDB.fire_event_id == FireEventDB.id)
            .exists()
        )
        has_unmarked_satellite = (
            select(FireEventSatelliteEvidenceDB.id)
            .join(SatelliteHotspotDB, SatelliteHotspotDB.id == FireEventSatelliteEvidenceDB.satellite_hotspot_id)
            .where(
                FireEventSatelliteEvidenceDB.fire_event_id == FireEventDB.id,
                or_(SatelliteHotspotDB.satellite.is_(None), SatelliteHotspotDB.satellite != satellite_name),
            )
            .exists()
        )
        has_unmarked_news = (
            select(FireEventNewsEvidenceDB.id)
            .join(WildfireReportDB, WildfireReportDB.id == FireEventNewsEvidenceDB.wildfire_report_id)
            .where(
                FireEventNewsEvidenceDB.fire_event_id == FireEventDB.id,
                or_(WildfireReportDB.source_feed.is_(None), WildfireReportDB.source_feed != news_source_feed),
            )
            .exists()
        )

        with self._session_scope() as session:
            return tuple(
                session.execute(
                    select(FireEventDB.id)
                    .where(
                        FireEventDB.status.in_(status.value for status in _ACTIVE_STATUSES),
                        FireEventDB.updated_at < updated_before,
                        or_(has_satellite, has_news),
                        ~has_unmarked_satellite,
                        ~has_unmarked_news,
                    )
                    .order_by(FireEventDB.id)
                )
                .scalars()
                .all()
            )

    def get_recent(self, limit: int) -> tuple[StoredFireEvent, ...]:
        """Return the `limit` most recently detected FireEvents, newest first.

        Task A6: unlike get_active_events, this is NOT scoped to SUSPECTED/
        CONFIRMED - the Activity Feed shows recent FireEvent activity
        regardless of current status (a resolved fire is still a real past
        event). Ordered by detected_at desc, id desc - the same timestamp
        A5's FireEvent activity detail uses as occurred_at. Evidence traces
        are not loaded here (supporting_evidence is left empty on each
        result), matching get_active_events's own precedent, since the feed
        preview doesn't need them.
        """
        self._validate_limit(limit)
        with self._session_scope() as session:
            db_events = (
                session.execute(
                    select(FireEventDB)
                    .order_by(FireEventDB.detected_at.desc(), FireEventDB.id.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            return tuple(self._to_stored_event(db_event) for db_event in db_events)

    def get_active_fire_event_ids(self) -> tuple[int, ...]:
        """Return ids of all currently active (SUSPECTED/CONFIRMED) FireEvents.

        Unscoped by location: unlike get_active_events_near, this has no
        geographic filter. It exists for callers that need the full active
        set - e.g. identifying which FireEvents' planning may depend on a
        resource-wide status change, where no fixed-radius relationship
        between a resource and a FireEvent already exists in this model.
        Ordered by id for deterministic iteration.
        """
        with self._session_scope() as session:
            return tuple(
                sorted(
                    session.execute(
                        select(FireEventDB.id).where(
                            FireEventDB.status.in_(status.value for status in _ACTIVE_STATUSES)
                        )
                    ).scalars()
                )
            )

    def get_response_eligible_fire_event_ids(self) -> tuple[int, ...]:
        """Return ids of all RESPONSE-ELIGIBLE (CONFIRMED) FireEvents, ordered by id (Task 9A).

        The set global planning, resource allocation and resource commitment operate on. SUSPECTED events are
        active for monitoring (get_active_fire_event_ids) but are deliberately absent here.
        """
        with self._session_scope() as session:
            return tuple(
                sorted(
                    session.execute(
                        select(FireEventDB.id).where(
                            FireEventDB.status.in_(status.value for status in RESPONSE_ELIGIBLE_STATUSES)
                        )
                    ).scalars()
                )
            )

    def _insert_missing_evidence_refs(
        self,
        session: Session,
        fire_event_id: int,
        evidence_refs: tuple[FireEvidenceRef, ...],
    ) -> None:
        existing_refs = set(self._get_evidence_refs(session, fire_event_id))
        for evidence_ref in evidence_refs:
            if evidence_ref in existing_refs:
                continue
            if evidence_ref.evidence_type is FireEvidenceType.SATELLITE:
                session.add(
                    FireEventSatelliteEvidenceDB(
                        fire_event_id=fire_event_id,
                        satellite_hotspot_id=evidence_ref.evidence_id,
                    )
                )
            elif evidence_ref.evidence_type is FireEvidenceType.NEWS:
                session.add(
                    FireEventNewsEvidenceDB(
                        fire_event_id=fire_event_id,
                        wildfire_report_id=evidence_ref.evidence_id,
                    )
                )
            else:
                raise FireEventRepositoryError(f"Unsupported evidence type: {evidence_ref.evidence_type!r}.")
            existing_refs.add(evidence_ref)

    @staticmethod
    def _to_db_event(event: FireEvent) -> FireEventDB:
        return FireEventDB(
            latitude=event.latitude,
            longitude=event.longitude,
            detected_at=event.detected_at,
            updated_at=event.updated_at,
            status=event.status.value,
            detection_confidence=event.detection_confidence,
            methodology=event.methodology,
            methodology_version=event.methodology_version,
            location_name=event.location_name,
        )

    @classmethod
    def _to_stored_event(
        cls,
        db_event: FireEventDB,
        evidence_refs: tuple[FireEvidenceRef, ...] = (),
    ) -> StoredFireEvent:
        return StoredFireEvent(
            id=db_event.id,
            event=FireEvent(
                latitude=db_event.latitude,
                longitude=db_event.longitude,
                detected_at=cls._ensure_aware_datetime(db_event.detected_at),
                updated_at=cls._ensure_aware_datetime(db_event.updated_at),
                status=FireEventStatus(db_event.status),
                detection_confidence=db_event.detection_confidence,
                methodology=db_event.methodology,
                methodology_version=db_event.methodology_version,
                location_name=db_event.location_name,
            ),
            supporting_evidence=evidence_refs,
            created_at=cls._ensure_aware_datetime(db_event.created_at),
        )

    @staticmethod
    def _get_db_ml_assessment(session: Session, fire_event_id: int) -> FireEventMLAssessmentDB | None:
        return session.execute(
            FireEventRepository._select_ml_assessments(session).where(
                FireEventMLAssessmentDB.fire_event_id == fire_event_id
            )
        ).scalar_one_or_none()

    @staticmethod
    def _select_ml_assessments(session: Session):  # type: ignore[no-untyped-def]
        """SELECT of ML assessment rows; the deferred ai_hybrid_v5 audit columns load WITH the row when they exist.

        Those columns stay deferred so a database that has not run the
        ai_hybrid_v5 migration keeps working (legacy modes never reference
        them, and ai_hybrid_v5 fails with its actionable migration message).
        On a migrated database, reading them lazily cost one extra round trip
        per column per row, so they are loaded with the row instead - decided
        by whether the columns exist, never by the configured decision mode.
        """
        statement = select(FireEventMLAssessmentDB)
        if _ml_assessment_ai_columns_exist(session):
            statement = statement.options(undefer_group(AI_HYBRID_V5_COLUMN_GROUP))
        return statement

    @classmethod
    def _to_domain_ml_assessment(cls, db_assessment: FireEventMLAssessmentDB) -> FireEventMLAssessment:
        decision_mode = FireDetectionDecisionMode(db_assessment.decision_mode)
        ai_fields: dict = {}
        if decision_mode is FireDetectionDecisionMode.AI_HYBRID_V5:
            # Deferred Task 9B columns: read ONLY for ai_hybrid_v5 rows so legacy rows never need them.
            ai_fields = {
                "policy_version": db_assessment.policy_version,
                "policy_status": (
                    FireDetectionStatus(db_assessment.policy_status) if db_assessment.policy_status else None
                ),
                "history_available": db_assessment.history_available,
                "satellite_pass_count": db_assessment.satellite_pass_count,
                "current_satellite_pixel_count": db_assessment.current_satellite_pixel_count,
            }
        return FireEventMLAssessment(
            fire_event_id=db_assessment.fire_event_id,
            decision_mode=decision_mode,
            rule_status=FireDetectionStatus(db_assessment.rule_status),
            rule_confidence=db_assessment.rule_confidence,
            ml_available=db_assessment.ml_available,
            ml_probability=db_assessment.ml_probability,
            ml_model_name=db_assessment.ml_model_name,
            ml_model_version=db_assessment.ml_model_version,
            ml_feature_schema_version=db_assessment.ml_feature_schema_version,
            ml_failure_reason=db_assessment.ml_failure_reason,
            agreement=FireDetectionMLRuleAgreement(db_assessment.agreement),
            updated_at=cls._ensure_aware_datetime(db_assessment.updated_at),
            **ai_fields,
        )

    @staticmethod
    def _get_db_event(session: Session, fire_event_id: int) -> FireEventDB | None:
        return (
            session.execute(
                select(FireEventDB)
                .options(
                    selectinload(FireEventDB.satellite_evidence),
                    selectinload(FireEventDB.news_evidence),
                )
                .where(FireEventDB.id == fire_event_id)
            )
            .scalars()
            .one_or_none()
        )

    @classmethod
    def _get_evidence_refs(cls, session: Session, fire_event_id: int) -> tuple[FireEvidenceRef, ...]:
        satellite_refs = [
            FireEvidenceRef(FireEvidenceType.SATELLITE, satellite_hotspot_id)
            for satellite_hotspot_id in session.execute(
                select(FireEventSatelliteEvidenceDB.satellite_hotspot_id).where(
                    FireEventSatelliteEvidenceDB.fire_event_id == fire_event_id
                )
            ).scalars()
        ]
        news_refs = [
            FireEvidenceRef(FireEvidenceType.NEWS, wildfire_report_id)
            for wildfire_report_id in session.execute(
                select(FireEventNewsEvidenceDB.wildfire_report_id).where(
                    FireEventNewsEvidenceDB.fire_event_id == fire_event_id
                )
            ).scalars()
        ]
        return cls._sort_evidence_refs(tuple(satellite_refs + news_refs))

    @staticmethod
    def _normalize_evidence_refs(evidence_refs: Iterable[FireEvidenceRef]) -> tuple[FireEvidenceRef, ...]:
        try:
            refs = tuple(evidence_refs)
        except TypeError as exc:
            raise FireEventRepositoryError("evidence refs must be iterable.") from exc
        for evidence_ref in refs:
            if not isinstance(evidence_ref, FireEvidenceRef):
                raise FireEventRepositoryError(f"evidence must contain FireEvidenceRef items, got {evidence_ref!r}.")
        if len(set(refs)) != len(refs):
            raise FireEventRepositoryError("evidence refs must not contain duplicates.")
        return FireEventRepository._sort_evidence_refs(refs)

    @staticmethod
    def _sort_evidence_refs(evidence_refs: tuple[FireEvidenceRef, ...]) -> tuple[FireEvidenceRef, ...]:
        return tuple(sorted(evidence_refs, key=lambda ref: (ref.evidence_type.value, ref.evidence_id)))

    @staticmethod
    def _validate_event(event: FireEvent) -> None:
        if not isinstance(event, FireEvent):
            raise FireEventRepositoryError(f"event must be a FireEvent, got {event!r}.")

    @staticmethod
    def _validate_fire_event_id(fire_event_id: int) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise FireEventRepositoryError(f"fire_event_id must be a positive integer, got {fire_event_id!r}.")

    @staticmethod
    def _normalize_fire_event_ids(field_name: str, ids: tuple[int, ...]) -> tuple[int, ...]:
        try:
            values = tuple(ids)
        except TypeError as exc:
            raise FireEventRepositoryError(f"{field_name} must be iterable.") from exc
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise FireEventRepositoryError(f"{field_name} must contain positive integer ids, got {value!r}.")
        return values

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise FireEventRepositoryError(f"Invalid limit: {limit!r}. Must be a positive integer.")

    @staticmethod
    def _validate_aware_datetime(field_name: str, value: object) -> None:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise FireEventRepositoryError(f"{field_name} must be a timezone-aware datetime, got {value!r}.")

    @staticmethod
    def _validate_radius_km(radius_km: object) -> None:
        if isinstance(radius_km, bool) or not isinstance(radius_km, (int, float)) or not math.isfinite(radius_km):
            raise FireEventRepositoryError(f"radius_km must be a finite number, got {radius_km!r}.")
        if radius_km <= 0:
            raise FireEventRepositoryError(f"radius_km must be greater than 0, got {radius_km!r}.")

    @staticmethod
    def _validate_coordinate(field_name: str, value: object, minimum: float, maximum: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise FireEventRepositoryError(f"{field_name} must be a finite number, got {value!r}.")
        if not minimum <= value <= maximum:
            raise FireEventRepositoryError(f"{field_name} must be within [{minimum}, {maximum}], got {value!r}.")

    @staticmethod
    def _ensure_aware_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @staticmethod
    def _haversine_distance_km(
        first_latitude: float,
        first_longitude: float,
        second_latitude: float,
        second_longitude: float,
    ) -> float:
        first_latitude_rad = math.radians(first_latitude)
        second_latitude_rad = math.radians(second_latitude)
        latitude_delta = math.radians(second_latitude - first_latitude)
        longitude_delta = math.radians(second_longitude - first_longitude)
        a = (
            math.sin(latitude_delta / 2) ** 2
            + math.cos(first_latitude_rad)
            * math.cos(second_latitude_rad)
            * math.sin(longitude_delta / 2) ** 2
        )
        return _EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

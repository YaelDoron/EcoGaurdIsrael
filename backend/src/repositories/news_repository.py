"""Persistence layer for wildfire news reports."""
from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from sqlalchemy import case, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.database.connection import get_session_factory
from src.database.models.wildfire_report_db import WildfireReportDB
from src.models.fire_report import WildfireReport
from src.repositories.exceptions import NewsRepositoryError

logger = logging.getLogger(__name__)

_ISO_OFFSET_WITHOUT_MINUTES_RE = re.compile(r"([+-]\d{2})$")


@dataclass(frozen=True)
class SaveNewsReportResult:
    """Result of NewsRepository.save_report()."""

    report: WildfireReport
    is_duplicate: bool


@dataclass(frozen=True)
class StoredWildfireReport:
    """Persisted wildfire report with database identity and evidence timestamp."""

    id: int
    report: WildfireReport
    observed_at: datetime


class NewsRepository:
    """Persists and retrieves WildfireReport objects via SQLAlchemy."""

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

    def save_report(self, report: WildfireReport) -> SaveNewsReportResult:
        """Insert a news report unless its source URL already exists."""
        with self._session_scope() as session:
            existing = self._find_by_source_url(session, report.source_url)
            if existing is not None:
                logger.info("Skipped duplicate wildfire news report: %s", report.source_url)
                return SaveNewsReportResult(report=self._to_domain_report(existing), is_duplicate=True)

            db_report = WildfireReportDB(
                source_url=report.source_url,
                source_feed=report.source_feed,
                title=report.title,
                summary=report.summary,
                location_name=report.location_name,
                latitude=report.latitude,
                longitude=report.longitude,
                published_at=report.published_at,
                fetched_at=report.fetched_at,
            )
            session.add(db_report)

            try:
                session.flush()
            except IntegrityError:
                session.rollback()
                logger.info("Duplicate wildfire news report detected on write: %s", report.source_url)
                existing = self._find_by_source_url(session, report.source_url)
                if existing is None:
                    raise NewsRepositoryError(
                        "Wildfire news report write failed and duplicate row could not be found."
                    ) from None
                return SaveNewsReportResult(report=self._to_domain_report(existing), is_duplicate=True)
            except SQLAlchemyError as exc:
                raise NewsRepositoryError("Wildfire news report write failed.") from exc

            logger.info("Stored wildfire news report: %s", report.source_url)
            return SaveNewsReportResult(report=self._to_domain_report(db_report), is_duplicate=False)

    def exists_by_source_url(self, source_url: str) -> bool:
        """Return True when a report with the given source URL is already stored."""
        with self._session_scope() as session:
            return self._find_by_source_url(session, source_url) is not None

    def get_by_source_url(self, source_url: str) -> WildfireReport | None:
        """Return a report by source URL, or None if no report is stored."""
        with self._session_scope() as session:
            db_report = self._find_by_source_url(session, source_url)
            return self._to_domain_report(db_report) if db_report is not None else None

    def get_recent_reports(
        self,
        as_of: datetime,
        lookback_minutes: int,
    ) -> list[StoredWildfireReport]:
        """Return persisted reports by published_at, falling back to fetched_at, newest first."""
        self._validate_recent_query(as_of, lookback_minutes)
        start_time = as_of - timedelta(minutes=lookback_minutes)
        observed_at = case(
            (WildfireReportDB.published_at.is_not(None), WildfireReportDB.published_at),
            else_=WildfireReportDB.fetched_at,
        )

        with self._session_scope() as session:
            rows = (
                session.execute(
                    select(WildfireReportDB, observed_at.label("observed_at"))
                    .where(
                        observed_at >= start_time,
                        observed_at <= as_of,
                    )
                    .order_by(observed_at.desc(), WildfireReportDB.id.desc())
                )
                .all()
            )
            return [
                StoredWildfireReport(
                    id=db_report.id,
                    report=self._to_domain_report(db_report),
                    observed_at=self._ensure_aware_datetime(row_observed_at),
                )
                for db_report, row_observed_at in rows
            ]

    def get_recent(self, limit: int) -> tuple[StoredWildfireReport, ...]:
        """Return the `limit` most recent reports by observed_at, newest first.

        Task A6: a bounded "recent across all reports" read for the
        Activity Feed - distinct from get_recent_reports's time-windowed
        query. Same observed_at computed-column semantics (published_at,
        falling back to fetched_at) and the same deterministic tie-break
        (observed_at desc, id desc) as get_recent_reports/get_by_id.
        """
        self._validate_limit(limit)
        observed_at = case(
            (WildfireReportDB.published_at.is_not(None), WildfireReportDB.published_at),
            else_=WildfireReportDB.fetched_at,
        )
        with self._session_scope() as session:
            rows = (
                session.execute(
                    select(WildfireReportDB, observed_at.label("observed_at"))
                    .order_by(observed_at.desc(), WildfireReportDB.id.desc())
                    .limit(limit)
                )
                .all()
            )
            return tuple(
                StoredWildfireReport(
                    id=db_report.id,
                    report=self._to_domain_report(db_report),
                    observed_at=self._ensure_aware_datetime(row_observed_at),
                )
                for db_report, row_observed_at in rows
            )

    def get_by_id(self, report_id: int) -> StoredWildfireReport | None:
        """Return a persisted wildfire report by database id, or None if absent."""
        self._validate_report_id(report_id)
        with self._session_scope() as session:
            db_report = session.get(WildfireReportDB, report_id)
            if db_report is None:
                return None
            return StoredWildfireReport(
                id=db_report.id,
                report=self._to_domain_report(db_report),
                observed_at=self._report_observed_at(db_report),
            )

    @staticmethod
    def _find_by_source_url(session: Session, source_url: str) -> WildfireReportDB | None:
        return session.execute(
            select(WildfireReportDB).where(WildfireReportDB.source_url == source_url)
        ).scalar_one_or_none()

    @staticmethod
    def _to_domain_report(db_report: WildfireReportDB) -> WildfireReport:
        return WildfireReport(
            source_url=db_report.source_url,
            source_feed=db_report.source_feed,
            title=db_report.title,
            summary=db_report.summary,
            location_name=db_report.location_name,
            latitude=db_report.latitude,
            longitude=db_report.longitude,
            published_at=NewsRepository._ensure_aware_datetime(db_report.published_at),
            fetched_at=NewsRepository._ensure_aware_datetime(db_report.fetched_at),
        )

    @staticmethod
    def _ensure_aware_datetime(value: datetime | str | None) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = NewsRepository._parse_legacy_datetime_string(value)
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @staticmethod
    def _report_observed_at(db_report: WildfireReportDB) -> datetime:
        observed_at = db_report.published_at if db_report.published_at is not None else db_report.fetched_at
        return NewsRepository._ensure_aware_datetime(observed_at)

    @staticmethod
    def _parse_legacy_datetime_string(value: str) -> datetime:
        normalized = value.strip()
        if not normalized:
            raise NewsRepositoryError("Stored wildfire news timestamp is empty.")

        iso_candidate = _ISO_OFFSET_WITHOUT_MINUTES_RE.sub(r"\1:00", normalized.replace("Z", "+00:00"))
        try:
            return datetime.fromisoformat(iso_candidate)
        except ValueError:
            pass

        try:
            return parsedate_to_datetime(normalized)
        except (TypeError, ValueError) as exc:
            raise NewsRepositoryError(f"Stored wildfire news timestamp is invalid: {value!r}") from exc

    @staticmethod
    def _validate_recent_query(as_of: datetime, lookback_minutes: int) -> None:
        if not isinstance(as_of, datetime):
            raise NewsRepositoryError(f"as_of must be a datetime, got {as_of!r}.")
        if (
            isinstance(lookback_minutes, bool)
            or not isinstance(lookback_minutes, int)
            or lookback_minutes <= 0
        ):
            raise NewsRepositoryError(
                f"lookback_minutes must be a positive integer, got {lookback_minutes!r}."
            )

    @staticmethod
    def _validate_report_id(report_id: int) -> None:
        if isinstance(report_id, bool) or not isinstance(report_id, int) or report_id <= 0:
            raise NewsRepositoryError(f"report_id must be a positive integer, got {report_id!r}.")

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise NewsRepositoryError(f"Invalid limit: {limit!r}. Must be a positive integer.")

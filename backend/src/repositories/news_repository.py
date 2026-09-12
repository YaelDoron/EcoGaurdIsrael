"""Persistence layer for wildfire news reports."""
from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from sqlalchemy import select
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

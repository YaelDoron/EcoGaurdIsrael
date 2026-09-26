"""ML assessment reads load the ai_hybrid_v5 audit columns with the row when the database has them.

The five Task 9B columns are deferred so an un-migrated database keeps
working (legacy modes never reference them; ai_hybrid_v5 then fails with its
actionable migration message). On a migrated database they were lazy-loaded
one round trip per column per row - 20 extra remote round trips for 4 active
fire cards - so they are now loaded with the row there. The decision depends
on the columns existing, never on the configured decision mode.
"""
from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from src.database.base import Base
from src.database.models.fire_event_ml_assessment_db import FireEventMLAssessmentDB  # noqa: F401 - registers the table
from src.repositories.fire_event_repository import FireEventRepository

AI_COLUMNS = (
    "policy_version",
    "policy_status",
    "history_available",
    "satellite_pass_count",
    "current_satellite_pixel_count",
)


def _compiled_select(session: Session) -> str:
    statement = FireEventRepository._select_ml_assessments(session)  # noqa: SLF001
    return str(statement.compile(dialect=postgresql.dialect()))


def test_migrated_database_selects_the_audit_columns_in_the_same_query():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine, tables=[FireEventMLAssessmentDB.__table__])

    with Session(engine) as session:
        sql = _compiled_select(session)

    for column in AI_COLUMNS:
        assert f"fire_event_ml_assessments.{column}" in sql


def test_unmigrated_database_keeps_the_audit_columns_deferred():
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE fire_event_ml_assessments (id INTEGER PRIMARY KEY, fire_event_id INTEGER)"))

    with Session(engine) as session:
        sql = _compiled_select(session)

    for column in AI_COLUMNS:
        assert f"fire_event_ml_assessments.{column}" not in sql


def test_the_column_check_runs_once_per_engine():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine, tables=[FireEventMLAssessmentDB.__table__])
    statements: list[str] = []

    from sqlalchemy import event

    @event.listens_for(engine, "before_cursor_execute")
    def _record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    with Session(engine) as session:
        _compiled_select(session)
        first = len(statements)
        _compiled_select(session)
        _compiled_select(session)

    assert first >= 1
    assert len(statements) == first

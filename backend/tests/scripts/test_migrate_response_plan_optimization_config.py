"""Tests for scripts/migrate_response_plan_optimization_config.py.

Column-addition idempotency is validated against a real SQLAlchemy engine
(SQLite) - `inspect(connection).get_columns()`/`ALTER TABLE ... ADD COLUMN`
are backend-agnostic enough for this control flow to be genuinely exercised.

The CHECK constraint step is NOT validated against SQLite: SQLite does not
support `ALTER TABLE ... ADD CONSTRAINT` at all (confirmed directly - it is
a hard syntax error), so pretending SQLite can execute that DDL would be
dishonest test coverage. Instead, `add_missing_constraint`'s decision logic
(add when missing, skip when already present, and the exact SQL emitted) is
tested against a lightweight fake connection/inspector - i.e. by mocking the
database connection, per the audit's explicitly allowed approach - and the
constraint's own SQL syntax was separately verified correct for PostgreSQL
by static compilation of the ORM model (see the audit report), not by
executing it here.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError

import scripts.migrate_response_plan_optimization_config as migration_script

TABLE_NAME = migration_script.TABLE_NAME
CONSTRAINT_NAME = migration_script.CONSTRAINT_NAME
COLUMN_NAMES = tuple(name for name, _ in migration_script.COLUMNS)


@pytest.fixture
def sqlite_engine():
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


def create_legacy_table(engine) -> None:
    """A response_plans table shaped like the pre-migration (live Neon) schema."""
    with engine.begin() as connection:
        connection.execute(
            text(
                f"CREATE TABLE {TABLE_NAME} ("
                "id INTEGER PRIMARY KEY, "
                "fire_event_id INTEGER NOT NULL, "
                "route_planning_run_id INTEGER NOT NULL, "
                "random_seed INTEGER NOT NULL, "
                "status TEXT NOT NULL"
                ")"
            )
        )


# ---------------------------------------------------------------------------
# add_missing_columns - real SQLite engine, real ALTER TABLE ADD COLUMN
# ---------------------------------------------------------------------------


def test_add_missing_columns_adds_all_eight_when_none_exist(sqlite_engine):
    create_legacy_table(sqlite_engine)

    with sqlite_engine.begin() as connection:
        added = migration_script.add_missing_columns(connection)

    assert set(added) == set(COLUMN_NAMES)
    with sqlite_engine.connect() as connection:
        columns = {c["name"] for c in inspect(connection).get_columns(TABLE_NAME)}
    assert set(COLUMN_NAMES) <= columns


def test_add_missing_columns_is_a_no_op_when_all_already_exist(sqlite_engine):
    create_legacy_table(sqlite_engine)
    with sqlite_engine.begin() as connection:
        migration_script.add_missing_columns(connection)

    with sqlite_engine.begin() as connection:
        second_added = migration_script.add_missing_columns(connection)

    assert second_added == ()


def test_add_missing_columns_running_twice_in_a_row_is_safe(sqlite_engine):
    create_legacy_table(sqlite_engine)

    with sqlite_engine.begin() as connection:
        migration_script.add_missing_columns(connection)
    with sqlite_engine.begin() as connection:
        migration_script.add_missing_columns(connection)  # must not raise (e.g. duplicate-column error)

    with sqlite_engine.connect() as connection:
        columns = [c["name"] for c in inspect(connection).get_columns(TABLE_NAME)]
    for column_name in COLUMN_NAMES:
        assert columns.count(column_name) == 1


def test_add_missing_columns_adds_only_the_missing_subset(sqlite_engine):
    create_legacy_table(sqlite_engine)
    with sqlite_engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE {TABLE_NAME} ADD COLUMN population_size INTEGER"))
        connection.execute(text(f"ALTER TABLE {TABLE_NAME} ADD COLUMN generation_count INTEGER"))

    with sqlite_engine.begin() as connection:
        added = migration_script.add_missing_columns(connection)

    assert "population_size" not in added
    assert "generation_count" not in added
    assert set(added) == set(COLUMN_NAMES) - {"population_size", "generation_count"}


# ---------------------------------------------------------------------------
# add_missing_constraint - decision logic only, via a mocked connection
# (SQLite cannot execute ALTER TABLE ADD CONSTRAINT at all - confirmed).
# ---------------------------------------------------------------------------


class _FakeCheckConstraintInspector:
    def __init__(self, constraint_names) -> None:
        self._constraint_names = tuple(constraint_names)

    def get_check_constraints(self, table_name):
        assert table_name == TABLE_NAME
        return [{"name": name} for name in self._constraint_names]


class _RecordingConnection:
    """Fake connection: records executed SQL text without running anything."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    def execute(self, statement):
        self.executed.append(str(statement))


def test_add_missing_constraint_adds_it_when_absent(monkeypatch):
    monkeypatch.setattr(
        migration_script, "inspect", lambda connection: _FakeCheckConstraintInspector(constraint_names=())
    )
    connection = _RecordingConnection()

    added = migration_script.add_missing_constraint(connection)

    assert added is True
    assert len(connection.executed) == 1
    assert f"ADD CONSTRAINT {CONSTRAINT_NAME}" in connection.executed[0]
    assert "CHECK" in connection.executed[0]
    for column_name in COLUMN_NAMES:
        assert column_name in connection.executed[0]


def test_add_missing_constraint_is_a_no_op_when_already_present(monkeypatch):
    monkeypatch.setattr(
        migration_script,
        "inspect",
        lambda connection: _FakeCheckConstraintInspector(constraint_names=(CONSTRAINT_NAME,)),
    )
    connection = _RecordingConnection()

    added = migration_script.add_missing_constraint(connection)

    assert added is False
    assert connection.executed == []


def test_sqlite_does_not_support_add_constraint_confirming_the_mock_is_necessary(sqlite_engine):
    """Documents WHY add_missing_constraint cannot be exercised end-to-end
    against SQLite: ALTER TABLE ADD CONSTRAINT is a hard syntax error there."""
    create_legacy_table(sqlite_engine)

    with pytest.raises(OperationalError):
        with sqlite_engine.begin() as connection:
            connection.execute(text(f"ALTER TABLE {TABLE_NAME} ADD CONSTRAINT ck_x CHECK (1 = 1)"))


# ---------------------------------------------------------------------------
# main() - status output
# ---------------------------------------------------------------------------


def test_main_prints_a_status_line_for_every_column_and_the_constraint(monkeypatch, sqlite_engine, capsys):
    create_legacy_table(sqlite_engine)
    monkeypatch.setattr(migration_script, "get_engine", lambda: sqlite_engine)
    # add_missing_constraint would fail against real SQLite (see test above);
    # stub it so main()'s orchestration/printing is exercised without that.
    monkeypatch.setattr(migration_script, "add_missing_constraint", lambda connection: True)

    exit_code = migration_script.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    for column_name in COLUMN_NAMES:
        assert f"{TABLE_NAME}.{column_name}: added" in output
    assert f"{CONSTRAINT_NAME}: added" in output

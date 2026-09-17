"""Tests for scripts/migrate_epic_5_foreign_keys.py.

Idempotency-relevant decision logic (does the table exist, does the
referenced table exist, does the constraint already exist) is validated
against a real SQLAlchemy engine (SQLite) - table/FK introspection is
backend-agnostic enough for this control flow to be genuinely exercised.

The actual `ALTER TABLE ... ADD CONSTRAINT ... FOREIGN KEY` statement is NOT
validated against SQLite: SQLite does not support adding a foreign key to
an existing table via ALTER TABLE at all (confirmed directly below - it is
a hard syntax error), so pretending SQLite can execute that DDL would be
dishonest test coverage. Instead, the add/failure-handling logic is
exercised through this same real SQLite engine deliberately taking the
"failed" branch (SQLite raises on the DDL every time), proving the
per-constraint try/except does not abort the rest of the run - and the
happy-path "added" outcome plus the constraint's SQL syntax are separately
verified correct for PostgreSQL by the live Neon migration itself, not by
executing it here.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

import scripts.migrate_epic_5_foreign_keys as migration_script

FOREIGN_KEYS = migration_script.FOREIGN_KEYS


@pytest.fixture
def sqlite_engine():
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


def create_full_chain(engine) -> None:
    """A minimal set of tables shaped like the live (pre-FK) Epic 5 schema:
    every column FOREIGN_KEYS references already exists, just without the
    constraint - exactly the state migrate_response_plan_optimization_config
    and the pre-FND-05 ORM models left the live Neon schema in."""
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE fire_events (id INTEGER PRIMARY KEY)"))
        connection.execute(text("CREATE TABLE response_target_sets (id INTEGER PRIMARY KEY)"))
        connection.execute(text("CREATE TABLE firefighting_resources (id TEXT PRIMARY KEY)"))
        connection.execute(
            text("CREATE TABLE route_planning_runs (id INTEGER PRIMARY KEY, fire_event_id INTEGER)")
        )
        connection.execute(
            text(
                "CREATE TABLE response_plans (id INTEGER PRIMARY KEY, route_planning_run_id INTEGER, "
                "fire_event_id INTEGER, response_target_set_id INTEGER)"
            )
        )
        connection.execute(text("CREATE TABLE response_targets (id INTEGER PRIMARY KEY)"))
        connection.execute(
            text("CREATE TABLE route_results (id INTEGER PRIMARY KEY, resource_id TEXT, response_target_id INTEGER)")
        )
        connection.execute(
            text(
                "CREATE TABLE response_actions (id INTEGER PRIMARY KEY, resource_id TEXT, "
                "response_target_id INTEGER, route_result_id INTEGER)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE plan_comparisons (id INTEGER PRIMARY KEY, fire_event_id INTEGER, "
                "optimized_plan_id INTEGER, route_planning_run_id INTEGER, response_target_set_id INTEGER)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE response_plan_uncovered_targets (id INTEGER PRIMARY KEY, response_target_id INTEGER)"
            )
        )


# ---------------------------------------------------------------------------
# Coverage of the FOREIGN_KEYS table itself
# ---------------------------------------------------------------------------


def test_foreign_keys_table_has_all_ten_entries_with_unique_names():
    assert len(FOREIGN_KEYS) == 10
    names = [entry[0] for entry in FOREIGN_KEYS]
    assert len(names) == len(set(names))


def test_foreign_keys_table_covers_every_column_identified_in_fnd_05():
    covered = {(table, column) for _, table, column, _, _ in FOREIGN_KEYS}
    assert covered == {
        ("response_plans", "route_planning_run_id"),
        ("response_actions", "resource_id"),
        ("response_actions", "response_target_id"),
        ("response_actions", "route_result_id"),
        ("route_results", "resource_id"),
        ("plan_comparisons", "fire_event_id"),
        ("plan_comparisons", "optimized_plan_id"),
        ("plan_comparisons", "route_planning_run_id"),
        ("plan_comparisons", "response_target_set_id"),
        ("response_plan_uncovered_targets", "response_target_id"),
    }


# ---------------------------------------------------------------------------
# add_missing_foreign_keys - table-existence decision logic (real SQLite)
# ---------------------------------------------------------------------------


def test_skips_every_entry_when_no_tables_exist(sqlite_engine):
    results = migration_script.add_missing_foreign_keys(sqlite_engine)

    assert len(results) == len(FOREIGN_KEYS)
    assert all(outcome.startswith("skipped") for outcome in results.values())


def test_skips_when_only_the_child_table_is_missing(sqlite_engine):
    with sqlite_engine.begin() as connection:
        connection.execute(text("CREATE TABLE fire_events (id INTEGER PRIMARY KEY)"))
        connection.execute(text("CREATE TABLE response_target_sets (id INTEGER PRIMARY KEY)"))
        connection.execute(text("CREATE TABLE firefighting_resources (id TEXT PRIMARY KEY)"))
        connection.execute(text("CREATE TABLE route_planning_runs (id INTEGER PRIMARY KEY)"))
        connection.execute(text("CREATE TABLE response_plans (id INTEGER PRIMARY KEY)"))
        # response_actions itself never created.

    results = migration_script.add_missing_foreign_keys(sqlite_engine)

    assert results["response_actions_resource_id_fkey"] == "skipped ('response_actions' table does not exist)"


def test_skips_when_only_the_referenced_table_is_missing(sqlite_engine):
    with sqlite_engine.begin() as connection:
        connection.execute(text("CREATE TABLE response_plans (id INTEGER PRIMARY KEY, route_planning_run_id INTEGER)"))
        # route_planning_runs itself never created.

    results = migration_script.add_missing_foreign_keys(sqlite_engine)

    assert results["response_plans_route_planning_run_id_fkey"] == (
        "skipped (referenced table 'route_planning_runs' does not exist)"
    )


# ---------------------------------------------------------------------------
# add_missing_foreign_keys - per-constraint failure isolation
# ---------------------------------------------------------------------------


def test_every_constraint_addition_is_attempted_and_reported_independently(sqlite_engine):
    """SQLite cannot execute ALTER TABLE ADD CONSTRAINT ... FOREIGN KEY at
    all (see module docstring / the syntax-error test below), so every
    entry in FOREIGN_KEYS fails here - but critically, ALL TEN are still
    attempted and reported, proving one failure never aborts the rest."""
    create_full_chain(sqlite_engine)

    results = migration_script.add_missing_foreign_keys(sqlite_engine)

    assert len(results) == len(FOREIGN_KEYS)
    assert all(outcome.startswith("failed") for outcome in results.values())
    for constraint_name, table_name, column_name, ref_table, ref_column in FOREIGN_KEYS:
        assert table_name in results[constraint_name]
        assert column_name in results[constraint_name]
        assert ref_table in results[constraint_name]


def test_running_twice_in_a_row_reports_the_same_failures_both_times(sqlite_engine):
    """Not idempotent in the sense of succeeding, but safe to re-run: no
    crash, no duplicate side effects, same per-constraint report each time."""
    create_full_chain(sqlite_engine)

    first = migration_script.add_missing_foreign_keys(sqlite_engine)
    second = migration_script.add_missing_foreign_keys(sqlite_engine)

    assert set(first) == set(second)
    assert all(outcome.startswith("failed") for outcome in second.values())


def test_sqlite_does_not_support_alter_table_add_foreign_key(sqlite_engine):
    """Documents WHY add_missing_foreign_keys cannot be exercised
    end-to-end to a successful "added" outcome against SQLite."""
    create_full_chain(sqlite_engine)

    with pytest.raises(OperationalError):
        with sqlite_engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE response_plans ADD CONSTRAINT fk_x "
                    "FOREIGN KEY (route_planning_run_id) REFERENCES route_planning_runs (id)"
                )
            )


# ---------------------------------------------------------------------------
# add_missing_foreign_keys - already-present short-circuit (fake inspector)
# ---------------------------------------------------------------------------


class _FakeForeignKeyInspector:
    def __init__(self, existing_names) -> None:
        self._existing_names = set(existing_names)

    def has_table(self, table_name):
        return True

    def get_foreign_keys(self, table_name):
        return [{"name": name} for name in self._existing_names]


def test_constraint_exists_true_when_name_present():
    inspector = _FakeForeignKeyInspector(existing_names=("response_plans_route_planning_run_id_fkey",))

    assert migration_script._constraint_exists(  # noqa: SLF001
        inspector, "response_plans", "response_plans_route_planning_run_id_fkey"
    )


def test_constraint_exists_false_when_name_absent():
    inspector = _FakeForeignKeyInspector(existing_names=())

    assert not migration_script._constraint_exists(  # noqa: SLF001
        inspector, "response_plans", "response_plans_route_planning_run_id_fkey"
    )


# ---------------------------------------------------------------------------
# main() - status output and exit code
# ---------------------------------------------------------------------------


def test_main_prints_a_status_line_for_every_constraint_and_reports_failure(monkeypatch, sqlite_engine, capsys):
    create_full_chain(sqlite_engine)
    monkeypatch.setattr(migration_script, "get_engine", lambda: sqlite_engine)

    exit_code = migration_script.main()

    assert exit_code == 1  # SQLite can't apply any of them -> non-zero, nothing silently swallowed
    output = capsys.readouterr().out
    for constraint_name, _, _, _, _ in FOREIGN_KEYS:
        assert f"{constraint_name}: failed" in output


def test_main_reports_success_when_every_constraint_is_already_present(monkeypatch, sqlite_engine, capsys):
    monkeypatch.setattr(migration_script, "get_engine", lambda: sqlite_engine)
    monkeypatch.setattr(
        migration_script,
        "add_missing_foreign_keys",
        lambda engine: {name: "already present" for name, _, _, _, _ in FOREIGN_KEYS},
    )

    exit_code = migration_script.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    for constraint_name, _, _, _, _ in FOREIGN_KEYS:
        assert f"{constraint_name}: already present" in output

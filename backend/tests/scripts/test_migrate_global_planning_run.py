"""Tests for scripts/migrate_global_planning_run.py (Stage 2 of the Global
Multi-Incident Optimizer refactor).

CREATE TABLE / ADD COLUMN / CREATE INDEX are backend-agnostic enough to be
genuinely exercised against SQLite. ALTER TABLE ADD CONSTRAINT ... FOREIGN
KEY is NOT: SQLite has no support for it at all (a hard syntax error,
confirmed directly below), matching this project's established precedent
for response_plans (migrate_epic_5_foreign_keys.py,
migrate_response_plan_optimization_config.py) - the FK step is expected to
report "failed: OperationalError" here and is verified for real only
against Neon (see the Stage 2 final report's Neon section).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError

import scripts.migrate_global_planning_run as migration_script
from src.database.base import Base
from src.database.models import (  # noqa: F401
    fire_event_db,
    fire_station_db,
    firefighting_resource_db,
    response_plan_db,
    response_target_db,
    response_target_set_db,
    route_planning_run_db,
    route_result_db,
)
from src.database.models.global_planning_run_db import GlobalPlanningRunDB
from src.database.models.global_planning_run_event_db import GlobalPlanningRunEventDB

RUN_TABLE_NAME = migration_script.RUN_TABLE_NAME
EVENT_TABLE_NAME = migration_script.EVENT_TABLE_NAME
RESPONSE_PLANS_TABLE_NAME = migration_script.RESPONSE_PLANS_TABLE_NAME
COLUMN_NAME = migration_script.COLUMN_NAME
FK_CONSTRAINT_NAME = migration_script.FK_CONSTRAINT_NAME
INDEX_NAME = migration_script.INDEX_NAME


@pytest.fixture
def pre_migration_engine():
    """Every table except the two new ones and response_plans, plus a
    response_plans table shaped like the pre-Stage-2 (live Neon) schema -
    i.e. every current column EXCEPT global_planning_run_id.

    response_plans is rebuilt via raw SQL rather than the ORM's Table
    object (as migrate_response_plan_optimization_config.py's own
    create_legacy_table already does for this same table) because SQLite
    embeds a FOREIGN KEY clause inline in CREATE TABLE and then refuses
    ALTER TABLE ... DROP COLUMN on any column that participates in one -
    there is no way to get a same-shaped-minus-one-column response_plans
    table out of the real declarative Table object on SQLite.
    """
    engine = create_engine("sqlite:///:memory:")
    tables_except_new = [
        table
        for table in Base.metadata.sorted_tables
        if table.name not in (RUN_TABLE_NAME, EVENT_TABLE_NAME, RESPONSE_PLANS_TABLE_NAME)
    ]
    Base.metadata.create_all(bind=engine, tables=tables_except_new)
    with engine.begin() as connection:
        connection.execute(
            text(
                f"CREATE TABLE {RESPONSE_PLANS_TABLE_NAME} ("
                "id INTEGER PRIMARY KEY, "
                "fire_event_id INTEGER NOT NULL, "
                "response_target_set_id INTEGER NOT NULL, "
                "route_planning_run_id INTEGER NOT NULL, "
                "generated_at TIMESTAMP NOT NULL, "
                "status TEXT NOT NULL, "
                "methodology TEXT NOT NULL, "
                "methodology_version TEXT NOT NULL, "
                "random_seed INTEGER NOT NULL, "
                "plan_score FLOAT, "
                "coverage_score FLOAT, "
                "average_eta_seconds FLOAT, "
                "population_size INTEGER, "
                "generation_count INTEGER, "
                "mutation_rate FLOAT, "
                "crossover_rate FLOAT, "
                "eta_reference_seconds FLOAT, "
                "initial_assignment_probability FLOAT, "
                "tournament_size INTEGER, "
                "elitism_count INTEGER, "
                "created_at TIMESTAMP NOT NULL"
                ")"
            )
        )
    yield engine
    engine.dispose()


def test_pre_migration_schema_is_missing_everything_new(pre_migration_engine):
    with pre_migration_engine.connect() as connection:
        inspector = inspect(connection)
        assert not inspector.has_table(RUN_TABLE_NAME)
        assert not inspector.has_table(EVENT_TABLE_NAME)
        columns = {c["name"] for c in inspector.get_columns(RESPONSE_PLANS_TABLE_NAME)}
        assert COLUMN_NAME not in columns


# ---------------------------------------------------------------------------
# migrate() - table/column/index steps (real SQLite DDL)
# ---------------------------------------------------------------------------


def test_migrate_creates_both_new_tables(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    results = migration_script.migrate()

    assert results["global_planning_runs_table"] == "created"
    assert results["global_planning_run_events_table"] == "created"
    with pre_migration_engine.connect() as connection:
        inspector = inspect(connection)
        assert inspector.has_table(RUN_TABLE_NAME)
        assert inspector.has_table(EVENT_TABLE_NAME)


def test_migrate_creates_expected_run_table_columns(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)
    migration_script.migrate()

    with pre_migration_engine.connect() as connection:
        columns = {c["name"] for c in inspect(connection).get_columns(RUN_TABLE_NAME)}
    assert columns == {
        "id",
        "started_at",
        "completed_at",
        "status",
        "trigger",
        "methodology",
        "methodology_version",
        "input_fingerprint",
        # Stage 6: demand/shortage historical persistence (see
        # migrate_global_planning_history.py) - a genuinely fresh CREATE
        # TABLE already includes these, since they are part of the current
        # ORM model.
        "raw_input_fingerprint",
        "optimization_policy_fingerprint",
        "random_seed",
        "ga_population_size",
        "ga_generation_count",
        "ga_mutation_rate",
        "ga_crossover_rate",
        "demand_scoring_policy_methodology",
        "demand_scoring_policy_version",
        "severity_demand_policy_methodology",
        "severity_demand_policy_version",
        "stability_policy_methodology",
        "stability_policy_version",
        "fitness_score",
        "coverage_score",
        "average_eta_seconds",
        "shortage_total_required",
        "shortage_total_desired",
        "shortage_total_assigned",
        "shortage_unmet_required",
        "shortage_unmet_desired",
        "shortage_candidate_assignable_resource_count",
        "shortage_committed_resource_count",
        "shortage_unavailable_resource_count",
        "shortage_locked_resources_preserved",
        "created_at",
    }


def test_migrate_creates_expected_run_event_table_columns_pk_and_fks(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)
    migration_script.migrate()

    with pre_migration_engine.connect() as connection:
        inspector = inspect(connection)
        columns = {c["name"] for c in inspector.get_columns(EVENT_TABLE_NAME)}
        fks = inspector.get_foreign_keys(EVENT_TABLE_NAME)
    assert columns == {
        "id",
        "global_planning_run_id",
        "fire_event_id",
        "event_order",
        "result_status",
        "response_plan_id",
        "local_state_fingerprint",
        "error_code",
        # Stage 6: demand/shortage historical persistence (see
        # migrate_global_planning_history.py).
        "severity_assessment_id",
        "severity_level",
        "severity_score",
        "demand_source",
        "demand_policy_methodology",
        "demand_policy_version",
        "minimum_resources",
        "desired_resources",
        "assigned_resources",
        "required_slots_covered",
        "required_slots_uncovered",
        "desired_slots_covered",
        "desired_slots_uncovered",
        "predicted_risk_slots_covered",
        "locked_resources_preserved",
        "coverage_score",
        "average_eta_seconds",
        "created_at",
    }
    fk_pairs = {(tuple(fk["constrained_columns"]), fk["referred_table"]) for fk in fks}
    assert (("global_planning_run_id",), RUN_TABLE_NAME) in fk_pairs
    assert (("fire_event_id",), "fire_events") in fk_pairs
    assert (("response_plan_id",), RESPONSE_PLANS_TABLE_NAME) in fk_pairs


def test_migrate_adds_the_response_plans_column_as_nullable(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    results = migration_script.migrate()

    assert results["response_plans.global_planning_run_id_column"] == "added"
    with pre_migration_engine.connect() as connection:
        columns = {c["name"]: c for c in inspect(connection).get_columns(RESPONSE_PLANS_TABLE_NAME)}
    assert COLUMN_NAME in columns
    assert columns[COLUMN_NAME]["nullable"] is True


def test_migrate_adds_the_response_plans_index(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    results = migration_script.migrate()

    assert results["response_plans.global_planning_run_id_index"] == "added"
    with pre_migration_engine.connect() as connection:
        indexed_columns = {
            column
            for index in inspect(connection).get_indexes(RESPONSE_PLANS_TABLE_NAME)
            for column in index["column_names"]
        }
    assert COLUMN_NAME in indexed_columns


# ---------------------------------------------------------------------------
# migrate() - FK step: documented SQLite limitation, graceful failure
# ---------------------------------------------------------------------------


def test_migrate_reports_the_fk_step_as_failed_on_sqlite_without_aborting_the_rest(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    results = migration_script.migrate()

    assert results["response_plans.global_planning_run_id_fk"].startswith("failed")
    # Every other step still succeeded despite the FK step failing.
    assert results["global_planning_runs_table"] == "created"
    assert results["global_planning_run_events_table"] == "created"
    assert results["response_plans.global_planning_run_id_column"] == "added"
    assert results["response_plans.global_planning_run_id_index"] == "added"


def test_sqlite_does_not_support_alter_table_add_foreign_key(pre_migration_engine):
    """Documents WHY the FK step cannot succeed against SQLite."""
    with pre_migration_engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE {RESPONSE_PLANS_TABLE_NAME} ADD COLUMN {COLUMN_NAME} INTEGER NULL"))

    with pytest.raises(OperationalError):
        with pre_migration_engine.begin() as connection:
            connection.execute(
                text(
                    f"ALTER TABLE {RESPONSE_PLANS_TABLE_NAME} ADD CONSTRAINT {FK_CONSTRAINT_NAME} "
                    f"FOREIGN KEY ({COLUMN_NAME}) REFERENCES {RUN_TABLE_NAME} (id)"
                )
            )


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_migrate_running_twice_in_a_row_is_safe_and_reports_already_present(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    migration_script.migrate()
    second = migration_script.migrate()

    assert second["global_planning_runs_table"] == "already present"
    assert second["global_planning_run_events_table"] == "already present"
    assert second["response_plans.global_planning_run_id_column"] == "already present"
    assert second["response_plans.global_planning_run_id_index"] == "already present"
    # FK never actually got added on SQLite, so it is attempted (and fails) every time - not "already present".
    assert second["response_plans.global_planning_run_id_fk"].startswith("failed")


def test_migrate_does_not_duplicate_columns_or_indexes_on_second_run(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    migration_script.migrate()
    migration_script.migrate()

    with pre_migration_engine.connect() as connection:
        inspector = inspect(connection)
        column_names = [c["name"] for c in inspector.get_columns(RESPONSE_PLANS_TABLE_NAME)]
        index_names = [index["name"] for index in inspector.get_indexes(RESPONSE_PLANS_TABLE_NAME)]
    assert column_names.count(COLUMN_NAME) == 1
    assert index_names.count(INDEX_NAME) == 1


# ---------------------------------------------------------------------------
# No backfill / historical plans preserved
# ---------------------------------------------------------------------------


def _insert_full_response_plan_chain(engine) -> int:
    """Insert one pre-existing FireEvent/ResponseTargetSet/RoutePlanningRun/
    ResponsePlan row using the pre-migration (no new column) schema.
    Returns the response_plans.id inserted."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO fire_events (latitude, longitude, detected_at, updated_at, status, "
                "detection_confidence, methodology, methodology_version, created_at) "
                "VALUES (32.7, 35.0, :now, :now, 'confirmed', 0.9, 'm', '1.0', :now)"
            ),
            {"now": now},
        )
        connection.execute(
            text(
                "INSERT INTO response_target_sets "
                "(fire_event_id, generated_at, methodology, methodology_version, created_at) "
                "VALUES (1, :now, 'm', '1.0', :now)"
            ),
            {"now": now},
        )
        connection.execute(
            text(
                "INSERT INTO route_planning_runs "
                "(fire_event_id, response_target_set_id, planned_at, methodology, methodology_version, "
                "resource_ids, created_at) "
                "VALUES (1, 1, :now, 'm', '1.0', '[]', :now)"
            ),
            {"now": now},
        )
        result = connection.execute(
            text(
                "INSERT INTO response_plans (fire_event_id, response_target_set_id, route_planning_run_id, "
                "generated_at, status, methodology, methodology_version, random_seed, created_at) "
                "VALUES (1, 1, 1, :now, 'complete', 'm', '1.0', 1, :now)"
            ),
            {"now": now},
        )
        return result.lastrowid if result.lastrowid is not None else 1


def test_migrate_preserves_historical_response_plans_with_null_global_run_id(monkeypatch, pre_migration_engine):
    plan_id = _insert_full_response_plan_chain(pre_migration_engine)
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    migration_script.migrate()

    with pre_migration_engine.connect() as connection:
        row = connection.execute(
            text(f"SELECT id, {COLUMN_NAME} FROM {RESPONSE_PLANS_TABLE_NAME} WHERE id = :id"), {"id": plan_id}
        ).one()
    assert row[0] == plan_id
    assert row[1] is None  # no backfill


# ---------------------------------------------------------------------------
# main() - status output
# ---------------------------------------------------------------------------


def test_main_prints_a_status_line_for_every_step(monkeypatch, pre_migration_engine, capsys):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    exit_code = migration_script.main()
    output = capsys.readouterr().out

    assert exit_code == 1  # the FK step fails on SQLite
    assert "global_planning_runs_table: created" in output
    assert "global_planning_run_events_table: created" in output
    assert "response_plans.global_planning_run_id_column: added" in output
    assert "response_plans.global_planning_run_id_fk: failed" in output
    assert "response_plans.global_planning_run_id_index: added" in output


def test_main_returns_zero_when_everything_already_present_and_fk_succeeded(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)
    monkeypatch.setattr(migration_script, "add_response_plans_fk", lambda connection: False)
    migration_script.migrate()
    # Re-patch after the first migrate() call used the real (failing) fk function for setup;
    # now simulate a database where the FK already exists (e.g. real Neon on a second run).

    exit_code = migration_script.main()

    assert exit_code == 0

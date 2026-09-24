"""Tests for scripts/migrate_add_fire_spread_insufficient_data_reason.py.

Uses plain ADD COLUMN / cross-dialect SQL, so - like the wildfire_signal_strength
migration - it is fully exercisable against a real SQLite engine. No live Neon
required.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text

import scripts.migrate_add_fire_spread_insufficient_data_reason as migration_script

TABLE_NAME = migration_script.TABLE_NAME
COLUMN_NAME = migration_script.COLUMN_NAME

# fire_spread_predictions as it exists on Neon before this migration - no reason column.
_PRE_MIGRATION_DDL = f"""
CREATE TABLE {TABLE_NAME} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fire_event_id INTEGER NOT NULL,
    severity_assessment_id INTEGER,
    predicted_at TEXT NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    status VARCHAR NOT NULL,
    methodology VARCHAR NOT NULL,
    methodology_version VARCHAR NOT NULL,
    effective_state_fingerprint VARCHAR(64),
    created_at TEXT NOT NULL
)
"""


@pytest.fixture
def pre_migration_engine():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text(_PRE_MIGRATION_DDL))
    yield engine
    engine.dispose()


@pytest.fixture
def post_migration_engine(pre_migration_engine):
    """An already-migrated database: the column is present."""
    with pre_migration_engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {COLUMN_NAME} VARCHAR NULL"))
    return pre_migration_engine


def _insert_legacy_row(engine, status: str = "insufficient_data") -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                f"INSERT INTO {TABLE_NAME} (fire_event_id, predicted_at, horizon_minutes, status, methodology, "
                "methodology_version, created_at) VALUES (474, '2026-09-23T19:11:19', 30, :status, "
                "'ECOGUARD_PROPAGATOR_CA', '1.0', '2026-09-23T19:11:30')"
            ),
            {"status": status},
        )


# --- column_exists / inspection ---


def test_column_does_not_exist_before_migration(pre_migration_engine):
    assert migration_script.column_exists(pre_migration_engine) is False


def test_column_exists_when_already_migrated(post_migration_engine):
    assert migration_script.column_exists(post_migration_engine) is True


def test_column_exists_raises_if_table_missing():
    engine = create_engine("sqlite:///:memory:")
    with pytest.raises(RuntimeError):
        migration_script.column_exists(engine)
    engine.dispose()


# --- --inspect-only ---


def test_inspect_only_detects_missing_column_and_changes_nothing(monkeypatch, pre_migration_engine, capsys):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    exit_code = migration_script.main(["--inspect-only"])

    assert exit_code == 0
    assert "MISSING" in capsys.readouterr().out
    assert migration_script.column_exists(pre_migration_engine) is False


def test_inspect_only_detects_existing_column(monkeypatch, post_migration_engine, capsys):
    monkeypatch.setattr(migration_script, "get_engine", lambda: post_migration_engine)

    assert migration_script.main(["--inspect-only"]) == 0
    assert "present" in capsys.readouterr().out


def test_mode_is_required():
    with pytest.raises(SystemExit):
        migration_script.main([])


# --- --apply ---


def test_apply_adds_one_nullable_column_without_default_or_index(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    assert migration_script.main(["--apply"]) == 0

    inspector = inspect(pre_migration_engine)
    column = next(c for c in inspector.get_columns(TABLE_NAME) if c["name"] == COLUMN_NAME)
    assert column["nullable"] is True
    assert column["default"] is None
    assert not any(COLUMN_NAME in index["column_names"] for index in inspector.get_indexes(TABLE_NAME))


def test_apply_is_idempotent_when_rerun(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    assert migration_script.main(["--apply"]) == 0
    assert migration_script.main(["--apply"]) == 0  # must not raise a duplicate-column error
    assert migration_script.column_exists(pre_migration_engine) is True


def test_migrate_is_a_no_op_when_already_present(monkeypatch, post_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: post_migration_engine)

    assert migration_script.migrate() is False


@pytest.mark.parametrize("status", ["insufficient_data", "valid", "inactive_event"])
def test_apply_never_backfills_historical_rows(monkeypatch, pre_migration_engine, status):
    _insert_legacy_row(pre_migration_engine, status=status)
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    migration_script.main(["--apply"])

    assert migration_script.row_count(pre_migration_engine) == 1
    with pre_migration_engine.connect() as connection:
        reason, kept_status = connection.execute(text(f"SELECT {COLUMN_NAME}, status FROM {TABLE_NAME}")).one()
    assert reason is None
    assert kept_status == status

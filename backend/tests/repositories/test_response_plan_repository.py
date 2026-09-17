"""Unit tests for ResponsePlanRepository using SQLite in-memory."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src.calculators.response_optimization.response_optimization_config import ResponseOptimizationConfig
from src.database.models.fire_event_db import FireEventDB
from src.database.models.response_action_db import ResponseActionDB
from src.database.models.response_plan_db import ResponsePlanDB
from src.database.models.response_plan_uncovered_target_db import ResponsePlanUncoveredTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.models import ResponseAction, ResponsePlan, ResponsePlanStatus
from src.repositories.exceptions import ResponsePlanRepositoryError
from src.repositories.response_plan_repository import ResponsePlanRepository, StoredResponsePlan

GENERATED_AT = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def repository(sqlite_session_factory) -> ResponsePlanRepository:
    return ResponsePlanRepository(sqlite_session_factory)


@pytest.fixture
def persisted_context(sqlite_session_factory):
    session = sqlite_session_factory()
    event = FireEventDB(
        latitude=32.731,
        longitude=35.046,
        detected_at=GENERATED_AT - timedelta(minutes=30),
        updated_at=GENERATED_AT - timedelta(minutes=5),
        status="confirmed",
        detection_confidence=0.9,
        methodology="TEST_DETECTION",
        methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    target_set = ResponseTargetSetDB(
        fire_event_id=event.id,
        generated_at=GENERATED_AT - timedelta(minutes=1),
        methodology="TEST_TARGETS",
        methodology_version="1.0",
    )
    session.add(target_set)
    session.commit()
    context = {"fire_event_id": event.id, "response_target_set_id": target_set.id}
    session.close()
    return context


def make_plan(context, **overrides) -> ResponsePlan:
    values = {
        "fire_event_id": context["fire_event_id"],
        "response_target_set_id": context["response_target_set_id"],
        "route_planning_run_id": 77,
        "generated_at": GENERATED_AT,
        "status": ResponsePlanStatus.COMPLETE,
        "methodology": "GENETIC_RESOURCE_ALLOCATION",
        "methodology_version": "1.0",
        "random_seed": 42,
        "actions": (
            ResponseAction("TRUCK-A", 10, 100),
            ResponseAction("TRUCK-B", 20, 200),
        ),
        "uncovered_target_ids": (),
        "plan_score": 81.5,
        "coverage_score": 100.0,
        "average_eta_seconds": 250.0,
    }
    values.update(overrides)
    return ResponsePlan(**values)


def rows(sqlite_session_factory, model):
    session = sqlite_session_factory()
    values = session.execute(select(model)).scalars().all()
    session.close()
    return values


def test_save_complete_response_plan_round_trips(repository, persisted_context, sqlite_session_factory):
    plan = make_plan(persisted_context)

    stored = repository.save(plan)

    assert isinstance(stored, StoredResponsePlan)
    assert stored.id > 0
    found = repository.get_by_id(stored.id)
    assert found.plan == plan
    assert found.plan.generated_at == GENERATED_AT
    assert len(rows(sqlite_session_factory, ResponseActionDB)) == 2


def test_save_partial_response_plan_with_uncovered_targets(repository, persisted_context):
    plan = make_plan(
        persisted_context,
        status=ResponsePlanStatus.PARTIAL,
        actions=(ResponseAction("TRUCK-A", 10, 100),),
        uncovered_target_ids=(20,),
        plan_score=50.0,
        coverage_score=60.0,
        average_eta_seconds=300.0,
    )

    stored = repository.save(plan)

    assert stored.plan.status is ResponsePlanStatus.PARTIAL
    assert stored.plan.uncovered_target_ids == (20,)


def test_save_no_feasible_assignments_with_zero_actions(repository, persisted_context):
    plan = make_plan(
        persisted_context,
        status=ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS,
        actions=(),
        uncovered_target_ids=(10, 20),
        plan_score=0.0,
        coverage_score=0.0,
        average_eta_seconds=None,
    )

    stored = repository.save(plan)

    assert stored.plan.actions == ()
    assert stored.plan.uncovered_target_ids == (10, 20)
    assert stored.plan.average_eta_seconds is None


def test_child_rows_belong_to_saved_plan(repository, persisted_context, sqlite_session_factory):
    stored = repository.save(make_plan(persisted_context))

    actions = rows(sqlite_session_factory, ResponseActionDB)

    assert {action.response_plan_id for action in actions} == {stored.id}


def test_scores_methodology_seed_and_uncovered_targets_round_trip(repository, persisted_context):
    plan = make_plan(
        persisted_context,
        status=ResponsePlanStatus.PARTIAL,
        actions=(ResponseAction("7", 10, 100),),
        uncovered_target_ids=(20, 30),
        random_seed=123,
        plan_score=42.25,
        coverage_score=50.0,
        average_eta_seconds=123.5,
    )

    found = repository.get_by_id(repository.save(plan).id)

    assert found.plan.random_seed == 123
    assert found.plan.methodology == "GENETIC_RESOURCE_ALLOCATION"
    assert found.plan.methodology_version == "1.0"
    assert found.plan.plan_score == pytest.approx(42.25)
    assert found.plan.coverage_score == pytest.approx(50.0)
    assert found.plan.average_eta_seconds == pytest.approx(123.5)
    assert found.plan.uncovered_target_ids == (20, 30)
    assert found.plan.actions[0].resource_id == "7"


# ---------------------------------------------------------------------------
# FND-04: optimization_config persistence and legacy-row semantics
# ---------------------------------------------------------------------------


def test_full_optimization_config_round_trips_exactly(repository, persisted_context):
    config = ResponseOptimizationConfig(
        population_size=20,
        generation_count=15,
        mutation_rate=0.19,
        crossover_rate=0.61,
        random_seed=55,
        eta_reference_seconds=444.0,
        initial_assignment_probability=0.5,
        tournament_size=3,
        elitism_count=2,
    )
    plan = make_plan(persisted_context, random_seed=55, optimization_config=config)

    found = repository.get_by_id(repository.save(plan).id)

    assert found.plan.optimization_config == config


def test_missing_optimization_config_round_trips_as_none(repository, persisted_context):
    plan = make_plan(persisted_context, optimization_config=None)

    found = repository.get_by_id(repository.save(plan).id)

    assert found.plan.optimization_config is None


def test_optimization_config_columns_persisted_all_or_none_at_the_db_row_level(
    repository, persisted_context, sqlite_session_factory
):
    config = ResponseOptimizationConfig(population_size=12, generation_count=8, random_seed=9)
    plan = make_plan(persisted_context, random_seed=9, optimization_config=config)
    stored_id = repository.save(plan).id

    session = sqlite_session_factory()
    db_row = session.get(ResponsePlanDB, stored_id)
    session.close()

    assert db_row.population_size == 12
    assert db_row.generation_count == 8
    assert db_row.mutation_rate == pytest.approx(config.mutation_rate)
    assert db_row.crossover_rate == pytest.approx(config.crossover_rate)
    assert db_row.eta_reference_seconds == pytest.approx(config.eta_reference_seconds)
    assert db_row.initial_assignment_probability == pytest.approx(config.initial_assignment_probability)
    assert db_row.tournament_size == config.tournament_size
    assert db_row.elitism_count == config.elitism_count


def test_genuinely_legacy_db_row_with_all_null_config_columns_loads_successfully(
    repository, persisted_context, sqlite_session_factory
):
    """Simulates a row persisted before FND-04 existed: inserted directly via
    the ORM model with every optimization-config column left NULL, bypassing
    ResponsePlanRepository.save/ResponsePlan entirely - exactly the shape a
    genuinely historical Neon row has."""
    session = sqlite_session_factory()
    legacy_row = ResponsePlanDB(
        fire_event_id=persisted_context["fire_event_id"],
        response_target_set_id=persisted_context["response_target_set_id"],
        route_planning_run_id=77,
        generated_at=GENERATED_AT,
        status="complete",
        methodology="GENETIC_RESOURCE_ALLOCATION",
        methodology_version="1.0",
        random_seed=1,
        plan_score=10.0,
        coverage_score=100.0,
        average_eta_seconds=120.0,
        # population_size, generation_count, ... all omitted -> NULL.
    )
    session.add(legacy_row)
    session.commit()
    legacy_id = legacy_row.id
    session.close()

    found = repository.get_by_id(legacy_id)

    assert found is not None
    assert found.plan.optimization_config is None
    assert found.plan.random_seed == 1  # random_seed itself predates this feature and is untouched


def test_history_excludes_other_fire_events_and_is_append_only(repository, persisted_context, sqlite_session_factory):
    first = repository.save(make_plan(persisted_context, generated_at=GENERATED_AT))
    second = repository.save(make_plan(persisted_context, generated_at=GENERATED_AT + timedelta(minutes=1)))

    other_context = persisted_context.copy()
    session = sqlite_session_factory()
    event = FireEventDB(
        latitude=32.0,
        longitude=35.0,
        detected_at=GENERATED_AT,
        updated_at=GENERATED_AT,
        status="confirmed",
        detection_confidence=0.8,
        methodology="TEST_DETECTION",
        methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    target_set = ResponseTargetSetDB(
        fire_event_id=event.id,
        generated_at=GENERATED_AT,
        methodology="TEST_TARGETS",
        methodology_version="1.0",
    )
    session.add(target_set)
    session.commit()
    other_context["fire_event_id"] = event.id
    other_context["response_target_set_id"] = target_set.id
    session.close()
    repository.save(make_plan(other_context, actions=(ResponseAction("TRUCK-Z", 99, 999),)))

    history = repository.get_for_fire_event(persisted_context["fire_event_id"])

    assert [item.id for item in history] == [second.id, first.id]
    assert repository.get_latest_for_fire_event(persisted_context["fire_event_id"]).id == second.id
    assert repository.get_by_id(first.id).plan == first.plan


def test_save_does_not_mutate_input_domain_object(repository, persisted_context):
    plan = make_plan(persisted_context)

    repository.save(plan)

    assert plan.actions == (ResponseAction("TRUCK-A", 10, 100), ResponseAction("TRUCK-B", 20, 200))
    assert plan.uncovered_target_ids == ()


def test_failure_while_saving_actions_rolls_back_entire_plan(
    repository,
    persisted_context,
    sqlite_session_factory,
    monkeypatch,
):
    plan = make_plan(persisted_context)

    def broken_action(response_plan_id, action_order, action):  # noqa: ANN001
        raise RuntimeError("action write failed")

    monkeypatch.setattr(ResponsePlanRepository, "_to_db_action", staticmethod(broken_action))

    with pytest.raises(RuntimeError):
        repository.save(plan)

    assert rows(sqlite_session_factory, ResponsePlanDB) == []
    assert rows(sqlite_session_factory, ResponseActionDB) == []
    assert rows(sqlite_session_factory, ResponsePlanUncoveredTargetDB) == []


def test_invalid_repository_arguments_rejected(repository):
    with pytest.raises(ResponsePlanRepositoryError):
        repository.save("not-a-plan")
    with pytest.raises(ResponsePlanRepositoryError):
        repository.get_by_id(0)
    with pytest.raises(ResponsePlanRepositoryError):
        repository.get_for_fire_event(True)
    with pytest.raises(ResponsePlanRepositoryError):
        repository.get_latest_for_fire_event(-1)

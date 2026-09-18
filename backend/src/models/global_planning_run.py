"""Domain model for one global planning cycle (Stage 2 of the Global
Multi-Incident Optimizer refactor).

A GlobalPlanningRun records that, at one instant, ALL then-active
FireEvents were considered together in one orchestration/audit boundary -
see src/services/global_planning/ for the full stage. Its `methodology` is
always "legacy_per_event_orchestration" for this stage (see
global_planning_config.py): the per-event routing/GA underneath is
completely unchanged, so this run is explicitly NOT a global-optimization
result and must never be mistaken for one.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real

from src.models.global_planning_run_status import GlobalPlanningRunStatus


@dataclass(frozen=True)
class GlobalPlanningRun:
    """One global planning cycle's bookkeeping record.

    The fields from `raw_input_fingerprint` through `shortage_locked_resources_preserved`
    (Stage 6, demand/shortage historical persistence) are the exact global
    optimization metadata and shortage snapshot for one successfully-planned
    cycle - see GlobalPlanningRunRepository.record_global_optimization_metadata.
    All None for a run that never reached the GA (NO_OP/NO_ACTIVE_EVENTS/pre-GA FAILED).
    """

    started_at: datetime
    completed_at: datetime | None
    status: GlobalPlanningRunStatus
    trigger: str
    methodology: str
    methodology_version: str
    input_fingerprint: str | None
    raw_input_fingerprint: str | None = None
    optimization_policy_fingerprint: str | None = None
    random_seed: int | None = None
    ga_population_size: int | None = None
    ga_generation_count: int | None = None
    ga_mutation_rate: float | None = None
    ga_crossover_rate: float | None = None
    demand_scoring_policy_methodology: str | None = None
    demand_scoring_policy_version: str | None = None
    severity_demand_policy_methodology: str | None = None
    severity_demand_policy_version: str | None = None
    stability_policy_methodology: str | None = None
    stability_policy_version: str | None = None
    fitness_score: float | None = None
    coverage_score: float | None = None
    average_eta_seconds: float | None = None
    shortage_total_required: int | None = None
    shortage_total_desired: int | None = None
    shortage_total_assigned: int | None = None
    shortage_unmet_required: int | None = None
    shortage_unmet_desired: int | None = None
    shortage_candidate_assignable_resource_count: int | None = None
    shortage_committed_resource_count: int | None = None
    shortage_unavailable_resource_count: int | None = None
    shortage_locked_resources_preserved: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.started_at, datetime) or self.started_at.tzinfo is None:
            raise ValueError(f"started_at must be a timezone-aware datetime, got {self.started_at!r}")
        if self.completed_at is not None:
            if not isinstance(self.completed_at, datetime) or self.completed_at.tzinfo is None:
                raise ValueError(f"completed_at must be a timezone-aware datetime, got {self.completed_at!r}")
        if not isinstance(self.status, GlobalPlanningRunStatus):
            raise ValueError(f"status must be a GlobalPlanningRunStatus, got {self.status!r}")
        if not isinstance(self.trigger, str) or not self.trigger.strip():
            raise ValueError(f"trigger must be a non-empty string, got {self.trigger!r}")
        if not isinstance(self.methodology, str) or not self.methodology.strip():
            raise ValueError(f"methodology must be a non-empty string, got {self.methodology!r}")
        if not isinstance(self.methodology_version, str) or not self.methodology_version.strip():
            raise ValueError(f"methodology_version must be a non-empty string, got {self.methodology_version!r}")
        if self.input_fingerprint is not None and (
            not isinstance(self.input_fingerprint, str) or not self.input_fingerprint.strip()
        ):
            raise ValueError(f"input_fingerprint must be a non-empty string or None, got {self.input_fingerprint!r}")

        for field_name in (
            "raw_input_fingerprint",
            "optimization_policy_fingerprint",
            "demand_scoring_policy_methodology",
            "demand_scoring_policy_version",
            "severity_demand_policy_methodology",
            "severity_demand_policy_version",
            "stability_policy_methodology",
            "stability_policy_version",
        ):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-empty string or None, got {value!r}")

        if self.random_seed is not None and (isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int)):
            raise ValueError(f"random_seed must be an integer or None, got {self.random_seed!r}")

        for field_name in (
            "ga_population_size",
            "ga_generation_count",
            "shortage_total_required",
            "shortage_total_desired",
            "shortage_total_assigned",
            "shortage_unmet_required",
            "shortage_unmet_desired",
            "shortage_candidate_assignable_resource_count",
            "shortage_committed_resource_count",
            "shortage_unavailable_resource_count",
            "shortage_locked_resources_preserved",
        ):
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{field_name} must be a non-negative integer or None, got {value!r}")

        for field_name in ("ga_mutation_rate", "ga_crossover_rate"):
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1):
                raise ValueError(f"{field_name} must be within [0, 1] or None, got {value!r}")

        if self.fitness_score is not None and (
            isinstance(self.fitness_score, bool) or not isinstance(self.fitness_score, Real) or not math.isfinite(self.fitness_score)
        ):
            raise ValueError(f"fitness_score must be a finite number or None, got {self.fitness_score!r}")

        if self.coverage_score is not None and (
            isinstance(self.coverage_score, bool)
            or not isinstance(self.coverage_score, Real)
            or not math.isfinite(self.coverage_score)
            or not 0 <= self.coverage_score <= 100
        ):
            raise ValueError(f"coverage_score must be within [0, 100] or None, got {self.coverage_score!r}")

        if self.average_eta_seconds is not None and (
            isinstance(self.average_eta_seconds, bool)
            or not isinstance(self.average_eta_seconds, Real)
            or not math.isfinite(self.average_eta_seconds)
            or self.average_eta_seconds < 0
        ):
            raise ValueError(f"average_eta_seconds must be a non-negative finite number or None, got {self.average_eta_seconds!r}")

"""Domain model for one FireEvent's membership row within a GlobalPlanningRun
(Stage 2 of the Global Multi-Incident Optimizer refactor).

One row per FireEvent captured in a run's initial active-event snapshot -
the audit trail for "what happened to every incident in this global
cycle," independent of whatever the FireEvent's status is by the time
someone queries it later. `result_status`/`response_plan_id`/
`local_state_fingerprint`/`error_code` are None until the orchestrator
records that member's child outcome (GlobalPlanningRunRepository.
record_member_result); every membership row created for a run's snapshot
is expected to be filled in before that run is finalized (Task 14:
per-event failure isolation - even a FAILED member is recorded, never
left blank).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from src.models.demand_source import DemandSource
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus


@dataclass(frozen=True)
class GlobalPlanningRunEvent:
    """One FireEvent's membership + recorded outcome within one GlobalPlanningRun.

    The fields from `severity_assessment_id` through `average_eta_seconds`
    (Stage 6, demand/shortage historical persistence) are this FireEvent's
    exact severity and demand-vs-supply snapshot AS OF this run - see
    GlobalPlanningRunRepository.record_member_result's `incident_demand`/
    `event_optimization_result` parameters. All None for a NO_OP/FAILED/
    INSUFFICIENT_DATA member whose cycle never reached (or never re-derived)
    this event's demand snapshot.
    """

    fire_event_id: int
    event_order: int
    result_status: GlobalPlanningRunEventStatus | None
    response_plan_id: int | None
    local_state_fingerprint: str | None
    error_code: str | None
    severity_assessment_id: int | None = None
    severity_level: FireSeverityLevel | None = None
    severity_score: float | None = None
    demand_source: DemandSource | None = None
    demand_policy_methodology: str | None = None
    demand_policy_version: str | None = None
    minimum_resources: int | None = None
    desired_resources: int | None = None
    assigned_resources: int | None = None
    required_slots_covered: int | None = None
    required_slots_uncovered: int | None = None
    desired_slots_covered: int | None = None
    desired_slots_uncovered: int | None = None
    predicted_risk_slots_covered: int | None = None
    locked_resources_preserved: int | None = None
    coverage_score: float | None = None
    average_eta_seconds: float | None = None

    @property
    def unmet_required(self) -> int | None:
        return self.required_slots_uncovered

    @property
    def unmet_desired(self) -> int | None:
        if self.required_slots_uncovered is None or self.desired_slots_uncovered is None:
            return None
        return self.required_slots_uncovered + self.desired_slots_uncovered

    def __post_init__(self) -> None:
        if isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {self.fire_event_id!r}")
        if isinstance(self.event_order, bool) or not isinstance(self.event_order, int) or self.event_order < 0:
            raise ValueError(f"event_order must be a non-negative integer, got {self.event_order!r}")
        if self.result_status is not None and not isinstance(self.result_status, GlobalPlanningRunEventStatus):
            raise ValueError(f"result_status must be a GlobalPlanningRunEventStatus or None, got {self.result_status!r}")
        if self.response_plan_id is not None and (
            isinstance(self.response_plan_id, bool) or not isinstance(self.response_plan_id, int) or self.response_plan_id <= 0
        ):
            raise ValueError(f"response_plan_id must be a positive integer or None, got {self.response_plan_id!r}")
        if self.local_state_fingerprint is not None and (
            not isinstance(self.local_state_fingerprint, str) or not self.local_state_fingerprint.strip()
        ):
            raise ValueError(
                f"local_state_fingerprint must be a non-empty string or None, got {self.local_state_fingerprint!r}"
            )
        if self.error_code is not None and (not isinstance(self.error_code, str) or not self.error_code.strip()):
            raise ValueError(f"error_code must be a non-empty string or None, got {self.error_code!r}")

        if self.severity_assessment_id is not None and (
            isinstance(self.severity_assessment_id, bool)
            or not isinstance(self.severity_assessment_id, int)
            or self.severity_assessment_id <= 0
        ):
            raise ValueError(
                f"severity_assessment_id must be a positive integer or None, got {self.severity_assessment_id!r}"
            )
        if self.severity_level is not None and not isinstance(self.severity_level, FireSeverityLevel):
            raise ValueError(f"severity_level must be a FireSeverityLevel or None, got {self.severity_level!r}")
        if self.severity_score is not None and (
            isinstance(self.severity_score, bool) or not isinstance(self.severity_score, Real) or not math.isfinite(self.severity_score)
        ):
            raise ValueError(f"severity_score must be a finite number or None, got {self.severity_score!r}")
        if self.demand_source is not None and not isinstance(self.demand_source, DemandSource):
            raise ValueError(f"demand_source must be a DemandSource or None, got {self.demand_source!r}")
        for field_name in ("demand_policy_methodology", "demand_policy_version"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-empty string or None, got {value!r}")

        for field_name in (
            "minimum_resources",
            "desired_resources",
            "assigned_resources",
            "required_slots_covered",
            "required_slots_uncovered",
            "desired_slots_covered",
            "desired_slots_uncovered",
            "predicted_risk_slots_covered",
            "locked_resources_preserved",
        ):
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{field_name} must be a non-negative integer or None, got {value!r}")

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
            raise ValueError(
                f"average_eta_seconds must be a non-negative finite number or None, got {self.average_eta_seconds!r}"
            )

"""Pure deterministic response-target generation.

The calculator converts the current active-fire location plus already
generated spread-risk candidate locations into operational targets. It does
not recalculate wildfire spread, access persistence, invoke agents, or perform
routing/resource-allocation work.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

from src.calculators.response_target.response_target_config import (
    ACTIVE_FIRE_BASE_PRIORITY,
    MIN_PREDICTED_TARGET_RISK_SCORE,
    PREDICTION_HORIZON_FACTORS,
    TARGET_DEDUP_DISTANCE_METERS,
)
from src.models.predicted_risk_target_candidate import PredictedRiskTargetCandidate
from src.models.response_target import ResponseTarget
from src.models.response_target_type import ResponseTargetType
from src.utils.geo import haversine_distance_km

_TARGET_TYPE_ORDER = {
    ResponseTargetType.ACTIVE_FIRE: 0,
    ResponseTargetType.PREDICTED_RISK: 1,
}
_NONE_SENTINEL = -1


@dataclass(frozen=True)
class _PrioritizedCandidate:
    candidate: PredictedRiskTargetCandidate
    priority_score: float


class ResponseTargetCalculator:
    """Build response targets for one FireEvent from normalized in-memory inputs."""

    def build_targets(
        self,
        *,
        fire_event_id: int,
        fire_latitude: float,
        fire_longitude: float,
        severity_score: float | None,
        predicted_candidates: Sequence[PredictedRiskTargetCandidate],
    ) -> tuple[ResponseTarget, ...]:
        """Return immutable, deterministically ordered response targets."""
        _validate_positive_int("fire_event_id", fire_event_id)
        _validate_finite_range("fire_latitude", fire_latitude, -90.0, 90.0)
        _validate_finite_range("fire_longitude", fire_longitude, -180.0, 180.0)
        if severity_score is not None:
            _validate_finite_range("severity_score", severity_score, 0.0, 100.0)
        if not isinstance(predicted_candidates, Sequence) or isinstance(predicted_candidates, (str, bytes)):
            raise ValueError(f"predicted_candidates must be a sequence, got {predicted_candidates!r}")

        active_target = ResponseTarget(
            fire_event_id=fire_event_id,
            target_type=ResponseTargetType.ACTIVE_FIRE,
            latitude=fire_latitude,
            longitude=fire_longitude,
            priority_score=_active_priority(severity_score),
        )

        prioritized_candidates = []
        for candidate in predicted_candidates:
            if not isinstance(candidate, PredictedRiskTargetCandidate):
                raise ValueError(f"predicted_candidates must contain PredictedRiskTargetCandidate, got {candidate!r}")
            if candidate.fire_event_id != fire_event_id:
                raise ValueError(
                    "predicted candidate fire_event_id must match calculation fire_event_id, got "
                    f"{candidate.fire_event_id!r} for calculation {fire_event_id!r}"
                )
            if candidate.risk_score < MIN_PREDICTED_TARGET_RISK_SCORE:
                continue
            prioritized_candidates.append(
                _PrioritizedCandidate(
                    candidate=candidate,
                    priority_score=_predicted_priority(candidate),
                )
            )

        retained = _deduplicate_predicted_candidates(
            prioritized_candidates=prioritized_candidates,
            fire_latitude=fire_latitude,
            fire_longitude=fire_longitude,
        )
        targets = [active_target]
        targets.extend(_build_predicted_target(prioritized) for prioritized in retained)
        targets.sort(key=_target_sort_key)
        return tuple(targets)


def _active_priority(severity_score: float | None) -> float:
    # Severity describes the current threat at the already detected fire. When
    # it is unavailable, V1 explicitly falls back to the active-fire base.
    if severity_score is None:
        return ACTIVE_FIRE_BASE_PRIORITY
    return ACTIVE_FIRE_BASE_PRIORITY + severity_score


def _predicted_priority(candidate: PredictedRiskTargetCandidate) -> float:
    # V1 uses Fire Spread's existing risk score and weights it by horizon
    # urgency; it does not recalculate spread probability.
    try:
        horizon_factor = PREDICTION_HORIZON_FACTORS[candidate.prediction_horizon_minutes]
    except KeyError as exc:
        raise ValueError(
            "prediction_horizon_minutes must be one of "
            f"{tuple(sorted(PREDICTION_HORIZON_FACTORS))}, got {candidate.prediction_horizon_minutes!r}"
        ) from exc
    return candidate.risk_score * horizon_factor


def _deduplicate_predicted_candidates(
    *,
    prioritized_candidates: Sequence[_PrioritizedCandidate],
    fire_latitude: float,
    fire_longitude: float,
) -> tuple[_PrioritizedCandidate, ...]:
    ordered = sorted(prioritized_candidates, key=_prioritized_candidate_sort_key)
    retained: list[_PrioritizedCandidate] = []

    for prioritized in ordered:
        candidate = prioritized.candidate
        if _distance_meters(fire_latitude, fire_longitude, candidate.latitude, candidate.longitude) <= (
            TARGET_DEDUP_DISTANCE_METERS
        ):
            continue
        if any(
            _distance_meters(candidate.latitude, candidate.longitude, kept.candidate.latitude, kept.candidate.longitude)
            <= TARGET_DEDUP_DISTANCE_METERS
            for kept in retained
        ):
            continue
        retained.append(prioritized)

    return tuple(retained)


def _prioritized_candidate_sort_key(prioritized: _PrioritizedCandidate) -> tuple[float, int, float, float, float, int, int]:
    candidate = prioritized.candidate
    return (
        -prioritized.priority_score,
        candidate.prediction_horizon_minutes,
        -candidate.risk_score,
        candidate.latitude,
        candidate.longitude,
        candidate.spread_prediction_id,
        candidate.spread_prediction_cell_id,
    )


def _build_predicted_target(prioritized: _PrioritizedCandidate) -> ResponseTarget:
    candidate = prioritized.candidate
    return ResponseTarget(
        fire_event_id=candidate.fire_event_id,
        target_type=ResponseTargetType.PREDICTED_RISK,
        latitude=candidate.latitude,
        longitude=candidate.longitude,
        priority_score=prioritized.priority_score,
        prediction_horizon_minutes=candidate.prediction_horizon_minutes,
        spread_prediction_id=candidate.spread_prediction_id,
        spread_prediction_cell_id=candidate.spread_prediction_cell_id,
    )


def _target_sort_key(target: ResponseTarget) -> tuple[float, int, int, float, float, int, int]:
    return (
        -target.priority_score,
        _TARGET_TYPE_ORDER[target.target_type],
        target.prediction_horizon_minutes if target.prediction_horizon_minutes is not None else _NONE_SENTINEL,
        target.latitude,
        target.longitude,
        target.spread_prediction_id if target.spread_prediction_id is not None else _NONE_SENTINEL,
        target.spread_prediction_cell_id if target.spread_prediction_cell_id is not None else _NONE_SENTINEL,
    )


def _distance_meters(
    first_latitude: float,
    first_longitude: float,
    second_latitude: float,
    second_longitude: float,
) -> float:
    return haversine_distance_km(first_latitude, first_longitude, second_latitude, second_longitude) * 1000.0


def _validate_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")


def _validate_finite_range(field_name: str, value: object, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number, got {value!r}")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field_name} must be within [{minimum}, {maximum}], got {value!r}")

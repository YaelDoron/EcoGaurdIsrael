"""Tests for ResponseTargetInputService."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.calculators.fire_spread.fire_spread_config import METHODOLOGY_NAME, METHODOLOGY_VERSION
from src.models import (
    FireEvent,
    FireEventStatus,
    FireSeverityAssessment,
    FireSeverityAssessmentStatus,
    FireSeverityLevel,
    FireSpreadPrediction,
    FireSpreadPredictionCell,
    FireSpreadPredictionStatus,
    ResponseTargetInputStatus,
)
from src.repositories.fire_event_repository import StoredFireEvent
from src.repositories.fire_severity_assessment_repository import StoredFireSeverityAssessment
from src.repositories.fire_spread_prediction_repository import (
    StoredFireSpreadPredictionCell,
    StoredFireSpreadPredictionWithCells,
)
from src.services.response_target import ResponseTargetInputService

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
EVENT_LAT = 32.731
EVENT_LON = 35.046

_UNSET = object()


class FakeFireEventRepository:
    def __init__(self, stored_event: StoredFireEvent | None) -> None:
        self.stored_event = stored_event
        self.calls: list[int] = []

    def get_by_id(self, fire_event_id: int):
        self.calls.append(fire_event_id)
        return self.stored_event


class FakeFireSeverityAssessmentRepository:
    def __init__(self, stored_assessment: StoredFireSeverityAssessment | None) -> None:
        self.stored_assessment = stored_assessment
        self.calls: list[tuple[int, datetime]] = []

    def get_latest_for_event_as_of(self, fire_event_id: int, as_of: datetime):
        self.calls.append((fire_event_id, as_of))
        return self.stored_assessment


class FakeFireSpreadPredictionRepository:
    def __init__(self, predictions_by_horizon=None) -> None:
        self.predictions_by_horizon = predictions_by_horizon or {}
        self.calls: list[tuple[int, int, datetime]] = []

    def get_latest_for_event_and_horizon_as_of(self, fire_event_id: int, horizon_minutes: int, as_of: datetime):
        self.calls.append((fire_event_id, horizon_minutes, as_of))
        return self.predictions_by_horizon.get(horizon_minutes)


def make_event(status=FireEventStatus.CONFIRMED, fire_event_id=10) -> StoredFireEvent:
    return StoredFireEvent(
        id=fire_event_id,
        event=FireEvent(
            latitude=EVENT_LAT,
            longitude=EVENT_LON,
            detected_at=AS_OF - timedelta(minutes=40),
            updated_at=AS_OF - timedelta(minutes=5),
            status=status,
            detection_confidence=0.85,
            methodology=FIRE_DETECTION_METHODOLOGY_NAME,
            methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
        ),
    )


def make_assessment(
    *,
    assessment_id=500,
    fire_event_id=10,
    status=FireSeverityAssessmentStatus.VALID,
    assessed_at=None,
    score=72.0,
) -> StoredFireSeverityAssessment:
    is_valid = status is FireSeverityAssessmentStatus.VALID
    return StoredFireSeverityAssessment(
        assessment_id=assessment_id,
        assessment=FireSeverityAssessment(
            fire_event_id=fire_event_id,
            assessed_at=assessed_at or (AS_OF - timedelta(minutes=10)),
            status=status,
            score=score if is_valid else None,
            level=FireSeverityLevel.HIGH if is_valid else None,
            methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
            methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
        ),
    )


def make_cell(
    *,
    latitude=32.74,
    longitude=35.06,
    risk_score=80.0,
    reached_step=1,
    cell_id=100,
) -> StoredFireSpreadPredictionCell:
    return StoredFireSpreadPredictionCell(
        cell_id=cell_id,
        cell=FireSpreadPredictionCell(
            latitude=latitude,
            longitude=longitude,
            spread_probability=risk_score / 100.0,
            spread_risk_score=risk_score,
            reached_step=reached_step,
            reached_minutes=reached_step * 5,
        ),
    )


def make_prediction(
    *,
    prediction_id=1000,
    fire_event_id=10,
    horizon_minutes=30,
    status=FireSpreadPredictionStatus.VALID,
    predicted_at=None,
    cells=_UNSET,
) -> StoredFireSpreadPredictionWithCells:
    stored_cells = (make_cell(),) if cells is _UNSET else tuple(cells)
    is_valid = status is FireSpreadPredictionStatus.VALID
    return StoredFireSpreadPredictionWithCells(
        id=prediction_id,
        prediction=FireSpreadPrediction(
            fire_event_id=fire_event_id,
            severity_assessment_id=500 if is_valid else None,
            predicted_at=predicted_at or (AS_OF - timedelta(minutes=5)),
            horizon_minutes=horizon_minutes,
            status=status,
            methodology=METHODOLOGY_NAME,
            methodology_version=METHODOLOGY_VERSION,
            cells=tuple(stored_cell.cell for stored_cell in stored_cells),
        ),
        cells=stored_cells,
        weather_observation_id=101 if is_valid else None,
    )


def build_service(
    *,
    stored_event=_UNSET,
    stored_assessment=_UNSET,
    predictions_by_horizon=None,
):
    event_repo = FakeFireEventRepository(make_event() if stored_event is _UNSET else stored_event)
    severity_repo = FakeFireSeverityAssessmentRepository(
        make_assessment() if stored_assessment is _UNSET else stored_assessment
    )
    spread_repo = FakeFireSpreadPredictionRepository(predictions_by_horizon)
    service = ResponseTargetInputService(
        fire_event_repository=event_repo,
        fire_severity_assessment_repository=severity_repo,
        fire_spread_prediction_repository=spread_repo,
    )
    return service, event_repo, severity_repo, spread_repo


@pytest.mark.parametrize("status", [FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED])
def test_active_fire_event_returns_ready(status):
    service, *_ = build_service(stored_event=make_event(status=status), predictions_by_horizon={})

    result = service.prepare_input(10, AS_OF)

    assert result.status is ResponseTargetInputStatus.READY
    assert result.input_data.fire_event_id == 10


@pytest.mark.parametrize("status", [FireEventStatus.RESOLVED, FireEventStatus.DISMISSED])
def test_inactive_event_returns_inactive(status):
    service, *_ = build_service(stored_event=make_event(status=status))

    result = service.prepare_input(10, AS_OF)

    assert result.status is ResponseTargetInputStatus.INACTIVE_EVENT
    assert result.input_data is None


def test_inactive_event_short_circuits_without_loading_severity_or_spread():
    service, _, severity_repo, spread_repo = build_service(stored_event=make_event(status=FireEventStatus.RESOLVED))

    service.prepare_input(10, AS_OF)

    assert severity_repo.calls == []
    assert spread_repo.calls == []


def test_missing_fire_event_raises_and_is_not_fabricated():
    service, *_ = build_service(stored_event=None)

    with pytest.raises(ValueError):
        service.prepare_input(10, AS_OF)


def test_naive_as_of_rejected():
    service, *_ = build_service(predictions_by_horizon={})

    with pytest.raises(ValueError):
        service.prepare_input(10, datetime(2026, 9, 14, 12, 0))


def test_timezone_aware_as_of_accepted():
    service, *_ = build_service(predictions_by_horizon={})

    assert service.prepare_input(10, AS_OF).status is ResponseTargetInputStatus.READY


def test_as_of_is_passed_to_severity_and_spread_repositories():
    service, _, severity_repo, spread_repo = build_service(predictions_by_horizon={})

    service.prepare_input(10, AS_OF)

    assert severity_repo.calls == [(10, AS_OF)]
    assert spread_repo.calls == [(10, 30, AS_OF), (10, 60, AS_OF)]


def test_no_severity_returns_ready_with_none_score():
    service, *_ = build_service(stored_assessment=None, predictions_by_horizon={})

    result = service.prepare_input(10, AS_OF)

    assert result.status is ResponseTargetInputStatus.READY
    assert result.input_data.severity_score is None


def test_latest_valid_severity_score_propagated_exactly():
    service, *_ = build_service(stored_assessment=make_assessment(score=88.5), predictions_by_horizon={})

    result = service.prepare_input(10, AS_OF)

    assert result.input_data.severity_score == 88.5


@pytest.mark.parametrize(
    "status", [FireSeverityAssessmentStatus.INSUFFICIENT_DATA, FireSeverityAssessmentStatus.INACTIVE_EVENT]
)
def test_latest_non_valid_severity_returns_none(status):
    service, *_ = build_service(stored_assessment=make_assessment(status=status), predictions_by_horizon={})

    result = service.prepare_input(10, AS_OF)

    assert result.input_data.severity_score is None


def test_newer_invalid_severity_state_prevents_fallback_to_older_valid_state():
    service, *_ = build_service(
        stored_assessment=make_assessment(status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA),
        predictions_by_horizon={},
    )

    assert service.prepare_input(10, AS_OF).input_data.severity_score is None


def test_cross_event_severity_row_fails_explicitly():
    service, *_ = build_service(stored_assessment=make_assessment(fire_event_id=11), predictions_by_horizon={})

    with pytest.raises(ValueError):
        service.prepare_input(10, AS_OF)


def test_no_predictions_returns_ready_with_zero_candidates():
    service, *_ = build_service(predictions_by_horizon={})

    result = service.prepare_input(10, AS_OF)

    assert result.input_data.predicted_candidates == ()


def test_latest_valid_30_minute_prediction_cells_convert_to_candidates():
    prediction = make_prediction(horizon_minutes=30, prediction_id=1000, cells=(make_cell(cell_id=101),))
    service, *_ = build_service(predictions_by_horizon={30: prediction})

    candidates = service.prepare_input(10, AS_OF).input_data.predicted_candidates

    assert len(candidates) == 1
    assert candidates[0].prediction_horizon_minutes == 30


def test_latest_valid_60_minute_prediction_cells_convert_to_candidates():
    prediction = make_prediction(horizon_minutes=60, prediction_id=2000, cells=(make_cell(cell_id=201),))
    service, *_ = build_service(predictions_by_horizon={60: prediction})

    candidates = service.prepare_input(10, AS_OF).input_data.predicted_candidates

    assert len(candidates) == 1
    assert candidates[0].prediction_horizon_minutes == 60


def test_both_horizons_return_candidates_from_both():
    prediction_30 = make_prediction(horizon_minutes=30, prediction_id=1000, cells=(make_cell(cell_id=101),))
    prediction_60 = make_prediction(horizon_minutes=60, prediction_id=2000, cells=(make_cell(cell_id=201),))
    service, *_ = build_service(predictions_by_horizon={30: prediction_30, 60: prediction_60})

    candidates = service.prepare_input(10, AS_OF).input_data.predicted_candidates

    assert [candidate.prediction_horizon_minutes for candidate in candidates] == [30, 60]


def test_valid_prediction_with_zero_cells_remains_ready_with_no_candidates_for_horizon():
    prediction = make_prediction(horizon_minutes=30, cells=())
    service, *_ = build_service(predictions_by_horizon={30: prediction})

    result = service.prepare_input(10, AS_OF)

    assert result.status is ResponseTargetInputStatus.READY
    assert result.input_data.predicted_candidates == ()


@pytest.mark.parametrize(
    "status", [FireSpreadPredictionStatus.INSUFFICIENT_DATA, FireSpreadPredictionStatus.INACTIVE_EVENT]
)
def test_latest_non_valid_prediction_returns_no_candidates_for_horizon(status):
    prediction = make_prediction(horizon_minutes=30, status=status, cells=())
    service, *_ = build_service(predictions_by_horizon={30: prediction})

    assert service.prepare_input(10, AS_OF).input_data.predicted_candidates == ()


def test_newer_non_valid_prediction_prevents_fallback_to_older_valid_prediction():
    prediction = make_prediction(horizon_minutes=30, status=FireSpreadPredictionStatus.INSUFFICIENT_DATA, cells=())
    service, *_ = build_service(predictions_by_horizon={30: prediction})

    assert service.prepare_input(10, AS_OF).input_data.predicted_candidates == ()


def test_candidate_preserves_traceability_and_cell_fields_exactly():
    cell = make_cell(latitude=32.75, longitude=35.07, risk_score=55.0, cell_id=321)
    prediction = make_prediction(horizon_minutes=60, prediction_id=123, cells=(cell,))
    service, *_ = build_service(predictions_by_horizon={60: prediction})

    candidate = service.prepare_input(10, AS_OF).input_data.predicted_candidates[0]

    assert candidate.fire_event_id == 10
    assert candidate.spread_prediction_id == 123
    assert candidate.spread_prediction_cell_id == 321
    assert candidate.prediction_horizon_minutes == 60
    assert candidate.latitude == 32.75
    assert candidate.longitude == 35.07
    assert candidate.risk_score == 55.0


def test_cell_below_response_target_threshold_is_still_returned_as_candidate():
    prediction = make_prediction(cells=(make_cell(risk_score=55.0),))
    service, *_ = build_service(predictions_by_horizon={30: prediction})

    candidates = service.prepare_input(10, AS_OF).input_data.predicted_candidates

    assert len(candidates) == 1
    assert candidates[0].risk_score == 55.0


def test_service_does_not_geographically_deduplicate_candidates():
    first = make_cell(latitude=32.74, longitude=35.06, cell_id=101)
    second = make_cell(latitude=32.74, longitude=35.06, cell_id=102)
    prediction = make_prediction(cells=(first, second))
    service, *_ = build_service(predictions_by_horizon={30: prediction})

    candidates = service.prepare_input(10, AS_OF).input_data.predicted_candidates

    assert [candidate.spread_prediction_cell_id for candidate in candidates] == [101, 102]


def test_same_coordinates_in_30_and_60_minute_predictions_remain_two_candidates():
    cell_30 = make_cell(latitude=32.74, longitude=35.06, cell_id=101)
    cell_60 = make_cell(latitude=32.74, longitude=35.06, cell_id=201)
    prediction_30 = make_prediction(horizon_minutes=30, prediction_id=1000, cells=(cell_30,))
    prediction_60 = make_prediction(horizon_minutes=60, prediction_id=2000, cells=(cell_60,))
    service, *_ = build_service(predictions_by_horizon={30: prediction_30, 60: prediction_60})

    candidates = service.prepare_input(10, AS_OF).input_data.predicted_candidates

    assert len(candidates) == 2


def test_predicted_cell_close_to_active_fire_is_still_returned():
    near_active = make_cell(latitude=EVENT_LAT, longitude=EVENT_LON, cell_id=101)
    prediction = make_prediction(cells=(near_active,))
    service, *_ = build_service(predictions_by_horizon={30: prediction})

    candidates = service.prepare_input(10, AS_OF).input_data.predicted_candidates

    assert len(candidates) == 1


def test_service_does_not_calculate_priority():
    prediction = make_prediction(cells=(make_cell(risk_score=80.0),))
    service, *_ = build_service(predictions_by_horizon={30: prediction})

    candidate = service.prepare_input(10, AS_OF).input_data.predicted_candidates[0]

    assert not hasattr(candidate, "priority_score")


def test_cross_event_prediction_fails_explicitly():
    prediction = make_prediction(fire_event_id=11)
    service, *_ = build_service(predictions_by_horizon={30: prediction})

    with pytest.raises(ValueError):
        service.prepare_input(10, AS_OF)


def test_service_does_not_use_geographic_closeness_to_attach_other_event_predictions():
    other_event_prediction = make_prediction(fire_event_id=11, cells=(make_cell(latitude=EVENT_LAT, longitude=EVENT_LON),))
    service, *_ = build_service(predictions_by_horizon={30: other_event_prediction})

    with pytest.raises(ValueError):
        service.prepare_input(10, AS_OF)


def test_same_persisted_state_produces_equal_input():
    prediction = make_prediction(cells=(make_cell(cell_id=101),))
    service, *_ = build_service(predictions_by_horizon={30: prediction})

    first = service.prepare_input(10, AS_OF).input_data
    second = service.prepare_input(10, AS_OF).input_data

    assert first == second


def test_candidate_ordering_is_deterministic():
    later = make_cell(latitude=32.75, longitude=35.08, cell_id=102)
    earlier = make_cell(latitude=32.74, longitude=35.07, cell_id=101)
    prediction = make_prediction(prediction_id=1000, cells=(later, earlier))
    service, *_ = build_service(predictions_by_horizon={30: prediction})

    candidates = service.prepare_input(10, AS_OF).input_data.predicted_candidates

    assert [candidate.spread_prediction_cell_id for candidate in candidates] == [101, 102]

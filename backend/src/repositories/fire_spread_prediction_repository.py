"""Persistence layer for wildfire-spread prediction runs and traceability."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload, sessionmaker

from src.database.connection import get_session_factory
from src.database.models.fire_spread_prediction_cell_db import FireSpreadPredictionCellDB
from src.database.models.fire_spread_prediction_db import FireSpreadPredictionDB
from src.database.models.fire_spread_prediction_weather_input_db import (
    FireSpreadPredictionWeatherInputDB,
)
from src.models.fire_spread_prediction import FireSpreadPrediction, FireSpreadPredictionCell
from src.models.fire_spread_prediction_status import FireSpreadPredictionStatus
from src.repositories.exceptions import FireSpreadPredictionRepositoryError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredFireSpreadPrediction:
    """Persisted FireSpreadPrediction plus database identity and weather traceability."""

    id: int
    prediction: FireSpreadPrediction
    weather_observation_id: int | None = None


class FireSpreadPredictionRepository:
    """Persists FireSpreadPrediction domain objects via SQLAlchemy."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        self._session_factory = session_factory or get_session_factory()

    @contextmanager
    def _session_scope(self) -> Iterator[Session]:
        """Run a block of work in a session, committing on success and rolling back on error."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_prediction(
        self,
        prediction: FireSpreadPrediction,
        weather_observation_id: int | None,
    ) -> StoredFireSpreadPrediction:
        """Atomically store a prediction, its cells (if any), and its weather trace row.

        `weather_observation_id` is `None` for INSUFFICIENT_DATA/INACTIVE_EVENT
        predictions that never selected a weather observation -- no trace row
        is fabricated in that case.
        """
        self._validate_save_request(prediction, weather_observation_id)

        with self._session_scope() as session:
            db_prediction = self._to_db_prediction(prediction)
            session.add(db_prediction)
            try:
                session.flush()
                for cell in prediction.cells:
                    session.add(self._to_db_cell(db_prediction.id, cell))
                if weather_observation_id is not None:
                    session.add(
                        FireSpreadPredictionWeatherInputDB(
                            prediction_id=db_prediction.id,
                            weather_observation_id=weather_observation_id,
                        )
                    )
                session.flush()
            except (IntegrityError, SQLAlchemyError) as exc:
                raise FireSpreadPredictionRepositoryError(
                    "Fire-spread prediction persistence failed."
                ) from exc

            logger.info(
                "Stored fire-spread prediction %s for FireEvent %s with status %s",
                db_prediction.id,
                prediction.fire_event_id,
                prediction.status.value,
            )
            return self._to_stored_prediction(db_prediction)

    def get_by_id(self, prediction_id: int) -> StoredFireSpreadPrediction | None:
        """Return a persisted spread prediction by database id, or None if absent."""
        self._validate_prediction_id(prediction_id)
        with self._session_scope() as session:
            db_prediction = self._get_db_prediction(session, prediction_id)
            return self._to_stored_prediction(db_prediction) if db_prediction is not None else None

    def get_latest_for_event_and_horizon(
        self,
        fire_event_id: int,
        horizon_minutes: int,
    ) -> StoredFireSpreadPrediction | None:
        """Return the latest persisted prediction for a FireEvent at one horizon."""
        self._validate_fire_event_id(fire_event_id)
        self._validate_horizon_minutes(horizon_minutes)
        with self._session_scope() as session:
            db_prediction = (
                session.execute(
                    select(FireSpreadPredictionDB)
                    .options(
                        selectinload(FireSpreadPredictionDB.cells),
                        selectinload(FireSpreadPredictionDB.weather_inputs),
                    )
                    .where(
                        FireSpreadPredictionDB.fire_event_id == fire_event_id,
                        FireSpreadPredictionDB.horizon_minutes == horizon_minutes,
                    )
                    .order_by(FireSpreadPredictionDB.predicted_at.desc(), FireSpreadPredictionDB.id.desc())
                    .limit(1)
                )
                .scalars()
                .one_or_none()
            )
            return self._to_stored_prediction(db_prediction) if db_prediction is not None else None

    @staticmethod
    def _to_db_prediction(prediction: FireSpreadPrediction) -> FireSpreadPredictionDB:
        return FireSpreadPredictionDB(
            fire_event_id=prediction.fire_event_id,
            severity_assessment_id=prediction.severity_assessment_id,
            predicted_at=prediction.predicted_at,
            horizon_minutes=prediction.horizon_minutes,
            status=prediction.status.value,
            methodology=prediction.methodology,
            methodology_version=prediction.methodology_version,
        )

    @staticmethod
    def _to_db_cell(prediction_id: int, cell: FireSpreadPredictionCell) -> FireSpreadPredictionCellDB:
        return FireSpreadPredictionCellDB(
            prediction_id=prediction_id,
            latitude=cell.latitude,
            longitude=cell.longitude,
            spread_probability=cell.spread_probability,
            spread_risk_score=cell.spread_risk_score,
            reached_step=cell.reached_step,
            reached_minutes=cell.reached_minutes,
        )

    @staticmethod
    def _get_db_prediction(session: Session, prediction_id: int) -> FireSpreadPredictionDB | None:
        return (
            session.execute(
                select(FireSpreadPredictionDB)
                .options(
                    selectinload(FireSpreadPredictionDB.cells),
                    selectinload(FireSpreadPredictionDB.weather_inputs),
                )
                .where(FireSpreadPredictionDB.id == prediction_id)
            )
            .scalars()
            .one_or_none()
        )

    @classmethod
    def _to_stored_prediction(cls, db_prediction: FireSpreadPredictionDB) -> StoredFireSpreadPrediction:
        cells = tuple(
            sorted(
                (
                    FireSpreadPredictionCell(
                        latitude=db_cell.latitude,
                        longitude=db_cell.longitude,
                        spread_probability=db_cell.spread_probability,
                        spread_risk_score=db_cell.spread_risk_score,
                        reached_step=db_cell.reached_step,
                        reached_minutes=db_cell.reached_minutes,
                    )
                    for db_cell in db_prediction.cells
                ),
                key=lambda cell: (cell.reached_step, cell.latitude, cell.longitude),
            )
        )
        weather_ids = tuple(trace.weather_observation_id for trace in db_prediction.weather_inputs)
        return StoredFireSpreadPrediction(
            id=db_prediction.id,
            prediction=FireSpreadPrediction(
                fire_event_id=db_prediction.fire_event_id,
                severity_assessment_id=db_prediction.severity_assessment_id,
                predicted_at=cls._ensure_aware_datetime(db_prediction.predicted_at),
                horizon_minutes=db_prediction.horizon_minutes,
                status=FireSpreadPredictionStatus(db_prediction.status),
                methodology=db_prediction.methodology,
                methodology_version=db_prediction.methodology_version,
                cells=cells,
            ),
            weather_observation_id=weather_ids[0] if weather_ids else None,
        )

    @staticmethod
    def _validate_save_request(prediction: FireSpreadPrediction, weather_observation_id: int | None) -> None:
        if not isinstance(prediction, FireSpreadPrediction):
            raise FireSpreadPredictionRepositoryError(
                f"prediction must be a FireSpreadPrediction, got {prediction!r}"
            )
        if weather_observation_id is not None and (
            isinstance(weather_observation_id, bool)
            or not isinstance(weather_observation_id, int)
            or weather_observation_id <= 0
        ):
            raise FireSpreadPredictionRepositoryError(
                "weather_observation_id must be a positive integer or None, "
                f"got {weather_observation_id!r}"
            )
        if prediction.status is FireSpreadPredictionStatus.VALID and weather_observation_id is None:
            raise FireSpreadPredictionRepositoryError(
                "VALID fire-spread predictions require a weather_observation_id."
            )

    @staticmethod
    def _validate_prediction_id(prediction_id: int) -> None:
        if isinstance(prediction_id, bool) or not isinstance(prediction_id, int) or prediction_id <= 0:
            raise FireSpreadPredictionRepositoryError(
                f"prediction_id must be a positive integer, got {prediction_id!r}."
            )

    @staticmethod
    def _validate_fire_event_id(fire_event_id: int) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise FireSpreadPredictionRepositoryError(
                f"fire_event_id must be a positive integer, got {fire_event_id!r}."
            )

    @staticmethod
    def _validate_horizon_minutes(horizon_minutes: int) -> None:
        if isinstance(horizon_minutes, bool) or not isinstance(horizon_minutes, int) or horizon_minutes <= 0:
            raise FireSpreadPredictionRepositoryError(
                f"horizon_minutes must be a positive integer, got {horizon_minutes!r}."
            )

    @staticmethod
    def _ensure_aware_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

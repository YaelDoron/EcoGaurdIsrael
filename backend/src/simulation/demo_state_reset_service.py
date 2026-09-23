"""Reset EcoGuard demo/simulation runtime state to a known clean baseline.

Task A1.6: operations_demo revealed that a shared, long-lived database
accumulates FireEvents/evidence/plans across runs, and FireDetectionAgent's
real (unmodified) evidence-correlation logic can then legitimately reject
new evidence that does not connect with a stale FireEvent's old evidence by
location and time. That validation is correct and is not touched here - the
fix is making the underlying operational state resettable to a clean
baseline before a demo run, not weakening detection.

DemoStateResetService is intentionally narrow:
- it is NOT a general-purpose production cleanup utility,
- it refuses to run unless `settings.ENABLE_DEMO_DATA_RESET` is explicitly
  true (see src/config/settings.py) - there is no APP_ENV concept in this
  project yet, so this flag is the sole safety gate, checked unconditionally
  every call, regardless of caller,
- it never infers safety from the DATABASE_URL hostname/database name -
  that would be a heuristic, not a guarantee,
- it deletes only runtime/operational demo data (see _DELETE_ORDER_MODELS'
  docstring below for the classification this was derived from) and always
  preserves: FireStationDB/FirefightingResourceDB row definitions,
  WeatherStationDB rows (deterministically re-derived by id per location by
  the simulation generators, not per-run data), and the road-network cache
  (GraphNodeDB/GraphEdgeDB) - re-fetching that from OSM is exactly the
  multi-second-to-tens-of-seconds cost Task A1.5 bounded and made
  efficient; deleting it on every reset would reintroduce that cost
  needlessly.
- firefighting resource *status* (a mutable operational field on an
  otherwise-static row, separate from the row's existence) is restored to
  the documented seeded baseline, AVAILABLE (see scripts/seed_fire_stations.py),
  since simulation runs and global planning legitimately mutate it away
  from AVAILABLE (SimulationOperationalCoordinator.scramble_resource_availability,
  ResourceStatusUpdateService).
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import delete, update
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import settings
from src.database.connection import get_session_factory
from src.database.models.fire_danger_assessment_db import FireDangerAssessmentDB
from src.database.models.fire_danger_assessment_weather_input_db import (
    FireDangerAssessmentWeatherInputDB,
)
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_event_ml_assessment_db import FireEventMLAssessmentDB
from src.database.models.fire_event_news_evidence_db import FireEventNewsEvidenceDB
from src.database.models.fire_event_satellite_evidence_db import FireEventSatelliteEvidenceDB
from src.database.models.fire_severity_assessment_db import FireSeverityAssessmentDB
from src.database.models.fire_severity_assessment_satellite_input_db import (
    FireSeverityAssessmentSatelliteInputDB,
)
from src.database.models.fire_severity_assessment_weather_input_db import (
    FireSeverityAssessmentWeatherInputDB,
)
from src.database.models.fire_spread_prediction_cell_db import FireSpreadPredictionCellDB
from src.database.models.fire_spread_prediction_db import FireSpreadPredictionDB
from src.database.models.fire_spread_prediction_weather_input_db import (
    FireSpreadPredictionWeatherInputDB,
)
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.global_planning_run_db import GlobalPlanningRunDB
from src.database.models.global_planning_run_event_db import GlobalPlanningRunEventDB
from src.database.models.plan_comparison_db import PlanComparisonDB
from src.database.models.resource_commitment_db import ResourceCommitmentDB
from src.database.models.response_action_db import ResponseActionDB
from src.database.models.response_plan_db import ResponsePlanDB
from src.database.models.response_plan_planning_state_db import ResponsePlanPlanningStateDB
from src.database.models.response_plan_uncovered_target_db import ResponsePlanUncoveredTargetDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.database.models.route_result_db import RouteResultDB
from src.database.models.satellite_hotspot_db import SatelliteHotspotDB
from src.database.models.weather_observation_db import WeatherObservationDB
from src.database.models.wildfire_report_db import WildfireReportDB
from src.models.resource_status import ResourceStatus

RESTORED_RESOURCE_STATUS = ResourceStatus.AVAILABLE

# FK-safe deletion order: deepest derived children first, deriving strictly
# from the actual FK graph audited across src/database/models (Task A1.6),
# not assumed. Bulk Core `delete()` is used (not ORM `session.delete()`),
# so relationship-level `cascade=` settings on the models do NOT apply here
# - order is enforced explicitly by this list instead.
#
# Deliberately absent (never deleted by a reset):
# - FireStationDB, FirefightingResourceDB: static reference data.
# - WeatherStationDB: simulation station ids are derived deterministically
#   from (location, station_index) with no run/time component - the same
#   row is reused and upserted by every run for a given location, never
#   duplicated - so it behaves as reusable reference data, not per-run state.
# - GraphNodeDB, GraphEdgeDB: the road-network cache Task A1.5 made
#   efficient to populate; re-fetching it from OSM on every demo reset would
#   reintroduce the exact cost that task removed.
_DELETE_ORDER_MODELS = (
    ResourceCommitmentDB,
    ResponsePlanUncoveredTargetDB,
    ResponsePlanPlanningStateDB,
    ResponseActionDB,
    PlanComparisonDB,
    GlobalPlanningRunEventDB,
    ResponsePlanDB,
    GlobalPlanningRunDB,
    RouteResultDB,
    RoutePlanningRunDB,
    ResponseTargetDB,
    FireSpreadPredictionCellDB,
    FireSpreadPredictionWeatherInputDB,
    FireSpreadPredictionDB,
    FireSeverityAssessmentWeatherInputDB,
    FireSeverityAssessmentSatelliteInputDB,
    FireSeverityAssessmentDB,
    ResponseTargetSetDB,
    FireEventSatelliteEvidenceDB,
    FireEventNewsEvidenceDB,
    FireEventMLAssessmentDB,
    FireEventDB,
    SatelliteHotspotDB,
    WildfireReportDB,
    FireDangerAssessmentWeatherInputDB,
    FireDangerAssessmentDB,
    WeatherObservationDB,
)


class DemoStateResetDisabledError(RuntimeError):
    """Raised when reset_demo_state() is called without ENABLE_DEMO_DATA_RESET=true."""


@dataclass(frozen=True)
class DemoStateResetResult:
    """Report of one reset_demo_state() call: what was deleted/restored."""

    deleted_counts: dict[str, int]
    resources_restored: int

    @property
    def total_deleted(self) -> int:
        return sum(self.deleted_counts.values())


class DemoStateResetService:
    """Reset EcoGuard runtime/operational demo state to a clean, repeatable baseline.

    Deletes only the tables classified RUNTIME in the Task A1.6 audit (see
    _DELETE_ORDER_MODELS above), preserves STATIC reference data and the
    CACHE road network, and restores firefighting-resource status to its
    seeded baseline. Runs as one transaction: commits only if every delete
    and the resource-status restore succeed, rolls back entirely on any
    failure, and always returns the connection to the pool.
    """

    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        self._session_factory = session_factory or get_session_factory()

    def reset_demo_state(self) -> DemoStateResetResult:
        """Delete runtime demo state and restore resource status, or raise.

        Raises DemoStateResetDisabledError without touching the database at
        all if ENABLE_DEMO_DATA_RESET is not explicitly true. Idempotent:
        calling this again on an already-clean database succeeds and
        reports zero counts.
        """
        if not settings.ENABLE_DEMO_DATA_RESET:
            raise DemoStateResetDisabledError(
                "Demo state reset refused: ENABLE_DEMO_DATA_RESET is not enabled. "
                "This deletes runtime wildfire/evidence/planning data and is intended "
                "only for a dedicated, disposable demo database or Neon branch - never "
                "the shared team development database. Set ENABLE_DEMO_DATA_RESET=true "
                "explicitly in that environment before calling reset_demo_state()."
            )

        session = self._session_factory()
        try:
            deleted_counts = self._delete_runtime_state(session)
            resources_restored = self._restore_resource_baseline(session)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return DemoStateResetResult(deleted_counts=deleted_counts, resources_restored=resources_restored)

    @staticmethod
    def _delete_runtime_state(session: Session) -> dict[str, int]:
        deleted_counts: dict[str, int] = {}
        for model in _DELETE_ORDER_MODELS:
            result = session.execute(delete(model))
            deleted_counts[model.__tablename__] = result.rowcount or 0
        return deleted_counts

    @staticmethod
    def _restore_resource_baseline(session: Session) -> int:
        result = session.execute(update(FirefightingResourceDB).values(status=RESTORED_RESOURCE_STATUS))
        return result.rowcount or 0

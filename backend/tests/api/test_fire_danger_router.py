"""API tests for the Fire Danger read endpoints (Task A4, Part 20).

Uses FastAPI's dependency_overrides to replace FireDangerQueryService with a
fake returning controlled read-model data - no real FFWI calculation,
FireDangerAssessmentAgent invocation, or DB access happens in these tests.
"""
from __future__ import annotations

import ast
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_fire_danger_query_service
from src.api.routers.fire_danger import fire_danger_router
from src.models.fire_danger_areas import (
    FireDangerAreaAssessmentSummary,
    FireDangerAreaSnapshot,
    FireDangerAreasResult,
)
from src.models.fire_danger_assessment_detail import (
    FireDangerAssessmentDetail,
    FireDangerAssessmentWeatherInputSummary,
)
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel

ASSESSED_AT = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)
AS_OF = datetime(2026, 9, 19, 12, 30, 0, tzinfo=timezone.utc)

AREAS_ENDPOINT = "/api/v1/fire-danger/areas/latest"


def single_area_endpoint(area_id: str) -> str:
    return f"/api/v1/fire-danger/areas/{area_id}/latest"


def assessment_endpoint(assessment_id) -> str:
    return f"/api/v1/fire-danger/assessments/{assessment_id}"


class FakeFireDangerQueryService:
    def __init__(self, *, all_areas_result=None, area_by_id=None, detail_by_id=None):
        self._all_areas_result = all_areas_result
        self._area_by_id = area_by_id or {}
        self._detail_by_id = detail_by_id or {}
        self.get_latest_for_all_areas_calls = 0
        self.get_latest_for_area_calls: list[str] = []
        self.get_assessment_detail_calls: list[int] = []

    def get_latest_for_all_areas(self):
        self.get_latest_for_all_areas_calls += 1
        return self._all_areas_result if self._all_areas_result is not None else FireDangerAreasResult(
            as_of=AS_OF, areas=()
        )

    def get_latest_for_area(self, area_id: str):
        self.get_latest_for_area_calls.append(area_id)
        return self._area_by_id.get(area_id)

    def get_assessment_detail(self, assessment_id: int):
        self.get_assessment_detail_calls.append(assessment_id)
        return self._detail_by_id.get(assessment_id)


def make_assessment_summary(**overrides) -> FireDangerAreaAssessmentSummary:
    values = dict(
        assessment_id=1,
        status=FireDangerAssessmentStatus.VALID,
        score=42.5,
        level=FireDangerLevel.VERY_HIGH,
        assessed_at=ASSESSED_AT,
        methodology="FOSBERG_FFWI",
        methodology_version="1.0",
    )
    values.update(overrides)
    return FireDangerAreaAssessmentSummary(**values)


def make_area(**overrides) -> FireDangerAreaSnapshot:
    values = dict(
        area_id="area-carmel",
        area_name="Carmel",
        area_latitude=32.731,
        area_longitude=35.046,
        area_radius_km=5.0,
        assessment=make_assessment_summary(),
    )
    values.update(overrides)
    return FireDangerAreaSnapshot(**values)


def make_detail(**overrides) -> FireDangerAssessmentDetail:
    values = dict(
        assessment_id=1,
        area_id="area-carmel",
        area_name="Carmel",
        area_latitude=32.731,
        area_longitude=35.046,
        area_radius_km=5.0,
        status=FireDangerAssessmentStatus.VALID,
        score=42.5,
        level=FireDangerLevel.VERY_HIGH,
        assessed_at=ASSESSED_AT,
        methodology="FOSBERG_FFWI",
        methodology_version="1.0",
        weather_inputs=(
            FireDangerAssessmentWeatherInputSummary(
                observation_id=101,
                station_external_id=1001,
                station_name="Station 1001",
                observed_at=datetime(2026, 9, 19, 11, 45),
            ),
        ),
    )
    values.update(overrides)
    return FireDangerAssessmentDetail(**values)


def client_for(service, *, raise_server_exceptions: bool = True) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_fire_danger_query_service] = lambda: service
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# 1 & 2. List endpoint success / multiple areas
# ---------------------------------------------------------------------------


def test_areas_latest_lists_multiple_areas():
    result = FireDangerAreasResult(
        as_of=AS_OF,
        areas=(
            make_area(area_id="area-carmel", area_name="Carmel"),
            make_area(area_id="area-golan", area_name="Golan", assessment=None),
        ),
    )
    client = client_for(FakeFireDangerQueryService(all_areas_result=result))

    response = client.get(AREAS_ENDPOINT)

    assert response.status_code == 200
    body = response.json()
    assert parse_dt(body["as_of"]) == AS_OF
    assert [area["area_id"] for area in body["areas"]] == ["area-carmel", "area-golan"]


# ---------------------------------------------------------------------------
# 3. Latest assessment selected (router trusts the query service; verify pass-through)
# ---------------------------------------------------------------------------


def test_areas_latest_passes_through_the_services_selected_assessment():
    result = FireDangerAreasResult(
        as_of=AS_OF,
        areas=(make_area(assessment=make_assessment_summary(assessment_id=77, score=61.0, level=FireDangerLevel.EXTREME)),),
    )
    client = client_for(FakeFireDangerQueryService(all_areas_result=result))

    body = client.get(AREAS_ENDPOINT).json()

    assessment = body["areas"][0]["assessment"]
    assert assessment["assessment_id"] == 77
    assert assessment["score"] == 61.0
    assert assessment["level"] == "extreme"


# ---------------------------------------------------------------------------
# 4. Single-area success
# ---------------------------------------------------------------------------


def test_single_area_success():
    area = make_area()
    client = client_for(FakeFireDangerQueryService(area_by_id={"area-carmel": area}))

    response = client.get(single_area_endpoint("area-carmel"))

    assert response.status_code == 200
    body = response.json()
    assert body["area_id"] == "area-carmel"
    assert body["assessment"]["score"] == 42.5


# ---------------------------------------------------------------------------
# 5. Known-area/no-assessment semantics: not reachable under this
# architecture (see fire_danger_query_service's module docstring) - the
# closest real equivalent is a known area whose latest assessment is
# INSUFFICIENT_DATA (assessment populated, score/level null).
# ---------------------------------------------------------------------------


def test_single_area_with_insufficient_data_assessment_has_null_score_and_level():
    area = make_area(
        assessment=make_assessment_summary(status=FireDangerAssessmentStatus.INSUFFICIENT_DATA, score=None, level=None)
    )
    client = client_for(FakeFireDangerQueryService(area_by_id={"area-carmel": area}))

    body = client.get(single_area_endpoint("area-carmel")).json()

    assert body["assessment"]["status"] == "insufficient_data"
    assert body["assessment"]["score"] is None
    assert body["assessment"]["level"] is None
    assert body["assessment"] is not None  # still a real, persisted assessment - not "no assessment"


# ---------------------------------------------------------------------------
# 6. Unknown-area error semantics
# ---------------------------------------------------------------------------


def test_single_area_unknown_area_returns_404_envelope():
    client = client_for(FakeFireDangerQueryService(area_by_id={}))

    response = client.get(single_area_endpoint("never-assessed"))

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "FIRE_DANGER_AREA_NOT_FOUND"


def test_assessment_detail_unknown_id_returns_404_envelope():
    client = client_for(FakeFireDangerQueryService(detail_by_id={}))

    response = client.get(assessment_endpoint(999))

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "FIRE_DANGER_ASSESSMENT_NOT_FOUND"


# ---------------------------------------------------------------------------
# 7. Empty DB behavior
# ---------------------------------------------------------------------------


def test_areas_latest_empty_database_returns_200_with_empty_list():
    result = FireDangerAreasResult(as_of=AS_OF, areas=())
    client = client_for(FakeFireDangerQueryService(all_areas_result=result))

    response = client.get(AREAS_ENDPOINT)

    assert response.status_code == 200
    assert response.json() == {"as_of": "2026-09-19T12:30:00Z", "areas": []}


# ---------------------------------------------------------------------------
# 8. Correct enum JSON values
# ---------------------------------------------------------------------------


def test_all_levels_serialize_as_stable_lowercase_strings():
    for level in (
        FireDangerLevel.LOW,
        FireDangerLevel.MODERATE,
        FireDangerLevel.HIGH,
        FireDangerLevel.VERY_HIGH,
        FireDangerLevel.EXTREME,
    ):
        area = make_area(assessment=make_assessment_summary(level=level))
        client = client_for(FakeFireDangerQueryService(area_by_id={"area-carmel": area}))

        body = client.get(single_area_endpoint("area-carmel")).json()

        assert body["assessment"]["level"] == level.value
        assert "FireDangerLevel" not in str(body)


# ---------------------------------------------------------------------------
# 9. Coordinates/radius
# ---------------------------------------------------------------------------


def test_single_area_returns_center_and_radius():
    area = make_area(area_latitude=31.774, area_longitude=35.139, area_radius_km=7.5)
    client = client_for(FakeFireDangerQueryService(area_by_id={"area-carmel": area}))

    body = client.get(single_area_endpoint("area-carmel")).json()

    assert body["center"] == {"latitude": 31.774, "longitude": 35.139}
    assert body["radius_km"] == 7.5


# ---------------------------------------------------------------------------
# 10. Timestamps are timezone-aware serialized ISO-8601
# ---------------------------------------------------------------------------


def test_timestamps_serialize_with_timezone_information():
    area = make_area()
    client = client_for(FakeFireDangerQueryService(area_by_id={"area-carmel": area}))

    body = client.get(single_area_endpoint("area-carmel")).json()

    value = body["assessment"]["assessed_at"]
    assert value.endswith("Z") or "+" in value[-6:]
    assert parse_dt(value).tzinfo is not None
    assert parse_dt(value) == ASSESSED_AT


# ---------------------------------------------------------------------------
# 11. No ORM/internal leakage
# ---------------------------------------------------------------------------


def test_single_area_response_contains_only_defined_dto_fields():
    area = make_area()
    client = client_for(FakeFireDangerQueryService(area_by_id={"area-carmel": area}))

    body = client.get(single_area_endpoint("area-carmel")).json()

    assert set(body.keys()) == {"area_id", "area_name", "center", "radius_km", "assessment"}
    assert set(body["assessment"].keys()) == {
        "assessment_id",
        "status",
        "score",
        "level",
        "assessed_at",
        "age_seconds",
        "methodology",
        "methodology_version",
    }
    forbidden = ("_sa_instance_state", "danger_level", "FireDangerAssessmentDB")
    body_text = str(body)
    for field in forbidden:
        assert field not in body_text


def test_assessment_detail_response_contains_only_defined_dto_fields():
    detail = make_detail()
    client = client_for(FakeFireDangerQueryService(detail_by_id={1: detail}))

    body = client.get(assessment_endpoint(1)).json()

    assert set(body.keys()) == {
        "assessment_id",
        "area_id",
        "area_name",
        "center",
        "radius_km",
        "status",
        "score",
        "level",
        "assessed_at",
        "age_seconds",
        "methodology",
        "methodology_version",
        "weather_inputs",
    }
    assert set(body["weather_inputs"][0].keys()) == {
        "observation_id",
        "station_external_id",
        "station_name",
        "observed_at",
    }


# ---------------------------------------------------------------------------
# 12. GET has no side effects
# ---------------------------------------------------------------------------


def test_get_requests_never_call_a_write_method():
    """The fake service exposes only read methods - if the router called
    anything else (e.g. an assessment/save method), this fake would raise
    AttributeError, which TestClient surfaces as a 500."""
    area = make_area()
    service = FakeFireDangerQueryService(area_by_id={"area-carmel": area})
    client = client_for(service)

    client.get(AREAS_ENDPOINT)
    client.get(single_area_endpoint("area-carmel"))

    assert not hasattr(service, "save_assessment")
    assert service.get_latest_for_all_areas_calls == 1
    assert service.get_latest_for_area_calls == ["area-carmel"]


def test_router_source_calls_no_write_operation():
    import inspect

    for route in fire_danger_router.routes:
        source = inspect.getsource(route.endpoint)
        for forbidden in (".save_assessment(", ".assess(", "FireDangerAssessmentAgent", "FFWICalculator"):
            assert forbidden not in source


def test_router_module_does_not_import_agents_calculators_or_simulation():
    forbidden_fragments = (
        "src.agents",
        "src.external",
        "src.simulation",
        "src.calculators",
        "sqlalchemy",
    )
    path = Path(__file__).resolve().parents[2] / "src" / "api" / "routers" / "fire_danger.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        module = ""
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
        elif isinstance(node, ast.Import):
            module = ",".join(alias.name for alias in node.names)
        if any(fragment in module for fragment in forbidden_fragments):
            violations.append(module)

    assert violations == []


# ---------------------------------------------------------------------------
# 13. Repeated GET returns same persisted state if DB did not change
# (age_seconds legitimately ticks up between calls - everything else must not)
# ---------------------------------------------------------------------------


def test_repeated_get_returns_same_persisted_fields():
    area = make_area()
    client = client_for(FakeFireDangerQueryService(area_by_id={"area-carmel": area}))

    first = client.get(single_area_endpoint("area-carmel")).json()
    time.sleep(0.01)
    second = client.get(single_area_endpoint("area-carmel")).json()

    for key in ("area_id", "area_name", "center", "radius_km"):
        assert first[key] == second[key]
    for key in ("assessment_id", "status", "score", "level", "assessed_at", "methodology", "methodology_version"):
        assert first["assessment"][key] == second["assessment"][key]
    assert second["assessment"]["age_seconds"] >= first["assessment"]["age_seconds"]


# ---------------------------------------------------------------------------
# OpenAPI
# ---------------------------------------------------------------------------


def test_all_three_endpoints_are_registered_in_openapi_schema():
    app = create_app()

    schema = app.openapi()

    assert AREAS_ENDPOINT in schema["paths"]
    assert "/api/v1/fire-danger/areas/{area_id}/latest" in schema["paths"]
    assert "/api/v1/fire-danger/assessments/{assessment_id}" in schema["paths"]

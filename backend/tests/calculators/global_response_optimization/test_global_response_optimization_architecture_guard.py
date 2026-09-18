"""Architecture guard for the Global GA package (Stage 4, Task 37).

Confirms src/calculators/global_response_optimization/ never imports
sqlalchemy, repositories, database, external providers, DijkstraCalculator,
or agents - it may only depend on GlobalPlanningInput/GlobalRouteMatrix/
other domain models and pure config/calculator code, matching Task 1's
"the GA operates purely on the supplied snapshot" boundary.
"""
from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_MODULE_FRAGMENTS = (
    "sqlalchemy",
    "src.repositories",
    "src.database",
    "src.agents",
    "src.simulation",
    "src.external",
    "road_network",
    "dijkstra",
)
FORBIDDEN_NAMES = (
    "DijkstraCalculator",
    "NodeMappingService",
    "RoadNetworkFetcher",
    "RoadNetworkRepository",
)

PACKAGE_DIR = Path(__file__).resolve().parents[3] / "src" / "calculators" / "global_response_optimization"


def _module_name(node: ast.AST) -> str:
    if isinstance(node, ast.ImportFrom):
        return node.module or ""
    if isinstance(node, ast.Import):
        return ",".join(alias.name for alias in node.names)
    return ""


def test_global_response_optimization_package_has_no_forbidden_imports():
    assert PACKAGE_DIR.is_dir(), f"expected package directory at {PACKAGE_DIR}"
    python_files = sorted(PACKAGE_DIR.glob("*.py"))
    assert python_files, "expected at least one source file in the Global GA package"

    violations = []
    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = _module_name(node)
            if module and any(fragment in module for fragment in FORBIDDEN_MODULE_FRAGMENTS):
                violations.append((path.name, module))
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                violations.append((path.name, node.id))

    assert violations == []


def test_global_response_optimization_package_imports_only_expected_model_roots():
    """Every src.models import in this package must come from an
    allow-listed domain model, not e.g. a persistence-adjacent one."""
    allowed_model_modules = {
        "src.models",
        "src.models.current_global_assignment",
        "src.models.demand_source",
        "src.models.dispatch_state",
        "src.models.fire_severity_level",
        "src.models.global_allocation_slot",
        "src.models.global_event_optimization_result",
        "src.models.global_event_resource_demand_result",
        "src.models.global_incident_demand",
        "src.models.global_optimization_result",
        "src.models.global_planning_input",
        "src.models.global_planning_resource",
        "src.models.global_planning_target",
        "src.models.global_resource_shortage",
        "src.models.global_response_action",
        "src.models.global_response_plan_chromosome",
        "src.models.global_route_matrix",
        "src.models.global_route_option",
        "src.models.optimization_validation",
        "src.models.resource_status",
        "src.models.response_target_type",
    }

    violations = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src.models"):
                if node.module not in allowed_model_modules:
                    violations.append((path.name, node.module))

    assert violations == []

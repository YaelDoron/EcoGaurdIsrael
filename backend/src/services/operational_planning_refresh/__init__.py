"""US 4.4 -> US 5.4 orchestration bridge.

Triggers a Response Planning Refresh (US 5.4) after a meaningful
Operational Refresh (US 4.4) completes, without either side owning the
other's internal logic: OperationalRefreshOrchestrator stays severity/
spread/target-only, and ResponsePlanningRefreshOrchestrator stays the only
place that decides NO_OP vs. a full routing/optimization/baseline cycle.

This package's __init__.py deliberately does not eagerly re-export its
contents, unlike this project's other service packages (e.g.
src/services/response_planning/__init__.py). Eagerly importing
ResponsePlanningRefreshOrchestrator here would pull in Epic 5's routing
package (via OperationalContextService/RoadNetworkFetcher) and its optional
osmnx dependency just to import this coordinator - the coordinator itself
depends only on a narrow Protocol (see operational_planning_refresh_ports.py)
precisely to avoid that. Import submodules directly:
    from src.services.operational_planning_refresh.operational_planning_refresh_coordinator import (
        OperationalPlanningRefreshCoordinator,
    )
Only operational_planning_refresh_production_factory.py needs the real
ResponsePlanningRefreshOrchestrator (to build one), and inherits its osmnx
dependency exactly as response_planning_production_factory.py already does.
"""
from __future__ import annotations

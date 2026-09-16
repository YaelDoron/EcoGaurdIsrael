"""Application services for input preparation and coordination.

Service exports are loaded lazily so importing one service subpackage does not
pull in unrelated operational dependencies.
"""
from __future__ import annotations

_EXPORTS = {
    "FireDangerInputService": "src.services.fire_danger",
    "haversine_distance_km": "src.services.fire_danger",
    "FireDetectionEvidenceService": "src.services.fire_detection",
    "OperationalContextService": "src.services.operational",
    "FireSeverityInputService": "src.services.fire_severity",
    "ResponseOptimizationInputService": "src.services.response_optimization",
    "ResponseOptimizationInputServiceError": "src.services.response_optimization",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value

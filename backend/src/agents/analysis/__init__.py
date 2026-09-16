"""Analysis agents for EcoGuard orchestration workflows.

Agent classes are loaded lazily so importing one analysis submodule does not
pull in unrelated service dependencies.
"""
from __future__ import annotations

_EXPORTS = {
    "FireDangerAssessmentAgent": "src.agents.analysis.fire_danger_assessment_agent",
    "FireDangerAssessmentResult": "src.agents.analysis.fire_danger_assessment_result",
    "FireDetectionAgent": "src.agents.analysis.fire_detection_agent",
    "FireDetectionResult": "src.agents.analysis.fire_detection_result",
    "FireSeverityAssessmentAgent": "src.agents.analysis.fire_severity_assessment_agent",
    "FireSpreadPredictionAgent": "src.agents.analysis.fire_spread_prediction_agent",
    "ResponseTargetGenerationAgent": "src.agents.analysis.response_target_generation_agent",
    "ResponseTargetGenerationResult": "src.agents.analysis.response_target_generation_result",
    "ResponseTargetGenerationStatus": "src.agents.analysis.response_target_generation_result",
    "ResponseOptimizationAgent": "src.agents.analysis.response_optimization_agent",
    "ResponseOptimizationResult": "src.agents.analysis.response_optimization_result",
    "ResponseOptimizationStatus": "src.agents.analysis.response_optimization_result",
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

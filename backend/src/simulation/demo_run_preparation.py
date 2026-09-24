"""The ONE way a demo simulation run is prepared: verify the reset is enabled, then reset runtime state.

Task 9A: a demo simulation must never begin on top of a previous run's FireEvents, evidence, event history, ML
assessments or derived response data. Every supported start path (the API via SimulationRunManager, and the CLI in
scripts/run_demo_simulation.py) calls these two functions BEFORE constructing or running a scenario, so the reset is
mandatory - there is no opt-out flag.

Fail-closed: when ENABLE_DEMO_DATA_RESET is not enabled, `require_demo_reset_enabled()` raises before any simulated
event exists. The run is never started "and merely logged as skipped".

What is reset is decided by DemoStateResetService's own table declaration (the source of truth); nothing here widens
or narrows it.
"""
from __future__ import annotations

from typing import Callable

from src.simulation.demo_state_reset_service import DemoStateResetResult, DemoStateResetService


class DemoResetRequiredError(RuntimeError):
    """A demo simulation cannot start because the mandatory runtime reset is not enabled."""


def require_demo_reset_enabled() -> None:
    """Raise DemoResetRequiredError unless ENABLE_DEMO_DATA_RESET is explicitly true (checked on every call)."""
    # Imported lazily and by module (not `from src.config import settings`): the package re-exports the object
    # under the same name as the submodule, and tests replace it on the submodule.
    import sys

    import src.config.settings  # noqa: F401 - ensures the submodule is in sys.modules

    settings = sys.modules["src.config.settings"].settings
    if not settings.ENABLE_DEMO_DATA_RESET:
        raise DemoResetRequiredError(
            "A demo simulation requires a clean runtime state, but ENABLE_DEMO_DATA_RESET is not enabled. "
            "Enable it (only on a dedicated, disposable demo database) to run a demo simulation."
        )


def prepare_clean_demo_state(
    reset_service_factory: Callable[[], object] = DemoStateResetService,
) -> DemoStateResetResult | None:
    """Reset all runtime/demo state (mandatory pre-step of every demo run). Raises if it cannot be done."""
    require_demo_reset_enabled()
    return reset_service_factory().reset_demo_state()

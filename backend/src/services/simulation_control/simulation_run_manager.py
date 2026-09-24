"""SimulationRunManager: lifecycle/state management for one background
demo simulation run at a time (Task A3).

Responsible ONLY for:
- ensuring a single run is active per process,
- owning current run metadata (a thread-safe, immutable snapshot),
- starting DemoSimulationRunner in a background worker,
- consuming its on_progress callback and folding it into the snapshot,
- storing the final result or a sanitized failure summary.

It does NOT implement Fire Detection, Severity, Spread, routing, or GA -
DemoSimulationRunner (src/simulation/demo_simulation_runner.py) remains the
one and only simulation execution implementation; this manager only calls
`.run(...)` on it, exactly like the CLI does. It does NOT know FastAPI/HTTP
(no status codes, no request/response objects) and does NOT produce
terminal output - see src/api/routers/simulation.py for the HTTP transport
built on top of this.

Process-local by design (Part 19/20 of the Task A3 audit): run state lives
in this process's memory only. A FastAPI restart loses it (DB writes
already made by the runner are unaffected); running with more than one
Uvicorn worker would give each worker its own independent manager, able to
each start their own run - this MVP requires a single worker. Neither
limitation is solved here.
"""
from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import logging
import random
import threading
from typing import Callable
from uuid import uuid4

from src.simulation.demo_simulation_runner import (
    DemoSimulationEventPhase,
    DemoSimulationEventProgress,
    DemoSimulationRunConfig,
    DemoSimulationRunner,
    DemoSimulationRunResult,
    DemoSimulationStatus,
)
from src.simulation.demo_run_preparation import DemoResetRequiredError, prepare_clean_demo_state, require_demo_reset_enabled
from src.simulation.demo_state_reset_service import DemoStateResetService
from src.simulation.simulation_presets import SimulationPreset, get_simulation_preset
from src.simulation.simulation_scenario_service import SimulationMode

logger = logging.getLogger(__name__)

RunnerFactory = Callable[[], "_RunnerLike"]
ResetServiceFactory = Callable[[], "_ResetServiceLike"]
ClockFn = Callable[[], datetime]


class _RunnerLike:  # pragma: no cover - structural typing helper only
    def run(self, scenario, config, scenario_started_at=None, on_progress=None) -> DemoSimulationRunResult: ...


class _ResetServiceLike:  # pragma: no cover - structural typing helper only
    def reset_demo_state(self): ...


class SimulationAlreadyRunningError(RuntimeError):
    """Raised by start_run() when a run is already PREPARING or RUNNING."""


class SimulationPresetNotFoundError(ValueError):
    """Raised by start_run() when the requested preset id is not registered."""

    def __init__(self, preset_id: str) -> None:
        super().__init__(f"Unknown simulation preset: {preset_id!r}")
        self.preset_id = preset_id


class SimulationResetDisabledError(RuntimeError):
    """Raised by start_run() when the MANDATORY demo reset cannot run because ENABLE_DEMO_DATA_RESET is not enabled."""


class SimulationResetRequiredError(ValueError):
    """Raised by start_run() when a caller asks to start a run WITHOUT the demo reset (no longer supported)."""


class SimulationRunState(Enum):
    """The manager's own lifecycle state - distinct from DemoSimulationStatus
    (the runner's result status). See _RUNNER_STATUS_TO_STATE for the
    explicit mapping between the two; they are never conflated.
    """

    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


_ACTIVE_STATES = (SimulationRunState.PREPARING, SimulationRunState.RUNNING)

# Range for auto-generated seeds (Part L) - any positive int the existing
# seeded generators/scenario builder can consume; comfortably below the
# platform's signed-32-bit range so it round-trips through any downstream
# int handling without special-casing.
_AUTO_SEED_MIN = 1
_AUTO_SEED_MAX = 2_147_483_647

_RUNNER_STATUS_TO_STATE = {
    DemoSimulationStatus.COMPLETED: SimulationRunState.COMPLETED,
    DemoSimulationStatus.COMPLETED_WITH_ERRORS: SimulationRunState.COMPLETED_WITH_ERRORS,
    DemoSimulationStatus.FAILED: SimulationRunState.FAILED,
}


@dataclass(frozen=True)
class SimulationCurrentEventSnapshot:
    """The most recently started-or-completed event, as of this snapshot."""

    event_index: int
    incident_id: str
    event_type: str
    timestamp_offset_sec: int


@dataclass(frozen=True)
class SimulationRunError:
    """A safe, sanitized failure summary - never a raw exception/traceback/credential."""

    code: str
    message: str


@dataclass(frozen=True)
class SimulationRunSnapshot:
    """Immutable, thread-safe snapshot of the current (or most recent) run.

    Handed out by reference (frozen dataclasses are safe to share) - callers
    never receive the manager's live, mutable internals.
    """

    run_id: str | None
    state: SimulationRunState
    preset_id: str | None
    seed: int | None
    mode: str
    simulation_duration_seconds: int | None
    events_total: int | None
    events_completed: int
    events_succeeded: int
    events_failed: int
    current_event: SimulationCurrentEventSnapshot | None
    started_at: datetime | None
    completed_at: datetime | None
    wall_clock_elapsed_seconds: float | None
    last_message: str | None
    error: SimulationRunError | None


def _idle_snapshot() -> SimulationRunSnapshot:
    return SimulationRunSnapshot(
        run_id=None,
        state=SimulationRunState.IDLE,
        preset_id=None,
        seed=None,
        mode="automatic",
        simulation_duration_seconds=None,
        events_total=None,
        events_completed=0,
        events_succeeded=0,
        events_failed=0,
        current_event=None,
        started_at=None,
        completed_at=None,
        wall_clock_elapsed_seconds=None,
        last_message=None,
        error=None,
    )


class SimulationRunManager:
    """Owns at most one active (PREPARING/RUNNING) demo simulation run.

    A process-local singleton is expected for this MVP (see
    `get_simulation_run_manager` in src/api/dependencies.py) - nothing here
    prevents multiple instances, but only one manager instance's state is
    meaningful per process, and only one Uvicorn worker should be used
    (Task A3, Part 20).
    """

    def __init__(
        self,
        *,
        runner_factory: RunnerFactory = DemoSimulationRunner,
        reset_service_factory: ResetServiceFactory = DemoStateResetService,
        clock_fn: ClockFn = lambda: datetime.now(timezone.utc),
        executor: Executor | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._snapshot = _idle_snapshot()
        self._runner_factory = runner_factory
        self._reset_service_factory = reset_service_factory
        self._clock_fn = clock_fn
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="simulation-run")
        # Task: when the API caller omits `seed`, start_run() picks one
        # itself (see _generate_seed) - this is simulation variability, not
        # cryptography, so plain `random` is fine. Tracked per-manager
        # (process-local, like everything else here) only to avoid
        # immediately repeating the same auto-generated seed twice in a row.
        self._last_auto_generated_seed: int | None = None

    def get_current_snapshot(self) -> SimulationRunSnapshot:
        """Return the latest known snapshot - IDLE if no run has ever started,
        otherwise the current or most recently finished run's state."""
        with self._lock:
            return self._snapshot

    def start_run(self, *, preset_id: str, seed: int | None = None, reset_demo_state: bool = True) -> SimulationRunSnapshot:
        """Validate and reserve a new run, then start it in the background.

        Task 9A - the demo reset is MANDATORY: every run first resets all runtime/demo state (FireEvents, evidence,
        event history, ML assessments, derived response data) so a new run can never inherit the previous one's Fire
        Detection history. `reset_demo_state` is kept ONLY for API/call-site compatibility and must be True; passing
        False raises SimulationResetRequiredError - there is no way to start a run without the reset. If
        ENABLE_DEMO_DATA_RESET is not enabled the start is refused (SimulationResetDisabledError) BEFORE a run is
        reserved and before any event is generated (fail-closed).

        `seed` is OPTIONAL: when the caller supplies one, it is used
        exactly as given (this is what the CLI script and reproducibility
        tests rely on). When omitted (`None` - the normal dashboard
        Start/Run Again request), a fresh seed is generated here via
        `_generate_seed()` before the run is reserved, so every field that
        already reports `seed` (the returned snapshot, `GET
        /simulation/runs/current`, and A6's `simulation.run.seed`) reflects
        the real seed this run actually used - never left null.

        Returns immediately (before the runner executes anything) with a
        PREPARING snapshot. Raises without reserving anything if:
        - the preset id is not registered (SimulationPresetNotFoundError),
        - the reset was declined (SimulationResetRequiredError),
        - the reset safety flag is off (SimulationResetDisabledError),
        - a run is already PREPARING/RUNNING (SimulationAlreadyRunningError).

        The PREPARING state is committed atomically under the same lock as
        the "no run active" check, so a second start_run() call - even one
        that arrives before the background worker has done anything at all -
        already observes this run as active.
        """
        preset = get_simulation_preset(preset_id)
        if preset is None:
            raise SimulationPresetNotFoundError(preset_id)

        if reset_demo_state is not True:
            raise SimulationResetRequiredError(
                "A demo simulation cannot start without the runtime reset; reset_demo_state must be true."
            )
        try:
            require_demo_reset_enabled()
        except DemoResetRequiredError as exc:
            raise SimulationResetDisabledError(str(exc)) from exc

        with self._lock:
            if self._snapshot.state in _ACTIVE_STATES:
                raise SimulationAlreadyRunningError(
                    f"A simulation run is already {self._snapshot.state.value}."
                )
            resolved_seed = seed if seed is not None else self._generate_seed()
            run_id = str(uuid4())
            self._snapshot = SimulationRunSnapshot(
                run_id=run_id,
                state=SimulationRunState.PREPARING,
                preset_id=preset.id,
                seed=resolved_seed,
                mode="automatic",
                simulation_duration_seconds=None,
                events_total=None,
                events_completed=0,
                events_succeeded=0,
                events_failed=0,
                current_event=None,
                started_at=None,
                completed_at=None,
                wall_clock_elapsed_seconds=None,
                last_message="Preparing simulation run.",
                error=None,
            )
            snapshot_to_return = self._snapshot

        self._executor.submit(self._execute, run_id, preset, resolved_seed)
        return snapshot_to_return

    def _generate_seed(self) -> int:
        """Pick a fresh integer seed for a run that omitted one explicitly.

        Plain, non-cryptographic randomness is intentional (Part L) - this
        is simulation variability, not a security boundary. Guards only
        against immediately repeating the previous auto-generated seed
        within this same process/manager instance; a caller who wants an
        exact, reproducible seed should simply pass one explicitly.
        """
        candidate = random.randint(_AUTO_SEED_MIN, _AUTO_SEED_MAX)
        while candidate == self._last_auto_generated_seed:
            candidate = random.randint(_AUTO_SEED_MIN, _AUTO_SEED_MAX)
        self._last_auto_generated_seed = candidate
        return candidate

    # -- background worker (never runs on the calling/request thread) ------

    def _execute(self, run_id: str, preset: SimulationPreset, seed: int) -> None:
        try:
            # Mandatory (Task 9A): nothing is generated or run until the reset has succeeded; a failed reset fails the run.
            self._replace_if_current(run_id, last_message="Resetting demo state.")
            prepare_clean_demo_state(self._reset_service_factory)

            scenario = preset.build(seed)
            started_at = self._clock_fn()
            self._replace_if_current(
                run_id,
                state=SimulationRunState.RUNNING,
                simulation_duration_seconds=scenario.duration_seconds,
                events_total=len(scenario.events),
                started_at=started_at,
                last_message="Simulation running.",
            )

            runner = self._runner_factory()
            config = DemoSimulationRunConfig(mode=SimulationMode.AUTOMATIC)
            result = runner.run(
                scenario,
                config,
                scenario_started_at=started_at,
                on_progress=lambda progress: self._apply_progress(run_id, progress),
            )
            self._apply_result(run_id, result)
        except Exception as exc:  # noqa: BLE001 - background worker boundary: never let this escape unlogged.
            logger.exception("Simulation run %s failed", run_id)
            self._apply_failure(run_id, exc)

    def _apply_progress(self, run_id: str, progress: DemoSimulationEventProgress) -> None:
        current_event = SimulationCurrentEventSnapshot(
            event_index=progress.event_index,
            incident_id=progress.incident_id,
            event_type=progress.event_type.value,
            timestamp_offset_sec=progress.timestamp_offset_sec,
        )
        with self._lock:
            if self._snapshot.run_id != run_id:
                return  # stale callback from a superseded run - ignore
            if progress.phase is DemoSimulationEventPhase.EVENT_STARTED:
                self._snapshot = replace(
                    self._snapshot,
                    current_event=current_event,
                    events_completed=progress.events_completed,
                    last_message=f"Started {progress.event_type.value} for {progress.incident_id}.",
                )
                return

            elapsed = None
            if self._snapshot.started_at is not None:
                elapsed = (self._clock_fn() - self._snapshot.started_at).total_seconds()
            self._snapshot = replace(
                self._snapshot,
                current_event=current_event,
                events_completed=progress.events_completed,
                events_succeeded=self._snapshot.events_succeeded + (1 if progress.success else 0),
                events_failed=self._snapshot.events_failed + (0 if progress.success else 1),
                wall_clock_elapsed_seconds=elapsed,
                last_message=(
                    f"Completed {progress.event_type.value} for {progress.incident_id} "
                    f"(success={progress.success})."
                ),
            )

    def _apply_result(self, run_id: str, result: DemoSimulationRunResult) -> None:
        state = _RUNNER_STATUS_TO_STATE[result.status]
        message = {
            SimulationRunState.COMPLETED: "Simulation run completed.",
            SimulationRunState.COMPLETED_WITH_ERRORS: "Simulation run completed with errors.",
            SimulationRunState.FAILED: "Simulation run failed.",
        }[state]
        self._replace_if_current(
            run_id,
            state=state,
            events_completed=result.events_executed,
            events_succeeded=result.events_succeeded,
            events_failed=result.events_failed,
            completed_at=result.simulation_completed_at,
            wall_clock_elapsed_seconds=result.wall_clock_elapsed_seconds,
            last_message=message,
        )

    def _apply_failure(self, run_id: str, exc: Exception) -> None:
        self._replace_if_current(
            run_id,
            state=SimulationRunState.FAILED,
            completed_at=self._clock_fn(),
            last_message="Simulation run failed.",
            error=SimulationRunError(code="SIMULATION_RUN_FAILED", message="Simulation run failed."),
        )

    def _replace_if_current(self, run_id: str, **changes: object) -> None:
        with self._lock:
            if self._snapshot.run_id != run_id:
                return
            self._snapshot = replace(self._snapshot, **changes)

"""The `presentation_demo` scenario: a fixed, paced timeline for live presentations.

`operations_demo` (simulation_scenario.build_operations_demo_scenario) draws
its locations, event order AND timing from the seed and ends after ~7-8
minutes, with its first events at T+0. For a live presentation that is the
wrong shape: the dashboard fills up before anyone can see it start, the
first fire can take minutes, and the run ends while the presenter is still
walking through Event Details / Spread / the Response Plan.

This builder keeps every simulation rule and only changes the SCHEDULE:

* The event timeline below is fixed (the same for every seed): a quiet
  start (first environmental update at T+4s), two active-fire incidents
  that are SUSPECTED on their first satellite pass (T+19s / T+24s) and
  CONFIRMED on their second (T+35s / T+41s) - both within the first ~90s
  including downstream processing - a third fire at ~3 minutes and a
  fourth (a high-risk area that later ignites) at ~8 minutes, then a
  steady cadence of one update every 26-36s until the 30-minute end - or
  until the operator presses Stop (SimulationRunManager.stop_run).
* The seed still drives every generated VALUE (weather, satellite
  properties, news text/coordinates) through the unchanged generators and
  each event's schedule seed key, so one seed always reproduces the same
  values, AI inputs and outcomes. PRESENTATION_DEFAULT_SEED is the approved
  seed for this timeline (see its comment).
* Each incident's evidence stays in its natural order (weather -> satellite
  -> news -> more satellite ...). One satellite pass yields one hotspot, so
  a fire can only be CONFIRMED after its second pass (AI Hybrid V5 requires
  at least 2 current pixels) - every fire gets a visible SUSPECTED phase
  first. Nothing here fabricates detection, severity, spread or planning
  results.
* All locations come from the canonical SIMULATION_LOCATIONS registry.
"""
from __future__ import annotations

from collections import defaultdict

from src.simulation.scenario_type import ScenarioType
from src.simulation.simulated_incident import SimulatedIncident
from src.simulation.simulation_event import SimulationEvent, SimulationEventType
from src.simulation.simulation_locations import SIMULATION_LOCATIONS
from src.simulation.simulation_scenario import SimulationScenario, build_multi_incident_scenario

PRESENTATION_DURATION_SECONDS = 1800

# The approved presentation seed for THIS timeline. Event values are seeded
# per scheduled event (type, index and offset), so a seed chosen for another
# schedule does not carry over; it was selected offline with the real
# generators, V5 feature extractor, HGB V5 model and AI Hybrid policy - never
# by changing a threshold - see scripts/select_presentation_seed.py.
PRESENTATION_DEFAULT_SEED = 594

W = SimulationEventType.WEATHER
S = SimulationEventType.SATELLITE
N = SimulationEventType.NEWS

JUDEAN_HILLS = "incident-judean-hills-01"
GALILEE = "incident-galilee-01"
GOLAN = "incident-golan-01"
CARMEL_HIGH_RISK = "incident-carmel-high-01"
CARMEL_FIRE = "incident-carmel-01"
JERUSALEM_FOREST = "incident-jerusalem-forest-moderate-01"

PRESENTATION_INCIDENTS: tuple[tuple[str, ScenarioType, str], ...] = (
    (JUDEAN_HILLS, ScenarioType.ACTIVE_FIRE, "judean_hills"),
    (GALILEE, ScenarioType.ACTIVE_FIRE, "galilee"),
    (GOLAN, ScenarioType.ACTIVE_FIRE, "golan"),
    # Carmel is a high-risk area first; the separate fire incident at the same
    # location only starts producing evidence ~9 minutes in.
    (CARMEL_HIGH_RISK, ScenarioType.HIGH_RISK_NO_FIRE, "carmel"),
    (CARMEL_FIRE, ScenarioType.ACTIVE_FIRE, "carmel"),
    (JERUSALEM_FOREST, ScenarioType.MODERATE_RISK_NO_FIRE, "jerusalem_forest"),
)

# (offset seconds, event type, incident) - the scripted first ~8 minutes.
# Both first fires are confirmed inside the first ~90 s. Every confirmation
# runs severity/spread/targets/planning inside its event (~25-35 s measured
# live) and a news event takes ~10 s, so only fire A's news precedes the
# first confirmation, the second fire's confirming pass is queued directly
# behind the first one (T+35 / T+41) with nothing in between, and fire B's
# news follows its confirmation.
PRESENTATION_OPENING: tuple[tuple[int, SimulationEventType, str], ...] = (
    # Quiet start, then the first environmental updates.
    (4, W, CARMEL_HIGH_RISK),
    (9, W, JUDEAN_HILLS),
    (14, W, GALILEE),
    # Fire A (Judean Hills) first satellite pass -> SUSPECTED.
    (19, S, JUDEAN_HILLS),
    # Fire B (Galilee) first satellite pass -> SUSPECTED.
    (24, S, GALILEE),
    (29, N, JUDEAN_HILLS),
    # Fire A second pass -> CONFIRMED -> severity, spread, targets, plans.
    (35, S, JUDEAN_HILLS),
    # Fire B second pass -> CONFIRMED (processed right after fire A's downstream work).
    (41, S, GALILEE),
    # Fire B's news corroboration follows its confirmation.
    (70, N, GALILEE),
    (75, W, JERUSALEM_FOREST),
    (100, W, GOLAN),
    (125, W, JUDEAN_HILLS),
    (150, N, JUDEAN_HILLS),
    # Fire C (Golan) appears: SUSPECTED, monitored until its second pass.
    (175, S, GOLAN),
    (200, W, CARMEL_HIGH_RISK),
    (225, W, GALILEE),
    (250, N, GOLAN),
    (275, S, JUDEAN_HILLS),
    (300, W, JERUSALEM_FOREST),
    (325, W, GOLAN),
    (350, N, GALILEE),
    (375, S, GALILEE),
    (400, W, CARMEL_HIGH_RISK),
    # The Carmel high-risk area escalates into a fire (fire D).
    (425, W, CARMEL_FIRE),
    (450, N, GOLAN),
    (475, S, CARMEL_FIRE),
)

# The steady phase after the opening: this rotation repeats until the end,
# one update per PRESENTATION_STEADY_GAPS entry (cycled). Every active fire
# keeps receiving weather, satellite and news updates; the risk-only area
# keeps its fire-danger updates.
PRESENTATION_STEADY_ROTATION: tuple[tuple[SimulationEventType, str], ...] = (
    (W, JUDEAN_HILLS),
    (N, CARMEL_FIRE),
    (W, GALILEE),
    (W, GOLAN),
    (S, JUDEAN_HILLS),
    (W, JERUSALEM_FOREST),
    (W, CARMEL_FIRE),
    (N, JUDEAN_HILLS),
    (S, GOLAN),
    (W, GALILEE),
    (N, GOLAN),
    (S, CARMEL_FIRE),
    (N, GALILEE),
    (S, GALILEE),
)
PRESENTATION_STEADY_GAPS: tuple[int, ...] = (30, 26, 34, 28, 36)


def build_presentation_timeline() -> tuple[tuple[int, SimulationEventType, str], ...]:
    """The full (offset, event type, incident id) schedule - identical for every seed."""
    timeline = list(PRESENTATION_OPENING)
    offset = timeline[-1][0]
    slot = 0
    while True:
        offset += PRESENTATION_STEADY_GAPS[slot % len(PRESENTATION_STEADY_GAPS)]
        if offset > PRESENTATION_DURATION_SECONDS:
            break
        event_type, incident_id = PRESENTATION_STEADY_ROTATION[slot % len(PRESENTATION_STEADY_ROTATION)]
        timeline.append((offset, event_type, incident_id))
        slot += 1
    return tuple(timeline)


def build_presentation_demo_scenario(seed: int = PRESENTATION_DEFAULT_SEED) -> SimulationScenario:
    """Build the paced presentation scenario (fixed schedule; `seed` drives generated values)."""
    incidents = tuple(
        SimulatedIncident(incident_id=incident_id, scenario_type=scenario_type, location=SIMULATION_LOCATIONS[key])
        for incident_id, scenario_type, key in PRESENTATION_INCIDENTS
    )
    events: list[SimulationEvent] = []
    per_incident_type_index: dict[tuple[str, SimulationEventType], int] = defaultdict(int)
    for offset_seconds, event_type, incident_id in build_presentation_timeline():
        key = (incident_id, event_type)
        events.append(
            SimulationEvent(
                offset_seconds=offset_seconds,
                event_type=event_type,
                incident_id=incident_id,
                source_event_index=per_incident_type_index[key],
            )
        )
        per_incident_type_index[key] += 1
    return build_multi_incident_scenario(
        incidents=incidents,
        events=events,
        duration_seconds=PRESENTATION_DURATION_SECONDS,
        seed=seed,
    )

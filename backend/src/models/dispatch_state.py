"""DispatchState: the minimum explicit state needed to distinguish a
tentatively-planned resource commitment from one EcoGuard already considers
physically dispatched/en route (Stage 6 of the Global Multi-Incident
Optimizer refactor).

Audited first (see Stage 6 audit): neither ResourceCommitment's mere
existence nor FirefightingResource.status (AVAILABLE/ASSIGNED/UNAVAILABLE -
a coarse, manually/externally-set operational flag, decoupled from
commitments and never written by plan activation) can answer "is this
resource still reassignable, or has EcoGuard already assumed it left the
station?" This two-value field is deliberately the smallest addition that
answers that question - not a larger resource state machine.
"""
from __future__ import annotations

from enum import Enum


class DispatchState(Enum):
    """One ResourceCommitment's reassignability.

    PLANNED: tentatively selected by a global optimization; still fully
    reassignable by ordinary global replanning (soft stability preference
    only, never a hard constraint).

    DISPATCHED: EcoGuard now assumes this resource has left its station and
    is travelling toward its committed FireEvent. Ordinary global replanning
    must treat this as a HARD constraint - the resource may be reconsidered
    for a different slot within the SAME FireEvent, but must never be moved
    to another FireEvent, and never released, except through a real
    operational lifecycle event (FireEvent resolve/dismiss, or an explicit
    future cancellation/release feature - never implicit recall through
    optimization).
    """

    PLANNED = "planned"
    DISPATCHED = "dispatched"

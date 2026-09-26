"""The approved presentation seed still tells the presentation story (offline, real model).

Replays the presentation timeline for PRESENTATION_DEFAULT_SEED through the
real generators, V5 feature extractor, HGB V5 model and AI Hybrid V5 policy
(scripts/select_presentation_seed.py) for a daytime and a night-time start.
Nothing is hard-coded: if a generator, feature, model or threshold change
altered the outcome, this fails instead of the presentation silently
changing.
"""
from __future__ import annotations

import pytest

from scripts.select_presentation_seed import _Predictor, evaluate
from src.simulation.presentation_scenario import PRESENTATION_DEFAULT_SEED


@pytest.fixture(scope="module")
def story():
    return evaluate(_Predictor(), PRESENTATION_DEFAULT_SEED)


def test_default_seed_satisfies_every_story_check_day_and_night(story):
    ok, notes = story

    assert ok, notes


def test_story_has_suspected_then_confirmed_fires_and_a_monitoring_only_fire(story):
    _, notes = story
    by_incident = {note.split(":")[0]: note for note in notes if note.startswith("incident-")}

    judean = by_incident["incident-judean-hills-01"]
    galilee = by_incident["incident-galilee-01"]
    golan = by_incident["incident-golan-01"]
    # Both opening fires: SUSPECTED on the first pass, CONFIRMED on the second (T+35 / T+41).
    assert "T+19 sat" in judean and "T+35 sat p=" in judean and judean.index("SUSPECTED") < judean.index("CONFIRMED")
    assert "T+24 sat" in galilee and "T+41 sat p=" in galilee and galilee.index("SUSPECTED") < galilee.index("CONFIRMED")
    assert judean.split("T+35 sat")[1].split("->")[0].strip().endswith("CONFIRMED")
    assert galilee.split("T+41 sat")[1].split("->")[0].strip().endswith("CONFIRMED")
    assert "SUSPECTED" in golan and "CONFIRMED" not in golan

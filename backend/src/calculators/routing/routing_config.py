"""Centralized EcoGuard V1 routing methodology constants.

Mirrors response_target_config.py / fire_spread_config.py: centralized and
versioned so future routing algorithm changes (e.g. a different shortest-path
strategy) can change the recorded methodology without touching callers.
"""
from __future__ import annotations

ROUTING_METHODOLOGY_NAME = "ECOGUARD_ROUTING_DIJKSTRA"
ROUTING_METHODOLOGY_VERSION = "1.0"

"""Centralized EcoGuard V1 response-target methodology constants.

These are application-level operational rules, not scientific wildfire
severity or spread standards. They are centralized and versioned so future
calibration can change policy without changing the pure calculator boundary.
"""
from __future__ import annotations

RESPONSE_TARGET_METHODOLOGY_NAME = "ECOGUARD_RESPONSE_TARGET_PRIORITY"
RESPONSE_TARGET_METHODOLOGY_VERSION = "1.0"

ACTIVE_FIRE_BASE_PRIORITY = 100.0

# Fire Spread's threshold decides whether a location belongs to a predicted
# spread result. This separate threshold decides whether an already predicted
# location is operationally important enough to become a response target.
MIN_PREDICTED_TARGET_RISK_SCORE = 60.0

# Risk expresses the importance/likelihood of predicted impact; the horizon
# factor expresses operational urgency. Unsupported horizons fail explicitly.
PREDICTION_HORIZON_FACTORS = {
    30: 1.00,
    60: 0.85,
}

TARGET_DEDUP_DISTANCE_METERS = 100.0

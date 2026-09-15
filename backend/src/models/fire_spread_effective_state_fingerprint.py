"""Validation for persisted fire-spread effective-state fingerprints."""
from __future__ import annotations

import re

_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate_effective_state_fingerprint(value: str | None) -> None:
    """Validate the persisted Task 1 SHA-256 fingerprint representation."""
    if value is None:
        return
    if not isinstance(value, str) or _SHA256_HEX_PATTERN.fullmatch(value) is None:
        raise ValueError("effective_state_fingerprint must be None or a 64-character lowercase SHA-256 hex digest.")

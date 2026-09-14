"""Types of direct evidence that may indicate an active wildfire."""
from __future__ import annotations

from enum import Enum


class FireEvidenceType(Enum):
    """Direct wildfire evidence source families.

    Weather/fire danger is intentionally absent: high FFWI is context, not
    direct evidence that an active wildfire exists.
    """

    SATELLITE = "satellite"
    NEWS = "news"

"""Result summary for a single SatelliteHotspotAgent.collect() run."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SatelliteHotspotCollectionResult:
    """Summary of one satellite hotspot collection cycle.

    `success` is False only when the cycle cannot begin because FIRMS
    retrieval failed. Per-detection mapping or persistence failures are counted
    in `detections_failed`, while the overall cycle remains successful because
    other detections could still be processed.

    Counter semantics:
    - detections_received: number of raw detections returned by FIRMSClient.
    - hotspots_saved: number of newly persisted satellite hotspots.
    - duplicates_skipped: number of detections already stored by repository identity.
    - ignored_outside_area: number of valid-coordinate detections outside configured bounds.
    - detections_failed: number of malformed-coordinate, mapping, or repository failures.
    """

    success: bool
    detections_received: int = 0
    hotspots_saved: int = 0
    duplicates_skipped: int = 0
    ignored_outside_area: int = 0
    detections_failed: int = 0
    error_message: Optional[str] = None

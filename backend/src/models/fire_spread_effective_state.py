"""Pure effective-state identity for wildfire-spread recalculation decisions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from src.calculators.fire_spread.fire_spread_config import METHODOLOGY_NAME, METHODOLOGY_VERSION
from src.models.fire_spread_effective_state_fingerprint import validate_effective_state_fingerprint
from src.models.fire_spread_fuel_class import FireSpreadFuelClass
from src.models.fire_spread_input import FireSpreadInput


@dataclass(frozen=True)
class FireSpreadEffectiveState:
    """Calculator-relevant identity for one FireEvent and prediction horizon.

    This model is constructed from the already prepared production
    FireSpreadInput. It does not retrieve FireEvents, Severity, Weather,
    vegetation, or repositories, and it deliberately excludes provenance IDs
    and processing timestamps that do not affect FireSpreadCalculator output.
    """

    fire_event_id: int
    origin_latitude: float
    origin_longitude: float
    wind_speed_kmh: float
    wind_direction_deg: float
    fuel_moisture_percent: float
    fuel_class: FireSpreadFuelClass
    horizon_minutes: int
    methodology: str = METHODOLOGY_NAME
    methodology_version: str = METHODOLOGY_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {self.fire_event_id!r}")
        if not isinstance(self.fuel_class, FireSpreadFuelClass):
            raise ValueError(f"fuel_class must be a FireSpreadFuelClass, got {self.fuel_class!r}")
        self._validate_non_empty_string("methodology", self.methodology)
        self._validate_non_empty_string("methodology_version", self.methodology_version)

    @classmethod
    def from_input(
        cls,
        *,
        fire_event_id: int,
        spread_input: FireSpreadInput,
        methodology: str = METHODOLOGY_NAME,
        methodology_version: str = METHODOLOGY_VERSION,
    ) -> "FireSpreadEffectiveState":
        """Build an effective state from an existing READY FireSpreadInput."""
        if not isinstance(spread_input, FireSpreadInput):
            raise ValueError(f"spread_input must be a FireSpreadInput, got {spread_input!r}")
        return cls(
            fire_event_id=fire_event_id,
            origin_latitude=spread_input.origin_latitude,
            origin_longitude=spread_input.origin_longitude,
            wind_speed_kmh=spread_input.wind_speed_kmh,
            wind_direction_deg=spread_input.wind_direction_deg,
            fuel_moisture_percent=spread_input.fuel_moisture_percent,
            fuel_class=spread_input.fuel_class,
            horizon_minutes=spread_input.horizon_minutes,
            methodology=methodology,
            methodology_version=methodology_version,
        )

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 hex digest of the canonical effective state."""
        payload = json.dumps(self._canonical_payload(), separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _canonical_payload(self) -> tuple[tuple[str, object], ...]:
        return (
            ("fire_event_id", self.fire_event_id),
            ("origin_latitude", self.origin_latitude),
            ("origin_longitude", self.origin_longitude),
            ("wind_speed_kmh", self.wind_speed_kmh),
            ("wind_direction_deg", self.wind_direction_deg),
            ("fuel_moisture_percent", self.fuel_moisture_percent),
            ("fuel_class", self.fuel_class.value),
            ("horizon_minutes", self.horizon_minutes),
            ("methodology", self.methodology),
            ("methodology_version", self.methodology_version),
        )

    @staticmethod
    def _validate_non_empty_string(field_name: str, value: object) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")


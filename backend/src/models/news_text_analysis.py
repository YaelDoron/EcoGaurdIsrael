"""Structured result of one LLM analysis pass over a news article's title/summary."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength


@dataclass(frozen=True)
class NewsTextAnalysis:
    """One article's LLM-derived location and wildfire signal strength.

    wildfire_signal_strength=None means no reliable analysis result is
    available (LLM call failed, response was malformed, or returned an
    unrecognized value) - distinct from NewsWildfireSignalStrength.NONE,
    which means the LLM successfully analyzed the text and found no
    meaningful active-fire signal. Callers must not conflate the two.
    """

    location_name: str | None
    wildfire_signal_strength: NewsWildfireSignalStrength | None

    def __post_init__(self) -> None:
        if self.location_name is not None and not isinstance(self.location_name, str):
            raise ValueError(f"location_name must be a string or None, got {self.location_name!r}")
        if self.wildfire_signal_strength is not None and not isinstance(
            self.wildfire_signal_strength, NewsWildfireSignalStrength
        ):
            raise ValueError(
                "wildfire_signal_strength must be a NewsWildfireSignalStrength or None, "
                f"got {self.wildfire_signal_strength!r}"
            )

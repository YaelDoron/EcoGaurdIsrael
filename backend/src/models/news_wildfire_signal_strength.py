"""Categorical strength of active-wildfire evidence expressed by a news article's own text."""
from __future__ import annotations

from enum import Enum


class NewsWildfireSignalStrength(Enum):
    """How strongly an article's TEXT claims an active wildfire is occurring.

    This is a semantic reading of the article's own wording, produced by an
    LLM (at runtime) or a deterministic synthetic generator (during ML
    training) - it is NEITHER ground truth NOR a calibrated probability. A
    STRONG article may still be wrong; a WEAK article may still describe a
    real fire. That overlap is intentional (see backend/docs/
    fire_detection_feature_representation_v3.md).

    NONE means the text was analyzed and found to lack meaningful active-fire
    evidence (e.g. retrospective/preventive articles, or an explicit report
    that a suspected fire was false) - distinct from no analysis being
    available at all (represented by this field being None/absent wherever
    it is used, e.g. on WildfireReport or FireDetectionEvidence).
    """

    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"

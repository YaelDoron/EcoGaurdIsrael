"""Deprecated: road network pre-seeding is no longer required.

The road network used to be pre-seeded here, in bulk, for each predefined
simulation location using osmnx. OperationalContextService now lazy-loads
the road network on demand: it checks RoadNetworkRepository for cached data
covering an incident's bounding box, and only on a cache miss fetches it
live from OSM via RoadNetworkFetcher (backend/src/services/operational/
road_network_fetcher.py) and caches the result for next time.

This script is kept as a placeholder so old invocations (docs, CI, muscle
memory) fail loudly with an explanation instead of silently disappearing.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEPRECATION_MESSAGE = (
    "Road network pre-seeding is no longer required. The system now "
    "automatically lazy-loads road networks on-demand via "
    "OperationalContextService."
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info(DEPRECATION_MESSAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

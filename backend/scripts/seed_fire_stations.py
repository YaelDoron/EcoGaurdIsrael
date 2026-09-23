"""Seed the fire_stations and firefighting_resources tables from governmental data.

Reads backend/src/database/data/firefighting_stations.json (the govermental
open-data export) and inserts a FireStationDB row per station plus fabricated
FirefightingResourceDB rows (fire trucks) so the fleet has something to
dispatch against. The number of trucks per station is derived from its
station_type (see RESOURCE_COUNTS_BY_STATION_TYPE).

Clean-slate: on every run, all existing fire_stations/firefighting_resources
rows are deleted before re-seeding, so re-running after a mapping change
re-applies the new quantities.

Usage:
    python scripts/seed_fire_stations.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import get_session, init_db
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.models.resource_status import ResourceStatus

DATA_PATH = Path(__file__).resolve().parents[1] / "src" / "database" / "data" / "firefighting_stations.json"

# Logical resource injection: counts follow the same hierarchy the
# station_type field already encodes (National HQ > District HQ > Regional
# > Substation), just sized up from the original baseline - a live check
# of the seeded fleet (409 trucks total: 1 HQ, 7 District, 33 Regional, 97
# Substation) showed even the OLD counts already vastly exceed a heavy
# 6-simultaneous-event scenario's total demand (roughly 20-24 desired
# resources, per this session's own live GlobalPlanningRun data) - so
# NATIONAL fleet size was never the real bottleneck for 100% coverage;
# geographic/routing reachability was (see GlobalPlanningInputBuilder's
# progressive expansion + RoadNetworkFetcher's tiling, this same session).
# This bump specifically grows the LOCAL, geographically-distributed tier
# (Substation, by far the most numerous - 97 of 138 stations) the most in
# absolute terms, since that's what actually helps a fire find enough
# LOCALLY-reachable supply without needing the expensive wide fetch at all.
DEFAULT_RESOURCE_COUNT = 4
RESOURCE_COUNTS_BY_STATION_TYPE = {
    "מטה": 12,  # National HQ
    "מחוז": 9,  # District HQ
    "אזורית": 7,  # Regional Station
    "משנה": 3,  # Substation
}


def resource_count_for(station_type: str | None) -> int:
    return RESOURCE_COUNTS_BY_STATION_TYPE.get(station_type, DEFAULT_RESOURCE_COUNT)


def main() -> None:
    init_db()
    records = load_station_records(DATA_PATH)

    stations_added = 0
    resources_added = 0

    with get_session() as session:
        session.query(FirefightingResourceDB).delete()
        session.query(FireStationDB).delete()

        for record in records:
            station_id = str(record["_id"])
            station_type = record.get("Station_Type")

            station = FireStationDB(
                id=station_id,
                name=record.get("Station") or record["Regional_Station"],
                latitude=record["Geo_Lat"],
                longitude=record["Geo_Lon"],
                station_type=station_type,
                address=record.get("Address"),
            )
            session.add(station)
            stations_added += 1

            for i in range(1, resource_count_for(station_type) + 1):
                session.add(
                    FirefightingResourceDB(
                        id=f"TRUCK-{station_id}-{i}",
                        station_id=station_id,
                        status=ResourceStatus.AVAILABLE,
                    )
                )
                resources_added += 1

        session.commit()

    print(f"Fire stations added: {stations_added}")
    print(f"Firefighting resources added: {resources_added}")


def load_station_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["result"]["records"]


if __name__ == "__main__":
    main()

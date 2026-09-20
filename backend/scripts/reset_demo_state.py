"""Reset EcoGuard demo/simulation runtime state to a known clean baseline.

DESTRUCTIVE. Deletes runtime wildfire/evidence/assessment/planning data
(see DemoStateResetService for the exact classification and FK-safe order).
Fire stations/firefighting-resource definitions, weather stations, and the
road-network cache are always preserved; firefighting-resource status is
restored to its seeded baseline (AVAILABLE).

Intended only for a dedicated, disposable demo database or Neon branch -
never the shared team development database. Refuses to run unless both:
- ENABLE_DEMO_DATA_RESET=true is set in the environment (see
  src/config/settings.py / .env.example), and
- --confirm is passed explicitly on the command line.

Example:
    python -m scripts.reset_demo_state --confirm
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TextIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import settings
from src.database.connection import DatabaseConfigurationError, init_db
from src.simulation.demo_state_reset_service import (
    DemoStateResetDisabledError,
    DemoStateResetResult,
    DemoStateResetService,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DESTRUCTIVE: reset EcoGuard demo runtime state to a clean baseline. "
            "Fire stations/resources, weather stations, and the road-network cache "
            "are preserved. Intended only for a dedicated demo database/Neon branch."
        )
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required. Explicitly confirms you intend to run this destructive reset.",
    )
    return parser.parse_args(argv)


def initialize_database() -> None:
    if not settings.DATABASE_URL:
        raise DatabaseConfigurationError("DATABASE_URL is not configured; cannot run demo state reset.")
    init_db()


def print_result(result: DemoStateResetResult, output: TextIO = sys.stdout) -> None:
    print("Road network cache: preserved", file=output)
    print("Fire stations/resources: preserved (resource status restored to AVAILABLE)", file=output)
    print("", file=output)
    print("Deleted:", file=output)
    for table_name, count in result.deleted_counts.items():
        print(f"- {table_name}: {count}", file=output)
    print(f"Resources restored to AVAILABLE: {result.resources_restored}", file=output)
    print("", file=output)
    print("Demo operational state is clean.", file=output)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("EcoGuard Demo State Reset")
    print(f"ENABLE_DEMO_DATA_RESET: {settings.ENABLE_DEMO_DATA_RESET}")
    print("")

    if not args.confirm:
        print(
            "Refused: this is a destructive runtime-demo cleanup "
            "(fire events, evidence, assessments, plans, global planning runs). "
            "Re-run with --confirm to proceed.",
            file=sys.stderr,
        )
        return 2

    try:
        initialize_database()
    except DatabaseConfigurationError as exc:
        print(f"Database configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        result = DemoStateResetService().reset_demo_state()
    except DemoStateResetDisabledError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero, never half-print success.
        print(f"Demo state reset failed: {exc}", file=sys.stderr)
        return 1

    print_result(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Request-scoped read-only database access for pure read endpoints.

Attached (router-level) to the GET-only routers whose services never write:
their many small repository reads then run on AUTOCOMMIT connections,
without a BEGIN/COMMIT round-trip pair per repository call - see
`src.database.connection.read_only_database_access`.

Deliberately `async`: FastAPI runs an async dependency in the request's own
task, so the context variable it sets is copied into the threadpool that
runs the (sync) endpoint and its services. A sync dependency would set it in
a separate threadpool context that the endpoint never sees.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from src.database.connection import read_only_database_access


async def read_only_request() -> AsyncIterator[None]:
    with read_only_database_access():
        yield

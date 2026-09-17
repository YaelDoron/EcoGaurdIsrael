"""FastAPI/Pydantic transport DTOs - the HTTP response contract.

Distinct from src/models (internal read/domain dataclasses) and from
src/mappers (which translates raw *external* provider data into internal
models). Schemas here only ever describe what the API sends back to a
client, and are built from internal models via a small explicit mapper
function colocated with each schema module - never returned directly.
"""

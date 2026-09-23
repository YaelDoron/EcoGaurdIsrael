"""API transport contract for Task 5 (`POST /api/v1/chatbot/ask`).

These Pydantic models are the HTTP request/response shape only - distinct
from `ChatMessage` (`src.agents.response.chatbot_agent`), which is
`ChatbotAgent`'s own internal history-item type. Converting between the two
is the router's job (a one-line, non-business-logic mapping), not this
module's.

`history` intentionally only accepts the "user"/"assistant" roles the
chatbot's own domain understands - never Gemini's own "model" turn role (a
Gemini protocol detail `ChatbotAgent` translates internally, never exposed
at this boundary) and never an arbitrary caller-supplied string.

`list[X]` (not `List[X]`) follows this project's more recent schema
convention (see e.g. `src.api.schemas.active_fire_events`,
`src.api.schemas.event_details`, `src.api.schemas.simulation_control`).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _require_non_blank(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty or whitespace-only.")
    return value


class ChatbotHistoryMessage(BaseModel):
    """One prior conversation turn, as sent by the caller.

    Mirrors `ChatMessage`'s own two fields exactly - this schema exists only
    to validate/shape the HTTP boundary, not to add new semantics.
    """

    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        return _require_non_blank(value, "content")


class ChatbotAskRequest(BaseModel):
    """Request body for `POST /api/v1/chatbot/ask`.

    `history` is optional and defaults to empty. It may contain more than
    `MAX_CHAT_HISTORY_MESSAGES` (`src.agents.response.chatbot_agent`)
    entries - `ChatbotAgent` is the one authoritative place that trims
    history, so this schema deliberately does not duplicate that limit or
    reject an oversized list.
    """

    question: str
    history: list[ChatbotHistoryMessage] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        return _require_non_blank(value, "question")


class ChatbotAskResponse(BaseModel):
    """Response body for `POST /api/v1/chatbot/ask` - the answer only.

    Deliberately minimal: no `fire_event_id`, no matched-event ids, no
    Gemini/model/provider metadata, no prompt/context. `ChatbotAgent`'s
    system instruction already tells Gemini never to expose `fire_event_id`
    in its answer text; this schema additionally makes it structurally
    impossible for any such field to leave the API regardless.
    """

    answer: str

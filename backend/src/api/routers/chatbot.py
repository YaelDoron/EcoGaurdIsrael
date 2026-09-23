"""Chatbot endpoint (Task 5).

A thin HTTP transport over the existing `ChatbotAgent`
(`src/agents/response/chatbot_agent.py`): no EcoGuard snapshot assembly, no
location matching, no Gemini call, and no domain calculation happens in
this module - `ChatbotAgent` already owns all of that (and itself only ever
reads via `ActiveFireEventsService`/`EventDetailsService`, never a
repository or domain agent - see that module's own docstring). This
router's only jobs are: validate the HTTP request via `ChatbotAskRequest`,
convert its `history` items into the existing `ChatMessage` type, call
`ChatbotAgent.ask(...)`, and map the result (or a Gemini failure) onto an
HTTP response.

Gemini failures are mapped to the shared Epic 6 API error envelope
(`{"error": {"code": ..., "message": ...}}`, see
`src.api.routers.simulation._error_response`) rather than FastAPI's default
`{"detail": ...}` shape - never a raw exception message, stack trace, API
key, or URL. `GeminiAuthenticationError`/`GeminiConfigurationError` (a
deployment/secret problem, never the caller's fault) map to 500;
`GeminiServiceUnavailableError` (timeout/connection/429/5xx - the upstream
is unreachable or overloaded) and `GeminiInvalidResponseError`/
`GeminiEmptyResponseError` (the upstream responded but not usably - still
not something the caller can fix by retrying with different input) both map
to 503, matching this task's "upstream/service failure, not a user
validation error" guidance. Any other `GeminiClientError` subclass falls
back to the same 503 handling. An exception that is not a `GeminiClientError`
at all is left to propagate to FastAPI/Starlette's own default exception
handling (generic 500, no internals leaked - see
`src.api.routers.response_plans`'s own docstring for this same convention).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.agents.response.chatbot_agent import ChatbotAgent, ChatMessage
from src.api.dependencies import get_chatbot_agent
from src.api.schemas.chatbot import ChatbotAskRequest, ChatbotAskResponse
from src.external.gemini.exceptions import (
    GeminiAuthenticationError,
    GeminiClientError,
    GeminiConfigurationError,
    GeminiEmptyResponseError,
    GeminiInvalidResponseError,
    GeminiServiceUnavailableError,
)

chatbot_router = APIRouter(prefix="/chatbot", tags=["chatbot"])

CHATBOT_CONFIGURATION_ERROR_CODE = "CHATBOT_CONFIGURATION_ERROR"
CHATBOT_CONFIGURATION_ERROR_MESSAGE = "The chatbot is not available on this deployment."
CHATBOT_UNAVAILABLE_CODE = "CHATBOT_UNAVAILABLE"
CHATBOT_UNAVAILABLE_MESSAGE = "The chatbot is temporarily unavailable. Please try again shortly."


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


@chatbot_router.post(
    "/ask",
    response_model=ChatbotAskResponse,
    summary="Ask EcoGuard's AI decision-support chatbot a question",
)
def ask_chatbot(
    request: ChatbotAskRequest,
    agent: ChatbotAgent = Depends(get_chatbot_agent),
) -> ChatbotAskResponse | JSONResponse:
    """Answer `request.question` from EcoGuard's current persisted state.

    `request.history` is converted to `ChatMessage` unchanged - trimming to
    the latest `MAX_CHAT_HISTORY_MESSAGES` entries remains `ChatbotAgent`'s
    own responsibility, not duplicated here. Nothing is persisted: history
    is used only for this one request.
    """
    history = [ChatMessage(role=item.role, content=item.content) for item in request.history]

    try:
        answer = agent.ask(request.question, history=history)
    except (GeminiAuthenticationError, GeminiConfigurationError):
        return _error_response(500, CHATBOT_CONFIGURATION_ERROR_CODE, CHATBOT_CONFIGURATION_ERROR_MESSAGE)
    except (GeminiServiceUnavailableError, GeminiInvalidResponseError, GeminiEmptyResponseError, GeminiClientError):
        return _error_response(503, CHATBOT_UNAVAILABLE_CODE, CHATBOT_UNAVAILABLE_MESSAGE)

    return ChatbotAskResponse(answer=answer)

"""API tests for `POST /api/v1/chatbot/ask` (Task 5).

Uses FastAPI's dependency_overrides to replace ChatbotAgent with a fake
returning a controlled answer (or raising a controlled GeminiClientError
subclass) - no real database access, no real Gemini API call, and no
domain calculation happens in these tests.
"""
from __future__ import annotations

import ast
from pathlib import Path

from fastapi.testclient import TestClient

from src.agents.response.chatbot_agent import ChatMessage
from src.api.app import create_app
from src.api.dependencies import get_chatbot_agent
from src.external.gemini.exceptions import (
    GeminiAuthenticationError,
    GeminiConfigurationError,
    GeminiEmptyResponseError,
    GeminiInvalidResponseError,
    GeminiServiceUnavailableError,
)

ENDPOINT = "/api/v1/chatbot/ask"


class FakeChatbotAgent:
    def __init__(self, answer: str = "The requested EcoGuard information.", *, raise_exc: Exception | None = None):
        self._answer = answer
        self._raise_exc = raise_exc
        self.calls: list[dict] = []

    def ask(self, question, *, history=()):
        self.calls.append({"question": question, "history": tuple(history)})
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._answer


def client_for(agent, *, raise_server_exceptions: bool = True) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_chatbot_agent] = lambda: agent
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


# ---------------------------------------------------------------------------
# 1-2: valid question, with/without history
# ---------------------------------------------------------------------------


def test_valid_question_with_no_history():
    agent = FakeChatbotAgent(answer="Two active fires are currently confirmed.")
    client = client_for(agent)

    response = client.post(ENDPOINT, json={"question": "What are the active fires?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Two active fires are currently confirmed."}
    assert agent.calls == [{"question": "What are the active fires?", "history": ()}]


def test_valid_question_with_history():
    agent = FakeChatbotAgent()
    client = client_for(agent)

    response = client.post(
        ENDPOINT,
        json={
            "question": "And what is its spread prediction?",
            "history": [
                {"role": "user", "content": "What is the status of the fire in Haifa?"},
                {"role": "assistant", "content": "The active event in Haifa is currently CONFIRMED."},
            ],
        },
    )

    assert response.status_code == 200
    sent_history = agent.calls[0]["history"]
    assert len(sent_history) == 2
    assert sent_history[0] == ChatMessage(role="user", content="What is the status of the fire in Haifa?")
    assert sent_history[1] == ChatMessage(
        role="assistant", content="The active event in Haifa is currently CONFIRMED."
    )


# ---------------------------------------------------------------------------
# 3-4: language passthrough
# ---------------------------------------------------------------------------


def test_hebrew_question_passes_through_unchanged():
    agent = FakeChatbotAgent()
    client = client_for(agent)

    response = client.post(ENDPOINT, json={"question": "מה מצב השריפה בחיפה?"})

    assert response.status_code == 200
    assert agent.calls[0]["question"] == "מה מצב השריפה בחיפה?"


def test_english_question_passes_through_unchanged():
    agent = FakeChatbotAgent()
    client = client_for(agent)

    response = client.post(ENDPOINT, json={"question": "What is the status of the fire in Haifa?"})

    assert response.status_code == 200
    assert agent.calls[0]["question"] == "What is the status of the fire in Haifa?"


# ---------------------------------------------------------------------------
# 5-6: history mapping / defaulting
# ---------------------------------------------------------------------------


def test_history_role_and_content_mapped_into_chat_message():
    agent = FakeChatbotAgent()
    client = client_for(agent)

    client.post(
        ENDPOINT,
        json={
            "question": "latest question",
            "history": [{"role": "user", "content": "earlier message"}],
        },
    )

    (message,) = agent.calls[0]["history"]
    assert isinstance(message, ChatMessage)
    assert message.role == "user"
    assert message.content == "earlier message"


def test_empty_history_defaults_to_empty_tuple():
    agent = FakeChatbotAgent()
    client = client_for(agent)

    response = client.post(ENDPOINT, json={"question": "What are the active fires?"})

    assert response.status_code == 200
    assert agent.calls[0]["history"] == ()


# ---------------------------------------------------------------------------
# 7-11: validation
# ---------------------------------------------------------------------------


def test_empty_question_rejected():
    agent = FakeChatbotAgent()
    client = client_for(agent)

    response = client.post(ENDPOINT, json={"question": ""})

    assert response.status_code == 422
    assert agent.calls == []


def test_whitespace_only_question_rejected():
    agent = FakeChatbotAgent()
    client = client_for(agent)

    response = client.post(ENDPOINT, json={"question": "   "})

    assert response.status_code == 422
    assert agent.calls == []


def test_invalid_history_role_rejected():
    agent = FakeChatbotAgent()
    client = client_for(agent)

    response = client.post(
        ENDPOINT,
        json={"question": "q", "history": [{"role": "model", "content": "hi"}]},
    )

    assert response.status_code == 422
    assert agent.calls == []


def test_empty_history_content_rejected():
    agent = FakeChatbotAgent()
    client = client_for(agent)

    response = client.post(
        ENDPOINT,
        json={"question": "q", "history": [{"role": "user", "content": ""}]},
    )

    assert response.status_code == 422
    assert agent.calls == []


def test_whitespace_only_history_content_rejected():
    agent = FakeChatbotAgent()
    client = client_for(agent)

    response = client.post(
        ENDPOINT,
        json={"question": "q", "history": [{"role": "user", "content": "   "}]},
    )

    assert response.status_code == 422
    assert agent.calls == []


# ---------------------------------------------------------------------------
# 12-13: response shape / no internal metadata
# ---------------------------------------------------------------------------


def test_response_contains_only_answer():
    agent = FakeChatbotAgent(answer="Only the answer.")
    client = client_for(agent)

    response = client.post(ENDPOINT, json={"question": "What are the active fires?"})

    assert response.json() == {"answer": "Only the answer."}
    assert set(response.json().keys()) == {"answer"}


def test_no_fire_event_id_or_internal_metadata_added_by_api():
    agent = FakeChatbotAgent(answer="The active event in Haifa is CONFIRMED.")
    client = client_for(agent)

    response = client.post(ENDPOINT, json={"question": "What is the status of the fire in Haifa?"})

    body = response.json()
    assert "fire_event_id" not in body
    assert "id" not in body
    assert "model" not in body
    assert "provider" not in body
    assert body == {"answer": "The active event in Haifa is CONFIRMED."}


# ---------------------------------------------------------------------------
# 14-17: Gemini failure mapping
# ---------------------------------------------------------------------------


def test_gemini_service_unavailable_maps_to_503():
    agent = FakeChatbotAgent(raise_exc=GeminiServiceUnavailableError("Gemini request timed out."))
    client = client_for(agent)

    response = client.post(ENDPOINT, json={"question": "What are the active fires?"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "CHATBOT_UNAVAILABLE"


def test_gemini_authentication_error_maps_safely_to_500():
    agent = FakeChatbotAgent(raise_exc=GeminiAuthenticationError("Gemini authentication failed."))
    client = client_for(agent)

    response = client.post(ENDPOINT, json={"question": "What are the active fires?"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "CHATBOT_CONFIGURATION_ERROR"


def test_gemini_configuration_error_maps_safely_to_500():
    agent = FakeChatbotAgent(raise_exc=GeminiConfigurationError("Gemini API key is not configured."))
    client = client_for(agent)

    response = client.post(ENDPOINT, json={"question": "What are the active fires?"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "CHATBOT_CONFIGURATION_ERROR"


def test_gemini_invalid_response_maps_to_503_not_a_validation_error():
    agent = FakeChatbotAgent(raise_exc=GeminiInvalidResponseError("Gemini returned an invalid JSON response."))
    client = client_for(agent)

    response = client.post(ENDPOINT, json={"question": "What are the active fires?"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "CHATBOT_UNAVAILABLE"


def test_gemini_empty_response_maps_to_503_not_a_validation_error():
    agent = FakeChatbotAgent(raise_exc=GeminiEmptyResponseError("Gemini response contained no usable text."))
    client = client_for(agent)

    response = client.post(ENDPOINT, json={"question": "What are the active fires?"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "CHATBOT_UNAVAILABLE"


def test_exception_responses_do_not_leak_secrets_or_raw_text():
    secret = "AQ.super-secret-gemini-key-do-not-leak"
    for exc in (
        GeminiAuthenticationError(f"Gemini authentication failed with key {secret}"),
        GeminiConfigurationError(f"Missing key {secret}"),
        GeminiServiceUnavailableError(f"Gemini service returned HTTP 503 for key {secret}"),
        GeminiInvalidResponseError(f"Invalid response, request used key {secret}"),
        GeminiEmptyResponseError(f"Empty response, request used key {secret}"),
    ):
        agent = FakeChatbotAgent(raise_exc=exc)
        client = client_for(agent)

        response = client.post(ENDPOINT, json={"question": "What are the active fires?"})

        assert secret not in response.text
        assert "Traceback" not in response.text
        assert response.json()["error"]["message"] in (
            "The chatbot is not available on this deployment.",
            "The chatbot is temporarily unavailable. Please try again shortly.",
        )


# ---------------------------------------------------------------------------
# 18: router delegates to the injected ChatbotAgent, no domain logic inline
# ---------------------------------------------------------------------------


def test_router_uses_injected_chatbot_agent_and_imports_no_domain_logic():
    forbidden_fragments = (
        "src.agents.analysis",
        "src.agents.collection",
        "src.agents.routing",
        "src.calculators",
        "src.repositories",
        "src.simulation",
        "src.database",
        "src.external.ims",
        "src.external.firms",
        "src.external.copernicus",
        "src.external.news",
        "src.external.geocoding",
    )
    path = (Path(__file__).resolve().parents[3] / "backend/src/api/routers/chatbot.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    violations = []
    imported_modules = []
    for node in ast.walk(tree):
        module = ""
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
        elif isinstance(node, ast.Import):
            module = ",".join(alias.name for alias in node.names)
        else:
            continue
        imported_modules.append(module)
        if any(fragment in module for fragment in forbidden_fragments):
            violations.append(module)

    assert violations == []
    # Positive check: the router really does delegate to the existing
    # ChatbotAgent rather than reimplementing its logic.
    assert any("src.agents.response.chatbot_agent" in module for module in imported_modules)
    assert "get_chatbot_agent" in source
    assert "ChatbotAgent(" not in source  # never constructed inline - only via the DI dependency


def test_fake_chatbot_agent_only_exposes_ask():
    agent = FakeChatbotAgent()
    for forbidden in ("get_active_events", "get_event_details", "generate_content", "detect", "optimize"):
        assert not hasattr(agent, forbidden)

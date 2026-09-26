"""Unit tests for GeminiClient.

All HTTP calls are mocked - no real network access is performed and no real
Gemini API key is required to run these tests.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import requests

from src.external.gemini.exceptions import (
    GeminiAuthenticationError,
    GeminiClientError,
    GeminiConfigurationError,
    GeminiEmptyResponseError,
    GeminiInvalidResponseError,
    GeminiServiceUnavailableError,
)
from src.external.gemini.gemini_client import GeminiClient

FAKE_API_KEY = "test-secret-gemini-key-123"
MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.1-flash-lite"
SAMPLE_CONTENTS = [{"role": "user", "parts": [{"text": "What is the status of the fire in Haifa?"}]}]


class FakeResponse:
    """Minimal stand-in for `requests.Response`, used to mock Gemini HTTP replies."""

    def __init__(self, status_code: int, json_data=None, invalid_json: bool = False, headers: dict | None = None):
        self.status_code = status_code
        self.ok = status_code < 400
        self.headers = headers or {}
        self._json_data = json_data
        self._invalid_json = invalid_json

    def json(self):
        if self._invalid_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._json_data


def success_body(text: str = "The fire in Haifa is currently CONFIRMED.") -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def quota_exhausted_body(metric: str = "generativelanguage.googleapis.com/generate_content_free_tier_requests") -> dict:
    """A real, verbatim-shaped quota-exhaustion error body (see Task 7D diagnostics)."""
    return {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "message": (
                "You exceeded your current quota, please check your plan and billing details. "
                f"Quota exceeded for metric: {metric}."
            ),
        }
    }


def transient_rate_limit_body() -> dict:
    """A 429 that is RESOURCE_EXHAUSTED but carries none of the quota/billing-specific
    wording - an ordinary transient per-minute rate limit, not exhausted quota."""
    return {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Resource has been exhausted (e.g. check quota)."}}


def make_client(
    api_key: str = FAKE_API_KEY,
    model: str = MODEL,
    timeout: int = 10,
    fallback_model: str = FALLBACK_MODEL,
) -> GeminiClient:
    return GeminiClient(api_key=api_key, model=model, timeout=timeout, fallback_model=fallback_model)


@pytest.fixture(autouse=True)
def no_real_sleep():
    """Every test in this module gets `time.sleep` patched automatically, so
    retry-backoff tests never actually wait real seconds. Tests that care
    about the delay take this fixture by name to assert on its call args."""
    with patch("src.external.gemini.gemini_client.time.sleep") as mock_sleep:
        yield mock_sleep


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_request_without_api_key_raises_configuration_error():
    client = make_client(api_key="")

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        with pytest.raises(GeminiConfigurationError, match="Gemini API key is not configured."):
            client.generate_content(SAMPLE_CONTENTS)

    mock_post.assert_not_called()


def test_request_without_contents_raises_client_error():
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        with pytest.raises(GeminiClientError):
            client.generate_content([])

    mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Successful generation
# ---------------------------------------------------------------------------


def test_generate_content_returns_generated_text():
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, success_body("Active fire near Haifa, status CONFIRMED."))
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "Active fire near Haifa, status CONFIRMED."


def test_generate_content_requests_configured_model_url():
    client = make_client(model="gemini-custom-model")

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, success_body())
        client.generate_content(SAMPLE_CONTENTS)

    called_url = mock_post.call_args.args[0]
    assert called_url == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-custom-model:generateContent"
    )


def test_generate_content_sends_configured_api_key():
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, success_body())
        client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_args.kwargs["params"] == {"key": FAKE_API_KEY}


def test_generate_content_applies_configured_timeout():
    client = make_client(timeout=42)

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, success_body())
        client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_args.kwargs["timeout"] == 42


def test_generate_content_sends_contents_unchanged():
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, success_body())
        client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_args.kwargs["json"]["contents"] == SAMPLE_CONTENTS
    assert "systemInstruction" not in mock_post.call_args.kwargs["json"]


def test_generate_content_sends_system_instruction_when_given():
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, success_body())
        client.generate_content(SAMPLE_CONTENTS, system_instruction="Answer only from supplied EcoGuard data.")

    sent_json = mock_post.call_args.kwargs["json"]
    assert sent_json["systemInstruction"] == {"parts": [{"text": "Answer only from supplied EcoGuard data."}]}


# ---------------------------------------------------------------------------
# Authentication failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [401, 403])
def test_authentication_error_status_codes_raise_authentication_error(status_code):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(status_code)
        with pytest.raises(GeminiAuthenticationError):
            client.generate_content(SAMPLE_CONTENTS)


# ---------------------------------------------------------------------------
# Service failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
def test_server_error_status_codes_raise_service_unavailable_error(status_code):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(status_code)
        with pytest.raises(GeminiServiceUnavailableError):
            client.generate_content(SAMPLE_CONTENTS)


def test_request_timeout_raises_service_unavailable_error():
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout()
        with pytest.raises(GeminiServiceUnavailableError):
            client.generate_content(SAMPLE_CONTENTS)


def test_connection_error_raises_service_unavailable_error():
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError()
        with pytest.raises(GeminiServiceUnavailableError):
            client.generate_content(SAMPLE_CONTENTS)


def test_other_request_exception_raises_client_error():
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.RequestException()
        with pytest.raises(GeminiClientError):
            client.generate_content(SAMPLE_CONTENTS)


def test_unexpected_error_status_code_raises_client_error():
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(400)
        with pytest.raises(GeminiClientError):
            client.generate_content(SAMPLE_CONTENTS)


# ---------------------------------------------------------------------------
# Invalid / empty responses
# ---------------------------------------------------------------------------


def test_invalid_json_response_raises_invalid_response_error():
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, invalid_json=True)
        with pytest.raises(GeminiInvalidResponseError):
            client.generate_content(SAMPLE_CONTENTS)


@pytest.mark.parametrize(
    "body",
    [
        {"candidates": []},
        {"candidates": [{"content": {"parts": []}}]},
        {"candidates": [{"content": {"parts": [{"text": ""}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "   "}]}}]},
        {"candidates": [{"content": {}}]},
        {},
        {"promptFeedback": {"blockReason": "SAFETY"}},
    ],
)
def test_missing_or_empty_generated_text_raises_empty_response_error(body):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, body)
        with pytest.raises(GeminiEmptyResponseError):
            client.generate_content(SAMPLE_CONTENTS)


# ---------------------------------------------------------------------------
# Security: the API key must never leak into exceptions, logs, or output
# ---------------------------------------------------------------------------


def test_exception_messages_never_contain_the_api_key():
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(500)
        with pytest.raises(GeminiServiceUnavailableError) as exc_info:
            client.generate_content(SAMPLE_CONTENTS)

    assert FAKE_API_KEY not in str(exc_info.value)


def test_logs_never_contain_the_api_key(caplog):
    client = make_client()

    with caplog.at_level("DEBUG"):
        with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
            mock_post.return_value = FakeResponse(200, success_body())
            client.generate_content(SAMPLE_CONTENTS)

    for record in caplog.records:
        assert FAKE_API_KEY not in record.getMessage()


def test_returned_text_never_contains_the_api_key():
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, success_body("Active fire near Haifa."))
        result = client.generate_content(SAMPLE_CONTENTS)

    assert FAKE_API_KEY not in result


# ---------------------------------------------------------------------------
# Bounded retry for transient failures
# ---------------------------------------------------------------------------


def test_success_on_first_attempt_makes_exactly_one_call_and_never_sleeps(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, success_body("ok"))
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "ok"
    assert mock_post.call_count == 1
    no_real_sleep.assert_not_called()


def test_one_transient_503_then_success_retries_exactly_once(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [FakeResponse(503), FakeResponse(200, success_body("Recovered answer."))]
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "Recovered answer."
    assert mock_post.call_count == 2
    no_real_sleep.assert_called_once()


def test_two_transient_503s_then_success_uses_all_three_attempts(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [FakeResponse(503), FakeResponse(503), FakeResponse(200, success_body("ok"))]
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "ok"
    assert mock_post.call_count == 3
    assert no_real_sleep.call_count == 2


def test_repeated_503_stops_after_exactly_three_attempts_and_raises(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(503)
        with pytest.raises(GeminiServiceUnavailableError):
            client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_count == 3
    assert no_real_sleep.call_count == 2


@pytest.mark.parametrize("status_code", [500, 502, 504])
def test_each_retryable_server_status_is_retried(status_code, no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [FakeResponse(status_code), FakeResponse(200, success_body("ok"))]
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "ok"
    assert mock_post.call_count == 2


def test_transient_429_without_quota_wording_is_retried(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [
            FakeResponse(429, transient_rate_limit_body()),
            FakeResponse(200, success_body("ok")),
        ]
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "ok"
    assert mock_post.call_count == 2


def test_quota_exhausted_429_is_not_retried(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(429, quota_exhausted_body())
        with pytest.raises(GeminiServiceUnavailableError):
            client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_count == 1
    no_real_sleep.assert_not_called()


def test_429_with_unparseable_body_is_treated_as_transient_and_retried(no_real_sleep):
    """A 429 whose body can't be read as the quota-exhaustion shape must never be
    assumed to be quota-exhausted - it stays retryable (fails open toward retry,
    not toward giving up)."""
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [
            FakeResponse(429, invalid_json=True),
            FakeResponse(200, success_body("ok")),
        ]
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "ok"
    assert mock_post.call_count == 2


def test_retry_after_header_is_respected_when_valid(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [
            FakeResponse(503, headers={"Retry-After": "3"}),
            FakeResponse(200, success_body("ok")),
        ]
        client.generate_content(SAMPLE_CONTENTS)

    no_real_sleep.assert_called_once_with(3.0)


def test_retry_after_header_is_capped_to_a_small_maximum(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [
            FakeResponse(503, headers={"Retry-After": "120"}),
            FakeResponse(200, success_body("ok")),
        ]
        client.generate_content(SAMPLE_CONTENTS)

    no_real_sleep.assert_called_once_with(5.0)


def test_invalid_retry_after_falls_back_to_the_default_backoff(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [
            FakeResponse(503, headers={"Retry-After": "not-a-number"}),
            FakeResponse(200, success_body("ok")),
        ]
        client.generate_content(SAMPLE_CONTENTS)

    no_real_sleep.assert_called_once_with(1.0)


@pytest.mark.parametrize("status_code", [401, 403])
def test_authentication_failure_is_not_retried(status_code, no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(status_code)
        with pytest.raises(GeminiAuthenticationError):
            client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_count == 1
    no_real_sleep.assert_not_called()


def test_invalid_successful_response_is_not_retried():
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, invalid_json=True)
        with pytest.raises(GeminiInvalidResponseError):
            client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_count == 1


def test_api_key_never_exposed_across_a_retry_sequence(caplog, no_real_sleep):
    client = make_client()

    with caplog.at_level("DEBUG"):
        with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
            mock_post.side_effect = [FakeResponse(503), FakeResponse(503), FakeResponse(200, success_body("ok"))]
            client.generate_content(SAMPLE_CONTENTS)

    for record in caplog.records:
        assert FAKE_API_KEY not in record.getMessage()


def test_timeout_then_success_retries_exactly_once(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [requests.exceptions.Timeout(), FakeResponse(200, success_body("Recovered."))]
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "Recovered."
    assert mock_post.call_count == 2
    no_real_sleep.assert_called_once()


def test_two_timeouts_then_success_uses_all_three_attempts(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [
            requests.exceptions.Timeout(),
            requests.exceptions.Timeout(),
            FakeResponse(200, success_body("ok")),
        ]
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "ok"
    assert mock_post.call_count == 3
    assert no_real_sleep.call_count == 2


def test_repeated_timeout_stops_after_exactly_three_attempts_and_raises(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout()
        with pytest.raises(GeminiServiceUnavailableError):
            client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_count == 3
    assert no_real_sleep.call_count == 2


def test_timeout_retry_uses_the_same_fixed_backoff_pattern(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout()
        with pytest.raises(GeminiServiceUnavailableError):
            client.generate_content(SAMPLE_CONTENTS)

    assert no_real_sleep.call_args_list == [((1.0,),), ((2.0,),)]


def test_timeout_then_transient_503_then_success_still_retries_correctly(no_real_sleep):
    """A timeout retry must not disturb the existing HTTP-failure retry rules -
    the two kinds of transient failure can be mixed within one bounded retry budget."""
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [
            requests.exceptions.Timeout(),
            FakeResponse(503),
            FakeResponse(200, success_body("ok")),
        ]
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "ok"
    assert mock_post.call_count == 3
    assert no_real_sleep.call_count == 2


def test_quota_exhausted_429_is_still_not_retried_after_timeout_change(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(429, quota_exhausted_body())
        with pytest.raises(GeminiServiceUnavailableError):
            client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_count == 1
    no_real_sleep.assert_not_called()


@pytest.mark.parametrize("status_code", [401, 403])
def test_authentication_failure_is_still_not_retried_after_timeout_change(status_code, no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(status_code)
        with pytest.raises(GeminiAuthenticationError):
            client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_count == 1
    no_real_sleep.assert_not_called()


def test_connection_error_is_still_not_retried_after_timeout_change(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError()
        with pytest.raises(GeminiServiceUnavailableError):
            client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_count == 1
    no_real_sleep.assert_not_called()


def test_api_key_never_exposed_when_timeout_retries_are_exhausted(caplog, no_real_sleep):
    client = make_client()

    with caplog.at_level("DEBUG"):
        with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout()
            with pytest.raises(GeminiServiceUnavailableError) as exc_info:
                client.generate_content(SAMPLE_CONTENTS)

    assert FAKE_API_KEY not in str(exc_info.value)
    for record in caplog.records:
        assert FAKE_API_KEY not in record.getMessage()


def test_api_key_never_exposed_when_retries_are_exhausted(caplog, no_real_sleep):
    client = make_client()

    with caplog.at_level("DEBUG"):
        with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
            mock_post.return_value = FakeResponse(503)
            with pytest.raises(GeminiServiceUnavailableError) as exc_info:
                client.generate_content(SAMPLE_CONTENTS)

    assert FAKE_API_KEY not in str(exc_info.value)
    for record in caplog.records:
        assert FAKE_API_KEY not in record.getMessage()


# ---------------------------------------------------------------------------
# Model fallback on the final bounded attempt (availability improvement only)
# ---------------------------------------------------------------------------


def _urls_called(mock_post) -> list[str]:
    return [call.args[0] for call in mock_post.call_args_list]


def test_primary_success_on_first_attempt_never_uses_fallback(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, success_body("ok"))
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "ok"
    assert mock_post.call_count == 1
    urls = _urls_called(mock_post)
    assert all(f"/models/{MODEL}:generateContent" in url for url in urls)
    assert all(FALLBACK_MODEL not in url for url in urls)


def test_primary_timeout_then_primary_success_never_uses_fallback(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [requests.exceptions.Timeout(), FakeResponse(200, success_body("ok"))]
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "ok"
    assert mock_post.call_count == 2
    urls = _urls_called(mock_post)
    assert all(f"/models/{MODEL}:generateContent" in url for url in urls)
    assert all(FALLBACK_MODEL not in url for url in urls)


def test_two_primary_timeouts_then_third_attempt_uses_fallback(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [
            requests.exceptions.Timeout(),
            requests.exceptions.Timeout(),
            FakeResponse(200, success_body("Fallback answer.")),
        ]
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "Fallback answer."
    assert mock_post.call_count == 3
    urls = _urls_called(mock_post)
    assert f"/models/{MODEL}:generateContent" in urls[0]
    assert f"/models/{MODEL}:generateContent" in urls[1]
    assert f"/models/{FALLBACK_MODEL}:generateContent" in urls[2]


def test_two_primary_503s_then_third_attempt_uses_fallback(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [
            FakeResponse(503),
            FakeResponse(503),
            FakeResponse(200, success_body("Fallback answer.")),
        ]
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "Fallback answer."
    assert mock_post.call_count == 3
    urls = _urls_called(mock_post)
    assert f"/models/{MODEL}:generateContent" in urls[0]
    assert f"/models/{MODEL}:generateContent" in urls[1]
    assert f"/models/{FALLBACK_MODEL}:generateContent" in urls[2]


def test_primary_timeout_then_503_then_third_attempt_uses_fallback(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [
            requests.exceptions.Timeout(),
            FakeResponse(503),
            FakeResponse(200, success_body("Fallback answer.")),
        ]
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "Fallback answer."
    assert mock_post.call_count == 3
    urls = _urls_called(mock_post)
    assert f"/models/{MODEL}:generateContent" in urls[0]
    assert f"/models/{MODEL}:generateContent" in urls[1]
    assert f"/models/{FALLBACK_MODEL}:generateContent" in urls[2]


def test_fallback_attempt_success_returns_normal_generated_text(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [
            FakeResponse(503),
            FakeResponse(503),
            FakeResponse(200, success_body("Active fire near Haifa, status CONFIRMED.")),
        ]
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "Active fire near Haifa, status CONFIRMED."


def test_fallback_attempt_also_fails_transiently_raises_existing_service_error(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [FakeResponse(503), FakeResponse(503), FakeResponse(503)]
        with pytest.raises(GeminiServiceUnavailableError):
            client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_count == 3
    urls = _urls_called(mock_post)
    assert f"/models/{FALLBACK_MODEL}:generateContent" in urls[2]


def test_quota_exhausted_429_never_uses_fallback(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(429, quota_exhausted_body())
        with pytest.raises(GeminiServiceUnavailableError):
            client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_count == 1
    urls = _urls_called(mock_post)
    assert all(FALLBACK_MODEL not in url for url in urls)
    no_real_sleep.assert_not_called()


@pytest.mark.parametrize("status_code", [401, 403])
def test_authentication_failure_never_uses_fallback(status_code, no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(status_code)
        with pytest.raises(GeminiAuthenticationError):
            client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_count == 1
    urls = _urls_called(mock_post)
    assert all(FALLBACK_MODEL not in url for url in urls)


def test_non_retryable_400_never_uses_fallback(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(400)
        with pytest.raises(GeminiClientError):
            client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_count == 1
    urls = _urls_called(mock_post)
    assert all(FALLBACK_MODEL not in url for url in urls)


def test_fallback_never_adds_a_fourth_attempt(no_real_sleep):
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(503)
        with pytest.raises(GeminiServiceUnavailableError):
            client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_count == 3


def test_fallback_disabled_when_configured_equal_to_primary_model(no_real_sleep):
    """If the fallback model is configured identically to the primary model,
    the final attempt still just calls the primary model again - no distinct
    fallback URL is ever produced."""
    client = make_client(fallback_model=MODEL)

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [FakeResponse(503), FakeResponse(503), FakeResponse(503)]
        with pytest.raises(GeminiServiceUnavailableError):
            client.generate_content(SAMPLE_CONTENTS)

    urls = _urls_called(mock_post)
    assert all(f"/models/{MODEL}:generateContent" in url for url in urls)


def test_fallback_log_message_names_only_models_and_never_the_api_key(caplog, no_real_sleep):
    client = make_client()

    with caplog.at_level("DEBUG"):
        with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
            mock_post.side_effect = [
                FakeResponse(503),
                FakeResponse(503),
                FakeResponse(200, success_body("ok")),
            ]
            client.generate_content(SAMPLE_CONTENTS)

    fallback_logs = [r for r in caplog.records if "fallback model" in r.getMessage()]
    assert len(fallback_logs) == 1
    assert MODEL in fallback_logs[0].getMessage()
    assert FALLBACK_MODEL in fallback_logs[0].getMessage()
    for record in caplog.records:
        assert FAKE_API_KEY not in record.getMessage()


# ---------------------------------------------------------------------------
# Gemma: thinkingConfig request shape + thought-part-aware extraction
# ---------------------------------------------------------------------------

GEMMA_MODEL = "gemma-4-26b-a4b-it"


def make_gemma_client(api_key: str = FAKE_API_KEY, timeout: int = 10) -> GeminiClient:
    # Fallback == primary, matching the production configuration: all
    # bounded attempts use the same Gemma model, never a distinct fallback.
    return GeminiClient(api_key=api_key, model=GEMMA_MODEL, timeout=timeout, fallback_model=GEMMA_MODEL)


def thought_and_answer_body(answer_text: str, thought_text: str = "") -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": thought_text, "thought": True},
                        {"text": answer_text},
                    ],
                    "role": "model",
                }
            }
        ]
    }


def test_gemma_request_includes_thinking_level_minimal(no_real_sleep):
    client = make_gemma_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, thought_and_answer_body("OK"))
        client.generate_content(SAMPLE_CONTENTS)

    sent_json = mock_post.call_args.kwargs["json"]
    assert sent_json["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "minimal"


def test_non_gemma_request_never_gets_generation_config(no_real_sleep):
    client = make_client()  # MODEL = "gemini-3.8-flash", not Gemma

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, success_body("ok"))
        client.generate_content(SAMPLE_CONTENTS)

    sent_json = mock_post.call_args.kwargs["json"]
    assert "generationConfig" not in sent_json


def test_empty_thought_part_then_answer_part_returns_only_the_answer(no_real_sleep):
    client = make_gemma_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, thought_and_answer_body("OK", thought_text=""))
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "OK"


def test_non_empty_thought_part_text_is_never_returned(no_real_sleep):
    client = make_gemma_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(
            200,
            thought_and_answer_body(
                "The current status is confirmed.",
                thought_text="The user wants me to summarize the event status. Let me check the data...",
            ),
        )
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "The current status is confirmed."
    assert "user wants" not in result
    assert "Let me check" not in result


def test_multiple_non_thought_text_parts_are_concatenated(no_real_sleep):
    client = make_gemma_client()
    body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "", "thought": True},
                        {"text": "Part one. "},
                        {"text": "Part two."},
                    ]
                }
            }
        ]
    }

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, body)
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "Part one. Part two."


def test_only_thought_parts_raises_empty_response_error(no_real_sleep):
    client = make_gemma_client()
    body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Some internal reasoning.", "thought": True},
                    ]
                }
            }
        ]
    }

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, body)
        with pytest.raises(GeminiEmptyResponseError):
            client.generate_content(SAMPLE_CONTENTS)


def test_classic_single_part_gemini_response_still_works_unaffected(no_real_sleep):
    """A plain, non-Gemma response with exactly one ordinary text part (no
    `thought` key at all) must keep working exactly as before this change."""
    client = make_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(200, success_body("Active fire near Haifa, status CONFIRMED."))
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "Active fire near Haifa, status CONFIRMED."


def test_all_three_retry_attempts_use_the_gemma_model(no_real_sleep):
    client = make_gemma_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [
            FakeResponse(503),
            FakeResponse(503),
            FakeResponse(200, thought_and_answer_body("OK")),
        ]
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "OK"
    urls = [call.args[0] for call in mock_post.call_args_list]
    assert len(urls) == 3
    assert all(f"/models/{GEMMA_MODEL}:generateContent" in url for url in urls)


def test_gemma_timeout_retry_behavior_is_unchanged(no_real_sleep):
    client = make_gemma_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [requests.exceptions.Timeout(), FakeResponse(200, thought_and_answer_body("OK"))]
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "OK"
    assert mock_post.call_count == 2
    no_real_sleep.assert_called_once_with(1.0)


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_gemma_server_error_retry_behavior_is_unchanged(status_code, no_real_sleep):
    client = make_gemma_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.side_effect = [FakeResponse(status_code), FakeResponse(200, thought_and_answer_body("OK"))]
        result = client.generate_content(SAMPLE_CONTENTS)

    assert result == "OK"
    assert mock_post.call_count == 2


def test_gemma_quota_exhausted_429_is_still_not_retried(no_real_sleep):
    client = make_gemma_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(429, quota_exhausted_body())
        with pytest.raises(GeminiServiceUnavailableError):
            client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_count == 1
    no_real_sleep.assert_not_called()


@pytest.mark.parametrize("status_code", [401, 403])
def test_gemma_authentication_failure_is_still_not_retried(status_code, no_real_sleep):
    client = make_gemma_client()

    with patch("src.external.gemini.gemini_client.requests.post") as mock_post:
        mock_post.return_value = FakeResponse(status_code)
        with pytest.raises(GeminiAuthenticationError):
            client.generate_content(SAMPLE_CONTENTS)

    assert mock_post.call_count == 1
    no_real_sleep.assert_not_called()

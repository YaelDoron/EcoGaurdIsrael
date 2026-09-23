"""Unit tests for TextProcessor (news_client.py).

All HTTP calls to the LLM provider are mocked. No real Groq/Gemini API key
or network access is required to run these tests.
"""
from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest
import requests

from src.external.news.news_client import TextProcessor

GROQ_LLM_CONFIG = {
    "provider": "groq",
    "model": "llama-3.1-8b-instant",
    "api_key_env": "TEST_NEWS_LLM_API_KEY",
    "temperature": 0.0,
    "max_tokens": 100,
    "translation_max_tokens": 600,
    "request_timeout_seconds": 15,
}


class FakeResponse:
    """Minimal stand-in for `requests.Response`."""

    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def groq_reply(content: dict | str) -> FakeResponse:
    text = content if isinstance(content, str) else json.dumps(content)
    return FakeResponse({"choices": [{"message": {"content": text}}]})


def gemini_reply(content: dict | str) -> FakeResponse:
    text = content if isinstance(content, str) else json.dumps(content)
    return FakeResponse({"candidates": [{"content": {"parts": [{"text": text}]}}]})


def make_processor(monkeypatch: pytest.MonkeyPatch, config: dict | None = None) -> TextProcessor:
    monkeypatch.setenv("TEST_NEWS_LLM_API_KEY", "fake-key")
    return TextProcessor(keywords=["שריפה", "יער"], llm_config=config or GROQ_LLM_CONFIG)


# ---------------------------------------------------------------------------
# is_relevant / extract_location (regression coverage for the pre-existing
# behavior, not previously unit-tested at this layer)
# ---------------------------------------------------------------------------


def test_is_relevant_matches_a_configured_keyword(monkeypatch: pytest.MonkeyPatch):
    processor = make_processor(monkeypatch)

    assert processor.is_relevant("שריפה גדולה בכרמל", "כוחות כיבוי בדרך") is True
    assert processor.is_relevant("תוצאות הבחירות", "סיכום היום") is False


def test_extract_location_returns_the_llm_location(monkeypatch: pytest.MonkeyPatch):
    processor = make_processor(monkeypatch)

    with patch("src.external.news.news_client.requests.post", return_value=groq_reply({"locationName": "כרמל"})) as post:
        location = processor.extract_location("שריפה בכרמל", "כוחות כיבוי בדרך")

    assert location == "כרמל"
    assert post.call_args.kwargs["json"]["max_tokens"] == 100


def test_extract_location_returns_none_on_llm_failure(monkeypatch: pytest.MonkeyPatch):
    processor = make_processor(monkeypatch)

    with patch("src.external.news.news_client.requests.post", side_effect=RuntimeError("network down")):
        location = processor.extract_location("שריפה בכרמל", "כוחות כיבוי בדרך")

    assert location is None


# ---------------------------------------------------------------------------
# reasoning_effort passthrough - Groq's currently available fast model
# (openai/gpt-oss-20b, replacing the decommissioned llama-3.1-8b-instant)
# is a reasoning model; reasoning_effort keeps its hidden reasoning pass
# short for this latency-sensitive use case. It must only ever be sent when
# explicitly configured, so providers/models that don't understand it are
# never sent a parameter they'd choke on.
# ---------------------------------------------------------------------------


def test_reasoning_effort_is_sent_when_configured(monkeypatch: pytest.MonkeyPatch):
    config = {**GROQ_LLM_CONFIG, "reasoning_effort": "low"}
    processor = make_processor(monkeypatch, config)

    with patch(
        "src.external.news.news_client.requests.post", return_value=groq_reply({"locationName": "כרמל"})
    ) as post:
        processor.extract_location("שריפה בכרמל", "כוחות כיבוי בדרך")

    assert post.call_args.kwargs["json"]["reasoning_effort"] == "low"


def test_reasoning_effort_is_omitted_when_not_configured(monkeypatch: pytest.MonkeyPatch):
    processor = make_processor(monkeypatch)

    with patch(
        "src.external.news.news_client.requests.post", return_value=groq_reply({"locationName": "כרמל"})
    ) as post:
        processor.extract_location("שריפה בכרמל", "כוחות כיבוי בדרך")

    assert "reasoning_effort" not in post.call_args.kwargs["json"]


# ---------------------------------------------------------------------------
# translate_report
# ---------------------------------------------------------------------------


def test_translate_report_returns_the_llm_english_translation(monkeypatch: pytest.MonkeyPatch):
    processor = make_processor(monkeypatch)
    payload = {"title": "Large fire in Carmel", "summary": "Firefighters en route.", "locationName": "Carmel"}

    with patch("src.external.news.news_client.requests.post", return_value=groq_reply(payload)):
        title, summary, location = processor.translate_report("שריפה גדולה בכרמל", "כוחות כיבוי בדרך", "כרמל")

    assert (title, summary, location) == ("Large fire in Carmel", "Firefighters en route.", "Carmel")


def test_translate_report_sends_the_larger_translation_token_budget_not_the_extraction_one(
    monkeypatch: pytest.MonkeyPatch,
):
    processor = make_processor(monkeypatch)
    payload = {"title": "Title", "summary": "Summary", "locationName": None}

    with patch("src.external.news.news_client.requests.post", return_value=groq_reply(payload)) as post:
        processor.translate_report("כותרת", "תקציר", None)

    assert post.call_args.kwargs["json"]["max_tokens"] == 600


def test_translate_report_maps_an_explicit_null_location_to_none(monkeypatch: pytest.MonkeyPatch):
    processor = make_processor(monkeypatch)
    payload = {"title": "A wildfire", "summary": "No location given.", "locationName": None}

    with patch("src.external.news.news_client.requests.post", return_value=groq_reply(payload)):
        _title, _summary, location = processor.translate_report("שריפה", "ללא מיקום", None)

    assert location is None


def test_translate_report_falls_back_to_the_original_text_on_llm_failure(monkeypatch: pytest.MonkeyPatch):
    processor = make_processor(monkeypatch)

    with patch("src.external.news.news_client.requests.post", side_effect=RuntimeError("network down")):
        result = processor.translate_report("שריפה גדולה בכרמל", "כוחות כיבוי בדרך", "כרמל")

    assert result == ("שריפה גדולה בכרמל", "כוחות כיבוי בדרך", "כרמל")


def test_translate_report_falls_back_to_the_original_text_on_malformed_json(monkeypatch: pytest.MonkeyPatch):
    processor = make_processor(monkeypatch)

    with patch("src.external.news.news_client.requests.post", return_value=groq_reply("not valid json")):
        result = processor.translate_report("שריפה גדולה בכרמל", "כוחות כיבוי בדרך", "כרמל")

    assert result == ("שריפה גדולה בכרמל", "כוחות כיבוי בדרך", "כרמל")


def test_translate_report_falls_back_field_by_field_when_the_llm_reply_is_only_partially_valid(
    monkeypatch: pytest.MonkeyPatch,
):
    """A missing/blank field must not discard an otherwise-good translation of the other fields."""
    processor = make_processor(monkeypatch)
    payload = {"title": "Large fire in Carmel", "summary": ""}  # summary blank, locationName key absent entirely

    with patch("src.external.news.news_client.requests.post", return_value=groq_reply(payload)):
        title, summary, location = processor.translate_report("שריפה גדולה בכרמל", "כוחות כיבוי בדרך", "כרמל")

    assert title == "Large fire in Carmel"
    assert summary == "כוחות כיבוי בדרך"  # fell back to the original Hebrew summary
    assert location == "כרמל"  # fell back to the original Hebrew location


def test_translate_report_strips_a_markdown_code_fence(monkeypatch: pytest.MonkeyPatch):
    processor = make_processor(monkeypatch)
    fenced = "```json\n" + json.dumps({"title": "Fire", "summary": "Details.", "locationName": "Carmel"}) + "\n```"

    with patch("src.external.news.news_client.requests.post", return_value=groq_reply(fenced)):
        result = processor.translate_report("שריפה", "פרטים", "כרמל")

    assert result == ("Fire", "Details.", "Carmel")


def test_translate_report_works_against_the_gemini_provider_too(monkeypatch: pytest.MonkeyPatch):
    config = {**GROQ_LLM_CONFIG, "provider": "gemini"}
    processor = make_processor(monkeypatch, config)
    payload = {"title": "Fire", "summary": "Details.", "locationName": "Carmel"}

    with patch("src.external.news.news_client.requests.post", return_value=gemini_reply(payload)) as post:
        result = processor.translate_report("שריפה", "פרטים", "כרמל")

    assert result == ("Fire", "Details.", "Carmel")
    assert post.call_args.kwargs["json"]["generationConfig"]["maxOutputTokens"] == 600


# ---------------------------------------------------------------------------
# translate_location_name
# ---------------------------------------------------------------------------


def test_translate_location_name_returns_the_llm_english_name(monkeypatch: pytest.MonkeyPatch):
    processor = make_processor(monkeypatch)

    with patch("src.external.news.news_client.requests.post", return_value=groq_reply({"name": "Judean Hills"})) as post:
        name = processor.translate_location_name("הרי יהודה")

    assert name == "Judean Hills"
    # Uses the tight extraction budget, not the larger translation one -
    # a bare place name needs far less headroom than a full article.
    assert post.call_args.kwargs["json"]["max_tokens"] == 100


def test_translate_location_name_falls_back_to_the_original_on_llm_failure(monkeypatch: pytest.MonkeyPatch):
    processor = make_processor(monkeypatch)

    with patch("src.external.news.news_client.requests.post", side_effect=RuntimeError("network down")):
        name = processor.translate_location_name("הרי יהודה")

    assert name == "הרי יהודה"


def test_translate_location_name_falls_back_to_the_original_on_malformed_json(monkeypatch: pytest.MonkeyPatch):
    processor = make_processor(monkeypatch)

    with patch("src.external.news.news_client.requests.post", return_value=groq_reply("not valid json")):
        name = processor.translate_location_name("הרי יהודה")

    assert name == "הרי יהודה"


# ---------------------------------------------------------------------------
# Translation memoization - once a place name is translated successfully, a
# later transient LLM failure for that exact same text must reuse the
# known-good English name instead of regressing to the raw original.
# ---------------------------------------------------------------------------


def test_translate_location_name_reuses_a_cached_translation_without_calling_the_llm_again(
    monkeypatch: pytest.MonkeyPatch,
):
    processor = make_processor(monkeypatch)

    with patch(
        "src.external.news.news_client.requests.post", return_value=groq_reply({"name": "Judean Hills"})
    ) as post:
        first = processor.translate_location_name("הרי יהודה")
    assert first == "Judean Hills"
    assert post.call_count == 1

    with patch("src.external.news.news_client.requests.post") as post_again:
        second = processor.translate_location_name("הרי יהודה")

    assert second == "Judean Hills"
    post_again.assert_not_called()


def test_translate_location_name_falls_back_to_the_last_known_good_translation_not_raw_hebrew(
    monkeypatch: pytest.MonkeyPatch,
):
    processor = make_processor(monkeypatch)

    with patch("src.external.news.news_client.requests.post", return_value=groq_reply({"name": "Golan Heights"})):
        processor.translate_location_name("רמת הגולן")

    with patch("src.external.news.news_client.requests.post", side_effect=RuntimeError("rate limited")):
        name = processor.translate_location_name("רמת הגולן")

    assert name == "Golan Heights"


def test_translate_report_stores_its_successful_location_translation_in_the_shared_cache(
    monkeypatch: pytest.MonkeyPatch,
):
    processor = make_processor(monkeypatch)

    with patch(
        "src.external.news.news_client.requests.post",
        return_value=groq_reply({"title": "Fire near Golan Heights", "summary": "Smoke seen.", "locationName": "Golan Heights"}),
    ):
        processor.translate_report("שריפה ליד רמת הגולן", "עשן נצפה.", "רמת הגולן")

    with patch("src.external.news.news_client.requests.post", side_effect=RuntimeError("rate limited")):
        name = processor.translate_location_name("רמת הגולן")

    assert name == "Golan Heights"


def test_translate_report_falls_back_to_the_cached_location_when_the_whole_call_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    processor = make_processor(monkeypatch)

    with patch(
        "src.external.news.news_client.requests.post", return_value=groq_reply({"name": "Golan Heights"})
    ):
        processor.translate_location_name("רמת הגולן")

    with patch("src.external.news.news_client.requests.post", side_effect=RuntimeError("rate limited")):
        title, summary, location_name = processor.translate_report(
            "עדכון נוסף על השריפה ברמת הגולן", "האש מתפשטת.", "רמת הגולן"
        )

    # title/summary have no prior cache (unique per report) so they fall back
    # to the original text, but the location_name - the field the bug report
    # was specifically about - reuses the last known-good English name.
    assert title == "עדכון נוסף על השריפה ברמת הגולן"
    assert location_name == "Golan Heights"


# ---------------------------------------------------------------------------
# Retry logic: only genuinely transient failures (network errors, 429, 5xx)
# are retried, up to 3 attempts with exponential backoff; a permanent
# failure (4xx other than 429 - the exact shape of the earlier
# decommissioned-model 404) fails fast on the first attempt instead of
# wasting 2 more attempts and delaying the existing fallback.
# ---------------------------------------------------------------------------


def _http_error(status_code: int) -> requests.exceptions.HTTPError:
    response = Mock()
    response.status_code = status_code
    return requests.exceptions.HTTPError(response=response)


def test_translate_location_name_retries_a_transient_connection_error_and_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
):
    processor = make_processor(monkeypatch)
    monkeypatch.setattr("src.external.news.news_client.time.sleep", lambda _seconds: None)

    with patch(
        "src.external.news.news_client.requests.post",
        side_effect=[requests.exceptions.ConnectionError("reset"), groq_reply({"name": "Judean Hills"})],
    ) as post:
        name = processor.translate_location_name("הרי יהודה")

    assert name == "Judean Hills"
    assert post.call_count == 2


def test_translate_location_name_retries_a_429_and_then_succeeds(monkeypatch: pytest.MonkeyPatch):
    processor = make_processor(monkeypatch)
    monkeypatch.setattr("src.external.news.news_client.time.sleep", lambda _seconds: None)

    with patch(
        "src.external.news.news_client.requests.post",
        side_effect=[_http_error(429), groq_reply({"name": "Judean Hills"})],
    ) as post:
        name = processor.translate_location_name("הרי יהודה")

    assert name == "Judean Hills"
    assert post.call_count == 2


def test_translate_location_name_retries_a_5xx_and_then_succeeds(monkeypatch: pytest.MonkeyPatch):
    processor = make_processor(monkeypatch)
    monkeypatch.setattr("src.external.news.news_client.time.sleep", lambda _seconds: None)

    with patch(
        "src.external.news.news_client.requests.post",
        side_effect=[_http_error(503), _http_error(502), groq_reply({"name": "Judean Hills"})],
    ) as post:
        name = processor.translate_location_name("הרי יהודה")

    assert name == "Judean Hills"
    assert post.call_count == 3


def test_translate_location_name_falls_back_after_exhausting_all_retries(monkeypatch: pytest.MonkeyPatch):
    processor = make_processor(monkeypatch)
    monkeypatch.setattr("src.external.news.news_client.time.sleep", lambda _seconds: None)

    with patch(
        "src.external.news.news_client.requests.post",
        side_effect=requests.exceptions.Timeout("no response"),
    ) as post:
        name = processor.translate_location_name("הרי יהודה")

    assert name == "הרי יהודה"  # exhausted retries -> original fail-safe fallback, unchanged behavior
    assert post.call_count == 3  # exactly _MAX_LLM_ATTEMPTS, never more


def test_translate_location_name_does_not_retry_a_permanent_404(monkeypatch: pytest.MonkeyPatch):
    """The exact failure shape of the earlier decommissioned-model incident:
    retrying a 404 three times would only add latency (nothing about
    retrying fixes a nonexistent model), so it must fail on the first
    attempt, not the third."""
    processor = make_processor(monkeypatch)
    monkeypatch.setattr("src.external.news.news_client.time.sleep", lambda _seconds: None)

    with patch("src.external.news.news_client.requests.post", side_effect=_http_error(404)) as post:
        name = processor.translate_location_name("הרי יהודה")

    assert name == "הרי יהודה"
    assert post.call_count == 1  # no retry at all


def test_retry_backoff_delays_grow_exponentially(monkeypatch: pytest.MonkeyPatch):
    processor = make_processor(monkeypatch)
    sleep_calls: list[float] = []
    monkeypatch.setattr("src.external.news.news_client.time.sleep", sleep_calls.append)

    with patch(
        "src.external.news.news_client.requests.post",
        side_effect=requests.exceptions.Timeout("no response"),
    ):
        processor.translate_location_name("הרי יהודה")

    assert len(sleep_calls) == 2  # slept between attempts 1->2 and 2->3, never after the last attempt
    assert sleep_calls[1] > sleep_calls[0]  # exponential, not constant, backoff

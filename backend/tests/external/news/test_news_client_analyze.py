"""Tests for TextProcessor.analyze: signal-strength parsing and safe failure.

analyze never raises and never fabricates NONE: "the text was analyzed and has no
wildfire signal" (NONE) is different from "no reliable analysis is available" (None).
All HTTP is mocked; no API key or network is needed.
"""
from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest
import requests

from src.external.news.news_client import TextProcessor
from src.models.news_text_analysis import NewsTextAnalysis
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength

LLM_CONFIG = {
    "provider": "groq",
    "model": "test-model",
    "api_key_env": "TEST_NEWS_LLM_API_KEY",
    "temperature": 0.0,
    "max_tokens": 100,
    "request_timeout_seconds": 15,
}
PAYLOAD = {"locationName": "כרמל", "wildfireSignalStrength": "strong"}
UNAVAILABLE = NewsTextAnalysis(location_name=None, wildfire_signal_strength=None)


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def groq_reply(content) -> FakeResponse:
    text = content if isinstance(content, str) else json.dumps(content)
    return FakeResponse({"choices": [{"message": {"content": text}}]})


def gemini_reply(content) -> FakeResponse:
    text = content if isinstance(content, str) else json.dumps(content)
    return FakeResponse({"candidates": [{"content": {"parts": [{"text": text}]}}]})


def http_error(status_code: int) -> requests.exceptions.HTTPError:
    response = Mock()
    response.status_code = status_code
    return requests.exceptions.HTTPError(response=response)


@pytest.fixture()
def processor(monkeypatch: pytest.MonkeyPatch) -> TextProcessor:
    monkeypatch.setenv("TEST_NEWS_LLM_API_KEY", "fake-key")
    monkeypatch.setattr("src.external.news.news_client.time.sleep", lambda _seconds: None)
    return TextProcessor(keywords=["שריפה"], llm_config=LLM_CONFIG)


POST = "src.external.news.news_client.requests.post"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("none", NewsWildfireSignalStrength.NONE),
        ("weak", NewsWildfireSignalStrength.WEAK),
        ("moderate", NewsWildfireSignalStrength.MODERATE),
        ("strong", NewsWildfireSignalStrength.STRONG),
        ("STRONG", NewsWildfireSignalStrength.STRONG),
        ("  Weak ", NewsWildfireSignalStrength.WEAK),
    ],
)
def test_every_signal_strength_is_parsed(processor, raw, expected):
    with patch(POST, return_value=groq_reply({"locationName": "כרמל", "wildfireSignalStrength": raw})):
        analysis = processor.analyze("שריפה", "פרטים")

    assert analysis.wildfire_signal_strength is expected
    assert analysis.location_name == "כרמל"


def test_a_none_signal_is_distinct_from_an_unavailable_analysis(processor):
    with patch(POST, return_value=groq_reply({"locationName": None, "wildfireSignalStrength": "none"})):
        analyzed_no_signal = processor.analyze("כתבה על מניעת שריפות", "טיפים")
    with patch(POST, side_effect=RuntimeError("down")):
        unavailable = processor.analyze("כתבה על מניעת שריפות", "טיפים")

    assert analyzed_no_signal.wildfire_signal_strength is NewsWildfireSignalStrength.NONE
    assert analyzed_no_signal.location_name is None
    assert unavailable.wildfire_signal_strength is None


@pytest.mark.parametrize("location", [None, "", "   "])
def test_a_blank_or_null_location_becomes_none(processor, location):
    with patch(POST, return_value=groq_reply({"locationName": location, "wildfireSignalStrength": "weak"})):
        analysis = processor.analyze("שריפה", "עשן")

    assert analysis.location_name is None
    assert analysis.wildfire_signal_strength is NewsWildfireSignalStrength.WEAK


def test_a_markdown_code_fence_is_stripped(processor):
    fenced = "```json\n" + json.dumps(PAYLOAD) + "\n```"

    with patch(POST, return_value=groq_reply(fenced)):
        analysis = processor.analyze("שריפה", "פרטים")

    assert analysis == NewsTextAnalysis("כרמל", NewsWildfireSignalStrength.STRONG)


@pytest.mark.parametrize(
    "bad_reply",
    [
        "not valid json",
        {"locationName": "כרמל"},  # signal strength missing
        {"locationName": "כרמל", "wildfireSignalStrength": None},
        {"locationName": "כרמל", "wildfireSignalStrength": "catastrophic"},  # unrecognised value
        {"locationName": "כרמל", "wildfireSignalStrength": 3},
        {"locationName": 42, "wildfireSignalStrength": "strong"},  # location of the wrong type
        [],  # valid JSON but not an object
    ],
)
def test_any_unusable_reply_is_unavailable_never_none(processor, bad_reply):
    with patch(POST, return_value=groq_reply(bad_reply)):
        analysis = processor.analyze("שריפה", "פרטים")

    assert analysis == UNAVAILABLE


def test_gemini_provider_is_supported_with_one_call(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEST_NEWS_LLM_API_KEY", "fake-key")
    processor = TextProcessor(keywords=["שריפה"], llm_config={**LLM_CONFIG, "provider": "gemini"})

    with patch(POST, return_value=gemini_reply(PAYLOAD)) as post:
        analysis = processor.analyze("שריפה", "פרטים")

    assert analysis == NewsTextAnalysis("כרמל", NewsWildfireSignalStrength.STRONG)
    assert post.call_count == 1
    assert post.call_args.kwargs["json"]["generationConfig"]["maxOutputTokens"] == 100


def test_an_unsupported_provider_is_unavailable_without_any_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEST_NEWS_LLM_API_KEY", "fake-key")
    processor = TextProcessor(keywords=["שריפה"], llm_config={**LLM_CONFIG, "provider": "carrier-pigeon"})

    with patch(POST) as post:
        analysis = processor.analyze("שריפה", "פרטים")

    assert analysis == UNAVAILABLE
    post.assert_not_called()


def test_braces_and_odd_text_in_the_article_do_not_break_the_prompt(processor):
    with patch(POST, return_value=groq_reply(PAYLOAD)) as post:
        analysis = processor.analyze("שריפה {בכרמל} %s {0}", '}{ \n "quotes"')

    assert analysis.wildfire_signal_strength is NewsWildfireSignalStrength.STRONG
    assert "שריפה {בכרמל} %s {0}" in post.call_args.kwargs["json"]["messages"][0]["content"]


def test_a_non_string_title_never_makes_analyze_raise(processor):
    with patch(POST, side_effect=RuntimeError("down")):
        assert processor.analyze(None, None) == UNAVAILABLE


def test_the_prompt_contains_the_article_and_asks_for_both_fields(processor):
    with patch(POST, return_value=groq_reply(PAYLOAD)) as post:
        processor.analyze("כותרת ייחודית", "תקציר ייחודי")

    prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
    assert "כותרת ייחודית" in prompt and "תקציר ייחודי" in prompt
    assert "locationName" in prompt and "wildfireSignalStrength" in prompt
    for value in ("none", "weak", "moderate", "strong"):
        assert f'"{value}"' in prompt


def test_a_transient_failure_is_retried_then_succeeds(processor):
    with patch(POST, side_effect=[requests.exceptions.Timeout("slow"), groq_reply(PAYLOAD)]) as post:
        analysis = processor.analyze("שריפה", "פרטים")

    assert analysis.wildfire_signal_strength is NewsWildfireSignalStrength.STRONG
    assert post.call_count == 2


def test_a_permanent_error_is_not_retried_and_is_unavailable(processor):
    with patch(POST, side_effect=http_error(401)) as post:
        analysis = processor.analyze("שריפה", "פרטים")

    assert analysis == UNAVAILABLE
    assert post.call_count == 1


def test_exhausted_retries_end_in_unavailable(processor):
    with patch(POST, side_effect=requests.exceptions.ConnectionError("reset")) as post:
        analysis = processor.analyze("שריפה", "פרטים")

    assert analysis == UNAVAILABLE
    assert post.call_count == 3


# --- construction: the configured provider's API key is required, and only that one ---


def test_the_configured_providers_api_key_is_required(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TEST_NEWS_LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="TEST_NEWS_LLM_API_KEY"):
        TextProcessor(keywords=["שריפה"], llm_config=LLM_CONFIG)


def test_the_bundled_news_config_names_its_provider_and_key_variable():
    from src.config.config import load_config

    llm = load_config()["llm"]

    assert llm["provider"] in {"groq", "gemini"}
    assert isinstance(llm["api_key_env"], str) and llm["api_key_env"].strip()

"""Unit tests for TextProcessor.analyze(). All HTTP calls are mocked - no real
Groq/Gemini access is required to run these tests."""
from __future__ import annotations

import json
from unittest.mock import patch

from src.external.news.news_client import TextProcessor
from src.models.news_text_analysis import NewsTextAnalysis
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength

API_KEY_ENV = "TEST_NEWS_LLM_API_KEY"


class FakeResponse:
    """Minimal stand-in for `requests.Response`."""

    def __init__(self, json_body: dict, status_code: int = 200):
        self._json_body = json_body
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._json_body


def make_processor(monkeypatch, **llm_overrides) -> TextProcessor:
    monkeypatch.setenv(API_KEY_ENV, "fake-key")
    llm_config = {
        "provider": "groq",
        "model": "test-model",
        "temperature": 0.0,
        "max_tokens": 100,
        "request_timeout_seconds": 15,
        "api_key_env": API_KEY_ENV,
    }
    llm_config.update(llm_overrides)
    return TextProcessor(keywords=["שריפה", "אש"], llm_config=llm_config)


def groq_response(content: str) -> FakeResponse:
    return FakeResponse({"choices": [{"message": {"content": content}}]})


def analysis_json(location_name, strength: str) -> str:
    return json.dumps({"locationName": location_name, "wildfireSignalStrength": strength})


# --- is_relevant (unchanged existing behavior) ---


def test_is_relevant_matches_keyword():
    processor = TextProcessor.__new__(TextProcessor)
    processor.keywords = ["שריפה"]

    assert processor.is_relevant("דיווח על שריפה בכרמל", "") is True
    assert processor.is_relevant("חדשות ספורט", "") is False


# --- analyze(): valid signal values ---


def test_analyze_returns_none_signal(monkeypatch):
    processor = make_processor(monkeypatch)
    with patch("requests.post", return_value=groq_response(analysis_json(None, "none"))):
        result = processor.analyze("title", "summary")

    assert result == NewsTextAnalysis(location_name=None, wildfire_signal_strength=NewsWildfireSignalStrength.NONE)


def test_analyze_returns_weak_signal(monkeypatch):
    processor = make_processor(monkeypatch)
    with patch("requests.post", return_value=groq_response(analysis_json("כרמל", "weak"))):
        result = processor.analyze("title", "summary")

    assert result == NewsTextAnalysis(location_name="כרמל", wildfire_signal_strength=NewsWildfireSignalStrength.WEAK)


def test_analyze_returns_moderate_signal(monkeypatch):
    processor = make_processor(monkeypatch)
    with patch("requests.post", return_value=groq_response(analysis_json("כרמל", "moderate"))):
        result = processor.analyze("title", "summary")

    assert result.wildfire_signal_strength is NewsWildfireSignalStrength.MODERATE


def test_analyze_returns_strong_signal(monkeypatch):
    processor = make_processor(monkeypatch)
    with patch("requests.post", return_value=groq_response(analysis_json("כרמל", "strong"))):
        result = processor.analyze("title", "summary")

    assert result.wildfire_signal_strength is NewsWildfireSignalStrength.STRONG


def test_analyze_handles_markdown_fenced_json(monkeypatch):
    processor = make_processor(monkeypatch)
    fenced = "```json\n" + analysis_json("כרמל", "strong") + "\n```"
    with patch("requests.post", return_value=groq_response(fenced)):
        result = processor.analyze("title", "summary")

    assert result == NewsTextAnalysis(location_name="כרמל", wildfire_signal_strength=NewsWildfireSignalStrength.STRONG)


def test_analyze_strips_whitespace_from_location():
    from src.external.news.news_client import TextProcessor as _TP

    analysis = _TP._parse_analysis(analysis_json("  כרמל  ", "strong"))

    assert analysis.location_name == "כרמל"


# --- analyze(): null location ---


def test_analyze_handles_null_location(monkeypatch):
    processor = make_processor(monkeypatch)
    with patch("requests.post", return_value=groq_response(analysis_json(None, "moderate"))):
        result = processor.analyze("title", "summary")

    assert result.location_name is None
    assert result.wildfire_signal_strength is NewsWildfireSignalStrength.MODERATE


# --- analyze(): failure modes must return the "unavailable" state, never raise ---


def test_analyze_returns_unavailable_state_on_invalid_signal_value(monkeypatch):
    processor = make_processor(monkeypatch)
    with patch("requests.post", return_value=groq_response(analysis_json("כרמל", "extreme"))):
        result = processor.analyze("title", "summary")

    assert result == NewsTextAnalysis(location_name=None, wildfire_signal_strength=None)


def test_analyze_returns_unavailable_state_on_malformed_json(monkeypatch):
    processor = make_processor(monkeypatch)
    with patch("requests.post", return_value=groq_response("not valid json at all")):
        result = processor.analyze("title", "summary")

    assert result == NewsTextAnalysis(location_name=None, wildfire_signal_strength=None)


def test_analyze_returns_unavailable_state_on_missing_signal_field(monkeypatch):
    processor = make_processor(monkeypatch)
    with patch("requests.post", return_value=groq_response(json.dumps({"locationName": "כרמל"}))):
        result = processor.analyze("title", "summary")

    assert result == NewsTextAnalysis(location_name=None, wildfire_signal_strength=None)


def test_analyze_returns_unavailable_state_on_provider_failure(monkeypatch):
    processor = make_processor(monkeypatch)
    with patch("requests.post", side_effect=ConnectionError("network down")):
        result = processor.analyze("title", "summary")

    assert result == NewsTextAnalysis(location_name=None, wildfire_signal_strength=None)


def test_analyze_returns_unavailable_state_on_http_error(monkeypatch):
    processor = make_processor(monkeypatch)
    with patch("requests.post", return_value=FakeResponse({}, status_code=500)):
        result = processor.analyze("title", "summary")

    assert result == NewsTextAnalysis(location_name=None, wildfire_signal_strength=None)


def test_analyze_never_raises_and_does_not_crash_the_caller(monkeypatch):
    processor = make_processor(monkeypatch)
    with patch("requests.post", side_effect=RuntimeError("boom")):
        result = processor.analyze("title", "summary")  # must not raise

    assert isinstance(result, NewsTextAnalysis)
    assert result.wildfire_signal_strength is None


# --- unsupported provider ---


def test_analyze_returns_unavailable_state_for_unsupported_provider(monkeypatch):
    processor = make_processor(monkeypatch, provider="unsupported-provider")

    result = processor.analyze("title", "summary")

    assert result == NewsTextAnalysis(location_name=None, wildfire_signal_strength=None)

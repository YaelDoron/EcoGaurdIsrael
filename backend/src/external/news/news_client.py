"""Fetches and parses configured RSS feeds."""
from __future__ import annotations

import logging
from typing import Any
import json
import os

import feedparser
import requests

from src.models.news_text_analysis import NewsTextAnalysis
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength

logger = logging.getLogger(__name__)

# The prompt template for one structured LLM analysis pass: location + how
# strongly the article's own text supports an active wildfire. One call
# covers both, rather than a separate call per field.
_ANALYSIS_PROMPT_TEMPLATE = """\
You are a news analyst reading Hebrew news articles about possible wildfires in Israel.

Judge ONLY the supplied title and summary below. Do not use outside knowledge about real events, and do not assume a wildfire is actually happening just because a wildfire-related word appears in the text.

Task 1 - location: extract a single geographical location (city, settlement, forest, mountain, or region in Israel) where the described event is taking place, if any.
- Clean Hebrew prepositions from the beginning of the location name (e.g., "בכרמל" -> "כרמל", "ליער ירושלים" -> "יער ירושלים").
- If no specific location is mentioned, use null.

Task 2 - wildfireSignalStrength: classify how strongly the ARTICLE TEXT ITSELF claims an active wildfire is occurring right now. Use exactly one of these four values:
- "none": the text does not actually provide meaningful evidence of an active wildfire (e.g. a retrospective article, a fire-prevention article, an unrelated use of a fire-related word, or an explicit report that a suspected fire was false).
- "weak": possible or indirect evidence only (e.g. smoke reported, a rumor, an unverified social-media report, "suspected" flames).
- "moderate": the text directly reports an active wildfire, but the information is still preliminary or indirect.
- "strong": the text explicitly describes an active wildfire with strong evidence, such as visible flames/active burning, a firefighting response, an evacuation due to an active fire, or an explicit official/emergency-service statement within the text.

Distinguish smoke, rumor, or suspicion ("weak") from explicit, current active-fire reporting ("moderate"/"strong"). A retrospective, preventive, or explicitly-false-alarm article is "none" even if it uses wildfire-related words.

Rules:
- Return ONLY a valid JSON object, with no extra text, no markdown, and no explanations.
- The format must be exactly: {{"locationName": "<location_name_or_null>", "wildfireSignalStrength": "<none|weak|moderate|strong>"}}

Title: {title}
Summary: {summary}
"""


class TextProcessor:
    """Filters articles by keyword relevance and extracts locations via a fast LLM."""

    def __init__(self, keywords: list[str], llm_config: dict):
        self.keywords = keywords
        self.provider = llm_config["provider"]
        self.model = llm_config["model"]
        self.temperature = llm_config.get("temperature", 0.0)
        self.max_tokens = llm_config.get("max_tokens", 100)
        self.timeout = llm_config.get("request_timeout_seconds", 15)

        api_key_env = llm_config["api_key_env"]
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(
                f"Missing LLM API key: environment variable '{api_key_env}' is not set. "
                "Add it to your .env file."
            )

    def is_relevant(self, title: str, summary: str) -> bool:
        """Keyword filter: True if any target keyword appears in the title or summary."""
        text = f"{title} {summary}"
        return any(keyword in text for keyword in self.keywords)

    def analyze(self, title: str, summary: str) -> NewsTextAnalysis:
        """One LLM call returning location + wildfire signal strength for one article.

        Never raises. On ANY failure (network/API error, malformed JSON, an
        unrecognized wildfireSignalStrength value) returns
        NewsTextAnalysis(location_name=None, wildfire_signal_strength=None) -
        an explicit "analysis unavailable" state. It never fabricates NONE,
        which has the different meaning "analyzed, no signal found".
        """
        prompt = _ANALYSIS_PROMPT_TEMPLATE.format(title=title, summary=summary)
        try:
            if self.provider == "groq":
                raw_content = self._call_groq(prompt)
            elif self.provider == "gemini":
                raw_content = self._call_gemini(prompt)
            else:
                raise ValueError(f"Unsupported llm.provider in config: '{self.provider}'")
            return self._parse_analysis(raw_content)
        except Exception:
            logger.exception("LLM structured news analysis failed for title: %r", title[:80])
            return NewsTextAnalysis(location_name=None, wildfire_signal_strength=None)

    # Call the Groq API (or Gemini) to extract location information from the article.
    def _call_groq(self, prompt: str) -> str:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _call_gemini(self, prompt: str) -> str:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            params={"key": self.api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens,
                    "responseMimeType": "application/json",
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]

    # Parse the raw LLM response into a NewsTextAnalysis. Raises on malformed JSON, a
    # non-string locationName, or a missing/unrecognized wildfireSignalStrength -
    # analyze() catches this and converts it to the "unavailable" state.
    @staticmethod
    def _parse_analysis(raw_content: str) -> NewsTextAnalysis:
        content = raw_content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        data = json.loads(content)

        location = data.get("locationName")
        if location is not None:
            if not isinstance(location, str):
                raise ValueError(f"locationName must be a string or null, got {location!r}")
            location = location.strip() or None

        raw_strength = data.get("wildfireSignalStrength")
        if not isinstance(raw_strength, str):
            raise ValueError(f"wildfireSignalStrength must be a string, got {raw_strength!r}")
        try:
            strength = NewsWildfireSignalStrength(raw_strength.strip().lower())
        except ValueError:
            raise ValueError(f"Unsupported wildfireSignalStrength value: {raw_strength!r}") from None

        return NewsTextAnalysis(location_name=location, wildfire_signal_strength=strength)

# Initialization of the RSSFetcher is handled in the NewsMonitoringAgent constructor.
class RSSFetcher:
    def __init__(self, feeds: list[dict[str, str]]):
        self.feeds = feeds

    # Fetch every configured feed and return a flat list of all the reports.
    def fetch_all(self) -> list[dict[str, Any]]:
        """Fetch every configured feed and return a flat list of raw entries."""
        entries: list[dict[str, Any]] = []
        for feed in self.feeds:
            name, url = feed["name"], feed["url"]
            try:
                parsed = feedparser.parse(url) # Feedparser analyzes the feed and returns a structured object.
            except Exception:
                logger.exception("Failed to fetch feed '%s' (%s)", name, url)
                continue

            if parsed.bozo and not parsed.entries: # If one of the sites is down or there is an internal error, log a warning and skip to the next feed.
                logger.warning(
                    "Feed '%s' returned no usable entries: %s",
                    name,
                    getattr(parsed, "bozo_exception", "unknown parse error"),
                )
                continue

            # For each entry in the feed, extract the relevant fields and append to the entries list.
            for entry in parsed.entries:
                entries.append(
                    {
                        "source_feed": name,
                        "title": entry.get("title", "").strip(),
                        "summary": entry.get("summary", entry.get("description", "")).strip(),
                        "link": entry.get("link", "").strip(),
                        "published": entry.get("published", ""),
                    }
                )
        logger.info("Fetched %d total entries from %d feeds", len(entries), len(self.feeds))
        return entries

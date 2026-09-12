"""Fetches and parses configured RSS feeds."""
from __future__ import annotations

import logging
from typing import Any
import json
import os

import feedparser
import requests

logger = logging.getLogger(__name__)

# The prompt template for the LLM to extract a location from the article text.
_LOCATION_PROMPT_TEMPLATE = """\
You are a news analyst parsing Hebrew news articles to extract a single geographical location (city, settlement, forest, mountain, or region in Israel) where the described wildfire event is taking place.

Rules:
- Return ONLY a valid JSON object, with no extra text, no markdown, and no explanations.
- The format must be exactly: {{"locationName": "<location_name>"}}
- Clean Hebrew prepositions from the beginning of the location name (e.g., "בכרמל" -> "כרמל", "ליער ירושלים" -> "יער ירושלים").
- If no specific location is mentioned in the text, return: {{"locationName": null}}

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

    def extract_location(self, title: str, summary: str) -> str | None:
        """Ask the LLM for the location mentioned in the article. Returns None on any failure
        or when no location is present, never raises."""
        prompt = _LOCATION_PROMPT_TEMPLATE.format(title=title, summary=summary)
        try:
            if self.provider == "groq":
                raw_content = self._call_groq(prompt)
            elif self.provider == "gemini":
                raw_content = self._call_gemini(prompt)
            else:
                raise ValueError(f"Unsupported llm.provider in config: '{self.provider}'")
            return self._parse_location(raw_content)
        except Exception:
            logger.exception("LLM location extraction failed for title: %r", title[:80])
            return None

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

    # Parse the raw LLM response into a location name. Returns None on any failure or if no location is found.
    @staticmethod
    def _parse_location(raw_content: str) -> str | None:
        content = raw_content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        data = json.loads(content)
        location = data.get("locationName")
        if location is None:
            return None
        location = location.strip()
        return location or None

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

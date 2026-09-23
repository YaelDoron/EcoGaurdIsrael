"""Fetches and parses configured RSS feeds."""
from __future__ import annotations

import logging
from typing import Any
import json
import os
import time

import feedparser
import requests

logger = logging.getLogger(__name__)

# Retry only genuinely transient failures - a network-level error (timeout,
# connection reset), a rate limit (429), or a transient server error (5xx).
# A 4xx other than 429 (bad request, invalid/missing API key, or a
# decommissioned model returning 404 - the exact failure this codebase hit
# earlier) is a PERMANENT configuration problem: retrying it 3 times would
# only add latency without any chance of success, and would delay the
# existing "TRANSLATION FALLBACK TRIGGERED" warning that's meant to be an
# immediate, clear diagnostic signal. Those fail fast, on the first attempt.
_MAX_LLM_ATTEMPTS = 3
_LLM_RETRY_BASE_DELAY_SECONDS = 0.5


def _is_transient_llm_error(exc: Exception) -> bool:
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
        return response is not None and (response.status_code == 429 or response.status_code >= 500)
    return False

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

# The prompt template for the LLM to translate an already-extracted article
# into English for the operational dashboard. Kept as a separate call/prompt
# from location extraction (rather than combined) so a translation failure
# can never affect the (already geocoded) location_name extraction, and so
# each prompt stays focused and easy to reason about independently.
_TRANSLATION_PROMPT_TEMPLATE = """\
You are translating a Hebrew wildfire news article into English for an operational dashboard read by English-speaking wildfire dispatchers.

Rules:
- Return ONLY a valid JSON object, with no extra text, no markdown, and no explanations.
- The format must be exactly: {{"title": "<english_title>", "summary": "<english_summary>", "locationName": <english_location_name_or_null>}}
- Translate naturally and concisely - never add information that is not present in the source text, and never summarize further than the original.
- Render proper nouns (place names, organizations) in their standard English form, e.g. "כרמל" -> "Carmel", "כבאות והצלה" -> "Fire and Rescue Services".
- Geographical names are proper nouns: transliterate them using their standard official Israeli government form, never a literal word-for-word translation - e.g. "מטה יהודה" -> "Mateh Yehuda" (not "Mate Yehuda" or "Mateh Yehuda Regional Council"), "פתח תקווה" -> "Petah Tikva" (not "Petach Tikva").
- If "Location name" below is "(none)", return "locationName": null. Otherwise translate it the same way as any other place name.
- If the input text is already in English, return it unchanged (not re-translated or altered).

Title: {title}
Summary: {summary}
Location name: {location_name}
"""

# For translating a standalone place name with no surrounding article text
# (e.g. a SatelliteHotspot's own location_name) - same idea as the location
# half of _TRANSLATION_PROMPT_TEMPLATE, but without an empty title/summary
# confusing the prompt.
_LOCATION_NAME_TRANSLATION_PROMPT_TEMPLATE = """\
Translate the following Hebrew geographical place name (city, settlement, forest, mountain, or region in Israel) into its standard English name.

Rules:
- Return ONLY a valid JSON object, with no extra text, no markdown, and no explanations.
- The format must be exactly: {{"name": "<english_name>"}}
- Use the standard English form, e.g. "הרי יהודה" -> "Judean Hills", "יער ירושלים" -> "Jerusalem Forest".
- Geographical names are proper nouns: transliterate them using their standard official Israeli government form, never a literal word-for-word translation - e.g. "מטה יהודה" -> "Mateh Yehuda" (not "Mate Yehuda" or "Mateh Yehuda Regional Council"), "פתח תקווה" -> "Petah Tikva" (not "Petach Tikva").
- If the input is already in English, return it unchanged.

Location name: {location_name}
"""


class TextProcessor:
    """Filters articles by keyword relevance and extracts locations via a fast LLM."""

    def __init__(self, keywords: list[str], llm_config: dict):
        self.keywords = keywords
        self.provider = llm_config["provider"]
        self.model = llm_config["model"]
        self.temperature = llm_config.get("temperature", 0.0)
        self.max_tokens = llm_config.get("max_tokens", 100)
        # Translating a title+summary needs far more headroom than the
        # short {"locationName": "..."} extraction reply - a separate
        # budget so raising it can never silently loosen extract_location's
        # own (deliberately tight) max_tokens.
        self.translation_max_tokens = llm_config.get("translation_max_tokens", 600)
        self.timeout = llm_config.get("request_timeout_seconds", 15)
        # Groq-specific, optional (e.g. "low"/"medium"/"high" for a
        # reasoning-capable model like openai/gpt-oss-20b) - only added to
        # the request payload when configured, so a non-reasoning model/
        # provider is never sent a parameter it doesn't understand.
        self.reasoning_effort = llm_config.get("reasoning_effort")

        api_key_env = llm_config["api_key_env"]
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(
                f"Missing LLM API key: environment variable '{api_key_env}' is not set. "
                "Add it to your .env file."
            )

        # Memoizes successful place-name translations, keyed by the exact
        # original-language source string. The demo simulation re-translates
        # the SAME small set of Hebrew place names on every single event
        # tick (one call per generated report/hotspot); under frequent
        # auto-polling this multiplies LLM call volume enough to hit
        # transient failures/rate limits. Once a name has been translated
        # successfully once, this cache lets every later call for that exact
        # text return the known-good English translation instantly - no LLM
        # call, so no chance of a fresh failure - and, on a genuine LLM
        # failure for a name that WAS previously translated, the fallback
        # path below prefers this cached value over the raw original, so a
        # display that has already shown a correct English name never
        # flickers back to the raw source text. Only ever grows across this
        # instance's lifetime; failures are deliberately never cached so a
        # later retry can still succeed.
        self._location_translation_cache: dict[str, str] = {}

    def is_relevant(self, title: str, summary: str) -> bool:
        """Keyword filter: True if any target keyword appears in the title or summary."""
        text = f"{title} {summary}"
        return any(keyword in text for keyword in self.keywords)

    def extract_location(self, title: str, summary: str) -> str | None:
        """Ask the LLM for the location mentioned in the article. Returns None on any failure
        or when no location is present, never raises."""
        prompt = _LOCATION_PROMPT_TEMPLATE.format(title=title, summary=summary)
        try:
            raw_content = self._call_llm(prompt, max_tokens=self.max_tokens)
            return self._parse_location(raw_content)
        except Exception:
            logger.exception("LLM location extraction failed for title: %r", title[:80])
            return None

    def translate_report(self, title: str, summary: str, location_name: str | None) -> tuple[str, str, str | None]:
        """Translate a Hebrew article's title/summary/location_name to English via the LLM.

        Dashboard operators are English-speaking and a static dictionary
        (see frontend stationTranslations.ts) only covers known station/
        region vocabulary - it cannot translate freeform incoming news text.
        This is the dynamic counterpart: translation happens once, here, at
        ingestion, so every downstream reader (API, frontend) sees English
        without needing its own translation step.

        On ANY failure (LLM error, malformed/incomplete JSON, missing API
        key) this returns the ORIGINAL title/summary unchanged - a
        best-effort presentation concern that must never block a report
        from being saved. `location_name` is the one exception: if this
        exact location_name was already translated successfully by an
        earlier call (on this instance), that known-good English name is
        reused instead of the raw original - see `_location_translation_cache`
        on __init__ - so a genuinely transient failure never regresses an
        already-correct display back to the source language.
        """
        fallback = (title, summary, self._cached_or_original_location(location_name))
        prompt = _TRANSLATION_PROMPT_TEMPLATE.format(
            title=title, summary=summary, location_name=location_name if location_name else "(none)"
        )
        try:
            raw_content = self._call_llm(prompt, max_tokens=self.translation_max_tokens)
            translated = self._parse_translation(raw_content, fallback)
        except Exception:
            logger.warning(
                "TRANSLATION FALLBACK TRIGGERED: LLM translation failed for title %r (location_name=%r) - "
                "persisting %s.",
                title[:80],
                location_name,
                "the last known-good English location name" if fallback[2] != location_name else "original text",
                exc_info=True,
            )
            return fallback

        self._remember_location_translation(location_name, translated[2])
        return translated

    def _cached_or_original_location(self, location_name: str | None) -> str | None:
        if location_name is None:
            return None
        return self._location_translation_cache.get(location_name, location_name)

    def _remember_location_translation(self, original: str | None, translated: str | None) -> None:
        if original is not None and translated is not None and translated != original:
            self._location_translation_cache[original] = translated

    def translate_location_name(self, location_name: str) -> str:
        """Translate a single Hebrew place name to English via the LLM.

        Used where only a bare place name exists with no surrounding
        article (e.g. a SatelliteHotspot's own location_name), as opposed
        to translate_report's title+summary+location_name. A cache hit for
        this exact location_name (see `_location_translation_cache` on
        __init__) is returned immediately without calling the LLM at all.
        On any failure (LLM error, malformed JSON, missing field) this
        returns the last known-good English translation for this exact
        text if one was ever produced, otherwise the ORIGINAL location_name
        unchanged - the same never-raise, never-block contract as
        extract_location/translate_report, extended so a transient failure
        can never regress an already-correct translation back to the
        source language.
        """
        cached = self._location_translation_cache.get(location_name)
        if cached is not None:
            return cached

        prompt = _LOCATION_NAME_TRANSLATION_PROMPT_TEMPLATE.format(location_name=location_name)
        try:
            raw_content = self._call_llm(prompt, max_tokens=self.max_tokens)
            data = self._parse_json_object(raw_content)
            name = data.get("name")
            translated = name.strip() if isinstance(name, str) and name.strip() else location_name
        except Exception:
            logger.warning(
                "TRANSLATION FALLBACK TRIGGERED: LLM location-name translation failed for %r - persisting "
                "the original, untranslated text.",
                location_name[:80],
                exc_info=True,
            )
            return location_name

        self._remember_location_translation(location_name, translated)
        return translated

    def _call_llm(self, prompt: str, *, max_tokens: int) -> str:
        """Dispatch to the configured provider, retrying up to
        _MAX_LLM_ATTEMPTS times on a transient failure (see
        _is_transient_llm_error) with exponential backoff between attempts.
        A non-transient failure (bad config, invalid API key, an unknown
        provider) raises immediately on the first attempt - only the class
        of error a retry could plausibly fix is ever retried. The final
        attempt's exception always propagates to the caller unchanged (the
        translate_*/extract_location methods' own try/except is what turns
        that into the original-text fallback - this method itself never
        swallows a failure)."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_LLM_ATTEMPTS):
            try:
                if self.provider == "groq":
                    return self._call_groq(prompt, max_tokens=max_tokens)
                if self.provider == "gemini":
                    return self._call_gemini(prompt, max_tokens=max_tokens)
                raise ValueError(f"Unsupported llm.provider in config: '{self.provider}'")
            except Exception as exc:  # noqa: BLE001 - classified below; re-raised when not retryable.
                last_exc = exc
                if not _is_transient_llm_error(exc) or attempt == _MAX_LLM_ATTEMPTS - 1:
                    raise
                delay = _LLM_RETRY_BASE_DELAY_SECONDS * (2**attempt)
                logger.warning(
                    "Transient LLM call failure (attempt %d/%d) - retrying in %.1fs: %s",
                    attempt + 1,
                    _MAX_LLM_ATTEMPTS,
                    delay,
                    exc,
                )
                time.sleep(delay)
        raise last_exc  # pragma: no cover - loop always returns or raises above

    # Call the Groq API to run one prompt. `max_tokens` is per-call (not
    # self.max_tokens) so extract_location's tight budget and
    # translate_report's much larger one never interfere with each other.
    def _call_groq(self, prompt: str, *, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _call_gemini(self, prompt: str, *, max_tokens: int) -> str:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            params={"key": self.api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": max_tokens,
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
        data = TextProcessor._parse_json_object(raw_content)
        location = data.get("locationName")
        if location is None:
            return None
        location = location.strip()
        return location or None

    @staticmethod
    def _parse_translation(
        raw_content: str, fallback: tuple[str, str, str | None]
    ) -> tuple[str, str, str | None]:
        fallback_title, fallback_summary, fallback_location = fallback
        data = TextProcessor._parse_json_object(raw_content)

        title = data.get("title")
        translated_title = title.strip() if isinstance(title, str) and title.strip() else fallback_title

        summary = data.get("summary")
        translated_summary = summary.strip() if isinstance(summary, str) and summary.strip() else fallback_summary

        if "locationName" not in data:
            translated_location = fallback_location
        else:
            raw_location = data["locationName"]
            if raw_location is None:
                translated_location = None
            elif isinstance(raw_location, str) and raw_location.strip():
                translated_location = raw_location.strip()
            else:
                translated_location = fallback_location

        return translated_title, translated_summary, translated_location

    @staticmethod
    def _parse_json_object(raw_content: str) -> dict:
        content = raw_content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        return json.loads(content)

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

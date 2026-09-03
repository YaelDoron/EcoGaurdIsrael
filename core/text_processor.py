"""Keyword filtering and LLM-based location extraction (geoparsing) from Hebrew text."""
import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

_LOCATION_PROMPT_TEMPLATE = """\
אתה מנתח כתבות חדשות בעברית ומחלץ מהן שם מקום גיאוגרפי יחיד (עיר, יישוב, יער, הר או אזור בישראל) \
שבו מתרחש אירוע השרפה המתואר.

כללים:
- החזר אך ורק אובייקט JSON תקין, ללא טקסט נוסף, ללא markdown וללא הסברים.
- הפורמט חייב להיות בדיוק: {{"locationName": "<שם המקום>"}}
- נקה את שם המקום מתחיליות יחס בעברית כגון "ב", "ל", "מ" (למשל "בכרמל" -> "כרמל", "ליער ירושלים" -> "יער ירושלים").
- אם לא מוזכר מקום ספציפי בטקסט, החזר: {{"locationName": null}}

כותרת: {title}
תקציר: {summary}
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

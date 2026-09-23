"""HTTP client for the Google Gemini `generateContent` API.

This module is responsible ONLY for sending a generation request to Gemini
over HTTP and returning the generated text it responds with. It has no
knowledge of the database, repositories, FireEvents, active fires,
locations, ML scores, or any other EcoGuard domain concept, and it builds
no prompts of its own - that all belongs to a later caller layer
(ChatbotAgent), built on top of this client.

Bounded retry: a transient server failure (429 rate-limit/500/502/503/504,
or a `requests.Timeout`) is retried up to `_MAX_ATTEMPTS` times with a
short backoff before becoming the same `GeminiServiceUnavailableError` this
client always raised - see `generate_content`'s own docstring for the exact
policy. The final bounded attempt calls a configured fallback model instead
of the primary one, but only when every attempt before it already failed
transiently - an availability improvement only, never a general model
switch. The public interface (constructor, `generate_content`, exception
types) is otherwise unchanged.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

from src.config.settings import settings
from src.external.gemini.exceptions import (
    GeminiAuthenticationError,
    GeminiClientError,
    GeminiConfigurationError,
    GeminiEmptyResponseError,
    GeminiInvalidResponseError,
    GeminiServiceUnavailableError,
)

logger = logging.getLogger(__name__)

_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_AUTHENTICATION_ERROR_STATUS_CODES = (401, 403)
_SERVICE_UNAVAILABLE_STATUS_CODES = (429, 500, 502, 503, 504)
_RETRYABLE_SERVER_STATUS_CODES = (500, 502, 503, 504)
_RATE_LIMIT_STATUS_CODE = 429

# Bounded retry only - never unbounded. Index 0 is the delay before the 2nd
# attempt, index 1 the delay before the 3rd (final) attempt.
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (1.0, 2.0)
# A valid `Retry-After` header is preferred over the fixed backoff above,
# but always clamped to this so one request can never sleep for an
# excessive period regardless of what the provider asks for.
_MAX_RETRY_AFTER_SECONDS = 5.0

# A 429 is only treated as "clearly quota-exhausted" (and therefore NOT
# retried - retrying within the same request cannot help) when the
# provider's own error body carries BOTH the RESOURCE_EXHAUSTED status AND
# one of these quota/billing-specific phrases (observed verbatim from a
# real exhausted-quota response during Task 7D diagnostics) - see
# `_is_quota_exhausted`. Any other 429 (an ordinary transient rate-limit
# blip) is retried like any other transient server failure.
_QUOTA_EXHAUSTED_MESSAGE_MARKERS = (
    "exceeded your current quota",
    "quota exceeded for metric",
    "check your plan and billing",
)


class GeminiClient:
    """Thin HTTP client for Google Gemini's `generateContent` API.

    Accepts an already-built Gemini `contents` turn sequence (and an
    optional system instruction) and returns the generated text as-is.
    Building EcoGuard-specific prompts, context, or conversation-history
    semantics is the caller's responsibility, not this client's - `contents`
    is passed straight to Gemini unchanged, whether it holds a single
    question or a full system-prompt/context/question/history sequence.
    """

    def __init__(
        self,
        api_key: str = settings.GEMINI_API_KEY,
        model: str = settings.GEMINI_MODEL,
        timeout: int = settings.GEMINI_REQUEST_TIMEOUT,
        fallback_model: str = settings.GEMINI_FALLBACK_MODEL,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.fallback_model = fallback_model

    def generate_content(
        self,
        contents: list[dict[str, Any]],
        *,
        system_instruction: str | None = None,
    ) -> str:
        """Send one `generateContent` request and return the generated text.

        `contents` must already be in Gemini's own turn shape (e.g.
        `[{"role": "user", "parts": [{"text": "..."}]}]`) - this method
        does not interpret or validate its meaning, only forwards it.
        `system_instruction`, when given, is sent via Gemini's dedicated
        `systemInstruction` field (kept separate from `contents`, matching
        Gemini's own API shape) so a fixed instruction prompt never has to
        be spliced into the turn history by this client.

        Never returns fabricated content: raises a typed
        `GeminiClientError` subclass for any failure, including a
        successful HTTP response that carries no usable generated text.

        Bounded retry: a transient server failure (HTTP 500/502/503/504, or
        a 429 that does not clearly indicate quota exhaustion - see
        `_is_quota_exhausted`) or a `requests.Timeout` is retried up to
        `_MAX_ATTEMPTS` (3) times total, waiting ~1s then ~2s between
        attempts (or, for an HTTP failure, the response's own `Retry-After`
        value, clamped to `_MAX_RETRY_AFTER_SECONDS`, when present and
        valid). Authentication/configuration failures, a clearly
        quota-exhausted 429, a connection error, and an invalid/empty
        successful response are never retried - the same typed exception is
        raised immediately, exactly as before this behavior was added.

        Model fallback: the final bounded attempt is sent to
        `self.fallback_model` instead of `self.model` - but only because
        reaching that final attempt at all already means every attempt
        before it failed transiently (a non-transient failure always raises
        immediately and never reaches it). This is a pure availability
        improvement, not a general model switch: it never changes which
        failures are retried, never adds a 4th attempt, and never applies
        to a non-transient failure. If the fallback attempt also fails, the
        final attempt's failure still becomes the same
        `GeminiServiceUnavailableError` this client always raised for these
        cases.
        """
        if not self.api_key:
            raise GeminiConfigurationError("Gemini API key is not configured.")
        if not contents:
            raise GeminiClientError("contents must not be empty.")

        payload: dict[str, Any] = {"contents": contents}
        if system_instruction is not None:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            model = self._model_for_attempt(attempt)
            if model != self.model:
                logger.error(
                    "Gemini primary model unavailable after transient failures; "
                    "using fallback model (primary=%s, fallback=%s)",
                    self.model,
                    model,
                )
            url = f"{_API_BASE_URL}/models/{model}:generateContent"
            request_payload = self._payload_for_model(payload, model)

            try:
                response = requests.post(
                    url,
                    params={"key": self.api_key},
                    json=request_payload,
                    timeout=self.timeout,
                )
            except requests.exceptions.Timeout as exc:
                if attempt < _MAX_ATTEMPTS:
                    delay = _BACKOFF_SECONDS[attempt - 1]
                    logger.error(
                        "Gemini request timed out - retrying in %.1fs (attempt %d/%d, model=%s)",
                        delay,
                        attempt,
                        _MAX_ATTEMPTS,
                        model,
                    )
                    time.sleep(delay)
                    continue
                logger.error("Gemini request timed out (model=%s)", model)
                raise GeminiServiceUnavailableError("Gemini request timed out.") from exc
            except requests.exceptions.ConnectionError as exc:
                logger.error("Gemini connection failed (model=%s)", model)
                raise GeminiServiceUnavailableError("Could not connect to Gemini service.") from exc
            except requests.exceptions.RequestException as exc:
                logger.error("Gemini request failed (model=%s)", model)
                raise GeminiClientError("Gemini request failed.") from exc

            if attempt < _MAX_ATTEMPTS and self._is_retryable(response):
                delay = self._retry_delay_seconds(response, attempt - 1)
                logger.error(
                    "Gemini service returned HTTP %s - retrying in %.1fs (attempt %d/%d, model=%s)",
                    response.status_code,
                    delay,
                    attempt,
                    _MAX_ATTEMPTS,
                    model,
                )
                time.sleep(delay)
                continue

            self._raise_for_status(response)

            try:
                body = response.json()
            except ValueError as exc:
                logger.error("Gemini returned an invalid (non-JSON) response (model=%s)", model)
                raise GeminiInvalidResponseError("Gemini returned an invalid JSON response.") from exc

            return self._extract_text(body)

        # Unreachable: every loop iteration above either `continue`s,
        # returns, or raises - kept only so this function's control flow is
        # unambiguous to static analysis.
        raise GeminiServiceUnavailableError("Gemini request failed after repeated retries.")

    def _model_for_attempt(self, attempt: int) -> str:
        """Which model this attempt should call. Only the final bounded
        attempt ever uses `self.fallback_model`, and only when it is
        actually configured to something other than the primary model -
        reaching the final attempt at all already means every attempt
        before it failed transiently, since a non-transient failure always
        raises immediately from within the loop and never reaches here."""
        if attempt == _MAX_ATTEMPTS and self.fallback_model and self.fallback_model != self.model:
            return self.fallback_model
        return self.model

    @staticmethod
    def _payload_for_model(payload: dict[str, Any], model: str) -> dict[str, Any]:
        """Gemma-family models need `generationConfig.thinkingConfig.thinkingLevel`
        set to `"minimal"` - without it, Gemma returns a large visible
        reasoning trace as part of the answer instead of a clean final
        response (see `_extract_text`, which separately skips any
        `thought: true` part regardless of model, as defense in depth).
        Every other model is sent the exact same payload as before,
        unmodified - this never touches `contents`/`systemInstruction`."""
        if "gemma" not in model.lower():
            return payload
        request_payload = dict(payload)
        generation_config = dict(request_payload.get("generationConfig", {}))
        thinking_config = dict(generation_config.get("thinkingConfig", {}))
        thinking_config["thinkingLevel"] = "minimal"
        generation_config["thinkingConfig"] = thinking_config
        request_payload["generationConfig"] = generation_config
        return request_payload

    @classmethod
    def _is_retryable(cls, response: requests.Response) -> bool:
        """Whether this response's status is worth retrying at all - see
        `generate_content`'s own docstring for the exact policy."""
        status_code = response.status_code
        if status_code in _RETRYABLE_SERVER_STATUS_CODES:
            return True
        if status_code == _RATE_LIMIT_STATUS_CODE:
            return not cls._is_quota_exhausted(response)
        return False

    @staticmethod
    def _is_quota_exhausted(response: requests.Response) -> bool:
        """True only when a 429 response's own error body clearly indicates
        the API key's quota is exhausted (retrying cannot help within this
        request) rather than an ordinary transient rate-limit blip. Any
        failure to parse/recognize the body means "not clearly
        quota-exhausted" (stays retryable) - never the reverse, and this
        never raises."""
        try:
            body = response.json()
        except ValueError:
            return False
        error = body.get("error") if isinstance(body, dict) else None
        if not isinstance(error, dict) or error.get("status") != "RESOURCE_EXHAUSTED":
            return False
        message = error.get("message")
        if not isinstance(message, str):
            return False
        lowered = message.lower()
        return any(marker in lowered for marker in _QUOTA_EXHAUSTED_MESSAGE_MARKERS)

    @staticmethod
    def _retry_delay_seconds(response: requests.Response, backoff_index: int) -> float:
        """The response's own `Retry-After` (clamped to `_MAX_RETRY_AFTER_SECONDS`)
        when present and valid, otherwise the fixed ~1s/~2s backoff."""
        retry_after = GeminiClient._parse_retry_after(response)
        if retry_after is not None:
            return min(retry_after, _MAX_RETRY_AFTER_SECONDS)
        return _BACKOFF_SECONDS[backoff_index]

    @staticmethod
    def _parse_retry_after(response: requests.Response) -> float | None:
        """Only the numeric-seconds form of `Retry-After` is supported (the
        HTTP-date form is not expected from this API) - anything else
        (missing, non-numeric, negative) safely yields `None`."""
        raw = response.headers.get("Retry-After")
        if raw is None:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return value if value >= 0 else None

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        """Translate a Gemini HTTP error status code into a Gemini-specific exception."""
        status_code = response.status_code

        if status_code in _AUTHENTICATION_ERROR_STATUS_CODES:
            logger.error("Gemini authentication failed with HTTP %s", status_code)
            raise GeminiAuthenticationError("Gemini authentication failed.")
        if status_code in _SERVICE_UNAVAILABLE_STATUS_CODES:
            logger.error("Gemini service returned HTTP %s", status_code)
            raise GeminiServiceUnavailableError(f"Gemini service returned HTTP {status_code}.")
        if not response.ok:
            logger.error("Gemini request failed with unexpected HTTP %s", status_code)
            raise GeminiClientError(f"Gemini request failed with HTTP {status_code}.")

    @staticmethod
    def _extract_text(body: Any) -> str:
        """Defensively pull the generated text out of a parsed Gemini response body.

        A 200 response is not, by itself, evidence of usable content (e.g.
        an empty `candidates` list, a safety-blocked response, or an
        unexpected body shape) - any structure that doesn't yield a
        non-empty string raises `GeminiEmptyResponseError` rather than
        returning `None`/`""`.

        Never assumes the answer is `parts[0]`: a `thought: true` part
        (Gemma-family models, when `thinkingConfig` is set - see
        `_payload_for_model`) carries internal reasoning, not the
        user-facing answer, and its text is always skipped regardless of
        position or content - this is the one place that content can never
        leak from, independent of which model produced it. The remaining
        non-thought parts' text is concatenated in order, which is also
        exactly what an ordinary Gemini response with a single plain text
        part reduces to.
        """
        try:
            candidates = body["candidates"]
            parts = candidates[0]["content"]["parts"]
            text_segments = [
                part["text"]
                for part in parts
                if isinstance(part, dict)
                and not part.get("thought")
                and isinstance(part.get("text"), str)
                and part["text"]
            ]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Gemini response contained no usable generated text.")
            raise GeminiEmptyResponseError("Gemini response contained no usable generated text.") from exc

        text = "".join(text_segments)

        if not text.strip():
            logger.error("Gemini response contained no usable generated text.")
            raise GeminiEmptyResponseError("Gemini response contained no usable generated text.")

        return text

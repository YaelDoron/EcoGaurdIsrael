"""Tests for Copernicus Data Space land-cover client."""
from __future__ import annotations

import math

import pytest
import requests

from src.external.copernicus import (
    CopernicusAuthenticationError,
    CopernicusClientError,
    CopernicusConfigurationError,
    CopernicusInvalidResponseError,
    CopernicusLandCoverClient,
    CopernicusServiceUnavailableError,
)
from src.external.copernicus.copernicus_land_cover_client import (
    COPERNICUS_LAND_COVER_TIME_FROM,
    COPERNICUS_LAND_COVER_TIME_TO,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, json_error: Exception | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


def token_payload(token="token-1", expires_in=3600):
    return {"access_token": token, "expires_in": expires_in}


def statistics_payload(**means):
    bands = {}
    ordered_categories = [
        "tree",
        "shrub",
        "grass",
        "crops",
        "bare",
        "built_up",
        "permanent_water",
        "seasonal_water",
        "moss_lichen",
        "snow",
    ]
    for index, category in enumerate(ordered_categories):
        bands[f"B{index}"] = {"stats": {"mean": means.get(category, 0.0)}}
    return {"data": [{"outputs": {"vegetation": {"bands": bands}}}]}


def make_client():
    return CopernicusLandCoverClient(
        client_id="client-id",
        client_secret="client-secret",
        token_url="https://token.example.test",
        statistics_url="https://stats.example.test",
        collection_id="collection-id",
        timeout=7,
    )


def test_obtains_oauth_token_using_client_credentials(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("token.example.test"):
            return FakeResponse(payload=token_payload())
        return FakeResponse(payload=statistics_payload(tree=0.5))

    monkeypatch.setattr(requests, "post", fake_post)

    make_client().get_land_cover_statistics(32.731, 35.046, 1.0)

    token_call = calls[0]
    assert token_call[0] == "https://token.example.test"
    assert token_call[1]["data"] == {
        "grant_type": "client_credentials",
        "client_id": "client-id",
        "client_secret": "client-secret",
    }
    assert token_call[1]["timeout"] == 7


def test_reuses_valid_cached_token(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if "token" in url:
            return FakeResponse(payload=token_payload())
        return FakeResponse(payload=statistics_payload(tree=0.5))

    monkeypatch.setattr(requests, "post", fake_post)
    client = make_client()

    # Different coordinates on the second call - the statistics results cache
    # (a separate performance pass) is keyed on coordinates, so identical
    # coordinates would short-circuit before ever reaching requests.post,
    # which would not exercise token reuse at all.
    client.get_land_cover_statistics(32.731, 35.046, 1.0)
    client.get_land_cover_statistics(33.0, 36.0, 1.0)

    assert [url for url, _ in calls].count("https://token.example.test") == 1
    assert [url for url, _ in calls].count("https://stats.example.test") == 2


def test_expired_token_is_refreshed(monkeypatch):
    calls = []
    tokens = iter(["token-1", "token-2"])

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if "token" in url:
            return FakeResponse(payload=token_payload(token=next(tokens), expires_in=1))
        return FakeResponse(payload=statistics_payload(tree=0.5))

    monkeypatch.setattr(requests, "post", fake_post)
    client = make_client()

    # Different coordinates on the second call - see test_reuses_valid_cached_token.
    client.get_land_cover_statistics(32.731, 35.046, 1.0)
    client.get_land_cover_statistics(33.0, 36.0, 1.0)

    assert [url for url, _ in calls].count("https://token.example.test") == 2


# --- performance pass: in-process statistics cache -------------------------


def test_identical_request_is_served_from_cache_without_a_second_statistics_call(monkeypatch):
    stats_calls = []

    def fake_post(url, **kwargs):
        if "token" in url:
            return FakeResponse(payload=token_payload())
        stats_calls.append((url, kwargs))
        return FakeResponse(payload=statistics_payload(tree=0.5))

    monkeypatch.setattr(requests, "post", fake_post)
    client = make_client()

    first = client.get_land_cover_statistics(32.731, 35.046, 1.0)
    second = client.get_land_cover_statistics(32.731, 35.046, 1.0)

    assert len(stats_calls) == 1
    assert second is first


def test_different_coordinates_are_not_served_from_cache(monkeypatch):
    stats_calls = []

    def fake_post(url, **kwargs):
        if "token" in url:
            return FakeResponse(payload=token_payload())
        stats_calls.append((url, kwargs))
        return FakeResponse(payload=statistics_payload(tree=0.5))

    monkeypatch.setattr(requests, "post", fake_post)
    client = make_client()

    client.get_land_cover_statistics(32.731, 35.046, 1.0)
    client.get_land_cover_statistics(32.732, 35.046, 1.0)  # latitude changed
    client.get_land_cover_statistics(32.731, 35.047, 1.0)  # longitude changed
    client.get_land_cover_statistics(32.731, 35.046, 2.0)  # radius changed

    assert len(stats_calls) == 4


def test_none_result_is_also_cached(monkeypatch):
    """A location with no Copernicus data is itself a deterministic result
    for the same fixed dataset - it must not be re-queried on every call."""
    stats_calls = []

    def fake_post(url, **kwargs):
        if "token" in url:
            return FakeResponse(payload=token_payload())
        stats_calls.append((url, kwargs))
        return FakeResponse(payload={"data": []})

    monkeypatch.setattr(requests, "post", fake_post)
    client = make_client()

    first = client.get_land_cover_statistics(32.731, 35.046, 1.0)
    second = client.get_land_cover_statistics(32.731, 35.046, 1.0)

    assert first is None
    assert second is None
    assert len(stats_calls) == 1


def test_a_failed_request_is_not_cached_and_the_next_call_retries(monkeypatch):
    stats_calls = []

    def fake_post(url, **kwargs):
        if "token" in url:
            return FakeResponse(payload=token_payload())
        stats_calls.append((url, kwargs))
        if len(stats_calls) == 1:
            return FakeResponse(status_code=503)
        return FakeResponse(payload=statistics_payload(tree=0.5))

    monkeypatch.setattr(requests, "post", fake_post)
    client = make_client()

    with pytest.raises(CopernicusServiceUnavailableError):
        client.get_land_cover_statistics(32.731, 35.046, 1.0)

    result = client.get_land_cover_statistics(32.731, 35.046, 1.0)

    assert result is not None
    assert len(stats_calls) == 2


def test_statistics_api_called_with_bearer_token_collection_geometry_radius_and_2019_range(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if "token" in url:
            return FakeResponse(payload=token_payload(token="secret-token"))
        return FakeResponse(payload=statistics_payload(tree=0.5))

    monkeypatch.setattr(requests, "post", fake_post)

    make_client().get_land_cover_statistics(32.731, 35.046, 1.0)

    url, kwargs = calls[1]
    body = kwargs["json"]
    assert url == "https://stats.example.test"
    assert kwargs["headers"]["Authorization"].startswith("Bearer ")
    assert kwargs["headers"]["Authorization"].endswith("secret-token")
    assert body["input"]["data"][0]["type"] == "byoc-collection-id"
    assert body["input"]["data"][0]["dataFilter"]["timeRange"] == {
        "from": COPERNICUS_LAND_COVER_TIME_FROM,
        "to": COPERNICUS_LAND_COVER_TIME_TO,
    }
    assert body["aggregation"]["timeRange"] == {
        "from": COPERNICUS_LAND_COVER_TIME_FROM,
        "to": COPERNICUS_LAND_COVER_TIME_TO,
    }
    coordinates = body["input"]["bounds"]["geometry"]["coordinates"][0]
    longitudes = [point[0] for point in coordinates]
    latitudes = [point[1] for point in coordinates]
    assert min(longitudes) < 35.046 < max(longitudes)
    assert min(latitudes) < 32.731 < max(latitudes)
    assert (max(latitudes) - min(latitudes)) * 111.2 == pytest.approx(2.0, rel=0.02)


def test_successful_statistics_response_is_parsed(monkeypatch):
    def fake_post(url, **kwargs):
        if "token" in url:
            return FakeResponse(payload=token_payload())
        return FakeResponse(payload=statistics_payload(tree=0.6, grass=0.2, built_up=0.1))

    monkeypatch.setattr(requests, "post", fake_post)

    result = make_client().get_land_cover_statistics(32.731, 35.046, 1.0)

    assert result is not None
    assert result.dataset_year == 2019
    assert result.radius_km == pytest.approx(1.0)
    assert {item.category: item.fraction for item in result.cover_fractions} == {
        "tree": 0.6,
        "grass": 0.2,
        "built_up": 0.1,
    }


def test_missing_credentials_raise_configuration_error():
    client = CopernicusLandCoverClient(client_id="", client_secret="", token_url="t", statistics_url="s")

    with pytest.raises(CopernicusConfigurationError):
        client.get_land_cover_statistics(32.731, 35.046, 1.0)


def test_token_authentication_failure_handled_without_secret_in_error(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse(status_code=401))

    with pytest.raises(CopernicusAuthenticationError) as exc_info:
        make_client().get_land_cover_statistics(32.731, 35.046, 1.0)

    assert "client-secret" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (400, CopernicusClientError),
        (404, CopernicusClientError),
        (500, CopernicusServiceUnavailableError),
        (503, CopernicusServiceUnavailableError),
    ],
)
def test_statistics_http_errors_are_handled(monkeypatch, status_code, expected_error):
    def fake_post(url, **kwargs):
        if "token" in url:
            return FakeResponse(payload=token_payload())
        return FakeResponse(status_code=status_code)

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(expected_error):
        make_client().get_land_cover_statistics(32.731, 35.046, 1.0)


@pytest.mark.parametrize("exception", [requests.exceptions.Timeout(), requests.exceptions.ConnectionError()])
def test_statistics_timeout_and_connection_errors_are_handled(monkeypatch, exception):
    def fake_post(url, **kwargs):
        if "token" in url:
            return FakeResponse(payload=token_payload())
        raise exception

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(CopernicusServiceUnavailableError):
        make_client().get_land_cover_statistics(32.731, 35.046, 1.0)


def test_malformed_statistics_json_is_handled(monkeypatch):
    def fake_post(url, **kwargs):
        if "token" in url:
            return FakeResponse(payload=token_payload())
        return FakeResponse(json_error=ValueError("bad json"))

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(CopernicusInvalidResponseError):
        make_client().get_land_cover_statistics(32.731, 35.046, 1.0)


@pytest.mark.parametrize("payload", [{"data": []}, statistics_payload()])
def test_empty_or_no_data_response_returns_none(monkeypatch, payload):
    def fake_post(url, **kwargs):
        if "token" in url:
            return FakeResponse(payload=token_payload())
        return FakeResponse(payload=payload)

    monkeypatch.setattr(requests, "post", fake_post)

    assert make_client().get_land_cover_statistics(32.731, 35.046, 1.0) is None


def test_malformed_statistics_shape_is_handled(monkeypatch):
    def fake_post(url, **kwargs):
        if "token" in url:
            return FakeResponse(payload=token_payload())
        return FakeResponse(payload={"unexpected": []})

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(CopernicusInvalidResponseError):
        make_client().get_land_cover_statistics(32.731, 35.046, 1.0)

"""KISClient의 환경변수 기반 캐시 토큰 초기화 회귀 테스트."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

import kis_agent.core.client as client_module
from kis_agent.core.client import KISClient


def test_cached_token_without_config_reapplies_environment_auth():
    client = object.__new__(KISClient)
    client.token_refresh_lock = threading.Lock()
    client.token = None
    client.token_expired = None
    client.config = None
    client.svr = "prod"

    cached = {
        "access_token": "cached",
        "access_token_token_expired": "2099-01-01 00:00:00",
    }
    with patch("kis_agent.core.client.os.getenv", return_value="key"), patch(
        "kis_agent.core.client.read_token", return_value=cached
    ), patch("kis_agent.core.client.auth") as auth, patch(
        "kis_agent.core.client.resolve_environment",
        return_value=("https://example.test", "prod"),
    ):
        client._initialize_token()

    auth.assert_called_once_with(svr="prod")
    assert client.token == "cached"
    assert client.base_url == "https://example.test"


def test_cached_token_with_explicit_config_reapplies_config_auth():
    client = object.__new__(KISClient)
    client.token_refresh_lock = threading.Lock()
    client.token = None
    client.token_expired = None
    client.config = SimpleNamespace(APP_KEY="key", BASE_URL="https://config.test")
    client.svr = "prod"
    cached = {
        "access_token": "cached",
        "access_token_token_expired": "2099-01-01 00:00:00",
    }

    with patch("kis_agent.core.client.read_token", return_value=cached), patch(
        "kis_agent.core.client.auth"
    ) as auth:
        client._initialize_token()

    auth.assert_called_once_with(config=client.config, svr="prod")
    assert client.base_url == "https://config.test"


def _request_client():
    client = object.__new__(KISClient)
    client.base_url = "https://example.test"
    client.is_real = True
    client.verbose = False
    client.enable_rate_limiter = False
    client.rate_limiter = None
    client._check_and_refresh_token = MagicMock()
    client._enforce_rate_limit = MagicMock()
    return client


def _response(status_code, payload):
    response = MagicMock(status_code=status_code, text="error", headers={})
    response.json.return_value = payload
    return response


def test_http_error_and_request_exception_retry_then_succeed():
    env = SimpleNamespace(my_token="token", my_app="app", my_sec="secret")
    success = _response(200, {"rt_cd": "0"})
    http_error = _response(500, {"rt_cd": "500", "msg1": "server error"})
    request = httpx.Request("GET", "https://example.test/test")

    with patch.object(client_module, "getTREnv", return_value=env), patch.object(
        client_module.time, "sleep"
    ) as sleep, patch.object(
        client_module.httpx, "request", side_effect=[http_error, success]
    ):
        assert _request_client().make_request("/test", "TR", {}, retries=2) == {
            "rt_cd": "0"
        }
    sleep.assert_called_once_with(0.2)

    with patch.object(client_module, "getTREnv", return_value=env), patch.object(
        client_module.time, "sleep"
    ) as sleep, patch.object(
        client_module.httpx,
        "request",
        side_effect=[httpx.RequestError("offline", request=request), success],
    ):
        assert _request_client().make_request("/test", "TR", {}, retries=2) == {
            "rt_cd": "0"
        }
    sleep.assert_called_once_with(0.2)


def test_zero_retries_rejects_request_without_network_call():
    env = SimpleNamespace(my_token="token", my_app="app", my_sec="secret")
    with patch.object(client_module, "getTREnv", return_value=env), patch.object(
        client_module.httpx, "request"
    ) as request, pytest.raises(Exception, match="Unknown error after retries"):
        _request_client().make_request("/test", "TR", {}, retries=0)
    request.assert_not_called()


def test_auth_headers_tolerate_missing_token_and_config_attrs():
    """Lightweight clients without token/config must fall back to getTREnv()."""
    env = SimpleNamespace(
        my_token="Bearer env-token", my_app="env-app", my_sec="env-sec"
    )
    client = _request_client()  # no token / config attributes

    with patch.object(client_module, "getTREnv", return_value=env):
        headers = client._auth_headers()

    assert headers == {
        "authorization": "Bearer env-token",
        "appkey": "env-app",
        "appsecret": "env-sec",
    }


def test_auth_headers_prefer_per_client_credentials_over_global_env():
    """Two clients must keep distinct token/app key/secret despite shared getTREnv()."""
    leaked = SimpleNamespace(
        my_token="Bearer leaked-last-agent",
        my_app="leaked-app",
        my_sec="leaked-sec",
    )
    client_a = _request_client()
    client_a.token = "token-a"
    client_a.config = SimpleNamespace(APP_KEY="key-a", APP_SECRET="sec-a")
    client_b = _request_client()
    client_b.token = "token-b"
    client_b.config = SimpleNamespace(APP_KEY="key-b", APP_SECRET="sec-b")

    with patch.object(client_module, "getTREnv", return_value=leaked):
        headers_a = client_a._auth_headers()
        headers_b = client_b._auth_headers()
        # Token refresh path must also re-bind to the same client credentials.
        client_a.token = "token-a-refreshed"
        headers_a_refreshed = client_a._auth_headers()

    assert headers_a == {
        "authorization": "Bearer token-a",
        "appkey": "key-a",
        "appsecret": "sec-a",
    }
    assert headers_b == {
        "authorization": "Bearer token-b",
        "appkey": "key-b",
        "appsecret": "sec-b",
    }
    assert headers_a_refreshed == {
        "authorization": "Bearer token-a-refreshed",
        "appkey": "key-a",
        "appsecret": "sec-a",
    }
    assert headers_a["appkey"] != headers_b["appkey"]
    assert headers_a["authorization"] != leaked.my_token

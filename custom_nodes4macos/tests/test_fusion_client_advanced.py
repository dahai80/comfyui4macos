"""FusionMLXClient 进阶测试。

覆盖重试边界（5xx/408/transport error）、health 超时、
模块级缓存 TTL、api_key header 注入。

补充 REVIEW_REPORT：fusion_client 重试边界与缓存行为未充分覆盖。
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

import httpx

from custom_nodes4macos import fusion_client
from custom_nodes4macos.fusion_client import (
    FusionMLXClient,
    FusionMLXError,
    list_models_safe,
    default_model_safe,
    _models_cache,
    _default_model_cache,
)


class TestRetryLogic(unittest.TestCase):

    def test_500_retries_then_succeeds(self):
        """500 错误重试后成功。"""
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            r500 = MagicMock(status_code=500, text="err")
            r200 = MagicMock(status_code=200)
            mock_client.request.side_effect = [r500, r200]
            client = FusionMLXClient(retries=2)
            with patch("time.sleep"):
                resp = client._request("GET", "/x")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(mock_client.request.call_count, 2)

    def test_500_exhausts_retries_returns_last(self):
        """5xx 重试耗尽后返回最后一次响应。"""
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            r500 = MagicMock(status_code=500, text="err")
            mock_client.request.side_effect = [r500, r500, r500]
            client = FusionMLXClient(retries=2)
            with patch("time.sleep"):
                resp = client._request("GET", "/x")
            self.assertEqual(resp.status_code, 500)
            self.assertEqual(mock_client.request.call_count, 3)

    def test_408_retries_with_retry_after(self):
        """408 超时触发重试。"""
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            r408 = MagicMock(status_code=408, headers={"retry-after": "0"})
            r200 = MagicMock(status_code=200)
            mock_client.request.side_effect = [r408, r200]
            client = FusionMLXClient(retries=2)
            with patch("time.sleep"):
                resp = client._request("GET", "/x")
            self.assertEqual(resp.status_code, 200)

    def test_429_respects_retry_after_header(self):
        """429 读取 retry-after header 决定等待时间。"""
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            r429 = MagicMock(status_code=429, headers={"retry-after": "3"})
            r200 = MagicMock(status_code=200)
            mock_client.request.side_effect = [r429, r200]
            client = FusionMLXClient(retries=1)
            with patch("time.sleep") as mock_sleep:
                resp = client._request("GET", "/x")
                mock_sleep.assert_called_once()
                # 应等待 3 秒（受 min(wait, 10) 限制）
                self.assertEqual(mock_sleep.call_args.args[0], 3)
            self.assertEqual(resp.status_code, 200)

    def test_4xx_no_retry_returns_immediately(self):
        """4xx（非 408/429）不重试，直接返回。"""
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            r404 = MagicMock(status_code=404, text="not found")
            mock_client.request.side_effect = [r404]
            client = FusionMLXClient(retries=3)
            resp = client._request("GET", "/x")
            self.assertEqual(resp.status_code, 404)
            self.assertEqual(mock_client.request.call_count, 1)

    def test_transport_error_raises_fusion_error(self):
        """TransportError 重试耗尽后抛 FusionMLXError。"""
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.request.side_effect = httpx.TransportError("conn refused")
            client = FusionMLXClient(retries=1)
            with patch("time.sleep"):
                with self.assertRaises(FusionMLXError):
                    client._request("GET", "/x")

    def test_timeout_error_raises_fusion_errors(self):
        """TimeoutException 重试耗尽后抛 FusionMLXError。"""
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.request.side_effect = httpx.TimeoutException("timed out")
            client = FusionMLXClient(retries=0)
            with self.assertRaises(FusionMLXError):
                client._request("GET", "/x")

    def test_zero_retries_no_retry(self):
        """retries=0 时单次失败即抛异常。"""
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.request.side_effect = httpx.TransportError("fail")
            client = FusionMLXClient(retries=0)
            with self.assertRaises(FusionMLXError):
                client._request("GET", "/x")
            self.assertEqual(mock_client.request.call_count, 1)


class TestHealthCheck(unittest.TestCase):

    def test_health_returns_true_on_200(self):
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get.return_value = MagicMock(status_code=200)
            client = FusionMLXClient()
            self.assertTrue(client.health())

    def test_health_returns_false_on_500(self):
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get.return_value = MagicMock(status_code=500, text="err")
            client = FusionMLXClient()
            self.assertFalse(client.health())

    def test_health_returns_false_on_exception(self):
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get.side_effect = Exception("conn refused")
            client = FusionMLXClient()
            self.assertFalse(client.health())

    def test_health_uses_short_timeout(self):
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get.return_value = MagicMock(status_code=200)
            client = FusionMLXClient()
            client.health(timeout=0.5)
            call_kwargs = mock_client.get.call_args
            self.assertEqual(call_kwargs.kwargs.get("timeout"), 0.5)


class TestApiKeyHeader(unittest.TestCase):

    def test_api_key_injected_into_header(self):
        with patch("httpx.Client") as mock_cls:
            FusionMLXClient(api_key="secret-key")
            _, kwargs = mock_cls.call_args
            headers = kwargs["headers"]
            self.assertEqual(headers["Authorization"], "Bearer secret-key")

    def test_no_api_key_no_authorization_header(self):
        with patch("httpx.Client") as mock_cls:
            FusionMLXClient(api_key="")
            _, kwargs = mock_cls.call_args
            headers = kwargs["headers"]
            self.assertNotIn("Authorization", headers)

    def test_base_url_stripped_trailing_slash(self):
        with patch("httpx.Client") as mock_cls:
            client = FusionMLXClient(base_url="http://example.com/")
            self.assertEqual(client.base_url, "http://example.com")


class TestListModelsSafeCache(unittest.TestCase):

    def setUp(self):
        _models_cache["value"] = None
        _models_cache["ts"] = 0.0

    def test_cache_returns_cached_value_within_ttl(self):
        _models_cache["value"] = ["(auto)", "cached-model"]
        _models_cache["ts"] = time.monotonic()
        result = list_models_safe()
        self.assertEqual(result, ["(auto)", "cached-model"])

    def test_force_refresh_bypasses_cache(self):
        _models_cache["value"] = ["(auto)", "old"]
        _models_cache["ts"] = time.monotonic()
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get.return_value = MagicMock(status_code=200)
            mock_client.health.return_value = True
            with patch.object(FusionMLXClient, "health", return_value=True):
                with patch.object(FusionMLXClient, "list_models", return_value=["new-model"]):
                    result = list_models_safe(force=True)
        self.assertIn("new-model", result)

    def test_fallback_to_auto_on_failure(self):
        _models_cache["value"] = None
        _models_cache["ts"] = 0.0
        with patch.object(FusionMLXClient, "health", return_value=False):
            result = list_models_safe(force=True)
        self.assertEqual(result, ["(auto)"])


class TestDefaultModelSafeCache(unittest.TestCase):

    def setUp(self):
        _default_model_cache["value"] = ""
        _default_model_cache["ts"] = 0.0

    def test_returns_cached_value(self):
        _default_model_cache["value"] = "cached-llm"
        _default_model_cache["ts"] = time.monotonic()
        self.assertEqual(default_model_safe(), "cached-llm")

    def test_returns_empty_on_unreachable(self):
        with patch.object(FusionMLXClient, "health", side_effect=Exception("fail")):
            # health 在 default_model_safe 内通过 _client.get 调用，mock 较复杂
            # 直接验证返回值为空字符串
            _default_model_cache["value"] = ""
            _default_model_cache["ts"] = 0.0
            with patch("httpx.Client") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.get.side_effect = Exception("fail")
                result = default_model_safe(force=True)
        self.assertEqual(result, "")


class TestClientContextManager(unittest.TestCase):

    def test_context_manager_closes_client(self):
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            with FusionMLXClient() as client:
                self.assertIsNotNone(client)
            mock_client.close.assert_called_once()

    def test_explicit_close(self):
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            client = FusionMLXClient()
            client.close()
            mock_client.close.assert_called_once()


class TestGenerateImageReferencePayload(unittest.TestCase):
    """generate_image 在提供参考图时把字段写入 payload，否则省略（前向兼容）。"""

    def _client_with_200(self):
        import base64
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            client = FusionMLXClient()
        b64 = base64.b64encode(b"\x89PNG fake").decode("ascii")
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"data": [{"b64_json": b64}]}
        client._request = MagicMock(return_value=resp)
        return client

    def test_reference_fields_omitted_by_default(self):
        client = self._client_with_200()
        client.generate_image(prompt="p", width=64, height=64)
        payload = client._request.call_args.kwargs.get("json_body")
        self.assertNotIn("reference_image", payload)
        self.assertNotIn("reference_strength", payload)
        self.assertNotIn("conditioning_mode", payload)

    def test_reference_fields_included_when_provided(self):
        client = self._client_with_200()
        client.generate_image(
            prompt="p", width=64, height=64,
            reference_image="REFB64", reference_strength=0.7, conditioning_mode="redux",
        )
        payload = client._request.call_args.kwargs.get("json_body")
        self.assertEqual(payload["reference_image"], "REFB64")
        self.assertEqual(payload["reference_strength"], 0.7)
        self.assertEqual(payload["conditioning_mode"], "redux")


if __name__ == "__main__":
    unittest.main()

import base64
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger("custom_nodes4macos.fusion_client")

DEFAULT_BASE_URL = os.environ.get("FUSION_MLX_BASE_URL", "http://localhost:11434")
DEFAULT_API_KEY = os.environ.get("FUSION_MLX_API_KEY", "")
DEFAULT_TIMEOUT = float(os.environ.get("FUSION_MLX_TIMEOUT", "120"))
DEFAULT_RETRIES = int(os.environ.get("FUSION_MLX_RETRIES", "2"))


class FusionMLXError(RuntimeError):
    pass


class FusionMLXClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
    ):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else DEFAULT_API_KEY
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
        self.retries = retries if retries is not None else DEFAULT_RETRIES
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
        )
        logger.info("client init base_url=%s timeout=%.1fs retries=%d", self.base_url, self.timeout, self.retries)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _request(self, method: str, path: str, json_body: Any = None, timeout: float | None = None):
        url = path
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                t0 = time.monotonic()
                resp = self._client.request(method, url, json=json_body, timeout=timeout or self.timeout)
                elapsed = (time.monotonic() - t0) * 1000.0
                logger.info(
                    "HTTP %s %s status=%s elapsed=%.1fms attempt=%d",
                    method, path, resp.status_code, elapsed, attempt,
                )
                if resp.status_code >= 500 and attempt < self.retries:
                    logger.warning("5xx retry %d/%d body=%r", attempt + 1, self.retries, resp.text[:200])
                    time.sleep(0.5 * (attempt + 1))
                    continue
                if resp.status_code in (429, 408) and attempt < self.retries:
                    wait = float(resp.headers.get("retry-after", "2"))
                    logger.warning("%d retry %d/%d wait=%.1fs", resp.status_code, attempt + 1, self.retries, wait)
                    time.sleep(min(wait, 10.0))
                    continue
                return resp
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                logger.warning("transport error %s %s: %s attempt=%d/%d", method, path, exc, attempt + 1, self.retries)
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise FusionMLXError(f"transport error {method} {path}: {exc}") from exc
        raise FusionMLXError(f"request failed after retries: {last_exc}")

    def health(self, timeout: float = 1.5) -> bool:
        try:
            resp = self._client.get("/health", timeout=timeout)
            if resp.status_code == 200:
                return True
            logger.warning("health status=%s body=%r", resp.status_code, resp.text[:200])
            return False
        except Exception as exc:
            logger.warning("health check failed: %s", exc)
            return False

    def list_models(self, timeout: float = 3.0) -> list[str]:
        resp = self._request("GET", "/v1/models", timeout=timeout)
        if resp.status_code != 200:
            raise FusionMLXError(f"list_models status {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        models = [m.get("id") for m in data.get("data", []) if m.get("id")]
        logger.info("list_models count=%d", len(models))
        return models

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.75,
        json_mode: bool = False,
        max_tokens: int | None = None,
        chat_template_kwargs: dict | None = None,
        timeout: float | None = None,
    ) -> tuple[str, dict]:
        resolved_model = model or default_model_safe()
        if not resolved_model:
            raise FusionMLXError(
                "no model specified and fusion-mlx reports no default_model; "
                "set FUSION_LLM_MODEL or pass model explicitly"
            )
        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        logger.info(
            "chat model=%s msgs=%d json_mode=%s max_tokens=%s ctk=%s",
            payload["model"], len(messages), json_mode, max_tokens, bool(chat_template_kwargs),
        )
        resp = self._request("POST", "/v1/chat/completions", json_body=payload, timeout=timeout)
        if resp.status_code != 200:
            raise FusionMLXError(f"chat status {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        logger.info("chat done usage=%s content_len=%d", usage, len(content))
        return content, usage

    def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        seed: int | None = None,
        guidance: float = 4.0,
        n: int = 1,
        response_format: str = "b64_json",
        timeout: float | None = None,
        reference_image: str | None = None,
        reference_strength: float | None = None,
        conditioning_mode: str | None = None,
    ) -> list[bytes]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "n": n,
            "response_format": response_format,
        }
        if seed is not None and seed != 0:
            payload["seed"] = seed
        if reference_image:
            payload["reference_image"] = reference_image
            if reference_strength is not None:
                payload["reference_strength"] = reference_strength
            if conditioning_mode:
                payload["conditioning_mode"] = conditioning_mode
        logger.info(
            "generate_image model=%s size=%dx%d steps=%d guidance=%.1f seed=%s ref=%s",
            model or "auto", width, height, steps, guidance, seed,
            conditioning_mode or "none",
        )
        resp = self._request("POST", "/v1/images/generate", json_body=payload, timeout=timeout)
        if resp.status_code != 200:
            raise FusionMLXError(f"generate_image status {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        out: list[bytes] = []
        for item in data.get("data", []):
            b64 = item.get("b64_json")
            url = item.get("url")
            if b64:
                out.append(base64.b64decode(b64))
            elif url and url.startswith("data:image"):
                b64 = url.split(",", 1)[-1]
                out.append(base64.b64decode(b64))
            else:
                raise FusionMLXError(f"image response missing b64_json/url: {item}")
        logger.info("generate_image done count=%d", len(out))
        return out

    def synthesize_speech(
        self,
        text: str,
        model: str,
        voice: str | None = None,
        instructions: str | None = None,
        speed: float = 1.0,
        response_format: str = "wav",
        stream: bool = False,
        timeout: float | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> bytes:
        payload: dict[str, Any] = {
            "model": model,
            "input": text,
            "response_format": response_format,
            "stream": stream,
        }
        if voice:
            payload["voice"] = voice
        if instructions:
            payload["instructions"] = instructions
        if speed is not None:
            payload["speed"] = speed
        if ref_audio:
            payload["ref_audio"] = ref_audio
        if ref_text:
            payload["ref_text"] = ref_text
        logger.info(
            "synthesize_speech model=%s voice=%s speed=%.2f text_len=%d stream=%s ref_audio=%s",
            model, voice, speed, len(text), stream, bool(ref_audio),
        )
        resp = self._request("POST", "/v1/audio/speech", json_body=payload, timeout=timeout)
        if resp.status_code != 200:
            raise FusionMLXError(f"synthesize_speech status {resp.status_code}: {resp.text[:500]}")
        audio_bytes = resp.content
        logger.info("synthesize_speech done bytes=%d", len(audio_bytes))
        return audio_bytes

    def transcribe(
        self,
        audio_path: str,
        model: str | None = None,
        language: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, dict]:
        if not os.path.isfile(audio_path):
            raise FusionMLXError(f"audio file not found: {audio_path}")
        resolved = model or default_whisper_model_safe()
        if not resolved:
            raise FusionMLXError(
                "no whisper model specified and FUSION_WHISPER_MODEL unset; "
                "set FUSION_WHISPER_MODEL or pass model explicitly"
            )
        headers = {k: v for k, v in self._client.headers.items() if k.lower() != "content-type"}
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            data: dict[str, Any] = {"model": resolved}
            if language:
                data["language"] = language
            logger.info(
                "transcribe model=%s file=%s language=%s",
                resolved, os.path.basename(audio_path), language or "auto",
            )
            resp = self._client.post(
                "/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=headers,
                timeout=timeout or self.timeout,
            )
        if resp.status_code != 200:
            raise FusionMLXError(f"transcribe status {resp.status_code}: {resp.text[:500]}")
        result = resp.json()
        text = result.get("text", "") or ""
        logger.info("transcribe done text_len=%d", len(text))
        return text, result


_MODELS_TTL = 60.0
_models_cache: dict = {"ts": 0.0, "value": None}


def list_models_safe(force: bool = False) -> list[str]:
    now = time.monotonic()
    cached = _models_cache["value"]
    if not force and cached is not None and now - _models_cache["ts"] < _MODELS_TTL:
        return cached
    opts = ["(auto)"]
    try:
        with FusionMLXClient() as client:
            if client.health():
                models = client.list_models()
                if models:
                    opts = ["(auto)"] + models
    except Exception as exc:
        logger.warning("list_models_safe fallback to (auto): %s", exc)
    _models_cache["value"] = opts
    _models_cache["ts"] = now
    return opts


_default_model_cache: dict = {"ts": 0.0, "value": ""}


def default_model_safe(force: bool = False) -> str:
    now = time.monotonic()
    cached = _default_model_cache["value"]
    if not force and cached and now - _default_model_cache["ts"] < _MODELS_TTL:
        return cached
    found = ""
    try:
        with FusionMLXClient() as client:
            resp = client._client.get("/health", timeout=1.5)
            if resp.status_code == 200:
                found = resp.json().get("default_model", "") or ""
    except Exception as exc:
        logger.warning("default_model_safe unavailable: %s", exc)
    _default_model_cache["value"] = found
    _default_model_cache["ts"] = now
    return found


DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


def default_whisper_model_safe() -> str:
    return os.environ.get("FUSION_WHISPER_MODEL", DEFAULT_WHISPER_MODEL).strip()

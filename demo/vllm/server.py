"""FastAPI app for OpenAI-compatible vLLM gateway + live streaming UI."""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from demo.vllm.config import CONFIG


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="OpenAI-Compatible vLLM Gateway Demo")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class LiveStreamHub:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1024)
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    async def publish(self, event: str, payload: dict[str, Any]) -> None:
        message = _sse(event, payload)
        async with self._lock:
            subscribers = list(self._subscribers)
        stale: list[asyncio.Queue[str]] = []
        for queue in subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                stale.append(queue)
        if stale:
            async with self._lock:
                for queue in stale:
                    self._subscribers.discard(queue)


HUB = LiveStreamHub()
_MODEL_ID_CACHE: str | None = None


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(CONFIG.request_timeout_s, connect=CONFIG.connect_timeout_s)


def _passthrough_headers(request: Request) -> dict[str, str]:
    allowed = {"authorization", "x-request-id", "openai-organization"}
    headers = {k: v for k, v in request.headers.items() if k.lower() in allowed}
    return headers


async def _resolve_backend_model(client: httpx.AsyncClient, headers: dict[str, str]) -> str:
    global _MODEL_ID_CACHE
    url = f"{CONFIG.backend_base_url.rstrip('/')}/v1/models"
    for _ in range(3):
        try:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") or []
            if data and isinstance(data[0], dict) and isinstance(data[0].get("id"), str):
                model_id = str(data[0]["id"])
                _MODEL_ID_CACHE = model_id
                return model_id
        except Exception:  # noqa: BLE001
            await asyncio.sleep(0.2)

    if _MODEL_ID_CACHE:
        return _MODEL_ID_CACHE
    raise RuntimeError("No model id returned by backend /v1/models")


async def _check_health(base_url: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/health"
    async with httpx.AsyncClient(timeout=_timeout()) as client:
        try:
            resp = await client.get(url)
            return {
                "ok": resp.status_code == 200,
                "status_code": resp.status_code,
                "url": url,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "status_code": None,
                "url": url,
                "error": str(exc),
            }


_PROM_LINE_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)$")


def _parse_prometheus_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    labels: dict[str, str] = {}
    for item in raw.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        labels[key.strip()] = value.strip().strip('"')
    return labels


def _extract_metric(text: str, name: str) -> tuple[float | None, dict[str, str]]:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith(name):
            continue
        match = _PROM_LINE_RE.match(line)
        if not match:
            continue
        metric_name = match.group("name")
        if metric_name != name:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        labels = _parse_prometheus_labels(match.group("labels"))
        return value, labels
    return None, {}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/healthz")
async def healthz() -> JSONResponse:
    backend = await _check_health(CONFIG.backend_base_url)
    models_url = f"{CONFIG.backend_base_url.rstrip('/')}/v1/models"
    async with httpx.AsyncClient(timeout=_timeout()) as client:
        try:
            models_resp = await client.get(models_url)
            models_ok = models_resp.status_code == 200
            model_count = len((models_resp.json().get("data") or [])) if models_ok else None
        except Exception as exc:  # noqa: BLE001
            models_ok = False
            model_count = None
            models_error = str(exc)
        else:
            models_error = None

    ok = backend.get("ok") and models_ok
    payload: dict[str, Any] = {
        "ok": bool(ok),
        "gateway": {"ok": True, "host": CONFIG.host, "port": CONFIG.port},
        "backend": backend,
        "models": {
            "ok": models_ok,
            "url": models_url,
            "count": model_count,
        },
    }
    if models_error:
        payload["models"]["error"] = models_error
    return JSONResponse(status_code=200 if ok else 503, content=payload)


@app.get("/v1/models")
async def list_models(request: Request) -> Response:
    url = f"{CONFIG.backend_base_url.rstrip('/')}/v1/models"
    async with httpx.AsyncClient(timeout=_timeout()) as client:
        resp = await client.get(url, headers=_passthrough_headers(request))
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )


@app.get("/api/kv-cache")
async def kv_cache_metrics() -> JSONResponse:
    metrics_url = f"{CONFIG.backend_base_url.rstrip('/')}/metrics"
    timeout = httpx.Timeout(min(CONFIG.request_timeout_s, 5.0), connect=CONFIG.connect_timeout_s)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(metrics_url)
            resp.raise_for_status()
            text = resp.text
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                status_code=503,
                content={
                    "ok": False,
                    "error": str(exc),
                    "metrics_url": metrics_url,
                },
            )

    usage_raw, usage_labels = _extract_metric(text, "vllm:kv_cache_usage_perc")
    if usage_raw is None:
        usage_raw, usage_labels = _extract_metric(text, "vllm_kv_cache_usage_perc")

    cfg_raw, cfg_labels = _extract_metric(text, "vllm:cache_config_info")
    if cfg_raw is None:
        cfg_raw, cfg_labels = _extract_metric(text, "vllm_cache_config_info")

    labels = usage_labels if usage_labels else cfg_labels
    model_name = labels.get("model_name")

    num_gpu_blocks: int | None = None
    block_size_tokens: int | None = None
    if cfg_raw is not None:
        try:
            num_gpu_blocks = int(float(cfg_labels.get("num_gpu_blocks", "nan")))
        except Exception:  # noqa: BLE001
            num_gpu_blocks = None
        try:
            block_size_tokens = int(float(cfg_labels.get("block_size", "nan")))
        except Exception:  # noqa: BLE001
            block_size_tokens = None

    capacity_tokens: float | None = None
    if num_gpu_blocks is not None and block_size_tokens is not None:
        capacity_tokens = float(num_gpu_blocks * block_size_tokens)

    usage_fraction: float | None = None
    usage_percent: float | None = None
    if usage_raw is not None:
        usage_fraction = float(usage_raw)
        usage_percent = usage_fraction * 100.0

    used_tokens_estimate: float | None = None
    if usage_fraction is not None and capacity_tokens is not None:
        used_tokens_estimate = usage_fraction * capacity_tokens

    return JSONResponse(
        content={
            "ok": usage_fraction is not None,
            "metrics_url": metrics_url,
            "model_name": model_name,
            "usage_fraction": usage_fraction,
            "usage_percent": usage_percent,
            "num_gpu_blocks": num_gpu_blocks,
            "block_size_tokens": block_size_tokens,
            "capacity_tokens_estimate": capacity_tokens,
            "used_tokens_estimate": used_tokens_estimate,
            "updated_at_ms": int(time.time() * 1000),
        }
    )


@app.get("/api/live/stream")
async def live_stream() -> StreamingResponse:
    queue = await HUB.subscribe()

    async def event_generator() -> AsyncIterator[str]:
        try:
            yield _sse("connected", {"ok": True})
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield message
        finally:
            await HUB.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/completions")
async def completions_proxy(request: Request) -> Response:
    """Transparent proxy for /v1/completions (used by openclaw openai-completions API)."""
    payload = await request.json()
    # Dump first request prompt for debugging
    import logging as _clog
    _prompt_text = payload.get("prompt", "")
    _clog.getLogger("demo.completions_proxy").info(
        "completions request: prompt_len=%d max_tokens=%s stream=%s prompt_head=%.500s",
        len(_prompt_text) if isinstance(_prompt_text, str) else 0,
        payload.get("max_tokens"),
        payload.get("stream"),
        _prompt_text[:500] if isinstance(_prompt_text, str) else str(_prompt_text)[:500],
    )
    # Save full prompt to file for manual replay
    try:
        import pathlib, time as _t
        _dump_dir = pathlib.Path("/tmp/openclaw_prompts")
        _dump_dir.mkdir(exist_ok=True)
        _dump_file = _dump_dir / f"prompt_{int(_t.time())}.json"
        with open(_dump_file, "w") as _f:
            json.dump(payload, _f, ensure_ascii=False)
        _clog.getLogger("demo.completions_proxy").info("Saved prompt to %s", _dump_file)
    except Exception:
        pass
    stream = bool(payload.get("stream", False))
    url = f"{CONFIG.backend_base_url.rstrip('/')}/v1/completions"
    headers = _passthrough_headers(request)

    if not stream:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            resp = await client.post(url, json=payload, headers=headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )

    async def _relay() -> AsyncIterator[bytes | str]:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as upstream:
                async for line in upstream.aiter_lines():
                    yield f"{line}\n\n"

    return StreamingResponse(_relay(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    payload = await request.json()
    stream = bool(payload.get("stream", False))
    request_id = request.headers.get("x-request-id") or f"gw-{uuid.uuid4().hex[:12]}"

    # Provide sensible defaults for the demo UI while keeping client overrides.
    payload.setdefault("temperature", CONFIG.default_temperature)
    payload.setdefault("top_p", CONFIG.default_top_p)
    payload.setdefault("max_tokens", CONFIG.default_max_tokens)
    if CONFIG.default_seed is not None:
        payload.setdefault("seed", CONFIG.default_seed)

    url = f"{CONFIG.backend_base_url.rstrip('/')}/v1/chat/completions"
    headers = _passthrough_headers(request)

    # Allow clients to send an alias model name and resolve it at gateway.
    requested_model = payload.get("model")
    if not isinstance(requested_model, str) or requested_model in {"", "gateway-auto", "auto"}:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            payload["model"] = await _resolve_backend_model(client, headers)

    await HUB.publish(
        "request_started",
        {
            "request_id": request_id,
            "stream": stream,
            "model": payload.get("model"),
            "client": request.client.host if request.client else None,
            "prompt_preview": json.dumps(payload.get("messages", []), ensure_ascii=False)[:240],
        },
    )

    if not stream:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            resp = await client.post(url, json=payload, headers=headers)
            try:
                parsed = resp.json()
            except Exception:  # noqa: BLE001
                parsed = None
            text_out = ""
            if isinstance(parsed, dict):
                choices = parsed.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message") or {}
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        text_out = msg.get("content", "")
            await HUB.publish(
                "request_done",
                {
                    "request_id": request_id,
                    "stream": False,
                    "status": resp.status_code,
                    "text": text_out,
                },
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )

    async def relay_stream() -> AsyncIterator[bytes | str]:
        started_at = time.perf_counter()
        first_token_at: float | None = None
        token_count = 0
        saw_done = False

        async def publish_done(finish_reason: str | None = None) -> None:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
            ttft_ms = None
            if first_token_at is not None:
                ttft_ms = round((first_token_at - started_at) * 1000.0, 2)
            await HUB.publish(
                "request_done",
                {
                    "request_id": request_id,
                    "stream": True,
                    "token_count": token_count,
                    "elapsed_ms": elapsed_ms,
                    "ttft_ms": ttft_ms,
                    "finish_reason": finish_reason,
                },
            )

        async with httpx.AsyncClient(timeout=_timeout()) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as upstream:
                if upstream.status_code >= 400:
                    body = await upstream.aread()
                    err = body.decode("utf-8", errors="replace")
                    await HUB.publish(
                        "request_error",
                        {
                            "request_id": request_id,
                            "status": upstream.status_code,
                            "message": err,
                        },
                    )
                    yield f"data: {json.dumps({'error': {'message': err, 'status': upstream.status_code}})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                data_lines: list[str] = []
                async for raw_line in upstream.aiter_lines():
                    line = raw_line
                    yield f"{line}\n"

                    if line.startswith("data:"):
                        data_lines.append(line[len("data:") :].strip())
                        continue

                    if line != "":
                        continue

                    if not data_lines:
                        continue
                    data = "".join(data_lines)
                    data_lines = []

                    if data == "[DONE]":
                        saw_done = True
                        await publish_done()
                        continue

                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(obj, dict) and obj.get("error"):
                        await HUB.publish(
                            "request_error",
                            {
                                "request_id": request_id,
                                "message": str(obj.get("error")),
                            },
                        )
                        continue

                    choice = ((obj.get("choices") or [{}])[0]) if isinstance(obj, dict) else {}
                    if not isinstance(choice, dict):
                        continue
                    delta = (choice.get("delta") or {}).get("content", "")
                    if isinstance(delta, str) and delta:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        token_count += 1
                        await HUB.publish(
                            "token",
                            {
                                "request_id": request_id,
                                "text": delta,
                                "token_count": token_count,
                            },
                        )

                    finish_reason = choice.get("finish_reason")
                    if isinstance(finish_reason, str) and finish_reason:
                        await HUB.publish(
                            "request_finish",
                            {
                                "request_id": request_id,
                                "finish_reason": finish_reason,
                            },
                        )

                if not saw_done:
                    await publish_done()

    return StreamingResponse(
        relay_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

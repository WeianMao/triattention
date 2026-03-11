"""FastAPI app for OpenAI-compatible vLLM gateway + live streaming UI."""
from __future__ import annotations

import asyncio
import json
import logging
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

# Mutable forward-backend setting (which backend's response is returned to HTTP caller)
_forward_backend: str = CONFIG.forward_backend

BACKEND_URLS: dict[str, str] = {
    "baseline": CONFIG.backend_base_url,
    "triattention": CONFIG.triattention_backend_base_url,
}


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
_MODEL_ID_CACHE: dict[str, str] = {}
LOGGER = logging.getLogger("demo.gateway")


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(CONFIG.request_timeout_s, connect=CONFIG.connect_timeout_s)


def _passthrough_headers(request: Request) -> dict[str, str]:
    allowed = {"authorization", "x-request-id", "openai-organization"}
    headers = {k: v for k, v in request.headers.items() if k.lower() in allowed}
    return headers


async def _resolve_backend_model(
    client: httpx.AsyncClient, headers: dict[str, str], base_url: str | None = None,
) -> str:
    if base_url is None:
        base_url = CONFIG.backend_base_url
    url = f"{base_url.rstrip('/')}/v1/models"
    for _ in range(3):
        try:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") or []
            if data and isinstance(data[0], dict) and isinstance(data[0].get("id"), str):
                model_id = str(data[0]["id"])
                _MODEL_ID_CACHE[base_url] = model_id
                return model_id
        except Exception:  # noqa: BLE001
            await asyncio.sleep(0.2)

    if base_url in _MODEL_ID_CACHE:
        return _MODEL_ID_CACHE[base_url]
    raise RuntimeError(f"No model id returned by backend {url}")


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


async def _fetch_kv_metrics(base_url: str) -> dict[str, Any]:
    """Fetch and parse KV cache metrics from a single backend."""
    metrics_url = f"{base_url.rstrip('/')}/metrics"
    timeout = httpx.Timeout(min(CONFIG.request_timeout_s, 5.0), connect=CONFIG.connect_timeout_s)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(metrics_url)
            resp.raise_for_status()
            text = resp.text
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "error": str(exc),
                "metrics_url": metrics_url,
            }

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

    return {
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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/healthz")
async def healthz() -> JSONResponse:
    baseline_health, triattention_health = await asyncio.gather(
        _check_health(BACKEND_URLS["baseline"]),
        _check_health(BACKEND_URLS["triattention"]),
    )
    ok = baseline_health.get("ok") and triattention_health.get("ok")
    payload: dict[str, Any] = {
        "ok": bool(ok),
        "gateway": {"ok": True, "host": CONFIG.host, "port": CONFIG.port},
        "baseline": baseline_health,
        "triattention": triattention_health,
        "forward_backend": _forward_backend,
    }
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
async def kv_cache_metrics(backend: str | None = None) -> JSONResponse:
    if backend and backend in BACKEND_URLS:
        result = await _fetch_kv_metrics(BACKEND_URLS[backend])
        result["backend"] = backend
        return JSONResponse(content=result)

    # Return both backends
    baseline_result, triattention_result = await asyncio.gather(
        _fetch_kv_metrics(BACKEND_URLS["baseline"]),
        _fetch_kv_metrics(BACKEND_URLS["triattention"]),
    )
    baseline_result["backend"] = "baseline"
    triattention_result["backend"] = "triattention"
    return JSONResponse(content={
        "baseline": baseline_result,
        "triattention": triattention_result,
    })


@app.get("/api/forward-toggle")
async def get_forward_toggle() -> JSONResponse:
    return JSONResponse(content={"backend": _forward_backend})


@app.post("/api/forward-toggle")
async def set_forward_toggle(request: Request) -> JSONResponse:
    global _forward_backend
    body = await request.json()
    new_backend = body.get("backend", "")
    if new_backend not in ("baseline", "triattention"):
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid backend: {new_backend!r}. Must be 'baseline' or 'triattention'."},
        )
    _forward_backend = new_backend
    return JSONResponse(content={"backend": _forward_backend})


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
    import logging as _clog
    _prompt_text = payload.get("prompt", "")
    _clog.getLogger("demo.completions_proxy").info(
        "completions request: prompt_len=%d max_tokens=%s stream=%s prompt_head=%.500s",
        len(_prompt_text) if isinstance(_prompt_text, str) else 0,
        payload.get("max_tokens"),
        payload.get("stream"),
        _prompt_text[:500] if isinstance(_prompt_text, str) else str(_prompt_text)[:500],
    )
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
    _normalize_max_tokens(payload)
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


async def _stream_backend(
    backend_name: str,
    base_url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    request_id: str,
) -> None:
    """Stream from a backend and publish events to the hub. Fire-and-forget for secondary backend."""
    bid = f"{request_id}-{backend_name}"
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    # Resolve model for this backend
    async with httpx.AsyncClient(timeout=_timeout()) as client:
        backend_payload = dict(payload)
        requested_model = backend_payload.get("model")
        if not isinstance(requested_model, str) or requested_model in {"", "gateway-auto", "auto"}:
            try:
                backend_payload["model"] = await _resolve_backend_model(client, headers, base_url)
            except Exception:  # noqa: BLE001
                backend_payload["model"] = requested_model or "auto"

    started_at = time.perf_counter()
    first_token_at: float | None = None
    token_count = 0
    saw_done = False

    await HUB.publish(
        "request_started",
        {
            "request_id": bid,
            "backend": backend_name,
            "stream": True,
            "model": backend_payload.get("model"),
        },
    )

    async def publish_done(finish_reason: str | None = None) -> None:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
        ttft_ms = None
        if first_token_at is not None:
            ttft_ms = round((first_token_at - started_at) * 1000.0, 2)
        await HUB.publish(
            "request_done",
            {
                "request_id": bid,
                "backend": backend_name,
                "stream": True,
                "token_count": token_count,
                "elapsed_ms": elapsed_ms,
                "ttft_ms": ttft_ms,
                "finish_reason": finish_reason,
            },
        )

    try:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            async with client.stream("POST", url, json=backend_payload, headers=headers) as upstream:
                if upstream.status_code >= 400:
                    body = await upstream.aread()
                    err = body.decode("utf-8", errors="replace")
                    await HUB.publish(
                        "request_error",
                        {
                            "request_id": bid,
                            "backend": backend_name,
                            "status": upstream.status_code,
                            "message": err,
                        },
                    )
                    return

                data_lines: list[str] = []
                line_iter = upstream.aiter_lines().__aiter__()
                while True:
                    try:
                        raw_line = await asyncio.wait_for(
                            line_iter.__anext__(),
                            timeout=CONFIG.secondary_stream_idle_timeout_s,
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        await HUB.publish(
                            "request_error",
                            {
                                "request_id": bid,
                                "backend": backend_name,
                                "message": (
                                    "KV cache insufficient (likely preempted); "
                                    "request terminated by gateway."
                                ),
                            },
                        )
                        return
                    line = raw_line

                    if line.startswith("data:"):
                        data_lines.append(line[len("data:"):].strip())
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
                                "request_id": bid,
                                "backend": backend_name,
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
                                "request_id": bid,
                                "backend": backend_name,
                                "text": delta,
                                "token_count": token_count,
                            },
                        )

                    finish_reason = choice.get("finish_reason")
                    if isinstance(finish_reason, str) and finish_reason:
                        await HUB.publish(
                            "request_finish",
                            {
                                "request_id": bid,
                                "backend": backend_name,
                                "finish_reason": finish_reason,
                            },
                        )

                if not saw_done:
                    await publish_done()
    except Exception as exc:  # noqa: BLE001
        await HUB.publish(
            "request_error",
            {
                "request_id": bid,
                "backend": backend_name,
                "message": str(exc),
            },
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    payload = await request.json()
    try:
        import pathlib, time as _t, logging as _clog
        _msgs = payload.get("messages", [])
        _total_len = sum(len(m.get("content","")) for m in _msgs if isinstance(m, dict))
        _clog.getLogger("demo.chat_proxy").info(
            "chat request: msgs=%d total_content_len=%d max_tokens=%s stream=%s",
            len(_msgs), _total_len, payload.get("max_tokens"), payload.get("stream"),
        )
        _dump_dir = pathlib.Path("/tmp/openclaw_prompts")
        _dump_dir.mkdir(exist_ok=True)
        _dump_file = _dump_dir / f"chat_{int(_t.time())}.json"
        with open(_dump_file, "w") as _f:
            json.dump(payload, _f, ensure_ascii=False)
    except Exception:
        pass
    stream = bool(payload.get("stream", False))
    request_id = request.headers.get("x-request-id") or f"gw-{uuid.uuid4().hex[:12]}"

    payload.setdefault("temperature", CONFIG.default_temperature)
    payload.setdefault("top_p", CONFIG.default_top_p)
    payload.setdefault("max_tokens", CONFIG.default_max_tokens)
    _normalize_max_tokens(payload)
    if CONFIG.default_seed is not None:
        payload.setdefault("seed", CONFIG.default_seed)

    headers = _passthrough_headers(request)

    # Determine primary (forwarded) and secondary backends
    primary_name = _forward_backend
    secondary_name = "baseline" if primary_name == "triattention" else "triattention"
    primary_url = BACKEND_URLS[primary_name]
    secondary_url = BACKEND_URLS[secondary_name]

    # Resolve model for the primary backend
    requested_model = payload.get("model")
    if not isinstance(requested_model, str) or requested_model in {"", "gateway-auto", "auto"}:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            payload["model"] = await _resolve_backend_model(client, headers, primary_url)

    if not stream:
        # Non-streaming: send to both, return primary response
        url = f"{primary_url.rstrip('/')}/v1/chat/completions"
        # Fire off secondary in background
        asyncio.create_task(_stream_backend(secondary_name, secondary_url, payload, headers, request_id))

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
            bid = f"{request_id}-{primary_name}"
            await HUB.publish(
                "request_done",
                {
                    "request_id": bid,
                    "backend": primary_name,
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

    # Streaming: run secondary in background, primary inline
    asyncio.create_task(_stream_backend(secondary_name, secondary_url, payload, headers, request_id))

    async def relay_stream() -> AsyncIterator[bytes | str]:
        bid = f"{request_id}-{primary_name}"
        url = f"{primary_url.rstrip('/')}/v1/chat/completions"
        started_at = time.perf_counter()
        first_token_at: float | None = None
        token_count = 0
        saw_done = False

        # Resolve model for primary
        primary_payload = dict(payload)
        async with httpx.AsyncClient(timeout=_timeout()) as resolve_client:
            rm = primary_payload.get("model")
            if not isinstance(rm, str) or rm in {"", "gateway-auto", "auto"}:
                try:
                    primary_payload["model"] = await _resolve_backend_model(resolve_client, headers, primary_url)
                except Exception:  # noqa: BLE001
                    pass

        await HUB.publish(
            "request_started",
            {
                "request_id": bid,
                "backend": primary_name,
                "stream": True,
                "model": primary_payload.get("model"),
            },
        )

        async def publish_done(finish_reason: str | None = None) -> None:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
            ttft_ms = None
            if first_token_at is not None:
                ttft_ms = round((first_token_at - started_at) * 1000.0, 2)
            await HUB.publish(
                "request_done",
                {
                    "request_id": bid,
                    "backend": primary_name,
                    "stream": True,
                    "token_count": token_count,
                    "elapsed_ms": elapsed_ms,
                    "ttft_ms": ttft_ms,
                    "finish_reason": finish_reason,
                },
            )

        async with httpx.AsyncClient(timeout=_timeout()) as client:
            async with client.stream("POST", url, json=primary_payload, headers=headers) as upstream:
                if upstream.status_code >= 400:
                    body = await upstream.aread()
                    err = body.decode("utf-8", errors="replace")
                    await HUB.publish(
                        "request_error",
                        {
                            "request_id": bid,
                            "backend": primary_name,
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
                        data_lines.append(line[len("data:"):].strip())
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
                                "request_id": bid,
                                "backend": primary_name,
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
                                "request_id": bid,
                                "backend": primary_name,
                                "text": delta,
                                "token_count": token_count,
                            },
                        )

                    finish_reason = choice.get("finish_reason")
                    if isinstance(finish_reason, str) and finish_reason:
                        await HUB.publish(
                            "request_finish",
                            {
                                "request_id": bid,
                                "backend": primary_name,
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
def _normalize_max_tokens(payload: dict[str, Any], *, field: str = "max_tokens") -> None:
    """Clamp max_tokens/max_completion_tokens so they fit the 16K context."""
    limit = CONFIG.max_tokens_cap
    default_value = CONFIG.default_max_tokens

    raw_value = payload.get(field)
    if not isinstance(raw_value, int):
        raw_value = default_value
    raw_value = max(1, raw_value)
    clamped = min(raw_value, limit)
    if clamped != raw_value:
        LOGGER.info("Clamping %s from %s to %s (limit %s)", field, raw_value, clamped, limit)
    payload[field] = clamped

    # Propagate to complementary field (OpenAI API supports both names)
    other_field = "max_completion_tokens" if field == "max_tokens" else "max_tokens"
    other_value = payload.get(other_field)
    if isinstance(other_value, int):
        other_value = max(1, min(other_value, limit))
    else:
        other_value = clamped
    payload[other_field] = other_value

    # Some OpenClaw builds also set "max_completion_tokens" only
    if field != "max_tokens" and "max_tokens" not in payload:
        payload["max_tokens"] = other_value

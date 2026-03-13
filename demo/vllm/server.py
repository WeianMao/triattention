"""FastAPI app for OpenAI-compatible vLLM gateway + live streaming UI."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
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

_forward_backend: str = CONFIG.forward_backend

BACKEND_URLS: dict[str, str] = {
    "baseline": CONFIG.backend_base_url,
    "triattention": CONFIG.triattention_backend_base_url,
}

LOGGER = logging.getLogger("demo.gateway")
_MODEL_ID_CACHE: dict[str, str] = {}

_PROM_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)$",
)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.DOTALL)
_XML_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_RESULT_RE = re.compile(
    r"(?:tool result|observation|result)\s*:\s*(.+?)(?=\n\s*\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_FUNCTION_CALL_RE = re.compile(r"^\s*([a-zA-Z_][\w:-]*)\s*\((.*)\)\s*$", re.DOTALL)


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


@dataclass
class SessionInfo:
    session_id: str
    route_kind: str
    request_ids: set[str] = field(default_factory=set)
    tool_keys: set[str] = field(default_factory=set)
    result_keys: set[str] = field(default_factory=set)
    updated_at: float = field(default_factory=time.time)


class DemoSessionStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._bindings: dict[str, tuple[str, float]] = {}
        self._sessions: dict[str, SessionInfo] = {}
        self._window_s = 90.0

    async def resolve_session_id(
        self,
        *,
        request: Request,
        payload: dict[str, Any],
        request_id: str,
        route_kind: str,
    ) -> str:
        explicit = _extract_explicit_session_id(request, payload)
        now = time.time()
        async with self._lock:
            if explicit:
                session = self._sessions.setdefault(explicit, SessionInfo(explicit, route_kind))
                session.updated_at = now
                session.request_ids.add(request_id)
                return explicit

            if route_kind == "chat":
                session = self._sessions.setdefault(request_id, SessionInfo(request_id, route_kind))
                session.updated_at = now
                session.request_ids.add(request_id)
                return request_id

            client_host = request.client.host if request.client else "unknown"
            binding_key = f"{route_kind}:{client_host}"
            existing = self._bindings.get(binding_key)
            if existing and now - existing[1] <= self._window_s:
                session_id = existing[0]
            else:
                session_id = f"{route_kind}-{uuid.uuid4().hex[:10]}"

            self._bindings[binding_key] = (session_id, now)
            session = self._sessions.setdefault(session_id, SessionInfo(session_id, route_kind))
            session.updated_at = now
            session.request_ids.add(request_id)
            self._prune(now)
            return session_id

    async def mark_tool_seen(self, session_id: str, tool_key: str) -> bool:
        async with self._lock:
            session = self._sessions.setdefault(session_id, SessionInfo(session_id, "unknown"))
            if tool_key in session.tool_keys:
                session.updated_at = time.time()
                return False
            session.tool_keys.add(tool_key)
            session.updated_at = time.time()
            return True

    async def mark_result_seen(self, session_id: str, result_key: str) -> bool:
        async with self._lock:
            session = self._sessions.setdefault(session_id, SessionInfo(session_id, "unknown"))
            if result_key in session.result_keys:
                session.updated_at = time.time()
                return False
            session.result_keys.add(result_key)
            session.updated_at = time.time()
            return True

    def _prune(self, now: float) -> None:
        stale_binding_keys = [k for k, (_, ts) in self._bindings.items() if now - ts > self._window_s * 2]
        for key in stale_binding_keys:
            self._bindings.pop(key, None)
        stale_session_ids = [
            sid for sid, session in self._sessions.items() if now - session.updated_at > self._window_s * 8
        ]
        for sid in stale_session_ids:
            self._sessions.pop(sid, None)


@dataclass
class ToolDeltaState:
    tool_call_id: str
    tool_name: str = ""
    arguments_text: str = ""
    started: bool = False
    finished: bool = False


class CompletionStreamAnalyzer:
    def __init__(self) -> None:
        self.mode = "unknown"
        self.buffer = ""
        self.tool_state = ToolDeltaState(tool_call_id=f"tc-{uuid.uuid4().hex[:8]}")
        self.text_started = False

    def ingest(self, delta: str) -> list[tuple[str, dict[str, Any]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        if not delta:
            return events

        self.buffer += delta

        if self.mode == "unknown":
            candidate = _parse_tool_call_candidate(self.buffer)
            if candidate:
                self.mode = "tool"
            elif _looks_like_plain_text(self.buffer):
                self.mode = "text"
                events.append(("text_delta", {"text": self.buffer}))
                self.text_started = True
                self.buffer = ""
                return events
            elif len(self.buffer) > 160:
                self.mode = "text"
                events.append(("text_delta", {"text": self.buffer}))
                self.text_started = True
                self.buffer = ""
                return events

        if self.mode == "tool":
            candidate = _parse_tool_call_candidate(self.buffer)
            if not candidate:
                if len(self.buffer) > 240 and _looks_like_plain_text(self.buffer):
                    self.mode = "text"
                    events.append(("text_delta", {"text": self.buffer}))
                    self.text_started = True
                    self.buffer = ""
                return events
            name = candidate["name"]
            args_text = candidate["arguments_text"]
            if not self.tool_state.started:
                self.tool_state.started = True
                self.tool_state.tool_name = name
                events.append(
                    (
                        "tool_call_started",
                        {
                            "tool_call_id": self.tool_state.tool_call_id,
                            "tool_name": name,
                            "arguments_full": args_text,
                        },
                    ),
                )
            elif name and name != self.tool_state.tool_name:
                self.tool_state.tool_name = name

            if args_text.startswith(self.tool_state.arguments_text):
                delta_text = args_text[len(self.tool_state.arguments_text) :]
            else:
                delta_text = args_text
            self.tool_state.arguments_text = args_text
            if delta_text:
                events.append(
                    (
                        "tool_call_delta",
                        {
                            "tool_call_id": self.tool_state.tool_call_id,
                            "tool_name": self.tool_state.tool_name,
                            "arguments_delta": delta_text,
                            "arguments_full": self.tool_state.arguments_text,
                        },
                    ),
                )
            return events

        if self.mode == "text":
            events.append(("text_delta", {"text": delta}))
        return events

    def finalize(self) -> list[tuple[str, dict[str, Any]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        if self.mode == "unknown" and self.buffer:
            events.append(("text_delta", {"text": self.buffer}))
            self.buffer = ""
            self.mode = "text"
        if self.mode == "tool" and self.tool_state.started and not self.tool_state.finished:
            self.tool_state.finished = True
            events.append(
                (
                    "tool_call_finished",
                    {
                        "tool_call_id": self.tool_state.tool_call_id,
                        "tool_name": self.tool_state.tool_name,
                        "arguments_full": self.tool_state.arguments_text,
                    },
                ),
            )
        return events


HUB = LiveStreamHub()
SESSIONS = DemoSessionStore()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(CONFIG.request_timeout_s, connect=CONFIG.connect_timeout_s)


def _passthrough_headers(request: Request) -> dict[str, str]:
    allowed = {"authorization", "x-request-id", "openai-organization"}
    return {k: v for k, v in request.headers.items() if k.lower() in allowed}


def _extract_explicit_session_id(request: Request, payload: dict[str, Any]) -> str | None:
    header_keys = ("x-session-id", "x-openclaw-session-id", "openclaw-session-id")
    for key in header_keys:
        value = request.headers.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    payload_keys = ("session_id", "conversation_id", "thread_id")
    for key in payload_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in payload_keys:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _safe_json_dumps(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


async def _publish_live_event(event: str, payload: dict[str, Any]) -> None:
    payload.setdefault("timestamp_ms", _now_ms())
    await HUB.publish(event, payload)


async def _resolve_backend_model(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    base_url: str | None = None,
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
        "updated_at_ms": _now_ms(),
    }


def _normalize_max_tokens(payload: dict[str, Any], *, field: str = "max_tokens") -> None:
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

    other_field = "max_completion_tokens" if field == "max_tokens" else "max_tokens"
    other_value = payload.get(other_field)
    if isinstance(other_value, int):
        other_value = max(1, min(other_value, limit))
    else:
        other_value = clamped
    payload[other_field] = other_value

    if field != "max_tokens" and "max_tokens" not in payload:
        payload["max_tokens"] = other_value


def _clean_tool_blob(text: str) -> str:
    text = text.strip()
    xml_match = _XML_TOOL_RE.search(text)
    if xml_match:
        text = xml_match.group(1).strip()
    text = _FENCE_RE.sub("", text).strip()
    return text


def _parse_tool_call_candidate(text: str) -> dict[str, str] | None:
    blob = _clean_tool_blob(text)
    if not blob:
        return None

    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        obj = None

    if isinstance(obj, dict):
        function_obj = obj.get("function")
        if isinstance(function_obj, dict):
            name = function_obj.get("_name") or function_obj.get("name")
            args = function_obj.get("arguments")
            if name:
                if args is None:
                    args = {k: v for k, v in function_obj.items() if k not in {"_name", "name", "arguments"}}
                return {
                    "name": str(name),
                    "arguments_text": _safe_json_dumps(args if args is not None else {}),
                }

        for name_key, args_key in (
            ("name", "arguments"),
            ("tool_name", "arguments"),
            ("name", "parameters"),
            ("tool", "input"),
        ):
            name = obj.get(name_key)
            if name:
                args = obj.get(args_key, {})
                return {
                    "name": str(name),
                    "arguments_text": _safe_json_dumps(args),
                }

    function_match = _FUNCTION_CALL_RE.match(blob)
    if function_match:
        args_text = function_match.group(2).strip()
        if args_text:
            return {
                "name": function_match.group(1),
                "arguments_text": args_text,
            }
    return None


def _looks_like_plain_text(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    if stripped[0] in "{[<" or stripped.startswith("```"):
        return False
    if re.search(r"[。！？.!?]\s", stripped):
        return True
    if len(stripped) > 32 and " " in stripped:
        return True
    return False


def _summarize_text(text: str, *, limit: int = 160) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 1]}…"


async def _publish_tool_result(
    *,
    session_id: str,
    backend_name: str,
    request_id: str,
    tool_call_id: str,
    tool_name: str,
    result_text: str,
    origin: str,
) -> None:
    result_key = f"{tool_call_id}:{_short_hash(result_text)}"
    if not await SESSIONS.mark_result_seen(session_id, result_key):
        return
    await _publish_live_event(
        "tool_result",
        {
            "session_id": session_id,
            "backend": backend_name,
            "request_id": request_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "result_summary": _summarize_text(result_text),
            "result_text": result_text,
            "origin": origin,
        },
    )


async def _publish_prompt_hints(
    *,
    session_id: str,
    backend_name: str,
    request_id: str,
    route_kind: str,
    payload: dict[str, Any],
) -> None:
    if route_kind == "chat":
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            return
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    tool_call_id = str(tool_call.get("id") or f"history-tool-{index}")
                    function_obj = tool_call.get("function") or {}
                    tool_name = str(function_obj.get("name") or "tool")
                    arguments_text = _safe_json_dumps(function_obj.get("arguments", {}))
                    tool_key = f"{tool_call_id}:{tool_name}:{_short_hash(arguments_text)}"
                    if await SESSIONS.mark_tool_seen(session_id, tool_key):
                        await _publish_live_event(
                            "tool_call_finished",
                            {
                                "session_id": session_id,
                                "backend": backend_name,
                                "request_id": request_id,
                                "tool_call_id": tool_call_id,
                                "tool_name": tool_name,
                                "arguments_full": arguments_text,
                                "origin": "history",
                            },
                        )
            elif role == "tool":
                tool_call_id = str(message.get("tool_call_id") or f"tool-result-{index}")
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    await _publish_tool_result(
                        session_id=session_id,
                        backend_name=backend_name,
                        request_id=request_id,
                        tool_call_id=tool_call_id,
                        tool_name="tool",
                        result_text=content,
                        origin="history",
                    )
        return

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return

    for match in _RESULT_RE.finditer(prompt):
        result_text = match.group(1).strip()
        if not result_text:
            continue
        tool_call_id = f"prompt-result-{_short_hash(result_text)}"
        await _publish_tool_result(
            session_id=session_id,
            backend_name=backend_name,
            request_id=request_id,
            tool_call_id=tool_call_id,
            tool_name="tool",
            result_text=result_text,
            origin="prompt",
        )


async def _publish_chat_tool_delta(
    *,
    session_id: str,
    backend_name: str,
    request_id: str,
    tool_states: dict[str, ToolDeltaState],
    tool_delta: dict[str, Any],
) -> None:
    index = tool_delta.get("index", 0)
    tool_call_id = str(tool_delta.get("id") or f"tc-{index}")
    state = tool_states.setdefault(tool_call_id, ToolDeltaState(tool_call_id=tool_call_id))

    function_obj = tool_delta.get("function") or {}
    tool_name = function_obj.get("name")
    if isinstance(tool_name, str) and tool_name:
        state.tool_name = tool_name

    arguments_delta = function_obj.get("arguments")
    if isinstance(arguments_delta, str) and arguments_delta:
        state.arguments_text += arguments_delta

    if not state.started:
        state.started = True
        await _publish_live_event(
            "tool_call_started",
            {
                "session_id": session_id,
                "backend": backend_name,
                "request_id": request_id,
                "tool_call_id": tool_call_id,
                "tool_name": state.tool_name or "tool",
                "arguments_full": state.arguments_text,
                "origin": "stream",
            },
        )

    if isinstance(arguments_delta, str) and arguments_delta:
        await _publish_live_event(
            "tool_call_delta",
            {
                "session_id": session_id,
                "backend": backend_name,
                "request_id": request_id,
                "tool_call_id": tool_call_id,
                "tool_name": state.tool_name or "tool",
                "arguments_delta": arguments_delta,
                "arguments_full": state.arguments_text,
                "origin": "stream",
            },
        )


async def _finish_chat_tool_states(
    *,
    session_id: str,
    backend_name: str,
    request_id: str,
    tool_states: dict[str, ToolDeltaState],
) -> None:
    for tool_call_id, state in tool_states.items():
        if state.finished or not state.started:
            continue
        state.finished = True
        tool_key = f"{tool_call_id}:{state.tool_name}:{_short_hash(state.arguments_text)}"
        await SESSIONS.mark_tool_seen(session_id, tool_key)
        await _publish_live_event(
            "tool_call_finished",
            {
                "session_id": session_id,
                "backend": backend_name,
                "request_id": request_id,
                "tool_call_id": tool_call_id,
                "tool_name": state.tool_name or "tool",
                "arguments_full": state.arguments_text,
                "origin": "stream",
            },
        )


async def _publish_completion_stream_events(
    *,
    session_id: str,
    backend_name: str,
    request_id: str,
    analyzer: CompletionStreamAnalyzer,
    delta: str,
) -> None:
    for event_name, payload in analyzer.ingest(delta):
        payload.update(
            {
                "session_id": session_id,
                "backend": backend_name,
                "request_id": request_id,
                "origin": "completion-stream",
            },
        )
        await _publish_live_event(event_name, payload)


async def _finalize_completion_stream(
    *,
    session_id: str,
    backend_name: str,
    request_id: str,
    analyzer: CompletionStreamAnalyzer,
) -> None:
    for event_name, payload in analyzer.finalize():
        payload.update(
            {
                "session_id": session_id,
                "backend": backend_name,
                "request_id": request_id,
                "origin": "completion-stream",
            },
        )
        if event_name == "tool_call_finished":
            tool_key = f"{payload['tool_call_id']}:{payload['tool_name']}:{_short_hash(payload['arguments_full'])}"
            await SESSIONS.mark_tool_seen(session_id, tool_key)
        await _publish_live_event(event_name, payload)


async def _stream_completion_backend(
    *,
    backend_name: str,
    base_url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    request_id: str,
    session_id: str,
) -> None:
    bid = f"{request_id}-{backend_name}"
    analyzer = CompletionStreamAnalyzer()
    url = f"{base_url.rstrip('/')}/v1/completions"

    await _publish_live_event(
        "request_started",
        {
            "session_id": session_id,
            "request_id": bid,
            "backend": backend_name,
            "stream": True,
            "route_kind": "completions",
            "model": payload.get("model"),
            "phase": "tooling",
        },
    )
    await _publish_prompt_hints(
        session_id=session_id,
        backend_name=backend_name,
        request_id=bid,
        route_kind="completions",
        payload=payload,
    )

    try:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as upstream:
                if upstream.status_code >= 400:
                    body = await upstream.aread()
                    err = body.decode("utf-8", errors="replace")
                    await _publish_live_event(
                        "request_error",
                        {
                            "session_id": session_id,
                            "request_id": bid,
                            "backend": backend_name,
                            "status": upstream.status_code,
                            "message": err,
                            "route_kind": "completions",
                        },
                    )
                    return

                data_lines: list[str] = []
                async for raw_line in upstream.aiter_lines():
                    if raw_line.startswith("data:"):
                        data_lines.append(raw_line[len("data:") :].strip())
                        continue
                    if raw_line != "":
                        continue
                    if not data_lines:
                        continue
                    data = "".join(data_lines)
                    data_lines = []
                    if data == "[DONE]":
                        await _finalize_completion_stream(
                            session_id=session_id,
                            backend_name=backend_name,
                            request_id=bid,
                            analyzer=analyzer,
                        )
                        await _publish_live_event(
                            "request_done",
                            {
                                "session_id": session_id,
                                "request_id": bid,
                                "backend": backend_name,
                                "stream": True,
                                "finish_reason": "stop",
                                "route_kind": "completions",
                            },
                        )
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and obj.get("error"):
                        await _publish_live_event(
                            "request_error",
                            {
                                "session_id": session_id,
                                "request_id": bid,
                                "backend": backend_name,
                                "message": str(obj.get("error")),
                                "route_kind": "completions",
                            },
                        )
                        continue
                    choice = ((obj.get("choices") or [{}])[0]) if isinstance(obj, dict) else {}
                    if not isinstance(choice, dict):
                        continue
                    text_delta = choice.get("text")
                    if isinstance(text_delta, str) and text_delta:
                        await _publish_completion_stream_events(
                            session_id=session_id,
                            backend_name=backend_name,
                            request_id=bid,
                            analyzer=analyzer,
                            delta=text_delta,
                        )
                    finish_reason = choice.get("finish_reason")
                    if isinstance(finish_reason, str) and finish_reason:
                        await _publish_live_event(
                            "request_finish",
                            {
                                "session_id": session_id,
                                "request_id": bid,
                                "backend": backend_name,
                                "finish_reason": finish_reason,
                                "route_kind": "completions",
                            },
                        )
    except Exception as exc:  # noqa: BLE001
        await _publish_live_event(
            "request_error",
            {
                "session_id": session_id,
                "request_id": bid,
                "backend": backend_name,
                "message": str(exc),
                "route_kind": "completions",
            },
        )


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

    baseline_result, triattention_result = await asyncio.gather(
        _fetch_kv_metrics(BACKEND_URLS["baseline"]),
        _fetch_kv_metrics(BACKEND_URLS["triattention"]),
    )
    baseline_result["backend"] = "baseline"
    triattention_result["backend"] = "triattention"
    return JSONResponse(
        content={
            "baseline": baseline_result,
            "triattention": triattention_result,
        },
    )


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
            yield _sse("connected", {"ok": True, "timestamp_ms": _now_ms()})
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
    payload = await request.json()
    request_id = request.headers.get("x-request-id") or f"gw-{uuid.uuid4().hex[:12]}"
    session_id = await SESSIONS.resolve_session_id(
        request=request,
        payload=payload,
        request_id=request_id,
        route_kind="completions",
    )

    prompt_text = payload.get("prompt", "")
    LOGGER.info(
        "completions request: session=%s prompt_len=%d max_tokens=%s stream=%s prompt_head=%.500s",
        session_id,
        len(prompt_text) if isinstance(prompt_text, str) else 0,
        payload.get("max_tokens"),
        payload.get("stream"),
        prompt_text[:500] if isinstance(prompt_text, str) else str(prompt_text)[:500],
    )

    try:
        dump_dir = Path("/tmp/openclaw_prompts")
        dump_dir.mkdir(exist_ok=True)
        dump_file = dump_dir / f"prompt_{int(time.time())}.json"
        dump_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        LOGGER.debug("Failed to dump OpenClaw prompt payload", exc_info=True)

    _normalize_max_tokens(payload)
    stream = bool(payload.get("stream", False))
    headers = _passthrough_headers(request)
    primary_name = _forward_backend
    secondary_name = "baseline" if primary_name == "triattention" else "triattention"
    primary_url = BACKEND_URLS[primary_name]
    secondary_url = BACKEND_URLS[secondary_name]

    if not stream:
        asyncio.create_task(
            _stream_completion_backend(
                backend_name=secondary_name,
                base_url=secondary_url,
                payload=payload,
                headers=headers,
                request_id=request_id,
                session_id=session_id,
            ),
        )
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            resp = await client.post(f"{primary_url.rstrip('/')}/v1/completions", json=payload, headers=headers)
            try:
                parsed = resp.json()
            except Exception:  # noqa: BLE001
                parsed = None
            text_out = ""
            if isinstance(parsed, dict):
                choices = parsed.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    text_out = str(choices[0].get("text") or "")
            analyzer = CompletionStreamAnalyzer()
            bid = f"{request_id}-{primary_name}"
            await _publish_live_event(
                "request_started",
                {
                    "session_id": session_id,
                    "request_id": bid,
                    "backend": primary_name,
                    "stream": False,
                    "route_kind": "completions",
                    "model": payload.get("model"),
                    "phase": "tooling",
                },
            )
            await _publish_prompt_hints(
                session_id=session_id,
                backend_name=primary_name,
                request_id=bid,
                route_kind="completions",
                payload=payload,
            )
            await _publish_completion_stream_events(
                session_id=session_id,
                backend_name=primary_name,
                request_id=bid,
                analyzer=analyzer,
                delta=text_out,
            )
            await _finalize_completion_stream(
                session_id=session_id,
                backend_name=primary_name,
                request_id=bid,
                analyzer=analyzer,
            )
            await _publish_live_event(
                "request_done",
                {
                    "session_id": session_id,
                    "request_id": bid,
                    "backend": primary_name,
                    "stream": False,
                    "status": resp.status_code,
                    "text": text_out,
                    "finish_reason": "stop",
                    "route_kind": "completions",
                },
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )

    asyncio.create_task(
        _stream_completion_backend(
            backend_name=secondary_name,
            base_url=secondary_url,
            payload=payload,
            headers=headers,
            request_id=request_id,
            session_id=session_id,
        ),
    )

    async def relay() -> AsyncIterator[str]:
        bid = f"{request_id}-{primary_name}"
        analyzer = CompletionStreamAnalyzer()
        await _publish_live_event(
            "request_started",
            {
                "session_id": session_id,
                "request_id": bid,
                "backend": primary_name,
                "stream": True,
                "route_kind": "completions",
                "model": payload.get("model"),
                "phase": "tooling",
            },
        )
        await _publish_prompt_hints(
            session_id=session_id,
            backend_name=primary_name,
            request_id=bid,
            route_kind="completions",
            payload=payload,
        )

        async with httpx.AsyncClient(timeout=_timeout()) as client:
            async with client.stream("POST", f"{primary_url.rstrip('/')}/v1/completions", json=payload, headers=headers) as upstream:
                if upstream.status_code >= 400:
                    body = await upstream.aread()
                    err = body.decode("utf-8", errors="replace")
                    await _publish_live_event(
                        "request_error",
                        {
                            "session_id": session_id,
                            "request_id": bid,
                            "backend": primary_name,
                            "status": upstream.status_code,
                            "message": err,
                            "route_kind": "completions",
                        },
                    )
                    yield f"data: {json.dumps({'error': {'message': err, 'status': upstream.status_code}})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                data_lines: list[str] = []
                async for raw_line in upstream.aiter_lines():
                    yield f"{raw_line}\n\n"
                    if raw_line.startswith("data:"):
                        data_lines.append(raw_line[len("data:") :].strip())
                        continue
                    if raw_line != "":
                        continue
                    if not data_lines:
                        continue
                    data = "".join(data_lines)
                    data_lines = []
                    if data == "[DONE]":
                        await _finalize_completion_stream(
                            session_id=session_id,
                            backend_name=primary_name,
                            request_id=bid,
                            analyzer=analyzer,
                        )
                        await _publish_live_event(
                            "request_done",
                            {
                                "session_id": session_id,
                                "request_id": bid,
                                "backend": primary_name,
                                "stream": True,
                                "finish_reason": "stop",
                                "route_kind": "completions",
                            },
                        )
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and obj.get("error"):
                        await _publish_live_event(
                            "request_error",
                            {
                                "session_id": session_id,
                                "request_id": bid,
                                "backend": primary_name,
                                "message": str(obj.get("error")),
                                "route_kind": "completions",
                            },
                        )
                        continue
                    choice = ((obj.get("choices") or [{}])[0]) if isinstance(obj, dict) else {}
                    if not isinstance(choice, dict):
                        continue
                    text_delta = choice.get("text")
                    if isinstance(text_delta, str) and text_delta:
                        await _publish_completion_stream_events(
                            session_id=session_id,
                            backend_name=primary_name,
                            request_id=bid,
                            analyzer=analyzer,
                            delta=text_delta,
                        )
                    finish_reason = choice.get("finish_reason")
                    if isinstance(finish_reason, str) and finish_reason:
                        await _publish_live_event(
                            "request_finish",
                            {
                                "session_id": session_id,
                                "request_id": bid,
                                "backend": primary_name,
                                "finish_reason": finish_reason,
                                "route_kind": "completions",
                            },
                        )

        await _finalize_completion_stream(
            session_id=session_id,
            backend_name=primary_name,
            request_id=bid,
            analyzer=analyzer,
        )
        await _publish_live_event(
            "request_done",
            {
                "session_id": session_id,
                "request_id": bid,
                "backend": primary_name,
                "stream": True,
                "finish_reason": "stop",
                "route_kind": "completions",
            },
        )

    return StreamingResponse(relay(), media_type="text/event-stream")


async def _stream_backend(
    backend_name: str,
    base_url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    request_id: str,
    session_id: str,
    route_kind: str,
) -> None:
    bid = f"{request_id}-{backend_name}"
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

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
    tool_states: dict[str, ToolDeltaState] = {}

    await _publish_live_event(
        "request_started",
        {
            "session_id": session_id,
            "request_id": bid,
            "backend": backend_name,
            "stream": True,
            "model": backend_payload.get("model"),
            "route_kind": route_kind,
            "phase": "streaming",
        },
    )
    await _publish_prompt_hints(
        session_id=session_id,
        backend_name=backend_name,
        request_id=bid,
        route_kind=route_kind,
        payload=backend_payload,
    )

    async def publish_done(finish_reason: str | None = None) -> None:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
        ttft_ms = None
        if first_token_at is not None:
            ttft_ms = round((first_token_at - started_at) * 1000.0, 2)
        await _finish_chat_tool_states(
            session_id=session_id,
            backend_name=backend_name,
            request_id=bid,
            tool_states=tool_states,
        )
        await _publish_live_event(
            "request_done",
            {
                "session_id": session_id,
                "request_id": bid,
                "backend": backend_name,
                "stream": True,
                "token_count": token_count,
                "elapsed_ms": elapsed_ms,
                "ttft_ms": ttft_ms,
                "finish_reason": finish_reason,
                "route_kind": route_kind,
            },
        )

    try:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            async with client.stream("POST", url, json=backend_payload, headers=headers) as upstream:
                if upstream.status_code >= 400:
                    body = await upstream.aread()
                    err = body.decode("utf-8", errors="replace")
                    await _publish_live_event(
                        "request_error",
                        {
                            "session_id": session_id,
                            "request_id": bid,
                            "backend": backend_name,
                            "status": upstream.status_code,
                            "message": err,
                            "route_kind": route_kind,
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
                        await _publish_live_event(
                            "request_error",
                            {
                                "session_id": session_id,
                                "request_id": bid,
                                "backend": backend_name,
                                "message": (
                                    "KV cache insufficient (likely preempted); "
                                    "request terminated by gateway."
                                ),
                                "route_kind": route_kind,
                            },
                        )
                        return

                    if raw_line.startswith("data:"):
                        data_lines.append(raw_line[len("data:") :].strip())
                        continue
                    if raw_line != "":
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
                        await _publish_live_event(
                            "request_error",
                            {
                                "session_id": session_id,
                                "request_id": bid,
                                "backend": backend_name,
                                "message": str(obj.get("error")),
                                "route_kind": route_kind,
                            },
                        )
                        continue

                    choice = ((obj.get("choices") or [{}])[0]) if isinstance(obj, dict) else {}
                    if not isinstance(choice, dict):
                        continue
                    delta_obj = choice.get("delta") or {}
                    delta = delta_obj.get("content", "")
                    if isinstance(delta, str) and delta:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        token_count += 1
                        await _publish_live_event(
                            "text_delta",
                            {
                                "session_id": session_id,
                                "request_id": bid,
                                "backend": backend_name,
                                "text": delta,
                                "token_count": token_count,
                                "route_kind": route_kind,
                            },
                        )

                    tool_calls = delta_obj.get("tool_calls") or []
                    if isinstance(tool_calls, list):
                        for tool_delta in tool_calls:
                            if isinstance(tool_delta, dict):
                                await _publish_chat_tool_delta(
                                    session_id=session_id,
                                    backend_name=backend_name,
                                    request_id=bid,
                                    tool_states=tool_states,
                                    tool_delta=tool_delta,
                                )

                    finish_reason = choice.get("finish_reason")
                    if isinstance(finish_reason, str) and finish_reason:
                        await _publish_live_event(
                            "request_finish",
                            {
                                "session_id": session_id,
                                "request_id": bid,
                                "backend": backend_name,
                                "finish_reason": finish_reason,
                                "route_kind": route_kind,
                            },
                        )

                if not saw_done:
                    await publish_done()
    except Exception as exc:  # noqa: BLE001
        await _publish_live_event(
            "request_error",
            {
                "session_id": session_id,
                "request_id": bid,
                "backend": backend_name,
                "message": str(exc),
                "route_kind": route_kind,
            },
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    payload = await request.json()
    request_id = request.headers.get("x-request-id") or f"gw-{uuid.uuid4().hex[:12]}"
    session_id = await SESSIONS.resolve_session_id(
        request=request,
        payload=payload,
        request_id=request_id,
        route_kind="chat",
    )

    try:
        messages = payload.get("messages", [])
        total_len = sum(len(m.get("content", "")) for m in messages if isinstance(m, dict))
        LOGGER.info(
            "chat request: session=%s msgs=%d total_content_len=%d max_tokens=%s stream=%s",
            session_id,
            len(messages),
            total_len,
            payload.get("max_tokens"),
            payload.get("stream"),
        )
        dump_dir = Path("/tmp/openclaw_prompts")
        dump_dir.mkdir(exist_ok=True)
        dump_file = dump_dir / f"chat_{int(time.time())}.json"
        dump_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        LOGGER.debug("Failed to dump chat payload", exc_info=True)

    stream = bool(payload.get("stream", False))

    payload.setdefault("temperature", CONFIG.default_temperature)
    payload.setdefault("top_p", CONFIG.default_top_p)
    payload.setdefault("max_tokens", CONFIG.default_max_tokens)
    _normalize_max_tokens(payload)
    if CONFIG.default_seed is not None:
        payload.setdefault("seed", CONFIG.default_seed)

    headers = _passthrough_headers(request)

    primary_name = _forward_backend
    secondary_name = "baseline" if primary_name == "triattention" else "triattention"
    primary_url = BACKEND_URLS[primary_name]
    secondary_url = BACKEND_URLS[secondary_name]

    requested_model = payload.get("model")
    if not isinstance(requested_model, str) or requested_model in {"", "gateway-auto", "auto"}:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            payload["model"] = await _resolve_backend_model(client, headers, primary_url)

    if not stream:
        url = f"{primary_url.rstrip('/')}/v1/chat/completions"
        asyncio.create_task(
            _stream_backend(
                secondary_name,
                secondary_url,
                payload,
                headers,
                request_id,
                session_id,
                "chat",
            ),
        )

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
            await _publish_live_event(
                "request_started",
                {
                    "session_id": session_id,
                    "request_id": bid,
                    "backend": primary_name,
                    "stream": False,
                    "model": payload.get("model"),
                    "route_kind": "chat",
                },
            )
            await _publish_prompt_hints(
                session_id=session_id,
                backend_name=primary_name,
                request_id=bid,
                route_kind="chat",
                payload=payload,
            )
            if text_out:
                await _publish_live_event(
                    "text_delta",
                    {
                        "session_id": session_id,
                        "request_id": bid,
                        "backend": primary_name,
                        "text": text_out,
                        "token_count": len(text_out.split()),
                        "route_kind": "chat",
                    },
                )
            await _publish_live_event(
                "request_done",
                {
                    "session_id": session_id,
                    "request_id": bid,
                    "backend": primary_name,
                    "stream": False,
                    "status": resp.status_code,
                    "text": text_out,
                    "route_kind": "chat",
                },
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )

    asyncio.create_task(
        _stream_backend(
            secondary_name,
            secondary_url,
            payload,
            headers,
            request_id,
            session_id,
            "chat",
        ),
    )

    async def relay_stream() -> AsyncIterator[str]:
        bid = f"{request_id}-{primary_name}"
        url = f"{primary_url.rstrip('/')}/v1/chat/completions"
        started_at = time.perf_counter()
        first_token_at: float | None = None
        token_count = 0
        saw_done = False
        tool_states: dict[str, ToolDeltaState] = {}

        primary_payload = dict(payload)
        async with httpx.AsyncClient(timeout=_timeout()) as resolve_client:
            rm = primary_payload.get("model")
            if not isinstance(rm, str) or rm in {"", "gateway-auto", "auto"}:
                try:
                    primary_payload["model"] = await _resolve_backend_model(resolve_client, headers, primary_url)
                except Exception:  # noqa: BLE001
                    pass

        await _publish_live_event(
            "request_started",
            {
                "session_id": session_id,
                "request_id": bid,
                "backend": primary_name,
                "stream": True,
                "model": primary_payload.get("model"),
                "route_kind": "chat",
                "phase": "streaming",
            },
        )
        await _publish_prompt_hints(
            session_id=session_id,
            backend_name=primary_name,
            request_id=bid,
            route_kind="chat",
            payload=primary_payload,
        )

        async def publish_done(finish_reason: str | None = None) -> None:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
            ttft_ms = None
            if first_token_at is not None:
                ttft_ms = round((first_token_at - started_at) * 1000.0, 2)
            await _finish_chat_tool_states(
                session_id=session_id,
                backend_name=primary_name,
                request_id=bid,
                tool_states=tool_states,
            )
            await _publish_live_event(
                "request_done",
                {
                    "session_id": session_id,
                    "request_id": bid,
                    "backend": primary_name,
                    "stream": True,
                    "token_count": token_count,
                    "elapsed_ms": elapsed_ms,
                    "ttft_ms": ttft_ms,
                    "finish_reason": finish_reason,
                    "route_kind": "chat",
                },
            )

        async with httpx.AsyncClient(timeout=_timeout()) as client:
            async with client.stream("POST", url, json=primary_payload, headers=headers) as upstream:
                if upstream.status_code >= 400:
                    body = await upstream.aread()
                    err = body.decode("utf-8", errors="replace")
                    await _publish_live_event(
                        "request_error",
                        {
                            "session_id": session_id,
                            "request_id": bid,
                            "backend": primary_name,
                            "status": upstream.status_code,
                            "message": err,
                            "route_kind": "chat",
                        },
                    )
                    yield f"data: {json.dumps({'error': {'message': err, 'status': upstream.status_code}})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                data_lines: list[str] = []
                async for raw_line in upstream.aiter_lines():
                    yield f"{raw_line}\n"
                    if raw_line.startswith("data:"):
                        data_lines.append(raw_line[len("data:") :].strip())
                        continue
                    if raw_line != "":
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
                        await _publish_live_event(
                            "request_error",
                            {
                                "session_id": session_id,
                                "request_id": bid,
                                "backend": primary_name,
                                "message": str(obj.get("error")),
                                "route_kind": "chat",
                            },
                        )
                        continue

                    choice = ((obj.get("choices") or [{}])[0]) if isinstance(obj, dict) else {}
                    if not isinstance(choice, dict):
                        continue
                    delta_obj = choice.get("delta") or {}
                    delta = delta_obj.get("content", "")
                    if isinstance(delta, str) and delta:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        token_count += 1
                        await _publish_live_event(
                            "text_delta",
                            {
                                "session_id": session_id,
                                "request_id": bid,
                                "backend": primary_name,
                                "text": delta,
                                "token_count": token_count,
                                "route_kind": "chat",
                            },
                        )

                    tool_calls = delta_obj.get("tool_calls") or []
                    if isinstance(tool_calls, list):
                        for tool_delta in tool_calls:
                            if isinstance(tool_delta, dict):
                                await _publish_chat_tool_delta(
                                    session_id=session_id,
                                    backend_name=primary_name,
                                    request_id=bid,
                                    tool_states=tool_states,
                                    tool_delta=tool_delta,
                                )

                    finish_reason = choice.get("finish_reason")
                    if isinstance(finish_reason, str) and finish_reason:
                        await _publish_live_event(
                            "request_finish",
                            {
                                "session_id": session_id,
                                "request_id": bid,
                                "backend": primary_name,
                                "finish_reason": finish_reason,
                                "route_kind": "chat",
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

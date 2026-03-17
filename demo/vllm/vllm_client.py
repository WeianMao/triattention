"""Async helpers for streaming requests to vLLM OpenAI-compatible endpoints."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx


@dataclass
class StreamMetrics:
    started_at: float
    first_token_at: Optional[float] = None
    last_token_at: Optional[float] = None
    token_count: int = 0

    def on_token(self, token_text: str) -> None:
        now = time.perf_counter()
        if self.first_token_at is None:
            self.first_token_at = now
        self.last_token_at = now
        if token_text:
            self.token_count += 1

    def as_payload(self) -> dict[str, Any]:
        now = time.perf_counter()
        total_s = max(now - self.started_at, 0.0)
        ttft_ms = None
        tps = None

        if self.first_token_at is not None:
            ttft_ms = max((self.first_token_at - self.started_at) * 1000.0, 0.0)

        if self.first_token_at is not None and self.last_token_at is not None and self.last_token_at > self.first_token_at:
            decode_s = self.last_token_at - self.first_token_at
            tps = self.token_count / decode_s if decode_s > 0 else None

        return {
            "elapsed_ms": round(total_s * 1000.0, 2),
            "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
            "tps": round(tps, 2) if tps is not None else None,
            "token_count": self.token_count,
            "total_ms": round(total_s * 1000.0, 2),
        }


def _extract_text_delta(event_obj: dict[str, Any]) -> str:
    choices = event_obj.get("choices") or []
    if not choices:
        return ""

    choice = choices[0] or {}
    # /v1/completions format
    if isinstance(choice.get("text"), str):
        return choice.get("text", "")

    # /v1/chat/completions delta format (for compatibility)
    delta = choice.get("delta") or {}
    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
        return delta["content"]

    return ""


async def _resolve_model_name(client: httpx.AsyncClient, base_url: str) -> str:
    url = f"{base_url.rstrip('/')}/v1/models"
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        payload = resp.json()
        models = payload.get("data") or []
        if models and isinstance(models[0], dict) and isinstance(models[0].get("id"), str):
            return models[0]["id"]
    except Exception:
        pass
    return "dummy"


async def stream_vllm_completions(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: Optional[int],
) -> AsyncIterator[tuple[str, dict[str, Any] | str]]:
    """Yield parsed events from a vLLM streaming completion call.

    Yields tuple(event_type, payload):
    - ("token", "...")
    - ("done", metrics_payload)
    """
    async def _stream_with(url: str, body: dict[str, Any]) -> AsyncIterator[tuple[str, dict[str, Any] | str]]:
        metrics = StreamMetrics(started_at=time.perf_counter())
        async with client.stream("POST", url, json=body) as response:
            if response.status_code == 404:
                raise httpx.HTTPStatusError(
                    "Endpoint not found",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()

            async for raw_line in response.aiter_lines():
                if not raw_line or not raw_line.startswith("data:"):
                    continue

                chunk = raw_line[len("data:") :].strip()
                if not chunk:
                    continue

                if chunk == "[DONE]":
                    yield "done", metrics.as_payload()
                    return

                try:
                    event = json.loads(chunk)
                except json.JSONDecodeError:
                    continue

                delta = _extract_text_delta(event)
                if delta:
                    metrics.on_token(delta)
                    yield "token", delta

        # Defensive: if backend closes stream without [DONE]
        yield "done", metrics.as_payload()

    model_name = await _resolve_model_name(client, base_url)

    chat_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    chat_body: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    if seed is not None:
        chat_body["seed"] = seed

    async for event in _stream_with(chat_url, chat_body):
        yield event

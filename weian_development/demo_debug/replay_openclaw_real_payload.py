#!/usr/bin/env python3
"""Replay a captured OpenClaw chat payload against the demo gateway.

This script is intentionally stdlib-only so it can run in lightweight local
environments. It sends the captured payload to `/v1/chat/completions`,
captures the raw SSE stream, and writes a compact JSON report for debugging.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_PAYLOAD = Path(__file__).resolve().parent / "fixtures" / "openclaw_real_payload_20260313_2252.json"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010", help="Demo gateway base URL.")
    parser.add_argument("--payload-file", type=Path, default=DEFAULT_PAYLOAD, help="Captured chat payload JSON.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR / "replay_openclaw_real_payload.json",
        help="Path to JSON report.",
    )
    parser.add_argument("--timeout", type=float, default=180.0, help="Request timeout in seconds.")
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Payload must be a JSON object: {path}")
    return payload


def replay_stream(base_url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    started_at = time.time()
    raw_lines: list[str] = []
    data_events: list[str] = []
    done_seen = False
    content_fragments: list[str] = []
    tool_deltas: list[Any] = []
    finish_reasons: list[str] = []
    status_code: int | None = None
    error_text: str | None = None

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status_code = getattr(resp, "status", None)
            while True:
                line = resp.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").rstrip("\n")
                raw_lines.append(text)

                if not text.startswith("data:"):
                    continue
                data = text[5:].strip()
                data_events.append(data)

                if data == "[DONE]":
                    done_seen = True
                    continue

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choice = ((obj.get("choices") or [{}])[0]) if isinstance(obj, dict) else {}
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        content_fragments.append(content)
                    tool_calls = delta.get("tool_calls")
                    if tool_calls:
                        tool_deltas.append(tool_calls)
                finish_reason = choice.get("finish_reason")
                if isinstance(finish_reason, str) and finish_reason:
                    finish_reasons.append(finish_reason)
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        error_text = exc.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        error_text = str(exc)

    return {
        "request_url": url,
        "started_at": started_at,
        "elapsed_s": round(time.time() - started_at, 3),
        "status_code": status_code,
        "error_text": error_text,
        "done_seen": done_seen,
        "raw_line_count": len(raw_lines),
        "data_event_count": len(data_events),
        "content_char_count": sum(len(x) for x in content_fragments),
        "content_preview": "".join(content_fragments)[:4000],
        "tool_delta_count": len(tool_deltas),
        "tool_deltas_preview": tool_deltas[:20],
        "finish_reasons": finish_reasons,
        "raw_lines_tail": raw_lines[-200:],
    }


def main() -> int:
    args = parse_args()
    payload = load_payload(args.payload_file)
    result = {
        "payload_file": str(args.payload_file),
        "payload_summary": {
            "keys": sorted(payload.keys()),
            "stream": payload.get("stream"),
            "tool_count": len(payload.get("tools") or []),
            "message_roles": [m.get("role") for m in payload.get("messages") or [] if isinstance(m, dict)],
            "max_completion_tokens": payload.get("max_completion_tokens"),
        },
        "replay": replay_stream(args.base_url, payload, args.timeout),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(args.out)
    print(
        json.dumps(
            {
                "status_code": result["replay"]["status_code"],
                "done_seen": result["replay"]["done_seen"],
                "content_char_count": result["replay"]["content_char_count"],
                "tool_delta_count": result["replay"]["tool_delta_count"],
                "finish_reasons": result["replay"]["finish_reasons"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

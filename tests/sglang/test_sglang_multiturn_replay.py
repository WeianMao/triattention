# triattention-sglang-visibility: public
"""Multi-turn tool-calling replay test for TriAttention sglang backend.

This test replays the same fixtures used by the vLLM multiturn replay test
(tests/multiturn_replay/), but targets an sglang server instead.

The original test (test_multiturn_replay.py) is backend-agnostic — it speaks
the OpenAI-compatible chat completions API. This adapter:

  1. Starts an sglang server with TriAttention enabled (or uses --skip-server).
  2. Calls the original replay logic against the sglang endpoint.
  3. Validates that all turns complete without errors.

Usage:
    python tests/sglang/test_sglang_multiturn_replay.py \
        --model <model_path> \
        --port 8899 \
        [--tp 1] \
        [--kv-budget 512] \
        [--divide-length 128] \
        [--stats-path <path_to_stats.pt>] \
        [--fixture-dir tests/multiturn_replay/fixtures] \
        [--timeout 300] \
        [--skip-server]

Fixtures:
    By default, uses fixtures from tests/multiturn_replay/fixtures/ (the same
    fixtures as the vLLM test). Override with --fixture-dir.

Compatibility:
    - Does NOT modify the original vLLM test (tests/multiturn_replay/).
    - Shares fixtures by reference, not by copy.
    - The vLLM test continues to work independently (D-009).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# Default fixture directory: tests/multiturn_replay/fixtures/ relative to repo
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_DIR = REPO_ROOT / "tests" / "multiturn_replay" / "fixtures"


def wait_for_server(base_url: str, timeout: float = 300) -> bool:
    """Poll the server health endpoint until it responds or timeout."""
    health_url = f"{base_url}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2)
    return False


def start_server(
    model: str,
    port: int,
    tp: int,
    kv_budget: int,
    divide_length: int,
    stats_path: Optional[str],
) -> subprocess.Popen:
    """Start sglang server with TriAttention via the launcher."""
    env = os.environ.copy()
    env["ENABLE_TRIATTENTION"] = "1"
    env["TRIATTN_RUNTIME_KV_BUDGET"] = str(kv_budget)
    env["TRIATTN_RUNTIME_DIVIDE_LENGTH"] = str(divide_length)
    env["TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION"] = "1"
    env["TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM"] = "1"
    env["TRIATTN_RUNTIME_LOG_DECISIONS"] = "1"
    if stats_path:
        env["TRIATTN_RUNTIME_SPARSE_STATS_PATH"] = stats_path

    cmd = [
        sys.executable, "-m", "triattention.sglang",
        "--model-path", model,
        "--port", str(port),
        "--tp", str(tp),
        "--disable-radix-cache",
    ]

    print(f"[replay] Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def load_fixtures(fixture_dir: Path) -> list[tuple[str, dict]]:
    """Load turn_*.json fixtures from the given directory."""
    files = sorted(glob.glob(str(fixture_dir / "turn_*.json")))
    if not files:
        print(
            f"[FAIL] No turn_*.json files found in {fixture_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    fixtures = []
    for f in files:
        with open(f) as fh:
            fixtures.append((Path(f).name, json.load(fh)))
    return fixtures


def _flatten_content(content):
    """Flatten array-format content (OpenAI multimodal) to plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return content


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize messages: flatten array content fields to strings."""
    normalized = []
    for msg in messages:
        msg = dict(msg)
        if "content" in msg:
            msg["content"] = _flatten_content(msg["content"])
        normalized.append(msg)
    return normalized


def send_streaming_request(url: str, payload: dict, model: str) -> dict:
    """Send a streaming request and collect SSE chunks into a response."""
    payload = dict(payload)
    payload["model"] = model
    payload["stream"] = True
    payload["messages"] = _normalize_messages(payload.get("messages", []))

    content_parts: list[str] = []
    tool_calls_by_index: dict[int, dict] = {}
    finish_reason = None
    usage = None

    resp = requests.post(url, json=payload, stream=True, timeout=300)
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue

        if obj.get("usage"):
            usage = obj["usage"]

        choices = obj.get("choices", [])
        if not choices:
            continue
        choice = choices[0]

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

        delta = choice.get("delta", {})
        if delta.get("content"):
            content_parts.append(delta["content"])

        for tc_delta in delta.get("tool_calls", []):
            idx = tc_delta.get("index", 0)
            if idx not in tool_calls_by_index:
                tool_calls_by_index[idx] = {
                    "id": tc_delta.get("id", ""),
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            tc = tool_calls_by_index[idx]
            if tc_delta.get("id"):
                tc["id"] = tc_delta["id"]
            fn_delta = tc_delta.get("function", {})
            if fn_delta.get("name"):
                tc["function"]["name"] = fn_delta["name"]
            if fn_delta.get("arguments"):
                tc["function"]["arguments"] += fn_delta["arguments"]

    message: dict = {"role": "assistant"}
    content = "".join(content_parts)
    if content:
        message["content"] = content
    if tool_calls_by_index:
        message["tool_calls"] = [
            tool_calls_by_index[i] for i in sorted(tool_calls_by_index)
        ]

    return {
        "finish_reason": finish_reason,
        "message": message,
        "usage": usage,
    }


def run_replay(
    backend_url: str,
    model: str,
    fixture_dir: Path,
) -> tuple[int, int]:
    """Replay all fixtures and return (passed, total) counts."""
    fixtures = load_fixtures(fixture_dir)
    print(f"[replay] Loaded {len(fixtures)} fixtures from {fixture_dir}")
    print(f"[replay] Backend: {backend_url}")
    print(f"[replay] Model: {model}")
    print()

    passed = 0
    total = len(fixtures)

    for i, (fname, payload) in enumerate(fixtures, 1):
        n_msgs = len(payload.get("messages", []))
        print(f"=== Turn {i}/{total}: {fname} ({n_msgs} messages) ===")
        t0 = time.time()
        try:
            result = send_streaming_request(backend_url, payload, model)
            elapsed = time.time() - t0
            print(f"  status: OK ({elapsed:.1f}s)")

            finish = result.get("finish_reason", "?")
            msg = result.get("message", {})
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls", [])

            print(f"  finish_reason: {finish}")
            if content:
                preview = content[:200].replace("\n", "\\n")
                print(f"  content ({len(content)} chars): {preview}...")
            if tool_calls:
                print(f"  tool_calls: {len(tool_calls)}")
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args_preview = fn.get("arguments", "")[:100]
                    print(f"    - {name}({args_preview})")

            usage = result.get("usage")
            if usage:
                print(
                    f"  usage: prompt={usage.get('prompt_tokens')} "
                    f"completion={usage.get('completion_tokens')} "
                    f"total={usage.get('total_tokens')}"
                )

            passed += 1

        except requests.HTTPError as e:
            elapsed = time.time() - t0
            print(f"  status: HTTP {e.response.status_code} ({elapsed:.1f}s)")
            print(f"  error: {e.response.text[:500]}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  status: FAILED ({elapsed:.1f}s)")
            print(f"  error: {type(e).__name__}: {e}")
        print()

    return passed, total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-turn replay test for TriAttention sglang"
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name or path",
    )
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--kv-budget", type=int, default=512)
    parser.add_argument("--divide-length", type=int, default=128)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument(
        "--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR,
        help="Directory containing turn_*.json fixtures",
    )
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--skip-server", action="store_true",
                        help="Use an already-running server")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"
    backend_url = f"{base_url}/v1/chat/completions"
    proc = None

    try:
        # --- Start server ---
        if not args.skip_server:
            proc = start_server(
                model=args.model,
                port=args.port,
                tp=args.tp,
                kv_budget=args.kv_budget,
                divide_length=args.divide_length,
                stats_path=args.stats_path,
            )
            print(
                f"[replay] Waiting for server (timeout={args.timeout}s)..."
            )
            if not wait_for_server(base_url, timeout=args.timeout):
                print("[FAIL] Server did not start within timeout")
                return 1
            print("[OK] Server is ready")
        else:
            print(f"[replay] Using existing server at {base_url}")

        # --- Run replay ---
        passed, total = run_replay(backend_url, args.model, args.fixture_dir)

        print(f"=== Results: {passed}/{total} turns passed ===")
        if passed == total:
            print("[OK] Multiturn replay test PASSED")
            return 0
        else:
            print(f"[FAIL] {total - passed} turn(s) failed")
            return 1

    finally:
        if proc is not None:
            print("[replay] Shutting down server...")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            print("[replay] Server stopped")


if __name__ == "__main__":
    raise SystemExit(main())

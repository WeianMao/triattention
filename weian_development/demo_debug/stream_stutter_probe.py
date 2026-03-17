#!/usr/bin/env python3
"""Probe streaming stutter and correlate token gaps with backend log events.

This script does not modify runtime logic. It is purely diagnostic:
1) Send one streaming chat request to an OpenAI-compatible endpoint.
2) Record per-token arrival timestamps.
3) Optionally tail backend log and timestamp trigger-related lines.
4) Emit a JSON report for postmortem analysis.
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx


TRIGGER_PATTERNS = (
    "[TriAttention][TRIGGER]",
    "[TriAttention][DONE]",
    "[TriAttention][DECISION]",
    "TriAttention compression applied",
    "TriAttention compression summary",
    "Legacy V1 backend plugin registration is retired",
)

SEQ_RE = re.compile(r"seq_len=(\d+)")


@dataclass
class TokenEvent:
    idx: int
    t: float
    text_len: int


@dataclass
class GapEvent:
    idx_from: int
    idx_to: int
    gap_ms: float
    t_from: float
    t_to: float


@dataclass
class LogEvent:
    t: float
    line: str
    seq_len: Optional[int]


class LogWatcher(threading.Thread):
    def __init__(self, log_path: Path, out_q: queue.Queue[LogEvent]) -> None:
        super().__init__(daemon=True)
        self.log_path = log_path
        self.out_q = out_q
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.touch()
        with self.log_path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)
            while not self._stop.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                if any(p in line for p in TRIGGER_PATTERNS):
                    seq = None
                    m = SEQ_RE.search(line)
                    if m:
                        try:
                            seq = int(m.group(1))
                        except ValueError:
                            seq = None
                    self.out_q.put(
                        LogEvent(
                            t=time.time(),
                            line=line.rstrip("\n"),
                            seq_len=seq,
                        )
                    )


def _resolve_model(base_url: str, timeout_s: float) -> str:
    url = f"{base_url.rstrip('/')}/v1/models"
    with httpx.Client(timeout=timeout_s) as client:
        payload = client.get(url).raise_for_status().json()
    data = payload.get("data") or []
    if not data:
        raise RuntimeError("No model found from /v1/models")
    model_id = data[0].get("id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError("Invalid model id in /v1/models response")
    return model_id


def _stream_tokens(
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    timeout_s: float,
) -> Tuple[List[TokenEvent], Dict[str, Any]]:
    model_id = _resolve_model(base_url, timeout_s)
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "stream": True,
    }
    token_events: List[TokenEvent] = []
    finish_reason: Optional[str] = None
    started = time.time()
    with httpx.Client(timeout=timeout_s) as client:
        with client.stream("POST", url, json=body) as resp:
            resp.raise_for_status()
            data_lines: List[str] = []
            for raw_line in resp.iter_lines():
                line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())
                    continue
                if line != "":
                    continue
                if not data_lines:
                    continue
                data = "".join(data_lines)
                data_lines = []
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices or not isinstance(choices[0], dict):
                    continue
                delta = (choices[0].get("delta") or {}).get("content", "")
                if isinstance(delta, str) and delta:
                    token_events.append(
                        TokenEvent(
                            idx=len(token_events),
                            t=time.time(),
                            text_len=len(delta),
                        )
                    )
                fr = choices[0].get("finish_reason")
                if isinstance(fr, str) and fr:
                    finish_reason = fr
    finished = time.time()
    return token_events, {
        "model_id": model_id,
        "started_at": started,
        "finished_at": finished,
        "elapsed_ms": (finished - started) * 1000.0,
        "finish_reason": finish_reason,
    }


def _compute_gaps(events: List[TokenEvent]) -> List[GapEvent]:
    gaps: List[GapEvent] = []
    for i in range(1, len(events)):
        t0 = events[i - 1].t
        t1 = events[i].t
        gaps.append(
            GapEvent(
                idx_from=i - 1,
                idx_to=i,
                gap_ms=(t1 - t0) * 1000.0,
                t_from=t0,
                t_to=t1,
            )
        )
    return gaps


def _summarize_gaps(gaps: List[GapEvent], stutter_threshold_ms: float) -> Dict[str, Any]:
    if not gaps:
        return {
            "num_gaps": 0,
            "max_gap_ms": 0.0,
            "p95_gap_ms": 0.0,
            "stutter_threshold_ms": stutter_threshold_ms,
            "stutter_count": 0,
            "stutters": [],
        }
    values = sorted(g.gap_ms for g in gaps)
    p95_idx = min(len(values) - 1, int(0.95 * (len(values) - 1)))
    stutters = [g for g in gaps if g.gap_ms >= stutter_threshold_ms]
    return {
        "num_gaps": len(gaps),
        "max_gap_ms": max(values),
        "p95_gap_ms": values[p95_idx],
        "stutter_threshold_ms": stutter_threshold_ms,
        "stutter_count": len(stutters),
        "stutters": [asdict(s) for s in stutters],
    }


def _collect_log_events(out_q: queue.Queue[LogEvent]) -> List[LogEvent]:
    events: List[LogEvent] = []
    while True:
        try:
            events.append(out_q.get_nowait())
        except queue.Empty:
            break
    return events


def _map_stutters_to_log(
    gaps: List[GapEvent],
    log_events: List[LogEvent],
    stutter_threshold_ms: float,
) -> List[Dict[str, Any]]:
    stutters = [g for g in gaps if g.gap_ms >= stutter_threshold_ms]
    mapped: List[Dict[str, Any]] = []
    for s in stutters:
        hits = [
            e for e in log_events
            if s.t_from <= e.t <= s.t_to
        ]
        mapped.append(
            {
                "gap": asdict(s),
                "log_hits_in_gap_window": [asdict(h) for h in hits],
            }
        )
    return mapped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", type=str, required=True, help="OpenAI-compatible server base URL.")
    p.add_argument("--prompt-file", type=Path, default=Path("demo/prompt.txt"))
    p.add_argument("--prompt-index", type=int, default=0, help="Line index in prompt file.")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=888)
    p.add_argument("--timeout-s", type=float, default=900.0)
    p.add_argument("--backend-log", type=Path, default=None, help="Optional backend log path for trigger correlation.")
    p.add_argument("--stutter-threshold-ms", type=float, default=1500.0)
    p.add_argument("--output-json", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lines = [ln.strip() for ln in args.prompt_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"No usable prompt lines in {args.prompt_file}")
    if args.prompt_index < 0 or args.prompt_index >= len(lines):
        raise RuntimeError(f"prompt-index {args.prompt_index} out of range, total lines={len(lines)}")
    prompt = lines[args.prompt_index]

    log_q: queue.Queue[LogEvent] = queue.Queue()
    watcher: Optional[LogWatcher] = None
    if args.backend_log is not None:
        watcher = LogWatcher(args.backend_log, log_q)
        watcher.start()
        time.sleep(0.2)

    try:
        token_events, run_meta = _stream_tokens(
            base_url=args.base_url,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            timeout_s=args.timeout_s,
        )
    finally:
        if watcher is not None:
            watcher.stop()
            watcher.join(timeout=2.0)

    gaps = _compute_gaps(token_events)
    log_events = _collect_log_events(log_q)
    report: Dict[str, Any] = {
        "run_meta": run_meta,
        "prompt_preview": prompt[:200],
        "token_count": len(token_events),
        "gap_summary": _summarize_gaps(gaps, args.stutter_threshold_ms),
        "log_event_count": len(log_events),
        "log_events": [asdict(e) for e in log_events],
        "stutter_log_alignment": _map_stutters_to_log(
            gaps,
            log_events,
            args.stutter_threshold_ms,
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "token_count": report["token_count"],
        "max_gap_ms": report["gap_summary"]["max_gap_ms"],
        "p95_gap_ms": report["gap_summary"]["p95_gap_ms"],
        "stutter_count": report["gap_summary"]["stutter_count"],
        "log_event_count": report["log_event_count"],
        "output_json": str(args.output_json),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

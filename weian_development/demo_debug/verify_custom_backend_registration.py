#!/usr/bin/env python3
"""Verify whether TriAttention CUSTOM backend is actually registered in serve mode.

This is a diagnostic script only. It launches a short-lived `vllm serve` process
with `--attention-backend CUSTOM` and checks startup logs for:
1) retired plugin marker
2) CUSTOM backend registration failure
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python-bin", type=Path, required=True)
    p.add_argument("--vllm-bin", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--stats-path", type=Path, required=True)
    p.add_argument("--gpu-id", type=str, default="0")
    p.add_argument("--port", type=int, default=18012)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.15)
    p.add_argument("--timeout-s", type=float, default=180.0)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--log-path", type=Path, required=True)
    return p.parse_args()


def _contains_any(lines: List[str], needles: List[str]) -> Dict[str, bool]:
    joined = "\n".join(lines)
    return {n: (n in joined) for n in needles}


def main() -> None:
    args = parse_args()
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu_id,
            "VLLM_PLUGINS": "triattention",
            "TRIATTENTION_STATS_PATH": str(args.stats_path),
            "TRIATTENTION_KV_BUDGET": "256",
            "TRIATTENTION_DIVIDE_LENGTH": "64",
            "TRIATTENTION_WINDOW_SIZE": "64",
            "TRIATTENTION_PRUNING_MODE": "per_head",
            "TRIATTENTION_QUIET": "0",
            "TRIATTENTION_LOG_TRIGGER": "1",
            "TRIATTENTION_LOG_DECISIONS": "1",
        }
    )
    cmd = [
        str(args.vllm_bin),
        "serve",
        str(args.model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "float16",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        "1",
        "--trust-remote-code",
        "--enforce-eager",
        "--max-num-seqs",
        "1",
        "--attention-backend",
        "CUSTOM",
    ]

    with args.log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        deadline = time.time() + args.timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        if proc.poll() is None:
            proc.terminate()
            time.sleep(1)
            if proc.poll() is None:
                proc.kill()

    lines = args.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    patterns = [
        "Legacy V1 backend plugin registration is retired",
        "Backend CUSTOM must be registered before use",
        "Engine core initialization failed",
    ]
    flags = _contains_any(lines, patterns)
    report = {
        "cmd": cmd,
        "return_code": proc.returncode,
        "log_path": str(args.log_path),
        "flags": flags,
        "conclusion": (
            "custom_backend_not_registered_via_retired_plugin"
            if flags[patterns[0]] and flags[patterns[1]]
            else "inconclusive"
        ),
    }
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

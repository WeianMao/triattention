#!/usr/bin/env bash
set -euo pipefail

# Probe an already-running demo gateway and report repetition quality metrics.
# This does NOT start vLLM/gateway; it only sends one long prompt.

DEMO_BASE_URL="${DEMO_BASE_URL:-http://127.0.0.1:8125}"
DATASET_PATH="${DATASET_PATH:-/tmp/tri_diag/openclaw_like_dataset.jsonl}"
MAX_TOKENS="${MAX_TOKENS:-4000}"
OUTPUT_JSON="${OUTPUT_JSON:-/tmp/demo32b_openclaw_gateway_completion_check.json}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi

echo "[probe] demo_url=${DEMO_BASE_URL}"
echo "[probe] dataset=${DATASET_PATH}"
echo "[probe] max_tokens=${MAX_TOKENS}"

DEMO_BASE_URL="${DEMO_BASE_URL}" \
DATASET_PATH="${DATASET_PATH}" \
MAX_TOKENS="${MAX_TOKENS}" \
OUTPUT_JSON="${OUTPUT_JSON}" \
conda run -n trivllm python - <<'PY'
import itertools
import json
import os
import re
import requests

base = os.environ["DEMO_BASE_URL"].rstrip("/")
dataset = os.environ["DATASET_PATH"]
max_tokens = int(os.environ["MAX_TOKENS"])
out_file = os.environ["OUTPUT_JSON"]

prompt = json.loads(open(dataset, "r", encoding="utf-8").readline())["question"]
model = requests.get(f"{base}/v1/models", timeout=30).json()["data"][0]["id"]
body = {
    "model": model,
    "prompt": prompt,
    "max_tokens": max_tokens,
    "temperature": 0.2,
    "top_p": 0.95,
    "stream": False,
}
resp = requests.post(f"{base}/v1/completions", json=body, timeout=1800)
resp.raise_for_status()
obj = resp.json()
open(out_file, "w", encoding="utf-8").write(json.dumps(obj, ensure_ascii=False, indent=2))

text = obj["choices"][0]["text"]

ws = [x for x in re.split(r"\s+", text) if x]
max_ws = 1
cur = 1
prev = None
for tok in ws:
    if tok == prev:
        cur += 1
    else:
        cur = 1
        prev = tok
    if cur > max_ws:
        max_ws = cur

max_char = 1
max_char_c = ""
for c, g in itertools.groupby(text):
    n = sum(1 for _ in g)
    if n > max_char:
        max_char = n
        max_char_c = c

summary = {
    "usage": obj.get("usage"),
    "finish_reason": obj["choices"][0].get("finish_reason"),
    "text_chars": len(text),
    "max_same_ws_run": max_ws,
    "max_same_char_run": max_char,
    "max_same_char": max_char_c,
    "output_json": out_file,
}
print(summary)
print("HEAD:\n" + text[:1000])
print("TAIL:\n" + text[-1000:])
PY

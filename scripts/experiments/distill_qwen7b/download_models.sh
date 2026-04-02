#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RKV_ROOT="$(cd "${EXP_ROOT}/.." && pwd)"

export PYTHONPATH="${RKV_ROOT}:${PYTHONPATH:-}"

DRY_RUN="${DRY_RUN:-0}"
JOB_PARALLEL="${JOB_PARALLEL:-1}"
if ! [[ "${JOB_PARALLEL}" =~ ^[0-9]+$ ]] || [[ "${JOB_PARALLEL}" -lt 1 ]]; then
  echo "JOB_PARALLEL must be a positive integer (got: ${JOB_PARALLEL})" >&2
  exit 1
fi
export DRY_RUN
export JOB_PARALLEL

python - <<'PY'
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts import cli

try:
  from huggingface_hub import snapshot_download
except ImportError as exc:
  raise SystemExit("huggingface_hub is required to download models.") from exc

dry_run = os.environ.get("DRY_RUN", "0") == "1"
job_parallel = int(os.environ.get("JOB_PARALLEL", "1"))

targets = [("DeepSeek-R1-Distill-Qwen-7B", cli.MODEL_SPECS["DeepSeek-R1-Distill-Qwen-7B"])]

cli.MODELS_DIR.mkdir(parents=True, exist_ok=True)

def prepare_download(spec):
  model_name, repo_id = spec
  target_dir = cli.MODELS_DIR / model_name
  if target_dir.exists() and any(target_dir.iterdir()):
    print(f"[skip] {model_name} already present at {target_dir}")
    return
  target_dir.mkdir(parents=True, exist_ok=True)
  print(f"[download] {model_name} -> {target_dir}")
  if dry_run:
    return
  snapshot_download(
    repo_id=repo_id,
    local_dir=str(target_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
  )

with ThreadPoolExecutor(max_workers=job_parallel) as executor:
  futures = [executor.submit(prepare_download, spec) for spec in targets]
  for future in as_completed(futures):
    future.result()
PY

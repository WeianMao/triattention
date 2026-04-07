#!/usr/bin/env python3
"""CLI helpers for the TriAttention experiments wrapper (defaults-driven)."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required to run experiment scripts.") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
EXP_ROOT = REPO_ROOT / "experiments"
CONFIG_ROOT = REPO_ROOT / "triattention" / "configs" / "shared"
DEFAULTS_PATH = CONFIG_ROOT / "defaults.yaml"
BUDGETS_PATH = CONFIG_ROOT / "budgets.yaml"
RUNNER_DEFAULTS_PATH = CONFIG_ROOT / "runner_defaults.yaml"
MODELS_DIR = EXP_ROOT / "models"
LOGS_DIR = EXP_ROOT / "logs"
OUTPUTS_DIR = EXP_ROOT / "outputs"
STATS_DIR = EXP_ROOT / "stats"

MODEL_SPECS: Dict[str, str] = {
    "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
}

DATASETS = ["aime24", "aime25", "math500"]
MODES = ["fullkv", "r1kv", "snapkv", "triattention"]


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_default_budget(model_name: str | None = None) -> int:
    data = load_yaml(DEFAULTS_PATH)
    if "default_budget" not in data:
        raise ValueError(f"default_budget missing in {DEFAULTS_PATH}")
    if model_name is not None:
        by_model = data.get("default_budget_by_model", {})
        if isinstance(by_model, dict) and model_name in by_model:
            return int(by_model[model_name])
    return int(data["default_budget"])


def load_budgets() -> List[int]:
    data = load_yaml(BUDGETS_PATH)
    budgets = data.get("budgets")
    if not isinstance(budgets, list) or not budgets:
        raise ValueError(f"budgets missing in {BUDGETS_PATH}")
    return [int(value) for value in budgets]


def load_runner_defaults() -> dict:
    data = load_yaml(RUNNER_DEFAULTS_PATH)
    if "experiment" not in data or "runner_args" not in data:
        raise ValueError(f"experiment/runner_args missing in {RUNNER_DEFAULTS_PATH}")
    return data


def load_extra_config(extra_paths: List[Path] | None) -> dict:
    if not extra_paths:
        return {}
    merged = {"experiment": {}, "runner_args": {}, "run_tag": None}
    for path in extra_paths:
        data = load_yaml(path)
        if not data:
            continue
        if not isinstance(data, dict):
            raise ValueError(f"extra config must be a mapping: {path}")
        if "run_tag" in data:
            merged["run_tag"] = data.get("run_tag")
        if "experiment" in data or "runner_args" in data:
            experiment = data.get("experiment", {}) or {}
            runner_args = data.get("runner_args", {}) or {}
            if not isinstance(experiment, dict) or not isinstance(runner_args, dict):
                raise ValueError(f"extra config experiment/runner_args must be mappings: {path}")
            merged["experiment"].update(experiment)
            merged["runner_args"].update(runner_args)
        else:
            runner_args = dict(data)
            runner_args.pop("run_tag", None)
            merged["runner_args"].update(runner_args)
    if not merged["experiment"] and not merged["runner_args"] and merged["run_tag"] is None:
        return {}
    return merged


def dataset_max_length(dataset: str, defaults: dict) -> int:
    mapping = defaults.get("dataset_max_length", {})
    if dataset in mapping:
        return int(mapping[dataset])
    if dataset in {"aime24", "aime25"}:
        return 32768
    return 8192


def resolve_dataset_path(dataset: str) -> Path:
    candidates = [
        PROJECT_ROOT / f"{dataset}.jsonl",
        REPO_ROOT / "data" / f"{dataset}.jsonl",
    ]
    if dataset == "math500":
        candidates.extend(
            [
                PROJECT_ROOT / "math.jsonl",
                REPO_ROOT / "data" / "math.jsonl",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            if candidate.name != f"{dataset}.jsonl":
                print(
                    f"[warn] dataset {dataset} resolved to {candidate}",
                    file=sys.stderr,
                )
            return candidate
    hint = PROJECT_ROOT / f"{dataset}.jsonl"
    raise FileNotFoundError(
        f"Dataset file not found for {dataset}. Expected symlink at {hint}."
    )


def resolve_model_path(model_name: str) -> Path:
    return MODELS_DIR / model_name


def stats_path_for(
    dataset: str,
    model_name: str,
    budget: int,
    *,
    announce: bool = True,
) -> Path:
    # Keep stats dataset distinct from evaluation dataset.
    stats_dataset = "aime25" if dataset == "aime24" else "aime24"
    if announce:
        sys.stderr.write(
            f"[info] triattention stats dataset: eval={dataset} stats={stats_dataset}\n"
        )
    return STATS_DIR / stats_dataset / model_name / f"stats_budget_{budget}.pt"


def budget_tag(mode: str, budget: int | None) -> str:
    if mode == "fullkv":
        return "full"
    if budget is None:
        raise ValueError("budget is required for non-fullkv modes")
    return f"budget_{budget}"


def resolve_num_samples(runner_args: dict, dataset: str | None = None) -> int:
    value = runner_args.get("num_samples", 64)
    per_dataset = runner_args.get("num_samples_by_dataset", {})
    if dataset and isinstance(per_dataset, dict) and dataset in per_dataset:
        value = per_dataset[dataset]
    try:
        return int(value)
    except (TypeError, ValueError):
        return 64


def sample_tag(num_samples: int) -> str:
    return f"sample{num_samples}"

def sanitize_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    if not cleaned:
        return "custom"
    return cleaned[:64]

def tag_with_suffix(tag: str, suffix: str | None) -> str:
    if not suffix:
        return tag
    return f"{tag}_{sanitize_tag(suffix)}"


def resolve_stats_path(value: str) -> Path:
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable in stats path: {value}")
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = (EXP_ROOT / path).resolve()
    return path


def resolve_stats_override(
    extra_config: dict | None,
    stats_path_arg: str | None,
) -> Path | None:
    if stats_path_arg:
        return resolve_stats_path(stats_path_arg)
    if not extra_config:
        return None
    runner_args = extra_config.get("runner_args", {})
    if not isinstance(runner_args, dict):
        return None
    value = runner_args.get("triattention_stats_file")
    if not value:
        return None
    return resolve_stats_path(str(value))

def resolve_run_tag(extra_config: dict | None, run_tag_arg: str | None) -> str | None:
    if run_tag_arg:
        return sanitize_tag(str(run_tag_arg))
    if not extra_config:
        return None
    value = extra_config.get("run_tag")
    if value is None:
        return None
    return sanitize_tag(str(value))


def config_output_path(
    dataset: str,
    model_name: str,
    mode: str,
    budget: int | None,
    run_tag: str | None,
) -> Path:
    slug = model_name.lower().replace("/", "-").replace(" ", "-")
    tag = tag_with_suffix(budget_tag(mode, budget), run_tag)
    return EXP_ROOT / "configs" / "generated" / dataset / slug / f"{mode}_{tag}.yaml"


def apply_defaults(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    merged.update(overrides)
    return merged


def build_config(
    dataset: str,
    dataset_path: Path,
    model_name: str,
    model_path: Path,
    mode: str,
    budget: int | None,
    stats_path: Path | None,
    run_tag: str | None,
    defaults: dict,
    extra_config: dict | None,
) -> dict:
    tag = tag_with_suffix(budget_tag(mode, budget), run_tag)
    exp_defaults = defaults.get("experiment", {})
    runner_defaults = defaults.get("runner_args", {})
    num_samples = resolve_num_samples(runner_defaults, dataset)
    sample_dir = sample_tag(num_samples)
    log_dir = LOGS_DIR / dataset / model_name / sample_dir / mode / tag
    output_dir = OUTPUTS_DIR / dataset / model_name / sample_dir / mode / tag

    experiment = apply_defaults(
        exp_defaults,
        {
            "name": f"{dataset}_{model_name}_{mode}_{tag}",
            "log_dir": str(log_dir),
            "method_output_dir": str(output_dir),
        },
    )
    runner_args = apply_defaults(
        runner_defaults,
        {
            "output_dir": str(output_dir / "shards"),
            "dataset_path": str(dataset_path),
            "model_path": str(model_path),
            "max_length": dataset_max_length(dataset, defaults),
            "method": mode,
            "kv_budget": budget,
        },
    )
    runner_args["num_samples"] = num_samples
    runner_args.pop("num_samples_by_dataset", None)

    if extra_config:
        extra_experiment = extra_config.get("experiment", {})
        extra_runner_args = extra_config.get("runner_args", {})
        if not isinstance(extra_experiment, dict) or not isinstance(extra_runner_args, dict):
            raise ValueError("extra config experiment/runner_args must be mappings")
        experiment = apply_defaults(experiment, extra_experiment)
        runner_args = apply_defaults(runner_args, extra_runner_args)

    if mode == "fullkv":
        runner_args["kv_budget"] = None
    if mode == "triattention":
        if stats_path is None and "triattention_stats_file" not in runner_args:
            raise ValueError("stats_path is required for triattention mode")
        runner_args.setdefault("triattention_stats_file", str(stats_path) if stats_path else None)
        if "per_head_pruning" not in runner_args and "per_layer_perhead_pruning" not in runner_args:
            runner_args["per_head_pruning"] = True
        runner_args.setdefault("count_prompt_tokens", True)
        runner_args.setdefault("attention_layer_compression", True)
        runner_args.setdefault("slack_budget_trigger", True)
        runner_args.setdefault("triattention_normalize_scores", True)
        runner_args.setdefault("divide_length", 128)
        runner_args.setdefault("window_size", 128)
        runner_args.setdefault("round_window", 32)
        runner_args.setdefault("triattention_frequency_window", 65536)
        runner_args.setdefault("triattention_score_aggregation", "mean")
        runner_args.setdefault("pruning_seed", 0)

    return {"experiment": {**experiment, "runner_args": runner_args}}


def write_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def ensure_run_log(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log = log_dir / "run.log"
    shard_logs = sorted(log_dir.glob("*.log"))
    if not shard_logs:
        return
    if len(shard_logs) == 1:
        shutil.copyfile(shard_logs[0], run_log)
        return
    with run_log.open("w", encoding="utf-8") as handle:
        for shard_log in shard_logs:
            handle.write(f"=== {shard_log.name} ===\n")
            handle.write(shard_log.read_text(encoding="utf-8"))
            handle.write("\n")


def dispatch_run(config_path: Path, dataset: str, log_dir: Path, dry_run: bool) -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{pythonpath}" if pythonpath else str(REPO_ROOT)


    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "dispatch.py"),
        "--config",
        str(config_path),
        "--dataset",
        dataset,
    ]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return
    subprocess.check_call(cmd, cwd=str(REPO_ROOT), env=env)
    ensure_run_log(log_dir)


def validate_model_exists(model_name: str, dry_run: bool) -> Path:
    model_path = resolve_model_path(model_name)
    if dry_run:
        print(f"[dry-run] model path: {model_path}")
        return model_path
    if not model_path.exists() or not any(model_path.iterdir()):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run scripts/download_models_v2.sh first."
        )
    return model_path


def run_one(
    dataset: str,
    model_name: str,
    mode: str,
    budget: int | None,
    *,
    require_stats: bool,
    stats_path_arg: str | None,
    run_tag: str | None,
    defaults: dict,
    extra_config: dict | None,
    dry_run: bool,
) -> None:
    dataset_path = resolve_dataset_path(dataset)
    model_path = validate_model_exists(model_name, dry_run)
    tag = budget_tag(mode, budget)
    runner_defaults = defaults.get("runner_args", {})
    num_samples = resolve_num_samples(runner_defaults, dataset)
    sample_dir = sample_tag(num_samples)
    resolved_run_tag = resolve_run_tag(extra_config, run_tag)

    stats_path = None
    if mode == "triattention":
        if budget is None:
            raise ValueError("budget is required for triattention runs")
        stats_override = resolve_stats_override(extra_config, stats_path_arg)
        if stats_override is None:
            stats_path = stats_path_for(dataset, model_name, budget)
        else:
            stats_path = stats_override
            sys.stderr.write(f"[info] triattention stats override: {stats_path}\n")
        if require_stats and stats_path and not stats_path.exists():
            if dry_run:
                print(
                    f"[dry-run] missing stats for {dataset}/{model_name}/budget {budget}: {stats_path}",
                    file=sys.stderr,
                )
            else:
                raise FileNotFoundError(
                    f"TriAttention stats missing for {dataset}/{model_name}/budget {budget}. "
                    f"Run scripts/experiments/build_all_stats.sh first."
                )

    tag = tag_with_suffix(tag, resolved_run_tag)
    log_dir = LOGS_DIR / dataset / model_name / sample_dir / mode / tag
    output_dir = OUTPUTS_DIR / dataset / model_name / sample_dir / mode / tag

    config_path = config_output_path(dataset, model_name, mode, budget, resolved_run_tag)
    config = build_config(
        dataset,
        dataset_path,
        model_name,
        model_path,
        mode,
        budget,
        stats_path,
        resolved_run_tag,
        defaults,
        extra_config,
    )
    write_config(config, config_path)
    dispatch_run(config_path, dataset, log_dir, dry_run)

    output_dir.mkdir(parents=True, exist_ok=True)


def run_defaults(dry_run: bool) -> None:
    defaults = load_runner_defaults()
    for dataset in DATASETS:
        for model_name in MODEL_SPECS.keys():
            default_budget = load_default_budget(model_name)
            run_one(
                dataset,
                model_name,
                "fullkv",
                None,
                require_stats=False,
                stats_path_arg=None,
                run_tag=None,
                defaults=defaults,
                extra_config=None,
                dry_run=dry_run,
            )
            run_one(
                dataset,
                model_name,
                "r1kv",
                default_budget,
                require_stats=False,
                stats_path_arg=None,
                run_tag=None,
                defaults=defaults,
                extra_config=None,
                dry_run=dry_run,
            )
            run_one(
                dataset,
                model_name,
                "triattention",
                default_budget,
                require_stats=True,
                stats_path_arg=None,
                run_tag=None,
                defaults=defaults,
                extra_config=None,
                dry_run=dry_run,
            )


def run_sweep(dry_run: bool) -> None:
    defaults = load_runner_defaults()
    budgets = load_budgets()
    for dataset in DATASETS:
        for model_name in MODEL_SPECS.keys():
            for budget in budgets:
                run_one(
                    dataset,
                    model_name,
                    "r1kv",
                    budget,
                    require_stats=False,
                    stats_path_arg=None,
                    run_tag=None,
                    defaults=defaults,
                    extra_config=None,
                    dry_run=dry_run,
                )
            for budget in budgets:
                run_one(
                    dataset,
                    model_name,
                    "triattention",
                    budget,
                    require_stats=True,
                    stats_path_arg=None,
                    run_tag=None,
                    defaults=defaults,
                    extra_config=None,
                    dry_run=dry_run,
                )


def has_trace_data(trace_root: Path) -> bool:
    merged = trace_root / "merged" / "merged.jsonl"
    if merged.exists():
        return True
    shards = trace_root / "shards"
    if shards.exists() and any(shards.glob("*.jsonl")):
        return True
    if any(trace_root.glob("*.jsonl")):
        return True
    return False


def normalize_selection(
    selected: List[str] | None, allowed: List[str], kind: str
) -> List[str]:
    if not selected:
        return list(allowed)
    allowed_set = set(allowed)
    ordered: List[str] = []
    seen: set[str] = set()
    for value in selected:
        if value not in allowed_set:
            raise ValueError(f"Unsupported {kind}: {value}")
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_stats(
    dry_run: bool,
    models: List[str] | None = None,
    input_file: str | None = None,
    output_dir: str | None = None,
    max_length: int = 32768,
    job_parallel: int = 1,
) -> None:
    if job_parallel < 1:
        raise ValueError("job_parallel must be >= 1")

    if input_file is None:
        raise SystemExit(
            "build-stats requires --input pointing to a plain text calibration file.\n"
            "Example: python scripts/cli.py build-stats --input calibration_text.txt"
        )
    input_path = Path(input_file)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    out_dir = Path(output_dir) if output_dir else REPO_ROOT / "triattention" / "calibration"
    model_list = normalize_selection(models, list(MODEL_SPECS.keys()), "model")

    commands: List[Dict[str, object]] = []

    for model_name in model_list:
        model_path = validate_model_exists(model_name, dry_run)
        stats_path = out_dir / f"{model_name.lower().replace('-', '_')}_stats.pt"
        if stats_path.exists():
            print(f"[skip] Stats already exist: {stats_path}")
            continue
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "calibrate.py"),
            "--model",
            str(model_path),
            "--input",
            str(input_path),
            "--output",
            str(stats_path),
            "--max-length",
            str(max_length),
            "--device",
            "cuda",
            "--attn-implementation",
            "flash_attention_2",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".strip(":")
        commands.append(
            {
                "cmd": cmd,
                "cwd": str(REPO_ROOT),
                "env": env,
                "label": model_name,
            }
        )

    if not commands:
        print("[info] No pending stats jobs for requested targets.")
        return

    if dry_run:
        print(f"[dry-run] job_parallel={job_parallel}")
        batch_id = 1
        for idx in range(0, len(commands), job_parallel):
            labels = ", ".join(
                info["label"]  # type: ignore[index]
                for info in commands[idx : idx + job_parallel]
            )
            print(f"[dry-run] batch {batch_id}: {labels}")
            batch_id += 1
        for info in commands:
            cmd_str = " ".join(info["cmd"])  # type: ignore[index]
            print(f"[dry-run] {cmd_str}")
        return

    running: List[Tuple[subprocess.Popen, str]] = []

    def wait_for_first() -> None:
        if not running:
            return
        proc, label = running.pop(0)
        ret = proc.wait()
        if ret != 0:
            raise SystemExit(f"[error] Stats job {label} failed with status {ret}")

    for info in commands:
        proc = subprocess.Popen(info["cmd"], cwd=info["cwd"], env=info["env"])  # type: ignore[arg-type]
        running.append((proc, info["label"]))  # type: ignore[index]
        if len(running) >= job_parallel:
            wait_for_first()

    while running:
        wait_for_first()


def download_models() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required to download models.") from exc

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, repo_id in MODEL_SPECS.items():
        target_dir = MODELS_DIR / model_name
        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"[skip] {model_name} already present at {target_dir}")
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[download] {model_name} -> {target_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )


def resolve_budget_for_mode(mode: str, budget: int | None, model_name: str | None = None) -> int | None:
    if mode == "fullkv":
        return None
    if budget is not None:
        return int(budget)
    return load_default_budget(model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download-models", help="Download all required models.")
    subparsers.add_parser("run-default", help="Run all default-budget experiments.")
    subparsers.add_parser("run-sweep", help="Run all budget sweep experiments.")
    build_stats_parser = subparsers.add_parser(
        "build-stats", help="Calibrate TriAttention stats from plain text input."
    )
    build_stats_parser.add_argument(
        "--input",
        required=True,
        help="Plain text file for calibration input.",
    )
    build_stats_parser.add_argument(
        "--model",
        action="append",
        choices=list(MODEL_SPECS.keys()),
        help="Models to include (repeatable). Defaults to all.",
    )
    build_stats_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for stats files (default: calibration/).",
    )
    build_stats_parser.add_argument(
        "--max-length",
        type=int,
        default=32768,
        help="Maximum token length for calibration (default: 32768).",
    )
    build_stats_parser.add_argument(
        "--job-parallel",
        type=int,
        default=1,
        help="Maximum concurrent stats jobs.",
    )

    run_one_parser = subparsers.add_parser("run-one", help="Run a single dataset/model/method/budget.")
    run_one_parser.add_argument("--dataset", required=True, choices=DATASETS)
    run_one_parser.add_argument("--model", required=True, choices=MODEL_SPECS.keys())
    run_one_parser.add_argument("--method", required=True, choices=MODES)
    run_one_parser.add_argument("--budget", type=int, default=None)
    run_one_parser.add_argument(
        "--stats-path",
        default=None,
        help="Override TriAttention stats path (supports env vars).",
    )
    run_one_parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional suffix for output/log/config dirs to avoid collisions.",
    )
    run_one_parser.add_argument(
        "--extra-config",
        action="append",
        default=None,
        help="YAML config overrides to merge into runner_args or experiment.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "download-models":
        download_models()
        return
    if args.command == "run-default":
        run_defaults(args.dry_run)
        return
    if args.command == "run-sweep":
        run_sweep(args.dry_run)
        return
    if args.command == "build-stats":
        build_stats(
            args.dry_run,
            models=args.model,
            input_file=args.input,
            output_dir=args.output_dir,
            max_length=args.max_length,
            job_parallel=args.job_parallel,
        )
        return
    if args.command == "run-one":
        defaults = load_runner_defaults()
        budget = resolve_budget_for_mode(args.method, args.budget, args.model)
        extra_config = load_extra_config(
            [Path(path) for path in (args.extra_config or [])]
        )
        run_one(
            args.dataset,
            args.model,
            args.method,
            budget,
            require_stats=(args.method == "triattention"),
            stats_path_arg=args.stats_path,
            run_tag=args.run_tag,
            defaults=defaults,
            extra_config=extra_config,
            dry_run=args.dry_run,
        )
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

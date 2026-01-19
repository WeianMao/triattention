#!/usr/bin/env python3
"""CLI helpers for the R-KV SpeckV experiments wrapper (defaults-driven)."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required to run speckv_experiments scripts.") from exc

RKV_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RKV_ROOT.parent
EXP_ROOT = RKV_ROOT / "speckv_experiments"
CONFIG_ROOT = EXP_ROOT / "configs" / "shared"
DEFAULTS_PATH = CONFIG_ROOT / "defaults.yaml"
BUDGETS_PATH = CONFIG_ROOT / "budgets.yaml"
RUNNER_DEFAULTS_PATH = CONFIG_ROOT / "runner_defaults.yaml"
MODELS_DIR = EXP_ROOT / "models"
LOGS_DIR = EXP_ROOT / "logs"
OUTPUTS_DIR = EXP_ROOT / "outputs"
STATS_DIR = EXP_ROOT / "stats"

MODEL_SPECS: Dict[str, str] = {
    "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Qwen-14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
}

DATASETS = ["aime24", "aime25", "math500"]
MODES = ["fullkv", "rkv", "snapkv", "speckv"]


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_default_budget() -> int:
    data = load_yaml(DEFAULTS_PATH)
    if "default_budget" not in data:
        raise ValueError(f"default_budget missing in {DEFAULTS_PATH}")
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
    merged = {"experiment": {}, "runner_args": {}}
    for path in extra_paths:
        data = load_yaml(path)
        if not data:
            continue
        if not isinstance(data, dict):
            raise ValueError(f"extra config must be a mapping: {path}")
        if "experiment" in data or "runner_args" in data:
            experiment = data.get("experiment", {}) or {}
            runner_args = data.get("runner_args", {}) or {}
            if not isinstance(experiment, dict) or not isinstance(runner_args, dict):
                raise ValueError(f"extra config experiment/runner_args must be mappings: {path}")
            merged["experiment"].update(experiment)
            merged["runner_args"].update(runner_args)
        else:
            merged["runner_args"].update(data)
    if not merged["experiment"] and not merged["runner_args"]:
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
        RKV_ROOT / "HuggingFace" / "data" / f"{dataset}.jsonl",
    ]
    if dataset == "math500":
        candidates.extend(
            [
                PROJECT_ROOT / "math.jsonl",
                RKV_ROOT / "HuggingFace" / "data" / "math.jsonl",
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


def stats_path_for(dataset: str, model_name: str, budget: int) -> Path:
    # Keep stats dataset distinct from evaluation dataset.
    stats_dataset = "aime25" if dataset == "aime24" else "aime24"
    sys.stderr.write(
        f"[info] speckv stats dataset: eval={dataset} stats={stats_dataset}\n"
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


def config_output_path(dataset: str, model_name: str, mode: str, budget: int | None) -> Path:
    slug = model_name.lower().replace("/", "-").replace(" ", "-")
    tag = budget_tag(mode, budget)
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
    defaults: dict,
    extra_config: dict | None,
) -> dict:
    tag = budget_tag(mode, budget)
    exp_defaults = defaults.get("experiment", {})
    runner_defaults = defaults.get("runner_args", {})
    num_samples = resolve_num_samples(runner_defaults, dataset)
    sample_dir = sample_tag(num_samples)
    log_dir = LOGS_DIR / dataset / model_name / sample_dir / mode / tag
    output_dir = OUTPUTS_DIR / dataset / model_name / sample_dir / mode / tag

    experiment = apply_defaults(
        exp_defaults,
        {
            "name": f"speckv_{dataset}_{model_name}_{mode}_{tag}",
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
    if mode == "speckv":
        if stats_path is None and "sparse_stats_path" not in runner_args:
            raise ValueError("stats_path is required for speckv mode")
        runner_args.setdefault("sparse_stats_path", str(stats_path) if stats_path else None)
        if "per_head_pruning" not in runner_args and "per_layer_perhead_pruning" not in runner_args:
            runner_args["per_head_pruning"] = True
        runner_args.setdefault("include_prefill_in_budget", True)
        runner_args.setdefault("rkv_style_compression", True)
        runner_args.setdefault("rkv_style_slack_trigger", True)
        runner_args.setdefault("sparse_normalize_scores", True)
        runner_args.setdefault("divide_length", 128)
        runner_args.setdefault("window_size", 128)
        runner_args.setdefault("sparse_round_window", 32)
        runner_args.setdefault("sparse_offset_max_length", 65536)
        runner_args.setdefault("sparse_score_aggregation", "mean")
        runner_args.setdefault("sparse_head_limit", -1)
        runner_args.setdefault("sparse_seed", 0)

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
    env["PYTHONPATH"] = f"{RKV_ROOT}:{pythonpath}" if pythonpath else str(RKV_ROOT)
    env.setdefault("VLLM_PROCESS_NAME_PREFIX", "PD-L1_binder")

    cmd = [
        sys.executable,
        str(RKV_ROOT / "weian_development" / "rkv_sharded_dispatch.py"),
        "--config",
        str(config_path),
        "--dataset",
        dataset,
    ]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return
    subprocess.check_call(cmd, cwd=str(RKV_ROOT), env=env)
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
    log_dir = LOGS_DIR / dataset / model_name / sample_dir / mode / tag
    output_dir = OUTPUTS_DIR / dataset / model_name / sample_dir / mode / tag

    stats_path = None
    if mode == "speckv":
        if budget is None:
            raise ValueError("budget is required for speckv runs")
        stats_path = stats_path_for(dataset, model_name, budget)
        if require_stats and not stats_path.exists():
            if dry_run:
                print(
                    f"[dry-run] missing stats for {dataset}/{model_name}/budget {budget}: {stats_path}",
                    file=sys.stderr,
                )
            else:
                raise FileNotFoundError(
                    f"SpeckV stats missing for {dataset}/{model_name}/budget {budget}. "
                    f"Run scripts/build_all_stats_v2.sh first."
                )

    config_path = config_output_path(dataset, model_name, mode, budget)
    config = build_config(
        dataset,
        dataset_path,
        model_name,
        model_path,
        mode,
        budget,
        stats_path,
        defaults,
        extra_config,
    )
    write_config(config, config_path)
    dispatch_run(config_path, dataset, log_dir, dry_run)

    output_dir.mkdir(parents=True, exist_ok=True)


def run_defaults(dry_run: bool) -> None:
    defaults = load_runner_defaults()
    default_budget = load_default_budget()
    for dataset in DATASETS:
        for model_name in MODEL_SPECS.keys():
            run_one(
                dataset,
                model_name,
                "fullkv",
                None,
                require_stats=False,
                defaults=defaults,
                extra_config=None,
                dry_run=dry_run,
            )
            run_one(
                dataset,
                model_name,
                "rkv",
                default_budget,
                require_stats=False,
                defaults=defaults,
                extra_config=None,
                dry_run=dry_run,
            )
            run_one(
                dataset,
                model_name,
                "speckv",
                default_budget,
                require_stats=True,
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
                    "rkv",
                    budget,
                    require_stats=False,
                    defaults=defaults,
                    extra_config=None,
                    dry_run=dry_run,
                )
            for budget in budgets:
                run_one(
                    dataset,
                    model_name,
                    "speckv",
                    budget,
                    require_stats=True,
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
    datasets: List[str] | None = None,
    models: List[str] | None = None,
    job_parallel: int = 1,
) -> None:
    if job_parallel < 1:
        raise ValueError("job_parallel must be >= 1")

    defaults = load_runner_defaults()
    runner_defaults = defaults.get("runner_args", {})
    dataset_list = normalize_selection(datasets, DATASETS, "dataset")
    model_list = normalize_selection(models, list(MODEL_SPECS.keys()), "model")

    budgets = load_budgets()
    missing_fullkv: List[Tuple[str, str, Path]] = []
    commands: List[Dict[str, object]] = []

    for dataset in dataset_list:
        num_samples = resolve_num_samples(runner_defaults, dataset)
        sample_dir = sample_tag(num_samples)
        num_traces = 30  # Use 30 traces for stats building
        for model_name in model_list:
            fullkv_root = OUTPUTS_DIR / dataset / model_name / sample_dir / "fullkv" / "full"
            if not fullkv_root.exists() or not has_trace_data(fullkv_root):
                missing_fullkv.append((dataset, model_name, fullkv_root))
                continue

            model_path = validate_model_exists(model_name, dry_run)
            for budget in budgets:
                stats_path = stats_path_for(dataset, model_name, budget)
                if stats_path.exists():
                    continue
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(RKV_ROOT / "weian_development" / "rkv_sparse_round_calibrate.py"),
                    "--trace-root",
                    str(fullkv_root),
                    "--model-path",
                    str(model_path),
                    "--output-path",
                    str(stats_path),
                    "--num-traces",
                    str(num_traces),
                    "--attn-implementation",
                    "flash_attention_2",
                    "--dtype",
                    "bfloat16",
                    "--kv-budget",
                    str(budget),
                ]
                env = os.environ.copy()
                env.setdefault("VLLM_PROCESS_NAME_PREFIX", "PD-L1_binder")
                env["PYTHONPATH"] = f"{RKV_ROOT}:{env.get('PYTHONPATH', '')}".strip(":")
                commands.append(
                    {
                        "cmd": cmd,
                        "cwd": str(RKV_ROOT),
                        "env": env,
                        "label": f"{dataset}/{model_name}/budget_{budget}",
                    }
                )

    if missing_fullkv:
        for dataset, model_name, path in missing_fullkv:
            print(
                f"[error] Missing fullkv outputs for {dataset}/{model_name}: {path}",
                file=sys.stderr,
            )
        if not dry_run:
            raise SystemExit("FullKV outputs missing. Run scripts/run_all_default_v2.sh first.")

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


def resolve_budget_for_mode(mode: str, budget: int | None) -> int | None:
    if mode == "fullkv":
        return None
    if budget is not None:
        return int(budget)
    return load_default_budget()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download-models", help="Download all required models.")
    subparsers.add_parser("run-default", help="Run all default-budget experiments.")
    subparsers.add_parser("run-sweep", help="Run all budget sweep experiments.")
    build_stats_parser = subparsers.add_parser(
        "build-stats", help="Build SpeckV stats for all datasets/models."
    )
    build_stats_parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASETS,
        help="Datasets to include (repeatable). Defaults to all.",
    )
    build_stats_parser.add_argument(
        "--model",
        action="append",
        choices=list(MODEL_SPECS.keys()),
        help="Models to include (repeatable). Defaults to all.",
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
            datasets=args.dataset,
            models=args.model,
            job_parallel=args.job_parallel,
        )
        return
    if args.command == "run-one":
        defaults = load_runner_defaults()
        budget = resolve_budget_for_mode(args.method, args.budget)
        extra_config = load_extra_config(
            [Path(path) for path in (args.extra_config or [])]
        )
        run_one(
            args.dataset,
            args.model,
            args.method,
            budget,
            require_stats=(args.method == "speckv"),
            defaults=defaults,
            extra_config=extra_config,
            dry_run=args.dry_run,
        )
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

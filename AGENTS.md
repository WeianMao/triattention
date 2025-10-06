# Repository Guidelines

## Project Structure & Module Organization
- `deepconf/`: core library wrapping vLLM with DeepThink features; update `wrapper.py`, `utils.py`, or `processors.py` for algorithm changes.
- `examples/`: runnable reference scripts (`example_online.py`, `example_offline.py`, etc.) demonstrating online/offline workflows and analysis helpers.
- `scripts/`: automation tooling, including YAML config loader (`scripts/config_loader.py`), preset configs (`scripts/configs/`), and GPU dispatch/run wrappers (`scripts/yaml_runs/`).
- `docs/`: living documentation such as `deepconf_migration_notes.md` and component checklists.
- Datasets are expected via symlinks in repo root (e.g., `aime25.jsonl`, `brumo_2025.jsonl`).

## Build, Test, and Development Commands
- `python -m compileall deepconf examples`: quick syntax sanity check without running inference.
- `python examples/example_online.py --help`: inspect runtime switches for online DeepConf runs.
- `python scripts/yaml_runs/run_online_deepseek.sh --rid 0 --gpus 0,1`: launch multi-GPU online experiments using default YAML.
- Use `conda run -n dc python -m pip install -r requirements.txt` to sync dependencies inside the shared `dc` environment.
- Default conda environment for development is named `dc`; activate it before running tooling or scripts.

## Coding Style & Naming Conventions
- Python code targets 3.9+, uses 4-space indentation, descriptive snake_case for variables/functions, and CapWords for classes.
- Preserve ASCII unless existing files require UTF-8 text.
- Follow minimal docstrings and inline comments only when clarifying non-obvious logic; avoid restating the code.

## Testing Guidelines
- Prefer lightweight smoke checks that reuse example scripts with reduced budgets (e.g., `--qids 0 --max-workers 1`).
- Add targeted unit tests with `pytest` under a new `tests/` directory when introducing reusable utilities; mirror filename + `_test.py` naming.
- For features touching logprob persistence, validate both default (enabled) and `--no_store_logprobs` flows.

## Commit & Pull Request Guidelines
- Use Conventional Commit prefixes (e.g., `feat:`, `fix:`, `docs:`) as seen in existing history: `feat: add optional logprob persistence for offline runs`.
- Squash work into coherent commits with descriptive bodies summarizing key changes and verification steps.
- Pull requests should include: concise summary, relevant commands/run logs, configuration diffs (YAML or scripts), and mention of affected datasets or output directories.

## Agent-Specific Notes
- Honor existing YAML schema; add new keys under `common` or mode-specific sections to preserve backwards compatibility.
- Default scripts now store raw logprobs—remind users about disk impact and document any overrides (`--no_store_logprobs`).

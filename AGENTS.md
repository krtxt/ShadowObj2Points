# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core package code (Lightning modules, data modules, callbacks, utils). Hydra-configured model entry points live under `src/models`.
- `configs/`: Hydra defaults and experiment overrides (model, backbone, velocity_strategy, trainer, logger). `configs/config.yaml` is the main entry.
- `tests/`: pytest suites; follows `test_*.py` naming.
- `assets/`, `data/`, `outputs/`: input assets, raw/preprocessed data, and Hydra run artifacts/checkpoints.
- `train.py`: Lightning + Hydra training driver; ensures `src` is on `PYTHONPATH`.
- `notebooks/`, `docs/`, `benchmark_sampling_speed.py`: exploratory analysis, docs, and perf checks.

## Build, Test, and Development Commands
- Install (dev): `python -m pip install -e ".[dev]"` from repo root for editable package + lint/test tooling.
- Run training: `python train.py model=flow_matching_hand_dit backbone=ptv3_sparse datamodule=handencoder_dm_dex experiment_name=my_run` (override configs as needed; outputs to `outputs/<name>/<timestamp>`).
- Profile variants: use `experiments=profiler` override; resume via `auto_resume=true` in config.
- Tests: `pytest` or `pytest tests/test_xyz.py -k keyword` to target subsets.
- Format: `black .` (line length 100); lint optional via `flake8` and type-check with `mypy` when touching typed modules.

## Coding Style & Naming Conventions
- Language: Python 3.8+; 4-space indent; prefer type hints for public functions/classes.
- Formatting: `black` config in `pyproject.toml` (100 chars). Keep imports grouped/ordered; avoid unused symbols to pass `flake8`.
- Naming: snake_case for functions/vars, PascalCase for classes, UPPER_SNAKE for constants. Config group names mirror filenames under `configs/`.
- Docstrings: concise summaries for Lightning modules, data modules, and utils; include shape info for tensors where non-obvious.

## Testing Guidelines
- Framework: pytest. Place new tests under `tests/` with `test_*.py` files and `Test*` classes or `test_*` functions.
- Cover: add unit tests for data transforms, model utilities, and any new callbacks; prefer deterministic seeds (`seed_everything`) in Lightning tests.
- Running tips: use `pytest -q` for quick checks; `pytest --maxfail=1` to triage failures fast.

## Commit & Pull Request Guidelines
- Commit messages: follow the current style (`feat: ...`, `refactor: ...`, short imperative). Keep summaries under ~72 chars; include scoped Chinese/English descriptions if helpful.
- PRs: describe purpose, key config overrides, and expected metrics or qualitative results. Link issues/experiments, attach relevant Hydra command lines, and include tensorboard screenshots or sample outputs when UI/vis changes are involved.
- Checklist: code formatted, tests (or targeted subset) run, configs added to `configs/` with sensible defaults, and new assets documented.

项目对应的conda环境是cure

# Repository Guidelines

This document describes how to work effectively in the `shadowObj2Points` repository.

## Project Structure & Modules

- Core Python code lives in `src/` (e.g., `src/datamodules` for data loading and hand keypoint datasets).
- Training and experiment entry points are in `train.py` and `bin/` (e.g., `bin/attention.py`, `bin/train_legacy.py`).
- Configuration files are in `configs/` (Hydra/YAML configs for models, data modules, trainers, and experiments).
- Notebooks for exploration and debugging are under `notebooks/`.
- Large or generated artifacts should go into `data/` or `outputs/`, not into `src/`.

## Build, Test, and Development Commands

- Create and activate the environment from `environment.yml` (e.g., `conda env create -f environment.yml`).
- Install the package in editable mode with `pip install -e .` from the repo root.
- Run training via `python train.py` with Hydra overrides, for example:
  - `python train.py experiment=full_training`
- Run ad‑hoc scripts from `bin/` directly, e.g. `python bin/hand_points_reconstruction.py`.

## Coding Style & Naming Conventions

- Use Python 3 with 4‑space indentation and type hints where reasonable.
- Prefer descriptive snake_case for functions, variables, and modules; use PascalCase for classes.
- Follow existing patterns in `src/` for module layout and imports; avoid circular imports.
- Keep configuration names consistent with existing YAML files (e.g., `handencoder_dm_*`, `flow_matching_*`).

## Testing Guidelines

- Use `pytest` from the repository root (`pytest` or `pytest path/to/test_*.py`) when tests are present.
- Name test files `test_*.py` and test functions `test_*`.
- When adding new behavior in `src/`, add or update tests that cover the change and run them locally.

## Commit & Pull Request Guidelines

- Write concise, imperative commit messages (e.g., `Add hand encoder datamodule`, `Fix flow matching loss`).
- Keep changes focused; separate refactors from feature additions when possible.
- Pull requests should include: a short summary, key implementation details, any relevant configs or scripts, and before/after metrics or screenshots when applicable.
- Mention affected configs (e.g., `configs/experiment/full_training.yaml`) and entry points in the PR description so others can reproduce your results.

## Agent-Specific Instructions

- When modifying files, respect these guidelines and keep changes minimal and well scoped.
- Do not commit large data files or outputs; prefer referencing paths under `data/` or `outputs/` instead.

# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the core pipeline: datasets under `src/data`, training logic in `src/trainer_iterative.py`, and model wrappers in `src/model`.
- `configs/` hosts YAML presets (e.g., `configs/cirr_iterative.yaml`) that define dataset roots, augmentation knobs, and iteration schedules.
- Shell entry points such as `run_iterative_training_paratuning.sh`, `test_multiGPU_parallel.sh`, and `eval_cirr.sh` live at the repository root for reproducible workflows.
- `experiments/` stores generated checkpoints (`iteration_*`), cached negatives, and WANDB logs; keep each run isolated per experiment folder.
- Place analysis outputs in results/ and the cirr_test.json file under submission/ to keep version control lean.

## Build, Test, and Development Commands
- Create the baseline environment with `conda env create -f environment.yml` or, if the env already exists, `pip install -r requirements.txt`.
- Launch full training via `./run_iterative_training.sh cirr qwen2vl 2` (dataset, model tag, GPU count); pass an existing experiment path as the fourth argument to resume.
- For configurable local experiments, run `python train_iterative.py --dataset_config configs/cirr_iterative.yaml --fast_mode --max_iterations 1` to exercise the loop without long GPU sessions.
- Validate distributed changes with `bash test_multiGPU_parallel.sh`, which runs a two-GPU smoke pass and writes logs to `experiments/MultiGPU_Test_*`.
- Evaluate checkpoints using `bash eval_cirr.sh --model_path <checkpoint_dir> --output_file results/<run>.json`; add `--single-gpu` if you need to force non-distributed evaluation.

## Coding Style & Naming Conventions
- Target Python 3.10 (matching `environment.yml`), four-space indentation, and PEP 8 spacing; keep imports grouped by standard/library/local order.
- Prefer descriptive module names (`iterative_retrieval`, `hard_negatives`) and camel-case iteration folders (`IterativeCIRR_<model>_<timestamp>`).
- Reuse helper logging such as `print_master` and `logger` instances; keep emoji usage aligned with existing scripts but avoid them in core library code.
- Document functions with short docstrings and type hints as seen in `src/trainer_iterative.py`; avoid in-line comments unless explaining non-obvious logic.
- When adding configs, follow the existing YAML key style (`fast_mode_max_samples`, `production_save_steps`) and document defaults in the README.

## Testing Guidelines
- Run `python train_iterative.py ... --fast_mode` before opening a PR to confirm negative mining, augmentation, and checkpoint wiring remain intact.
- Use `bash test_multiGPU_parallel.sh` after touching distributed code; inspect the generated `test.log` and ensure key JSON artifacts appear.
- Dataset or retrieval tweaks should be validated with `python verify_candidates.py` to confirm coverage metrics still match expectations.
- Place ad-hoc regression scripts under the repository root as `test_<feature>.py` (see `test_text_generation.py`) and keep paths configurable through CLI flags.
- Record eval metrics in `results/` and link them in the PR so reviewers can reproduce or audit the numbers quickly.

## Commit & Pull Request Guidelines
- Follow the existing history: concise, present-tense summaries in Chinese or bilingual form (e.g., “修复caption保存路径混乱问题”).
- Group related changes with one commit per feature or fix; avoid mixing environment updates with model edits.
- PRs should include: a short description, config deltas, commands run, and links to output artifacts (experiment folder, result JSON).
- Mention any dependency additions explicitly and update both `environment.yml` and `requirements.txt` when required.
- Attach screenshots or log excerpts only when they clarify new behaviors; otherwise point reviewers to the relevant files (`experiments/<run>/train.log`, `results/*.json`).

## Environment & Data Tips
- Keep Hugging Face caches in `$HOME/.cache/huggingface` as set in the scripts; avoid hard-coding alternative paths in committed files.
- Large foundation model weights are referenced via absolute paths in scripts—factor them into `.env` variables if you need portability.
- When sharing configs, strip user-specific directories or replace them with tokens (e.g., `/path/to/cirr/images`) before committing.
- For WANDB, use project `iterative_composed_retrieval_production`; disable it locally with `export WANDB_DISABLED=true` during quick tests.

# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/`. Use `src/pipeline.py` for the end-to-end VOD scan, `src/timer/` for timer calibration, dataset building, CNN training, and inference, and `src/toast/` for toast ROI detection and template matching. Utility and one-off debugging scripts live in `scripts/`. Runtime configs are in `configs/`, reusable image assets in `templates/` and `static/`, generated previews in `debug/`, and labeled or intermediate datasets in `data/`. Large local VOD files and model artifacts such as `timer_model.pth` are kept at the repo root.

## Build, Test, and Development Commands
Create the Python environment with `conda env create -f environment.yml` and activate `forsen`. Run the full pipeline with `python -m src.pipeline --video part538.mp4`. Train the digit model with `python -m src.timer.train_timer_cnn --data_root data/timer_digits_raw --num_workers 0 --epochs 110 --yellow_thr 23`. Validate timer inference on a sampled frame with `python src/timer/test_infer.py --video input_lowres.mp4`. Check anchor-relative digit crops visually with `python scripts/test_random_timer_digits.py --video part538.mp4 --frame 73740`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, snake_case for modules, functions, variables, and CLI flags, and PascalCase for classes such as `TinyCharCNN`. Keep files focused on one pipeline stage or debugging task. Prefer explicit argument parsing for scripts, `pathlib.Path` for paths, and short comments only where image-processing logic is non-obvious. There is no enforced formatter in the repo today, so keep imports tidy and match surrounding style.

## Testing Guidelines
This repository uses script-based integration checks instead of a formal test suite. Before opening a PR, run the relevant validation script for the area you changed, especially `python src/timer/test_infer.py --video <vod>` and any matching script in `scripts/` or `src/timer/test_*.py`. Name new checks after the behavior they validate, for example `test_anchor_timer_rois.py`. Save generated debug images to `debug/` and avoid committing bulky transient outputs unless they document a real regression.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit messages, for example `added we need to go template` or `toast template matching works + started work to retrain CNN`. Keep the first line specific to one change. PRs should include: a concise summary, the VOD or asset used for validation, exact commands run, and screenshots or output snippets when a change affects template matching, OCR, or timer geometry.

## Data & Configuration Notes
Treat video files, cookies, API-backed OCR experiments, and model weights as local working assets unless the repo already tracks them intentionally. Put new calibration JSON under `configs/` and new reusable templates under `templates/`, with descriptive names such as `toast_templates/we_need_to_go.png`.

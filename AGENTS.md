# AGENTS.md

## Setup

```bash
uv sync
source .venv/bin/activate
```

Python **3.12 only** (`>=3.12,<3.13`). Managed by `uv`, `.python-version` pins to `3.12`.

## Commands

```bash
# Smoke test (fast, uses 20 samples)
python main.py train --max-samples 20
python main.py train --task detect --max-samples 20

# Full training
python main.py train                    # segmentation (default)
python main.py train --task detect      # detection
python main.py train --no-classes       # single "nucleus" class instead of 5 cell types

# Inference
python main.py predict path/to/image.png
python main.py predict image.png --task detect -o output.tiff
```

Installed entry point: `yolo26seg-skripsi train` / `yolo26seg-skripsi predict`.

No test suite. No linter/typechecker configured.

## Architecture

Four source files at repo root, no packages:

- **`main.py`** — argparse CLI, delegates to `train.train()` or `predict.predict()`
- **`data.py`** — streams PanNuke from HuggingFace, writes images + labels to `data/pannuke_yolo/`
- **`train.py`** — Ultralytics YOLO26s training (segment or detect)
- **`predict.py`** — single-image inference → instance label map

## Key gotchas

- **Dataset download is streaming and slow.** First run streams ~190k images from HuggingFace. Subsequent runs skip splits whose image dirs already exist. Use `--max-samples N` for any development iteration.
- **Detection uses symlinks for image paths.** Ultralytics replaces `/images/` with `/labels/` to find labels. Detection images live at `images/det_{split}/` (symlinks → `../{split}/`) so that `labels/det_{split}/` is discovered correctly. Do not flatten this structure.
- **Both label formats are always written.** `data.py` writes segmentation polygons and detection bboxes in one pass. Switching `--task` does not re-download data.
- **`data/` is gitignored** — the full dataset and model outputs under `runs/` are not committed.
- **Mosaic augmentation is disabled** (`mosaic=0.0` in `train.py:71`) — PanNuke nuclei are small and dense; mosaic creates unrealistic arrangements.
- **Model paths**: `runs/{segment|detect}/models/pannuke_yolo26{seg|det}/weights/best.pt`
- **Batch size is auto** (`batch=-1`) — Ultralytics picks the largest batch that fits in VRAM.
- **Pretrained weights** `yolo26s-seg.pt` and `yolo26s.pt` exist at repo root; these are Ultralytics pretrained checkpoints, not trained models.

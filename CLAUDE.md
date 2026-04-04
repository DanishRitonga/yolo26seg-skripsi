# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python **3.12** (`requires-python = ">=3.12,<3.13"`). The venv is managed with `uv`.

```bash
uv sync                  # install / sync dependencies
source .venv/bin/activate
```

## Commands

```bash
# Full training run (fold1+fold2 → train, fold3 → val)
python main.py train

# Quick smoke test (caps each fold at N samples, skips disk cache)
python main.py train --max-samples 20

# Train without class-aware head (single class "nucleus")
python main.py train --no-classes

# Inference on a single image
python main.py predict path/to/image.png
python main.py predict path/to/image.png -o output.tiff

# Installed entry point (same as python main.py)
yolo26seg-skripsi train
yolo26seg-skripsi predict path/to/image.png
```

There is no test suite.

## Architecture

Four files, no packages:

| File | Role |
|------|------|
| `main.py` | argparse CLI router — delegates to `train.train()` or `predict.predict()` |
| `data.py` | PanNuke loading, mask-to-polygon conversion, YOLO dataset on disk |
| `train.py` | YOLO26s-seg config, Ultralytics training call |
| `predict.py` | Single-image inference, mask-to-instance-map conversion |

### Data flow

**PanNuke** (HuggingFace `RationAI/PanNuke`) has three folds. `prepare_yolo_dataset` in `data.py`:

1. **First run** — streams fold from HuggingFace one sample at a time (`streaming=True`), converts each instance mask to polygon coordinates via `cv2.findContours`, writes images as PNG and labels as YOLO-format `.txt` files to `data/pannuke_yolo/{images,labels}/{train,val}/`.
2. **Subsequent runs** — skips folds whose image directories already exist and are non-empty.

`--max-samples` bypasses the cache entirely (partial data is never persisted).

### YOLO dataset format

Ultralytics expects:
- `data/pannuke_yolo/dataset.yaml` — paths + class names
- `data/pannuke_yolo/images/{train,val}/*.png`
- `data/pannuke_yolo/labels/{train,val}/*.txt` — one row per instance: `class_id x1 y1 x2 y2 ... xn yn` (normalised 0–1)

### YOLO training

`train.py` loads a pretrained `yolo26s-seg.pt` (COCO weights) and fine-tunes on the PanNuke dataset. Key hyperparameters: 200 epochs, batch 4, lr 3e-4, imgsz 256, mosaic disabled.

The trained model is saved by Ultralytics to `runs/segment/models/pannuke_yolo26seg/weights/best.pt`.

### Key details

- `CELL_TYPES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]` — 5 PanNuke classes, 0-indexed in label files.
- `--no-classes` sets all instances to class 0 and uses a single `nucleus` class name.
- Ultralytics early-stops by default (patience=100). Best model is always saved regardless.
- `predict.py` converts YOLO polygon masks back to integer instance label maps for output compatibility with the StarDist pipeline.

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
# Segmentation training (polygon masks) — default
python main.py train
python main.py train --max-samples 20          # quick smoke test
python main.py train --no-classes              # single class "nucleus"

# Detection training (bounding boxes)
python main.py train --task detect
python main.py train --task detect --max-samples 20

# Inference
python main.py predict path/to/image.png
python main.py predict path/to/image.png --task detect -o output.tiff

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
| `data.py` | PanNuke loading, mask-to-polygon/bbox conversion, YOLO dataset on disk |
| `train.py` | YOLO26s config and training for both `segment` and `detect` tasks |
| `predict.py` | Single-image inference, mask/bbox-to-instance-map conversion |

### Data flow

**PanNuke** (HuggingFace `RationAI/PanNuke`) has three folds. `prepare_yolo_dataset` in `data.py`:

1. **First run** — streams folds from HuggingFace one sample at a time (`streaming=True`), writes images as PNG and **both** label formats (polygons + bboxes) in a single pass.
2. **Subsequent runs** — skips folds whose image directories already exist.

### Data directory layout

```
data/pannuke_yolo/
  images/train/*.png          (actual images, fold1+fold2)
  images/val/*.png            (fold3)
  images/det_train/           (symlinks → ../train/)
  images/det_val/             (symlinks → ../val/)
  labels/train/*.txt          (segmentation: class_id x1 y1 ... xn yn)
  labels/val/*.txt
  labels/det_train/*.txt      (detection: class_id cx cy w h)
  labels/det_val/*.txt
  dataset.yaml                (segmentation config)
  dataset_det.yaml            (detection config)
```

Detection images live at `images/det_{split}/` (symlinks to `images/{split}/`) and detection labels at `labels/det_{split}/`. This is required because Ultralytics resolves label paths by replacing the exact substring `/images/` with `/labels/` in the image path. The nested `det_` prefix ensures the replacement works: `images/det_train/` → `labels/det_train/`.

### Tasks

| `--task` | Model | Label format | Output |
|----------|-------|-------------|--------|
| `segment` | `yolo26s-seg.pt` | Polygons | Instance masks |
| `detect` | `yolo26s.pt` | Bounding boxes | Bboxes + centres |

### Key details

- `CELL_TYPES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]` — 5 PanNuke classes, 0-indexed in label files.
- `--no-classes` sets all instances to class 0 and uses a single `nucleus` class name.
- Ultralytics early-stops by default (patience=100). Best model is always saved regardless.
- Model output paths: `runs/{segment|detect}/models/pannuke_yolo26{seg|det}/weights/best.pt`
- `predict.py` converts polygon masks to instance label maps; for detection models it marks bbox centres on the label map.

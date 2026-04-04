"""Train a YOLO26s-seg model on the PanNuke dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from data import prepare_yolo_dataset

# ── Hyperparameters ───────────────────────────────────────────────────────────
MODEL_VARIANT = "yolo26s-seg.pt"
MODEL_NAME = "pannuke_yolo26seg"
# Ultralytics saves under runs/segment/{project}/{name}/ by default.
MODEL_BASEDIR = Path("models")
MODEL_PATH = MODEL_BASEDIR / MODEL_NAME / "weights" / "best.pt"
TRAIN_EPOCHS = 200
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
IMGSZ = 256


# ── Training entry point ──────────────────────────────────────────────────────


def train(
    use_classes: bool = True,
    max_samples: int | None = None,
) -> None:
    from ultralytics import YOLO

    # ── Prepare dataset ──
    print("Preparing YOLO dataset...")
    dataset_yaml = prepare_yolo_dataset(use_classes=use_classes, max_samples=max_samples)

    # ── Load pretrained model ──
    model = YOLO(MODEL_VARIANT)

    n_classes = 5 if use_classes else 1
    print(
        f"\nModel '{MODEL_VARIANT}'  "
        f"classes={n_classes}  "
        f"epochs={TRAIN_EPOCHS}  "
        f"imgsz={IMGSZ}\n"
    )

    # ── Train ──
    model.train(
        data=str(dataset_yaml),
        epochs=TRAIN_EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        project=str(MODEL_BASEDIR),
        name=MODEL_NAME,
        exist_ok=True,
        # Disable mosaic — PanNuke nuclei are small and dense; cutting/pasting
        # patches tends to create unrealistic arrangements.
        mosaic=0.0,
    )

    # Ultralytics saves to runs/segment/{project}/{name}/ internally, but
    # the path below is what the user should reference for predict.
    actual_path = Path("runs/segment") / MODEL_BASEDIR / MODEL_NAME / "weights" / "best.pt"
    if actual_path.exists():
        print(f"\nDone. Best model saved to {actual_path}")
    else:
        print(f"\nDone. Model saved under runs/segment/{MODEL_BASEDIR}/{MODEL_NAME}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26s-seg on PanNuke")
    parser.add_argument(
        "--no-classes",
        action="store_true",
        help="Disable class-aware training (single class 'nucleus')",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit samples per fold for quick smoke tests (e.g. --max-samples 50)",
    )
    args = parser.parse_args()
    train(use_classes=not args.no_classes, max_samples=args.max_samples)

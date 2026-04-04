"""Train YOLO26s on PanNuke for both segmentation and detection tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from data import prepare_yolo_dataset, TASKS

# ── Hyperparameters ───────────────────────────────────────────────────────────
MODELS = {
    "segment": "yolo26s-seg.pt",
    "detect": "yolo26s.pt",
}
MODEL_NAMES = {
    "segment": "pannuke_yolo26seg",
    "detect": "pannuke_yolo26det",
}
MODEL_BASEDIR = Path("models")
TRAIN_EPOCHS = 200
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
IMGSZ = 256


# ── Training entry point ──────────────────────────────────────────────────────


def train(
    task: str = "segment",
    use_classes: bool = True,
    max_samples: int | None = None,
) -> None:
    from ultralytics import YOLO

    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got '{task}'")

    # ── Prepare dataset (writes both label formats in one pass) ──
    print("Preparing YOLO dataset...")
    yaml_paths = prepare_yolo_dataset(use_classes=use_classes, max_samples=max_samples)
    dataset_yaml = yaml_paths[task]

    model_variant = MODELS[task]
    model_name = MODEL_NAMES[task]

    # ── Load pretrained model ──
    model = YOLO(model_variant)

    n_classes = 5 if use_classes else 1
    print(
        f"\nModel '{model_variant}'  "
        f"task={task}  "
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
        name=model_name,
        exist_ok=True,
        # Disable mosaic — PanNuke nuclei are small and dense; cutting/pasting
        # patches tends to create unrealistic arrangements.
        mosaic=0.0,
    )

    # Ultralytics saves to runs/{task}/{project}/{name}/ internally
    actual_path = (
        Path(f"runs/{task}") / MODEL_BASEDIR / model_name / "weights" / "best.pt"
    )
    if actual_path.exists():
        print(f"\nDone. Best model saved to {actual_path}")
    else:
        print(f"\nDone. Model saved under runs/{task}/{MODEL_BASEDIR}/{model_name}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26s on PanNuke")
    parser.add_argument(
        "--task",
        choices=list(TASKS),
        default="segment",
        help="Task: segment (polygon masks) or detect (bounding boxes)",
    )
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
    train(task=args.task, use_classes=not args.no_classes, max_samples=args.max_samples)

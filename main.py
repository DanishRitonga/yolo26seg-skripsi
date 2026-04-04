"""CLI entry point for yolo26seg-skripsi."""

from __future__ import annotations

import argparse
from pathlib import Path

from train import MODEL_BASEDIR, MODEL_NAMES

# Default to segmentation model path
MODEL_PATH = (
    Path("runs/segment")
    / MODEL_BASEDIR
    / MODEL_NAMES["segment"]
    / "weights"
    / "best.pt"
)


def cmd_train(args: argparse.Namespace) -> None:
    from train import train

    train(
        task=args.task,
        use_classes=not args.no_classes,
        max_samples=args.max_samples,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    from predict import load_model, predict, summarize

    # Resolve model path based on task
    if args.model_path:
        model_path = args.model_path
    else:
        runs_dir = Path("runs") / args.task
        model_path = (
            runs_dir / MODEL_BASEDIR / MODEL_NAMES[args.task] / "weights" / "best.pt"
        )

    model = load_model(model_path)
    labels, details = predict(
        args.image,
        model=model,
        conf=args.conf,
        iou=args.iou,
    )

    summarize(labels, details)

    if args.output:
        import numpy as np
        from tifffile import imwrite

        imwrite(args.output, labels.astype(np.uint16))
        print(f"Saved → {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="yolo26seg-skripsi",
        description="YOLO26 nucleus segmentation & detection on PanNuke",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──
    t = sub.add_parser("train", help="Train YOLO26 on PanNuke (fold1+fold2 → fold3 val)")
    t.add_argument(
        "--task",
        choices=["segment", "detect"],
        default="segment",
        help="Task: segment (polygon masks) or detect (bounding boxes)",
    )
    t.add_argument(
        "--no-classes",
        action="store_true",
        help="Disable class-aware training",
    )

    def _positive_int(v: str) -> int:
        n = int(v)
        if n < 1:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
        return n

    t.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Cap samples per fold for quick tests (e.g. --max-samples 50)",
    )

    # ── predict ──
    p = sub.add_parser("predict", help="Run inference on a single image")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--output", "-o", metavar="FILE", help="Save instance map as TIFF")
    p.add_argument("--task", choices=["segment", "detect"], default="segment")
    p.add_argument("--model-path", default=None, help="Path to trained .pt model")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)


if __name__ == "__main__":
    main()

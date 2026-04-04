"""CLI entry point for yolo26seg-skripsi."""

from __future__ import annotations

import argparse
from pathlib import Path

from train import MODEL_BASEDIR, MODEL_NAME

# Ultralytics saves to runs/segment/{project}/{name}/ internally
MODEL_PATH = Path("runs/segment") / MODEL_BASEDIR / MODEL_NAME / "weights" / "best.pt"


def cmd_train(args: argparse.Namespace) -> None:
    from train import train

    train(use_classes=not args.no_classes, max_samples=args.max_samples)


def cmd_predict(args: argparse.Namespace) -> None:
    from predict import load_model, predict, summarize

    model_path = args.model_path or str(MODEL_PATH)
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
        description="YOLO26s-seg nucleus segmentation on PanNuke",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──
    t = sub.add_parser("train", help="Train YOLO26s-seg on PanNuke (fold1+fold2 → fold3 val)")
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

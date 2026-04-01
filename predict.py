"""Run StarDist inference on a single image."""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
from csbdeep.utils import normalize
from PIL import Image
from stardist.models import StarDist2D

from train import MODEL_BASEDIR, MODEL_NAME


def load_model(name: str = MODEL_NAME, basedir: str = MODEL_BASEDIR) -> StarDist2D:
    """Load a trained StarDist model from disk."""
    return StarDist2D(None, name=name, basedir=basedir)


def predict(
    image_path: str | Path,
    model: StarDist2D | None = None,
    prob_thresh: float | None = None,
    nms_thresh: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Run StarDist inference on a single image file.

    Args:
        image_path:  Path to any image readable by PIL (PNG, TIFF, JPG, …).
        model:       Pre-loaded model; loads the default trained model if None.
        prob_thresh: Detection probability threshold (uses model default if None).
        nms_thresh:  NMS IoU threshold (uses model default if None).

    Returns:
        labels:  (H, W) int32 instance label map.
        details: Dict containing 'coord', 'points', 'prob', and optionally 'class_id'.
    """
    if model is None:
        model = load_model()

    img = np.array(Image.open(image_path).convert("RGB")).astype(np.float32)
    img_norm = normalize(img, 1, 99.8, axis=(0, 1))

    kwargs: dict = {}
    if prob_thresh is not None:
        kwargs["prob_thresh"] = prob_thresh
    if nms_thresh is not None:
        kwargs["nms_thresh"] = nms_thresh

    labels, details = model.predict_instances(img_norm, **kwargs)
    return labels, details


def summarize(labels: np.ndarray, details: dict) -> None:
    """Print a short summary of prediction results to stdout."""
    from data import CELL_TYPES

    n_nuclei = int(labels.max())
    print(f"Detected {n_nuclei} nucleus instance(s).")

    if "class_id" in details and len(details["class_id"]) > 0:
        counts = collections.Counter(details["class_id"])
        print("Cell-type breakdown:")
        for cls_id in sorted(counts):
            name = CELL_TYPES[cls_id - 1] if 1 <= cls_id <= len(CELL_TYPES) else f"class_{cls_id}"
            print(f"  {name}: {counts[cls_id]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="StarDist inference on a single image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output", "-o", help="Save instance label map as TIFF")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-basedir", default=MODEL_BASEDIR)
    parser.add_argument("--prob-thresh", type=float, default=None)
    parser.add_argument("--nms-thresh", type=float, default=None)
    args = parser.parse_args()

    m = load_model(name=args.model_name, basedir=args.model_basedir)
    labels, details = predict(
        args.image, model=m,
        prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh,
    )

    summarize(labels, details)

    if args.output:
        from tifffile import imwrite
        imwrite(args.output, labels.astype(np.uint16))
        print(f"Saved → {args.output}")

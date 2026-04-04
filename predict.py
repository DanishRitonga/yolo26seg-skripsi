"""Run YOLO26s-seg inference on a single image."""

from __future__ import annotations

import collections
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from train import MODEL_NAME

# Ultralytics saves under runs/segment/{project}/{name}/ by default.
MODEL_PATH = Path("runs/segment/models") / MODEL_NAME / "weights" / "best.pt"


def load_model(path: Path | str = MODEL_PATH):
    """Load a trained YOLO segmentation model from disk."""
    from ultralytics import YOLO

    return YOLO(str(path))


def predict(
    image_path: str | Path,
    model=None,
    conf: float = 0.25,
    iou: float = 0.7,
) -> tuple[np.ndarray, dict]:
    """
    Run YOLO26s-seg inference on a single image file.

    Args:
        image_path:  Path to any image readable by PIL (PNG, TIFF, JPG, ...).
        model:       Pre-loaded YOLO model; loads the default trained model if None.
        conf:        Confidence threshold for detections.
        iou:         NMS IoU threshold.

    Returns:
        labels:  (H, W) int32 instance label map.
        details: Dict containing 'class_id', 'conf', 'points', and 'masks'.
    """
    if model is None:
        model = load_model()

    results = model(image_path, conf=conf, iou=iou, verbose=False)
    result = results[0]

    img = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]
    labels = np.zeros((H, W), dtype=np.int32)

    class_ids: list[int] = []
    confs: list[float] = []
    points: list[tuple[int, int]] = []

    if result.masks is not None:
        masks_data = result.masks.data.cpu().numpy()  # (N, H, W)
        boxes = result.boxes
        for idx in range(len(masks_data)):
            mask = masks_data[idx]
            # Resize mask to image size if needed
            if mask.shape != (H, W):
                from PIL import Image as PILImage

                mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((W, H), resample=PILImage.NEAREST)
                mask = np.array(mask_pil) > 127

            labels[mask] = idx + 1

            cls_id = int(boxes.cls[idx])
            class_ids.append(cls_id)
            confs.append(float(boxes.conf[idx]))

            # Bounding box centre as the "point"
            x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy()
            points.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))

    details: dict = {
        "class_id": np.array(class_ids),
        "prob": np.array(confs),
        "points": np.array(points) if points else np.empty((0, 2)),
    }
    return labels, details


def summarize(labels: np.ndarray, details: dict) -> None:
    """Print a short summary of prediction results to stdout."""
    from data import CELL_TYPES

    n_nuclei = int(labels.max())
    print(f"Detected {n_nuclei} nucleus instance(s).")

    class_ids = details.get("class_id")
    if class_ids is not None and len(class_ids) > 0:
        counts = collections.Counter(int(c) for c in class_ids)
        print("Cell-type breakdown:")
        for cls_id in sorted(counts):
            name = CELL_TYPES[cls_id] if 0 <= cls_id < len(CELL_TYPES) else f"class_{cls_id}"
            print(f"  {name}: {counts[cls_id]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO26s-seg inference on a single image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output", "-o", help="Save instance label map as TIFF")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    args = parser.parse_args()

    m = load_model(args.model_path)
    labels, details = predict(args.image, model=m, conf=args.conf, iou=args.iou)

    summarize(labels, details)

    if args.output:
        from tifffile import imwrite

        n_instances = int(labels.max())
        if n_instances > np.iinfo(np.uint16).max:
            warnings.warn(
                f"{n_instances} instances detected but uint16 max is "
                f"{np.iinfo(np.uint16).max}; instance IDs will be truncated in the output TIFF."
            )
        imwrite(args.output, labels.astype(np.uint16))
        print(f"Saved → {args.output}")

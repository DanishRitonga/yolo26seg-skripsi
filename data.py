"""PanNuke dataset loading and YOLO segmentation format conversion."""

from __future__ import annotations

import gc
import shutil
from pathlib import Path

import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

DATASET_ID = "RationAI/PanNuke"
DATA_DIR = Path("data") / "pannuke_yolo"

# PanNuke cell-type classes (0-indexed, matching `categories` field values)
CELL_TYPES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]


# ── Mask → polygon conversion ─────────────────────────────────────────────────


def _mask_to_polygons(mask: np.ndarray, img_h: int, img_w: int) -> list[list[float]]:
    """
    Extract normalised polygon contours from a binary instance mask.

    Returns a list of polygons, each polygon is a flat list [x1, y1, x2, y2, ...]
    with coordinates normalised to [0, 1].
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        # Need at least 3 points for a valid polygon
        if contour.shape[0] < 3:
            continue
        coords = contour.reshape(-1, 2).astype(np.float64)
        # Normalise to [0, 1]
        coords[:, 0] /= img_w
        coords[:, 1] /= img_h
        # Clamp to [0, 1]
        coords = np.clip(coords, 0.0, 1.0)
        polygons.append(coords.flatten().tolist())
    return polygons


def _write_label_file(
    path: Path,
    instances: list,
    categories: list[int],
    img_h: int,
    img_w: int,
    use_classes: bool,
) -> int:
    """
    Write a YOLO segmentation label file for one image.

    Each row: `class_id x1 y1 x2 y2 ... xn yn`

    Returns the number of instances written.
    """
    lines: list[str] = []
    for mask, cat in zip(instances, categories):
        m = np.array(mask)
        if m.ndim == 3:
            m = m[..., 0]
        m = (m > 0).astype(np.uint8)

        polygons = _mask_to_polygons(m, img_h, img_w)
        class_id = cat if use_classes else 0
        for poly in polygons:
            line = f"{class_id} " + " ".join(f"{v:.6f}" for v in poly)
            lines.append(line)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n" if lines else "")
    return len(lines)


def _write_dataset_yaml(path: Path, use_classes: bool) -> None:
    """Generate the Ultralytics dataset.yaml config file."""
    names = CELL_TYPES if use_classes else ["nucleus"]
    nc = len(names)

    lines = [
        f"path: {path.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {nc}",
        f"names: {names}",
    ]
    (path / "dataset.yaml").write_text("\n".join(lines) + "\n")


# ── Dataset preparation ────────────────────────────────────────────────────────


def _split_has_data(split: str) -> bool:
    """Check whether a split directory already contains images."""
    img_dir = DATA_DIR / "images" / split
    if not img_dir.exists():
        return False
    return any(img_dir.iterdir())


def _process_fold(
    fold_name: str,
    split: str,
    use_classes: bool,
    max_samples: int | None,
) -> None:
    """
    Stream one PanNuke fold and write images + labels to the YOLO dataset dir.
    """
    img_dir = DATA_DIR / "images" / split
    lbl_dir = DATA_DIR / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_ID, split=fold_name, streaming=True)
    if max_samples is not None:
        ds = ds.take(max_samples)

    n_instances = 0
    n_images = 0
    for i, sample in enumerate(tqdm(ds, desc=fold_name, unit="img")):
        img: Image.Image = sample["image"]
        img_w, img_h = img.size  # PIL gives (W, H)

        # Save image as PNG
        img_path = img_dir / f"{i:06d}.png"
        img.save(img_path)

        # Write label file
        instances = sample["instances"]
        categories = sample["categories"]

        if use_classes and len(instances) != len(categories):
            raise ValueError(
                f"{fold_name}[{i}]: {len(instances)} masks but {len(categories)} "
                "categories — instance-to-class mapping would be wrong"
            )

        lbl_path = lbl_dir / f"{i:06d}.txt"
        n_instances += _write_label_file(
            lbl_path, instances, categories, img_h, img_w, use_classes
        )

        if i % 100 == 0:
            gc.collect()

    n_images = i + 1
    print(f"  {fold_name} → {split}: {n_images} images, {n_instances} instances")


def prepare_yolo_dataset(
    use_classes: bool = True,
    max_samples: int | None = None,
) -> Path:
    """
    Prepare the full PanNuke YOLO segmentation dataset.

    fold1 + fold2 → train split
    fold3         → val split

    Returns the path to dataset.yaml.
    """
    splits = {"train": ["fold1", "fold2"], "val": ["fold3"]}

    for split, folds in splits.items():
        if max_samples is None and _split_has_data(split):
            print(f"Split '{split}' already exists, skipping download.")
            continue

        # Clear existing data if re-downloading
        for d in [DATA_DIR / "images" / split, DATA_DIR / "labels" / split]:
            if d.exists():
                shutil.rmtree(d)

        for fold in folds:
            _process_fold(fold, split, use_classes, max_samples)

    _write_dataset_yaml(DATA_DIR, use_classes)
    return DATA_DIR / "dataset.yaml"

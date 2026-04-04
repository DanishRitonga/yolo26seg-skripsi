"""PanNuke dataset loading and YOLO format conversion (segment + detect)."""

from __future__ import annotations

import gc
import os
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

TASKS = ("segment", "detect")


# ── Mask → polygon conversion ─────────────────────────────────────────────────


def _mask_to_polygons(mask: np.ndarray, img_h: int, img_w: int) -> list[list[float]]:
    """Extract normalised polygon contours from a binary instance mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        coords = contour.reshape(-1, 2).astype(np.float64)
        coords[:, 0] /= img_w
        coords[:, 1] /= img_h
        coords = np.clip(coords, 0.0, 1.0)
        polygons.append(coords.flatten().tolist())
    return polygons


# ── Mask → bbox conversion ────────────────────────────────────────────────────


def _mask_to_bbox(mask: np.ndarray, img_h: int, img_w: int) -> tuple[float, float, float, float] | None:
    """
    Extract a normalised bounding box (cx, cy, w, h) from a binary instance mask.
    Returns None if the mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return (cx, cy, w, h)


# ── Label file writers ────────────────────────────────────────────────────────


def _write_seg_labels(
    path: Path,
    instances: list,
    categories: list[int],
    img_h: int,
    img_w: int,
    use_classes: bool,
) -> int:
    """Write YOLO segmentation labels (polygon format). Returns instance count."""
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


def _write_det_labels(
    path: Path,
    instances: list,
    categories: list[int],
    img_h: int,
    img_w: int,
    use_classes: bool,
) -> int:
    """Write YOLO detection labels (bbox format: class_id cx cy w h). Returns instance count."""
    lines: list[str] = []
    for mask, cat in zip(instances, categories):
        m = np.array(mask)
        if m.ndim == 3:
            m = m[..., 0]
        m = (m > 0).astype(np.uint8)

        bbox = _mask_to_bbox(m, img_h, img_w)
        if bbox is None:
            continue
        class_id = cat if use_classes else 0
        cx, cy, w, h = bbox
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n" if lines else "")
    return len(lines)


# ── YAML + symlink helpers ────────────────────────────────────────────────────


def _write_yaml(path: Path, use_classes: bool, label_dir_prefix: str) -> Path:
    """Write a dataset.yaml for Ultralytics. Returns the yaml path."""
    names = CELL_TYPES if use_classes else ["nucleus"]
    nc = len(names)
    yaml_name = "dataset.yaml" if label_dir_prefix == "labels" else "dataset_det.yaml"

    if label_dir_prefix == "labels":
        img_dirs = ("images/train", "images/val", "images/test")
    else:
        # Detection labels live in labels/det_{split}/, so images must be
        # under images/det_{split}/ for Ultralytics' "images → labels" path
        # replacement to find them.
        img_dirs = ("images/det_train", "images/det_val", "images/det_test")

    lines = [
        f"path: {path.resolve()}",
        f"train: {img_dirs[0]}",
        f"val: {img_dirs[1]}",
        f"test: {img_dirs[2]}",
        f"nc: {nc}",
        f"names: {names}",
    ]
    yaml_path = path / yaml_name
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


def _ensure_image_symlinks(base: Path, split: str) -> None:
    """
    Create images/det_{split}/ populated with symlinks → ../{split}/.

    Ultralytics resolves label paths by replacing '/images/' with '/labels/'
    in the image path.  By placing detection images at images/det_{split}/
    and detection labels at labels/det_{split}/, the replacement works
    correctly:  images/det_train/  →  labels/det_train/.
    """
    det_img_dir = base / "images" / f"det_{split}"
    src_img_dir = base / "images" / split
    if det_img_dir.exists():
        return
    det_img_dir.mkdir(parents=True, exist_ok=True)
    for img_file in src_img_dir.iterdir():
        link = det_img_dir / img_file.name
        link.symlink_to(os.path.relpath(img_file, link.parent))


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
    """Stream one PanNuke fold and write images + both label formats."""
    img_dir = DATA_DIR / "images" / split
    seg_lbl_dir = DATA_DIR / "labels" / split
    det_lbl_dir = DATA_DIR / "labels" / f"det_{split}"
    img_dir.mkdir(parents=True, exist_ok=True)
    seg_lbl_dir.mkdir(parents=True, exist_ok=True)
    det_lbl_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_ID, split=fold_name, streaming=True)
    if max_samples is not None:
        ds = ds.take(max_samples)

    n_instances = 0
    i = -1
    for i, sample in enumerate(tqdm(ds, desc=fold_name, unit="img")):
        img: Image.Image = sample["image"]
        img_w, img_h = img.size

        img_path = img_dir / f"{i:06d}.png"
        img.save(img_path)

        instances = sample["instances"]
        categories = sample["categories"]

        if use_classes and len(instances) != len(categories):
            raise ValueError(
                f"{fold_name}[{i}]: {len(instances)} masks but {len(categories)} "
                "categories — instance-to-class mapping would be wrong"
            )

        # Segmentation labels (polygons)
        seg_lbl_path = seg_lbl_dir / f"{i:06d}.txt"
        n_instances += _write_seg_labels(
            seg_lbl_path, instances, categories, img_h, img_w, use_classes
        )

        # Detection labels (bboxes)
        det_lbl_path = det_lbl_dir / f"{i:06d}.txt"
        _write_det_labels(
            det_lbl_path, instances, categories, img_h, img_w, use_classes
        )

        if i % 100 == 0:
            gc.collect()

    n_images = i + 1 if i >= 0 else 0
    print(f"  {fold_name} → {split}: {n_images} images, {n_instances} instances")


def prepare_yolo_dataset(
    use_classes: bool = True,
    max_samples: int | None = None,
) -> dict[str, Path]:
    """
    Prepare the PanNuke YOLO dataset (both segmentation and detection labels).

    fold1 → train split
    fold2 → val split
    fold3 → test split

    Returns dict with paths to both dataset yaml files:
        {"segment": Path("data/pannuke_yolo/dataset.yaml"),
         "detect":  Path("data/pannuke_yolo/dataset_det.yaml")}
    """
    splits = {"train": ["fold1"], "val": ["fold2"], "test": ["fold3"]}

    for split, folds in splits.items():
        if max_samples is None and _split_has_data(split):
            print(f"Split '{split}' already exists, skipping download.")
            continue

        for d in [
            DATA_DIR / "images" / split,
            DATA_DIR / "labels" / split,
            DATA_DIR / "labels" / f"det_{split}",
        ]:
            if d.exists():
                shutil.rmtree(d)

        for fold in folds:
            _process_fold(fold, split, use_classes, max_samples)

    # Create symlinks so Ultralytics can find det labels via images/det_{split}/
    for split in ("train", "val", "test"):
        _ensure_image_symlinks(DATA_DIR, split)

    # Write both yaml configs
    seg_yaml = _write_yaml(DATA_DIR, use_classes, "labels")
    det_yaml = _write_yaml(DATA_DIR, use_classes, "labels_det")  # triggers det_ paths

    return {"segment": seg_yaml, "detect": det_yaml}

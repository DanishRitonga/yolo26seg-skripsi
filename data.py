"""PanNuke dataset loading and preprocessing for StarDist training."""

from __future__ import annotations

import numpy as np
from csbdeep.utils import normalize
from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "RationAI/PanNuke"

# PanNuke cell-type classes (0-indexed, matching `categories` field values)
CELL_TYPES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]


# ── Mask utilities ────────────────────────────────────────────────────────────


def _to_mask_array(mask) -> np.ndarray:
    """Convert a single binary instance mask (PIL or ndarray) to 2D uint8."""
    m = np.array(mask)
    if m.ndim == 3:
        m = m[..., 0]
    return m


def build_instance_map(
    instances: list, size: tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    Combine per-nucleus binary masks into a single integer instance label map.
    Each nucleus gets a unique positive integer ID; background = 0.
    Overlapping pixels are assigned to the last nucleus in the list.
    """
    H, W = size
    instance_map = np.zeros((H, W), dtype=np.int32)
    for idx, mask in enumerate(instances, start=1):
        m = _to_mask_array(mask)
        instance_map[m > 0] = idx
    return instance_map


def build_class_map(
    instances: list,
    categories: list[int],
    size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Build a per-pixel class label map aligned with the instance map.
    Values are 1-indexed (1–5 for the 5 cell types); 0 = background.
    """
    H, W = size
    class_map = np.zeros((H, W), dtype=np.int32)
    for mask, cat in zip(instances, categories):
        m = _to_mask_array(mask)
        class_map[m > 0] = int(cat) + 1  # shift to 1-indexed
    return class_map


# ── Dataset loading ───────────────────────────────────────────────────────────


def load_fold(
    fold_name: str,
    use_classes: bool = True,
    max_samples: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray] | None]:
    """
    Load and preprocess one PanNuke fold from Hugging Face.

    Args:
        fold_name:    One of "fold1", "fold2", "fold3".
        use_classes:  If True, also return per-pixel class maps for class-aware training.
        max_samples:  Optionally cap the number of samples (useful for quick tests).

    Returns:
        X:       List of float32 RGB images normalized to [0, 1], shape (H, W, 3).
        Y:       List of int32 instance label maps, shape (H, W).
        classes: List of int32 class maps, shape (H, W), or None if use_classes=False.
                 Class values are 1-indexed; 0 = background.
    """
    split = load_dataset(DATASET_ID, split=fold_name)
    if max_samples is not None:
        split = split.select(range(min(max_samples, len(split))))

    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    C: list[np.ndarray] | None = [] if use_classes else None

    for sample in tqdm(split, desc=fold_name, unit="img"):
        img = np.array(sample["image"]).astype(np.float32)  # (H, W, 3)
        img_norm = normalize(img, 1, 99.8, axis=(0, 1))

        # NOTE: adjust these field names if the dataset schema differs
        instances: list = sample["instances"]  # list of binary masks
        categories: list[int] = sample["categories"]  # list of int 0–4

        inst_map = build_instance_map(instances)

        X.append(img_norm)
        Y.append(inst_map)
        if use_classes:
            C.append(build_class_map(instances, categories))  # type: ignore[union-attr]

    return X, Y, C

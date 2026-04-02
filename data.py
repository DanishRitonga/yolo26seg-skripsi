"""PanNuke dataset loading and preprocessing for StarDist training."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
from csbdeep.utils import normalize
from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "RationAI/PanNuke"
DATA_DIR = Path("data")

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


def build_class_dict(categories: list[int]) -> dict[int, int]:
    """
    Build a dictionary mapping instance ID to class ID.
    Values are 1-indexed to match the instance map; 0 = background.
    """
    # enumerate(start=1) perfectly matches the instance IDs generated
    # in your build_instance_map function.
    return {idx: int(cat) + 1 for idx, cat in enumerate(categories, start=1)}


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


# ── Disk cache helpers ────────────────────────────────────────────────────────


def _fold_dir(fold_name: str) -> Path:
    return DATA_DIR / fold_name


def _cache_exists(fold_name: str) -> bool:
    d = _fold_dir(fold_name)
    return (d / "X.npy").exists() and (d / "Y.npy").exists()


def _save_fold(
    fold_name: str,
    X: list[np.ndarray],
    Y: list[np.ndarray],
    C: list[dict[int, int]] | None,
) -> None:
    d = _fold_dir(fold_name)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "X.npy", np.stack(X))
    np.save(d / "Y.npy", np.stack(Y))
    if C is not None:
        with open(d / "C.json", "w") as f:
            json.dump(C, f)
    print(f"Cached {fold_name} → {d}/")


def _load_fold_from_disk(
    fold_name: str,
    use_classes: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[int, int]] | None]:
    d = _fold_dir(fold_name)
    # mmap_mode='r' pages data from disk on demand — only accessed regions
    # are loaded into RAM, which acts as lazy loading for StarDist's random
    # patch sampling during training.
    X_arr = np.load(d / "X.npy", mmap_mode="r")  # (N, H, W, 3)
    Y_arr = np.load(d / "Y.npy", mmap_mode="r")  # (N, H, W)
    X = list(X_arr)
    Y = list(Y_arr)

    C: list[dict[int, int]] | None = None
    if use_classes:
        c_path = d / "C.json"
        if c_path.exists():
            with open(c_path) as f:
                raw = json.load(f)
            # JSON round-trips dict keys as strings; restore to int.
            C = [{int(k): v for k, v in entry.items()} for entry in raw]

    return X, Y, C


# ── Dataset loading ───────────────────────────────────────────────────────────


def load_fold(
    fold_name: str,
    use_classes: bool = True,
    max_samples: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[int, int]] | None]:
    """
    Load and preprocess one PanNuke fold from Hugging Face (streaming).

    Args:
        fold_name:    One of "fold1", "fold2", "fold3".
        use_classes:  If True, also return per-pixel class maps for class-aware training.
        max_samples:  Optionally cap the number of samples (useful for quick tests).

    Returns:
        X:       List of float32 RGB images normalized to [0, 1], shape (H, W, 3).
        Y:       List of int32 instance label maps, shape (H, W).
        classes: List of class dicts {instance_id: class_id}, or None if use_classes=False.
                 Class values are 1-indexed; 0 = background.
    """
    if max_samples is not None and max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")

    split = load_dataset(DATASET_ID, split=fold_name, streaming=True)
    if max_samples is not None:
        split = split.take(max_samples)

    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    C: list[dict[int, int]] | None = [] if use_classes else None

    for i, sample in enumerate(tqdm(split, desc=fold_name, unit="img")):
        img = np.array(sample["image"]).astype(np.float32)  # (H, W, 3)
        img_norm = normalize(img, 1, 99.8, axis=(0, 1))
        H, W = img_norm.shape[:2]
        del img

        # NOTE: adjust these field names if the dataset schema differs
        instances: list = sample["instances"]  # list of binary masks
        categories: list[int] = sample["categories"]  # list of int 0–4

        if use_classes and len(instances) != len(categories):
            raise ValueError(
                f"{fold_name}[{i}]: {len(instances)} masks but {len(categories)} "
                "categories — instance-to-class mapping would be wrong"
            )

        # Pass actual image size so masks with unexpected dimensions raise
        # a clear error instead of a cryptic numpy boolean-index shape mismatch.
        inst_map = build_instance_map(instances, size=(H, W))
        del instances

        X.append(img_norm)
        Y.append(inst_map)
        if use_classes:
            C.append(build_class_dict(categories))  # type: ignore[union-attr]

        # gc.collect() every N iterations — reference counting handles most
        # frees immediately; GC is only needed for any residual PIL cycles.
        if i % 100 == 0:
            gc.collect()

    return X, Y, C


def load_fold_cached(
    fold_name: str,
    use_classes: bool = True,
    max_samples: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[int, int]] | None]:
    """
    Load a PanNuke fold, using a local disk cache when available.

    On first call (or when max_samples is set), downloads and preprocesses
    from Hugging Face via streaming, then saves the full fold to data/<fold_name>/.
    On subsequent calls, loads directly from disk using memory-mapped arrays
    so only the pages touched during training are kept in RAM.

    Caching is skipped when max_samples is set to avoid storing partial data.
    """
    if max_samples is None and _cache_exists(fold_name):
        d = _fold_dir(fold_name)
        if use_classes and not (d / "C.json").exists():
            # Cache was built with --no-classes; re-download so class data is saved.
            print(f"Cache for {fold_name} has no class data; re-downloading...")
        else:
            print(f"Loading {fold_name} from disk cache ({d})...")
            return _load_fold_from_disk(fold_name, use_classes)

    X, Y, C = load_fold(fold_name, use_classes=use_classes, max_samples=max_samples)

    if max_samples is None:
        _save_fold(fold_name, X, Y, C)

    return X, Y, C

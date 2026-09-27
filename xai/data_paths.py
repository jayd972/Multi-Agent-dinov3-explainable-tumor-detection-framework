"""BRISC image-split discovery. Does not invent masks or change splits."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALID_EXT = {".jpg", ".jpeg", ".png"}


def list_split_images(root: Path) -> list[tuple[str, Path]]:
    items = []
    if not root.exists():
        return items
    class_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    sources = class_dirs or [root]
    for d in sources:
        label = d.name if d != root else "unknown"
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in VALID_EXT:
                items.append((label, p))
    return items

DEFAULT_DATA_ROOT = Path(os.environ.get("BRISC_DATA_ROOT", r"C:\Users\darji\Downloads\BT_Images"))
VAL_FRACTION = 0.15
VAL_SEED = 42
CLASS_ALIASES = {
    "glioma": "glioma",
    "glioma_tumor": "glioma",
    "meningioma": "meningioma",
    "meningioma_tumor": "meningioma",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary",
    "no": "no_tumor",
    "no_tumor": "no_tumor",
    "notumor": "no_tumor",
}


def data_root() -> Path:
    return Path(os.environ.get("BRISC_DATA_ROOT", DEFAULT_DATA_ROOT))


def _first_existing(*cands: Path) -> Path | None:
    for p in cands:
        if p.exists():
            return p
    return None


def test_dir() -> Path | None:
    root = data_root()
    return _first_existing(root / "Testing", root / "test")


def train_dir() -> Path | None:
    root = data_root()
    return _first_existing(root / "Training", root / "train")


def normalize_class(name: str) -> str:
    key = name.lower().replace("-", "_").replace(" ", "_")
    return CLASS_ALIASES.get(key, key)


def official_test_images() -> list[tuple[str, Path]]:
    d = test_dir()
    if d is None:
        return []
    return [(normalize_class(lab), p) for lab, p in list_split_images(d)]


def training_images() -> list[tuple[str, Path]]:
    d = train_dir()
    if d is None:
        return []
    return [(normalize_class(lab), p) for lab, p in list_split_images(d)]


def validation_images() -> list[tuple[str, Path]]:
    """Deterministic 15% of official Training, seed 42, stratified by class.

    This matches the notebook convention. Official Testing is never used as val.
    """
    items = training_images()
    if not items:
        return []
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        rng = __import__("random").Random(VAL_SEED)
        items = list(items)
        rng.shuffle(items)
        n = max(1, int(len(items) * VAL_FRACTION))
        return items[:n]
    labels = [lab for lab, _ in items]
    idx = list(range(len(items)))
    _, val_idx = train_test_split(
        idx, test_size=VAL_FRACTION, random_state=VAL_SEED, stratify=labels
    )
    return [items[i] for i in sorted(val_idx)]


CLASS_ORDER = ["glioma", "meningioma", "pituitary", "no_tumor"]


def n_per_class(items: list[tuple[str, Path]], n: int = 10) -> list[tuple[str, Path]]:
    """First n images per class after sorting by filename (deterministic, not cherry-picked)."""
    buckets: dict[str, list[tuple[str, Path]]] = {c: [] for c in CLASS_ORDER}
    extra: dict[str, list[tuple[str, Path]]] = {}
    for lab, path in sorted(items, key=lambda t: (t[0], t[1].name)):
        if lab in buckets:
            buckets[lab].append((lab, path))
        else:
            extra.setdefault(lab, []).append((lab, path))
    out: list[tuple[str, Path]] = []
    for lab in CLASS_ORDER:
        out.extend(buckets.get(lab, [])[:n])
    for lab in sorted(extra):
        out.extend(extra[lab][:n])
    return out


def first_image_per_class(items: list[tuple[str, Path]]) -> dict[str, tuple[str, Path]]:
    """Deterministic: first path after sorting by class then filename."""
    chosen: dict[str, tuple[str, Path]] = {}
    for lab, path in sorted(items, key=lambda t: (t[0], t[1].name)):
        if lab not in chosen:
            chosen[lab] = (lab, path)
    return chosen


def smoke_images() -> dict[str, tuple[str, Path]]:
    return first_image_per_class(official_test_images() or training_images())

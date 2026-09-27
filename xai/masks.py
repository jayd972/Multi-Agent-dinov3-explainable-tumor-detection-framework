"""Official BRISC segmentation mask loader.

Masks are NOT invented. If the official `segmentation_task/*/masks` tree is
absent, localization metrics are marked MASK_UNAVAILABLE.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

DEFAULT_IMAGE_ROOTS = [
    Path(r"C:\Users\darji\Downloads\BT_Images\Testing"),
    Path(r"C:\Users\darji\Downloads\BT_Images\test"),
    Path(r"C:\Users\darji\Downloads\BT_Images\Training"),
    Path(r"C:\Users\darji\Downloads\BT_Images\train"),
]

def _mask_search_roots() -> list[Path]:
    env = os.environ.get("BRISC_MASK_ROOT", "").strip()
    roots = []
    if env:
        roots.append(Path(env))
    roots.extend(
        [
            Path(r"C:\Users\darji\Downloads\BT_Images\segmentation_task\test\masks"),
            Path(r"C:\Users\darji\Downloads\BT_Images\segmentation_task\train\masks"),
            Path(r"C:\Users\darji\Downloads\brisc2025\segmentation_task\test\masks"),
            Path(r"C:\Users\darji\Downloads\brisc2025\segmentation_task\train\masks"),
        ]
    )
    return roots


def discover_mask_dir() -> Optional[Path]:
    for root in _mask_search_roots():
        if root.exists() and any(root.glob("*.png")):
            return root
    return None


def find_mask_for_image(image_path: Path) -> Optional[Path]:
    stem = Path(image_path).stem
    for mask_dir in _mask_search_roots():
        if not mask_dir.exists():
            continue
        cand = mask_dir / f"{stem}.png"
        if cand.exists():
            return cand
    parent = Path(image_path).parent
    for cand in (
        parent / f"{stem}.png",
        parent.parent / "masks" / f"{stem}.png",
        parent.parent.parent / "masks" / f"{stem}.png",
        parent.parent.parent / "segmentation_task" / "test" / "masks" / f"{stem}.png",
        parent.parent.parent / "segmentation_task" / "train" / "masks" / f"{stem}.png",
    ):
        if cand.exists():
            return cand
    return None


def load_mask(image_path: Path, size: Optional[tuple[int, int]] = None) -> Optional[np.ndarray]:
    mask_path = find_mask_for_image(image_path)
    if mask_path is None:
        return None
    mask = Image.open(mask_path).convert("L")
    if size is not None:
        mask = mask.resize(size, Image.NEAREST)
    arr = np.array(mask)
    return (arr > 0).astype(np.uint8)


def mask_status() -> dict:
    d = discover_mask_dir()
    return {
        "mask_dir": str(d) if d else None,
        "available": d is not None,
        "note": (
            "Official BRISC masks use identical basenames with .png extension "
            "under segmentation_task/{train,test}/masks/. "
            "Set BRISC_MASK_ROOT if they live elsewhere."
            if d is None
            else "Official mask directory found."
        ),
    }

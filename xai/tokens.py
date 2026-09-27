from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def num_register_tokens(backbone) -> int:
    cfg = getattr(backbone, "config", None)
    return int(getattr(cfg, "num_register_tokens", 0) or 0) if cfg is not None else 0


def patch_size(backbone) -> int:
    cfg = getattr(backbone, "config", None)
    return int(getattr(cfg, "patch_size", 16) or 16) if cfg is not None else 16


def image_size(backbone) -> int:
    cfg = getattr(backbone, "config", None)
    return int(getattr(cfg, "image_size", 224) or 224) if cfg is not None else 224


def token_layout(backbone, n_tokens: int) -> dict:
    n_reg = num_register_tokens(backbone)
    n_patches = n_tokens - 1 - n_reg
    gh, gw = patch_grid(n_patches, image_size(backbone), patch_size(backbone))
    return {
        "n_tokens": n_tokens,
        "cls_index": 0,
        "n_register_tokens": n_reg,
        "register_slice": [1, 1 + n_reg],
        "patch_slice": [1 + n_reg, n_tokens],
        "n_patches": n_patches,
        "grid_h": gh,
        "grid_w": gw,
        "token_order": "[CLS, registers..., patches...]",
    }


def patch_grid(n_patches: int, img_size: int, p_size: int) -> Tuple[int, int]:
    expected = (img_size // p_size) ** 2
    if n_patches == expected:
        s = img_size // p_size
        return s, s
    s = int(math.sqrt(n_patches))
    if s * s == n_patches:
        return s, s
    raise RuntimeError(
        f"Cannot reshape {n_patches} patch tokens into a grid "
        f"(expected {expected} for {img_size}x{img_size}, patch {p_size})."
    )


def strip_special_tokens(seq: torch.Tensor, backbone) -> torch.Tensor:
    """seq: (B, T, C) or (T,) -> patch tokens only."""
    n_reg = num_register_tokens(backbone)
    start = 1 + n_reg
    if seq.dim() == 1:
        return seq[start:]
    return seq[:, start:]


def force_eager_attention(backbone) -> str:
    before = getattr(getattr(backbone, "config", None), "_attn_implementation", None)
    for fn_name in ("set_attn_implementation", "set_default_attn_implementation"):
        fn = getattr(backbone, fn_name, None)
        if callable(fn):
            try:
                fn("eager")
                return f"set via {fn_name} (was {before})"
            except Exception:
                pass
    cfg = getattr(backbone, "config", None)
    if cfg is not None:
        cfg._attn_implementation = "eager"
        return f"set config._attn_implementation (was {before})"
    return "eager not set"


def upsample(heat: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(np.asarray(heat, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def normalize_minmax(heat: np.ndarray) -> np.ndarray:
    heat = np.asarray(heat, dtype=np.float32)
    lo, hi = float(np.min(heat)), float(np.max(heat))
    if hi - lo < 1e-12:
        raise RuntimeError("constant map cannot be normalized")
    return (heat - lo) / (hi - lo)

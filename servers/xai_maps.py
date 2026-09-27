"""
Shared attention-rollout and gradient-saliency maps for DINOv3.

The previous maps looked uniformly "diffused" for every image because of three
implementation bugs, not because the model has no spatial structure:

1. DINOv3 prepends register tokens after CLS. Treating CLS+registers+patches as
   a square grid (200 tokens) is not 14x14 and smears the map.
2. Full rollout from layer 0 multiplies in nearly-uniform early attention and
   washes out later spatial peaks.
3. Overlay used gamma < 1, which boosts background and hides peaks.

This module excludes register tokens, starts rollout at mid-depth, and uses
percentile clipping plus gamma > 1 for display.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _num_register_tokens(backbone) -> int:
    cfg = getattr(backbone, "config", None)
    if cfg is None:
        return 0
    return int(getattr(cfg, "num_register_tokens", 0) or 0)


def _force_eager_attention(backbone) -> None:
    for fn_name in ("set_attn_implementation", "set_default_attn_implementation"):
        fn = getattr(backbone, fn_name, None)
        if callable(fn):
            try:
                fn("eager")
                return
            except Exception:
                pass
    cfg = getattr(backbone, "config", None)
    if cfg is not None:
        try:
            cfg._attn_implementation = "eager"
        except Exception:
            pass


def _patch_grid_from_tokens(n_patches: int, image_size: int, patch_size: int) -> Tuple[int, int]:
    """Return (height, width) patch grid. Prefer the model's expected square grid."""
    expected = (image_size // patch_size) ** 2
    if n_patches == expected:
        s = image_size // patch_size
        return s, s
    s = int(math.sqrt(n_patches))
    if s * s == n_patches:
        return s, s
    raise RuntimeError(
        f"Cannot form a spatial grid from {n_patches} patch tokens "
        f"(expected {expected} for {image_size}x{image_size} with patch {patch_size})."
    )


def normalize_map(heat: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0, gamma: float = 1.8) -> np.ndarray:
    """Stretch a map so peaks stand out instead of a uniform wash."""
    heat = np.asarray(heat, dtype=np.float32)
    if heat.size == 0 or not np.isfinite(heat).any():
        return np.zeros_like(heat, dtype=np.float32)
    finite = heat[np.isfinite(heat)]
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi - lo < 1e-8:
        lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-8:
        return np.zeros_like(heat, dtype=np.float32)
    heat = np.clip((heat - lo) / (hi - lo), 0.0, 1.0)
    return np.power(heat, gamma).astype(np.float32)


def overlay_heatmap(pil: Image.Image, heat: np.ndarray, out_png, alpha: float = 0.50, sharpen: bool = True) -> None:
    import matplotlib.pyplot as plt

    if sharpen:
        heat = normalize_map(heat)
    else:
        heat = heat.astype(np.float32)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
        heat = np.power(heat, 0.6)

    cmap = plt.get_cmap("jet")
    colored = (cmap(heat)[..., :3] * 255).astype(np.uint8)
    hm = Image.fromarray(colored).resize(pil.size, Image.BICUBIC)
    base = pil.convert("RGB")
    # Keep anatomy visible; previous code darkened the MRI first and looked washed out.
    out = Image.blend(base, hm.convert("RGB"), alpha=alpha)
    from pathlib import Path
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    out.save(out_png)


def compute_attention_rollout(
    backbone,
    pixel_values: torch.Tensor,
    discard_first_layers: int = 6,
) -> np.ndarray:
    """
    Abnar & Zuidema attention rollout on patch tokens only.

    discard_first_layers=6 skips the first half of a 12-layer ViT-B, which is
    where attention is typically nearly uniform.
    """
    _force_eager_attention(backbone)
    device = pixel_values.device
    with torch.no_grad():
        outputs = backbone(pixel_values=pixel_values, output_attentions=True)
    attns = outputs.attentions
    if attns is None or len(attns) == 0:
        raise RuntimeError(
            "Model did not return attentions. DINOv3 defaults to sdpa; "
            "eager attention must be enabled."
        )

    n_reg = _num_register_tokens(backbone)
    start = min(max(discard_first_layers, 0), len(attns) - 1)
    rollout = None
    for layer_attn in attns[start:]:
        # (batch, heads, tokens, tokens) -> average heads
        A = layer_attn[0].mean(0)
        tokens = A.shape[-1]
        A = A + torch.eye(tokens, device=A.device, dtype=A.dtype)
        A = A / A.sum(dim=-1, keepdim=True)
        rollout = A if rollout is None else rollout @ A

    # Token order in DINOv2/v3: [CLS, registers..., patches...]
    patch_attn = rollout[0, 1 + n_reg :]
    n_patches = int(patch_attn.numel())
    cfg = backbone.config
    image_size = int(getattr(cfg, "image_size", 224) or 224)
    patch_size = int(getattr(cfg, "patch_size", 16) or 16)
    gh, gw = _patch_grid_from_tokens(n_patches, image_size, patch_size)
    heat = patch_attn.reshape(gh, gw).detach().cpu().float().numpy()
    return heat


def compute_last_layer_cls_attention(backbone, pixel_values: torch.Tensor) -> np.ndarray:
    """Last-layer CLS-to-patch attention. Often sharper than full rollout."""
    _force_eager_attention(backbone)
    with torch.no_grad():
        outputs = backbone(pixel_values=pixel_values, output_attentions=True)
    attns = outputs.attentions
    if attns is None or len(attns) == 0:
        raise RuntimeError("Model did not return attentions.")
    last = attns[-1][0].mean(0)  # (tokens, tokens)
    n_reg = _num_register_tokens(backbone)
    patch_attn = last[0, 1 + n_reg :]
    n_patches = int(patch_attn.numel())
    cfg = backbone.config
    image_size = int(getattr(cfg, "image_size", 224) or 224)
    patch_size = int(getattr(cfg, "patch_size", 16) or 16)
    gh, gw = _patch_grid_from_tokens(n_patches, image_size, patch_size)
    return patch_attn.reshape(gh, gw).detach().cpu().float().numpy()


def compute_gradient_saliency(
    model,
    pixel_values: torch.Tensor,
    target_class: int,
    n_smooth: int = 12,
    noise_sigma: float = 0.15,
) -> np.ndarray:
    """
    Class-targeted SmoothGrad saliency.

    Single-backprop ViT gradients are spatially noisy and look similar across
    images. Averaging noisy copies concentrates the map on stable peaks.
    """
    model.eval()
    acc = None
    for _ in range(max(1, n_smooth)):
        noise = torch.randn_like(pixel_values) * noise_sigma
        x = (pixel_values + noise).detach().requires_grad_(True)
        logits = model(pixel_values=x)
        score = logits[0, target_class]
        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        score.backward()
        g = x.grad.detach().abs().mean(dim=1)[0]
        acc = g if acc is None else acc + g
    sal = (acc / max(1, n_smooth)).cpu().numpy()
    return sal


def upsample_map(heat: np.ndarray, size: int = 224) -> np.ndarray:
    t = torch.from_numpy(np.asarray(heat, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()

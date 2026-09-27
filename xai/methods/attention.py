from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

from xai.methods._common import finish, predict_from_pixels
from xai.tokens import (
    force_eager_attention,
    image_size,
    num_register_tokens,
    patch_grid,
    patch_size,
    token_layout,
)


def _fuse_heads(attn_bht: torch.Tensor, how: str) -> torch.Tensor:
    """attn: (H, T, T) -> (T, T)"""
    if how == "mean":
        return attn_bht.mean(0)
    if how == "max":
        return attn_bht.max(0).values
    if how == "min":
        return attn_bht.min(0).values
    raise ValueError(how)


def run_attention_rollout(
    model,
    pixel_values: torch.Tensor,
    target_class: int,
    checkpoint_info: dict,
    config: dict | None = None,
    mask: np.ndarray | None = None,
    pil: Image.Image | None = None,
):
    """Abnar & Zuidema rollout with residual identity and row normalization.

    Not class-specific. target_class is recorded but does not change the map.
    """
    cfg = {"head_fusion": "mean", "discard_first_layers": 0, **(config or {})}
    t0 = time.time()
    backend_note = force_eager_attention(model.backbone)
    pred, probs = predict_from_pixels(model, pixel_values)
    with torch.no_grad():
        outputs = model.backbone(pixel_values=pixel_values, output_attentions=True)
    attns = outputs.attentions
    if attns is None or len(attns) == 0:
        raise RuntimeError("Attention rollout requires eager attentions")

    start = min(max(int(cfg["discard_first_layers"]), 0), len(attns) - 1)
    rollout = None
    for layer_attn in attns[start:]:
        A = _fuse_heads(layer_attn[0], cfg["head_fusion"])
        tokens = A.shape[-1]
        A = A + torch.eye(tokens, device=A.device, dtype=A.dtype)
        A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        rollout = A if rollout is None else rollout @ A

    n_reg = num_register_tokens(model.backbone)
    patch = rollout[0, 1 + n_reg :].detach().cpu().float().numpy()
    gh, gw = patch_grid(patch.size, image_size(model.backbone), patch_size(model.backbone))
    heat = patch.reshape(gh, gw)
    row_sums = rollout.sum(dim=-1).detach().cpu().numpy()
    h = w = pixel_values.shape[-1]
    return finish(
        method="attention_rollout",
        algorithm="Abnar & Zuidema attention rollout (not class-specific)",
        raw=heat,
        target_class=target_class,
        pred=pred,
        probs=probs,
        input_hw=(h, w),
        config={
            "head_fusion": cfg["head_fusion"],
            "discard_first_layers": start,
            "class_specific": False,
            "row_normalization": "verified",
            "residual_identity": True,
            "row_sum_mean": float(np.mean(row_sums)),
            "token_layout": token_layout(model.backbone, int(rollout.shape[0])),
            "attention_backend": backend_note,
            "checkpoint": checkpoint_info.get("checkpoint_path"),
            "interpretation": "Inspects attention flow. Not proof of prediction causality.",
        },
        t0=t0,
        warning="Attention rollout is not class-specific and is not a causal explanation.",
        upsample_to=h,
    )


def last_layer_attention(
    model,
    pixel_values: torch.Tensor,
    target_class: int,
    checkpoint_info: dict,
    config: dict | None = None,
    mask: np.ndarray | None = None,
    pil: Image.Image | None = None,
):
    """Supplementary last-layer CLS-to-patch attention (mean/max/min + per-head)."""
    cfg = {**(config or {})}
    t0 = time.time()
    force_eager_attention(model.backbone)
    pred, probs = predict_from_pixels(model, pixel_values)
    with torch.no_grad():
        outputs = model.backbone(pixel_values=pixel_values, output_attentions=True)
    last = outputs.attentions[-1][0]  # (H, T, T)
    n_reg = num_register_tokens(model.backbone)
    heads = []
    for hi in range(last.shape[0]):
        patch = last[hi, 0, 1 + n_reg :].detach().cpu().float().numpy()
        gh, gw = patch_grid(patch.size, image_size(model.backbone), patch_size(model.backbone))
        heads.append(patch.reshape(gh, gw))
    heads = np.stack(heads, axis=0)
    mean_map = heads.mean(0)
    max_map = heads.max(0)
    h = w = pixel_values.shape[-1]
    return finish(
        method="last_layer_attention",
        algorithm="last-layer CLS-to-patch attention (supplementary)",
        raw=mean_map,
        target_class=target_class,
        pred=pred,
        probs=probs,
        input_hw=(h, w),
        config={
            "n_heads": int(heads.shape[0]),
            "role": "supplementary",
            "class_specific": False,
            "checkpoint": checkpoint_info.get("checkpoint_path"),
        },
        t0=t0,
        extras={"per_head": heads, "mean": mean_map, "max": max_map, "min": heads.min(0)},
        warning="Supplementary only. Not the main explanation method.",
        upsample_to=h,
    )

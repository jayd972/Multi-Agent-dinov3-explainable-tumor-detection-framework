"""Chefer Transformer Attribution (fallback for AttnLRP).

Exact AttnLRP requires LRP-compatible custom implementations of every DINOv3
block (attention, residuals, LayerNorm, MLP). The HuggingFace DINOv3 module
uses fused SDPA / standard autograd ops and does not expose LRP rules.

Chefer et al. 2021 ("Transformer Interpretability Beyond Attention Visualization")
is implemented instead and named `chefer_attribution` everywhere.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

from xai.methods._common import class_score, finish, predict_from_pixels
from xai.tokens import (
    force_eager_attention,
    image_size,
    num_register_tokens,
    patch_grid,
    patch_size,
    token_layout,
)

ATTLRP_FALLBACK_REASON = (
    "Exact AttnLRP is not implemented because facebook/dinov3-vitb16-pretrain-lvd1689m "
    "is loaded through HuggingFace transformers with fused attention and standard LayerNorm/"
    "MLP modules. Those layers do not provide the relevance-conservation rules AttnLRP requires. "
    "Chefer Transformer Attribution is used as the named fallback."
)


def run_chefer_attribution(
    model,
    pixel_values: torch.Tensor,
    target_class: int,
    checkpoint_info: dict,
    config: dict | None = None,
    mask: np.ndarray | None = None,
    pil: Image.Image | None = None,
):
    cfg = {**(config or {})}
    t0 = time.time()
    backend_note = force_eager_attention(model.backbone)
    pred, probs = predict_from_pixels(model, pixel_values)

    model.zero_grad(set_to_none=True)
    outputs = model.backbone(pixel_values=pixel_values, output_attentions=True)
    attns = outputs.attentions
    if attns is None or len(attns) == 0:
        raise RuntimeError("Chefer attribution requires eager attentions")

    # Gradients of the class score w.r.t. each attention tensor
    cls = outputs.last_hidden_state[:, 0]
    logits = model.classifier(cls)
    score = logits[0, target_class]
    grads = torch.autograd.grad(score, attns, retain_graph=False)
    if any(g is None for g in grads):
        raise RuntimeError("Attention tensors did not receive gradients")

    tokens = attns[0].shape[-1]
    eye = torch.eye(tokens, device=pixel_values.device, dtype=attns[0].dtype)
    relevance = eye.clone()
    for attn, grad in zip(attns, grads):
        # attn, grad: (B, H, T, T)
        cam = (grad * attn).clamp(min=0).mean(dim=1)[0]  # (T, T)
        cam = cam + eye
        cam = cam / cam.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        relevance = relevance @ cam

    n_reg = num_register_tokens(model.backbone)
    patch = relevance[0, 1 + n_reg :].detach().cpu().float().numpy()
    gh, gw = patch_grid(patch.size, image_size(model.backbone), patch_size(model.backbone))
    heat = patch.reshape(gh, gw)
    h = w = pixel_values.shape[-1]
    return finish(
        method="chefer_attribution",
        algorithm="Chefer Transformer Attribution (fallback for AttnLRP)",
        raw=heat,
        target_class=target_class,
        pred=pred,
        probs=probs,
        input_hw=(h, w),
        config={
            "attnlrp_status": "NOT IMPLEMENTED",
            "fallback": "chefer_attribution",
            "fallback_reason": ATTLRP_FALLBACK_REASON,
            "token_layout": token_layout(model.backbone, tokens),
            "attention_backend": backend_note,
            "checkpoint": checkpoint_info.get("checkpoint_path"),
        },
        t0=t0,
        warning="AttnLRP unavailable; Chefer Transformer Attribution used.",
        upsample_to=h,
    )

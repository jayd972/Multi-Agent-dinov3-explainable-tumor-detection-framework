from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

from xai.methods._common import class_score, find_encoder_layers, finish, predict_from_pixels
from xai.tokens import (
    force_eager_attention,
    image_size,
    num_register_tokens,
    patch_grid,
    patch_size,
    token_layout,
)


def run_gradcam(
    model,
    pixel_values: torch.Tensor,
    target_class: int,
    checkpoint_info: dict,
    config: dict | None = None,
    mask: np.ndarray | None = None,
    pil: Image.Image | None = None,
):
    """ViT Grad-CAM on a configurable transformer block.

    Uses token activations of the selected block. CLS and register tokens are
    removed before the spatial reshape. Gradients are taken w.r.t. that block.
    """
    # Default -2: last-block patch tokens get no gradient because the head
    # reads only the CLS token after the final attention mix.
    cfg = {"target_block": -2, "relu": True, **(config or {})}
    t0 = time.time()
    backend_note = force_eager_attention(model.backbone)
    pred, probs = predict_from_pixels(model, pixel_values)

    layers = find_encoder_layers(model.backbone)
    requested = cfg["target_block"]
    block_idx = requested if requested >= 0 else len(layers) + requested
    if not (0 <= block_idx < len(layers)):
        raise IndexError(f"target_block {cfg['target_block']} out of range 0..{len(layers)-1}")

    n_reg = num_register_tokens(model.backbone)
    warning = None
    act = grads = None
    used_idx = block_idx
    for used_idx in range(block_idx, -1, -1):
        stored = {}

        def hook(_mod, _inp, out, _store=stored):
            hidden = out[0] if isinstance(out, tuple) else out
            _store["act"] = hidden
            hidden.retain_grad()

        handle = layers[used_idx].register_forward_hook(hook)
        try:
            model.zero_grad(set_to_none=True)
            score = class_score(model, pixel_values, target_class)
            if "act" not in stored:
                raise RuntimeError("Grad-CAM hook did not capture activations")
            score.backward()
            act = stored["act"]
            if act.grad is None:
                continue
            grads = act.grad[0, 1 + n_reg :]
            if float(grads.detach().abs().sum().cpu()) > 0:
                if used_idx != block_idx:
                    warning = (
                        f"Block {block_idx} patch tokens received no gradient "
                        f"(CLS-only head). Used block {used_idx} instead."
                    )
                break
        finally:
            handle.remove()
    else:
        raise RuntimeError(
            "No transformer block produced non-zero patch-token gradients. "
            "Enable eager attention and confirm blocks are on the forward path."
        )

    patches = act[0, 1 + n_reg :]
    weights = grads.mean(dim=0)
    cam = (patches * weights).sum(dim=-1)
    if cfg["relu"]:
        cam_relu = torch.relu(cam)
        if float((cam_relu.max() - cam_relu.min()).detach().cpu()) < 1e-12:
            warning = (warning + "; " if warning else "") + "ReLU Grad-CAM was constant; signed CAM used instead"
        else:
            cam = cam_relu
    cam_np = cam.detach().cpu().float().numpy()
    block_idx = used_idx

    n_patches = cam_np.size
    gh, gw = patch_grid(n_patches, image_size(model.backbone), patch_size(model.backbone))
    heat = cam_np.reshape(gh, gw)
    layout = token_layout(model.backbone, int(act.shape[1]))
    h = w = pixel_values.shape[-1]
    return finish(
        method="gradcam",
        algorithm="ViT-GradCAM (Selvaraju et al., token activations + reshape)",
        raw=heat,
        target_class=target_class,
        pred=pred,
        probs=probs,
        input_hw=(h, w),
        config={
            "target_block": block_idx,
            "n_blocks": len(layers),
            "relu": cfg["relu"],
            "normalization": "min-max after bilinear upsample",
            "token_layout": layout,
            "attention_backend": backend_note,
            "checkpoint": checkpoint_info.get("checkpoint_path"),
        },
        t0=t0,
        extras={"layer_received_gradients": True},
        upsample_to=h,
        warning=warning,
    )

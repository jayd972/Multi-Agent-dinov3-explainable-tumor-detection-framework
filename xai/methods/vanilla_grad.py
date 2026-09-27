"""Vanilla input gradient (the only input-gradient baseline).

There is no separate 'salience' algorithm in this repository. The older
`grad_saliency` / SmoothGrad outputs are this method (absolute input gradient,
optionally averaged over noise). They are not presented as a second major method.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

from xai.methods._common import finish, predict_from_pixels


def run_vanilla_gradient(
    model,
    pixel_values: torch.Tensor,
    target_class: int,
    checkpoint_info: dict,
    config: dict | None = None,
    mask: np.ndarray | None = None,
    pil: Image.Image | None = None,
):
    cfg = {"smooth_samples": 1, "noise_sigma": 0.0, **(config or {})}
    t0 = time.time()
    pred, probs = predict_from_pixels(model, pixel_values)
    n = max(1, int(cfg["smooth_samples"]))
    acc = None
    for _ in range(n):
        noise = torch.randn_like(pixel_values) * float(cfg["noise_sigma"])
        x = (pixel_values + noise).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(pixel_values=x)[0, target_class].backward()
        g = x.grad.detach().abs().mean(dim=1)[0]
        acc = g if acc is None else acc + g
    heat = (acc / n).cpu().numpy()
    algo = "vanilla input-gradient" if n == 1 else f"SmoothGrad (n={n})"
    h = w = pixel_values.shape[-1]
    return finish(
        method="vanilla_gradient",
        algorithm=algo,
        raw=heat,
        target_class=target_class,
        pred=pred,
        probs=probs,
        input_hw=(h, w),
        config={
            "smooth_samples": n,
            "noise_sigma": cfg["noise_sigma"],
            "role": "supplementary baseline",
            "not_a_second_major_method": True,
            "checkpoint": checkpoint_info.get("checkpoint_path"),
        },
        t0=t0,
        warning="Supplementary input-gradient baseline, not a second major XAI method.",
    )

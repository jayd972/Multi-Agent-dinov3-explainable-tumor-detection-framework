from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from xai.methods._common import finish, predict_from_pixels


def _baseline(pixel_values: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "zero":
        return torch.zeros_like(pixel_values)
    if kind == "mean":
        return pixel_values.mean(dim=(2, 3), keepdim=True).expand_as(pixel_values)
    if kind == "blur":
        return F.avg_pool2d(pixel_values, kernel_size=21, stride=1, padding=10)
    raise ValueError(f"Unknown IG baseline: {kind}")


def run_integrated_gradients(
    model,
    pixel_values: torch.Tensor,
    target_class: int,
    checkpoint_info: dict,
    config: dict | None = None,
    mask: np.ndarray | None = None,
    pil: Image.Image | None = None,
):
    cfg = {"baseline": "zero", "steps": 32, **(config or {})}
    t0 = time.time()
    pred, probs = predict_from_pixels(model, pixel_values)
    x = pixel_values.detach()
    base = _baseline(x, cfg["baseline"])
    steps = int(cfg["steps"])
    alphas = torch.linspace(0.0, 1.0, steps, device=x.device)
    grads = []
    for a in alphas:
        xi = (base + a * (x - base)).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        score = model(pixel_values=xi)[0, target_class]
        score.backward()
        grads.append(xi.grad.detach())
    avg_grad = torch.stack(grads, dim=0).mean(dim=0)
    ig = (x - base) * avg_grad
    with torch.no_grad():
        score_x = model(pixel_values=x)[0, target_class]
        score_b = model(pixel_values=base)[0, target_class]
    conv_delta = float((score_x - score_b - ig.sum()).item())

    ig_np = ig[0].detach().cpu().float().numpy()  # (C,H,W)
    pos = np.clip(ig_np, 0, None).sum(axis=0)
    neg = np.clip(-ig_np, 0, None).sum(axis=0)
    combined = pos  # positive evidence for the target class
    note = (
        "Combined map is the sum of positive channel attributions. "
        "Negative evidence is saved separately in negative_map and was not folded into abs()."
    )
    h = w = x.shape[-1]
    return finish(
        method="integrated_gradients",
        algorithm="Integrated Gradients (Sundararajan et al.)",
        raw=combined,
        target_class=target_class,
        pred=pred,
        probs=probs,
        input_hw=(h, w),
        config={
            "baseline": cfg["baseline"],
            "steps": steps,
            "convergence_delta": conv_delta,
            "combination": "positive-only sum over channels",
            "checkpoint": checkpoint_info.get("checkpoint_path"),
        },
        t0=t0,
        warning=note,
        positive_map=pos,
        negative_map=neg,
        extras={"convergence_delta": conv_delta, "steps": steps},
        upsample_to=None,
    )

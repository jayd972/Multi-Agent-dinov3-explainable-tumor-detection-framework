from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from xai.methods._common import finish, predict_from_pixels


def _fill(patch: torch.Tensor, kind: str, src: torch.Tensor) -> torch.Tensor:
    if kind == "zero":
        return torch.zeros_like(patch)
    if kind == "mean":
        return src.mean(dim=(2, 3), keepdim=True).expand_as(patch)
    if kind == "blur":
        return F.avg_pool2d(src, kernel_size=21, stride=1, padding=10)
    raise ValueError(kind)


def run_occlusion(
    model,
    pixel_values: torch.Tensor,
    target_class: int,
    checkpoint_info: dict,
    config: dict | None = None,
    mask: np.ndarray | None = None,
    pil: Image.Image | None = None,
):
    """Heatmap value = baseline_prob(target) - occluded_prob(target).

    Positive: masking that region *decreases* the target class probability
    (region supports the prediction). Negative: masking *increases* it.
    """
    cfg = {
        "patch_size": 16,
        "stride": 8,
        "batch_size": 16,
        "baseline": "zero",
        "mask_shape": "square",
        **(config or {}),
    }
    t0 = time.time()
    pred, probs = predict_from_pixels(model, pixel_values)
    with torch.no_grad():
        base_prob = torch.softmax(model(pixel_values=pixel_values), dim=1)[0, target_class].item()

    x = pixel_values
    _, _, h, w = x.shape
    ps, st = int(cfg["patch_size"]), int(cfg["stride"])
    coords = [(y, x0) for y in range(0, h - ps + 1, st) for x0 in range(0, w - ps + 1, st)]
    heat = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    for i in range(0, len(coords), cfg["batch_size"]):
        batch_coords = coords[i : i + cfg["batch_size"]]
        batch = x.repeat(len(batch_coords), 1, 1, 1)
        for j, (yy, xx) in enumerate(batch_coords):
            fill = _fill(batch[j : j + 1, :, yy : yy + ps, xx : xx + ps], cfg["baseline"], x)
            if cfg["mask_shape"] == "circle":
                yy_g, xx_g = torch.meshgrid(
                    torch.arange(ps, device=x.device),
                    torch.arange(ps, device=x.device),
                    indexing="ij",
                )
                circ = ((yy_g - ps / 2) ** 2 + (xx_g - ps / 2) ** 2) <= (ps / 2) ** 2
                orig = batch[j, :, yy : yy + ps, xx : xx + ps]
                batch[j, :, yy : yy + ps, xx : xx + ps] = torch.where(circ, fill[0], orig)
            else:
                batch[j, :, yy : yy + ps, xx : xx + ps] = fill[0]
        with torch.no_grad():
            occ = torch.softmax(model(pixel_values=batch), dim=1)[:, target_class]
        for j, (yy, xx) in enumerate(batch_coords):
            delta = base_prob - float(occ[j].item())
            heat[yy : yy + ps, xx : xx + ps] += delta
            count[yy : yy + ps, xx : xx + ps] += 1

    count = np.maximum(count, 1.0)
    heat = heat / count
    pos = np.clip(heat, 0, None)
    neg = np.clip(-heat, 0, None)
    return finish(
        method="occlusion",
        algorithm="sliding-window class-probability occlusion",
        raw=heat,
        target_class=target_class,
        pred=pred,
        probs=probs,
        input_hw=(h, w),
        config={
            **{k: cfg[k] for k in ("patch_size", "stride", "batch_size", "baseline", "mask_shape")},
            "value_meaning": "baseline_P(target) - occluded_P(target); positive = supportive region",
            "checkpoint": checkpoint_info.get("checkpoint_path"),
        },
        t0=t0,
        positive_map=pos,
        negative_map=neg,
    )

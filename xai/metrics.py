from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def _trapz(y) -> float:
    y = np.asarray(y, dtype=np.float64)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y))
    return float(np.trapz(y))


def _align(heat: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    heat = np.asarray(heat, dtype=np.float32)
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if heat.shape != mask.shape:
        scale = (mask.shape[0] / heat.shape[0], mask.shape[1] / heat.shape[1])
        heat = zoom(heat, scale, order=1)
    return heat, mask


def dice(heat_bin: np.ndarray, mask: np.ndarray) -> float:
    inter = np.logical_and(heat_bin, mask).sum()
    denom = heat_bin.sum() + mask.sum()
    return float(2 * inter / denom) if denom else 0.0


def iou(heat_bin: np.ndarray, mask: np.ndarray) -> float:
    inter = np.logical_and(heat_bin, mask).sum()
    union = np.logical_or(heat_bin, mask).sum()
    return float(inter / union) if union else 0.0


def pointing_game(heat: np.ndarray, mask: np.ndarray) -> float:
    heat, mask = _align(heat, mask)
    if mask.sum() == 0:
        return float("nan")
    y, x = np.unravel_index(int(np.argmax(heat)), heat.shape)
    return float(mask[y, x] > 0)


def relevance_mass(heat: np.ndarray, mask: np.ndarray) -> dict:
    heat, mask = _align(np.clip(heat, 0, None), mask)
    total = float(heat.sum()) + 1e-12
    inside = float(heat[mask > 0].sum())
    return {
        "mass_inside": inside / total,
        "mass_outside": 1.0 - inside / total,
        "mass_inside_raw": inside,
        "mass_outside_raw": total - inside,
    }


def threshold_map(heat: np.ndarray, t: float, normalize: str = "minmax") -> np.ndarray:
    h = np.asarray(heat, dtype=np.float32)
    if normalize == "minmax":
        lo, hi = float(h.min()), float(h.max())
        h = (h - lo) / (hi - lo + 1e-12)
    return (h >= t).astype(np.uint8)


def insertion_deletion_auc(
    model,
    pixel_values,
    heat: np.ndarray,
    target_class: int,
    n_steps: int = 20,
    device=None,
) -> dict:
    import torch

    heat = np.asarray(heat, dtype=np.float32)
    if heat.shape != tuple(pixel_values.shape[-2:]):
        heat = zoom(heat, (pixel_values.shape[-2] / heat.shape[0], pixel_values.shape[-1] / heat.shape[1]), order=1)
    order = np.argsort(heat.ravel())[::-1]
    x = pixel_values.detach().clone()
    mean_val = x.mean()
    n_pix = order.size
    step = max(1, n_pix // n_steps)

    def prob_of(t):
        with torch.no_grad():
            return float(torch.softmax(model(pixel_values=t), dim=1)[0, target_class].item())

    del_curve, ins_curve = [], []
    x_del = x.clone()
    x_ins = torch.full_like(x, float(mean_val))
    del_curve.append(prob_of(x_del))
    ins_curve.append(prob_of(x_ins))
    for s in range(n_steps):
        idx = order[s * step : (s + 1) * step]
        ys, xs = np.unravel_index(idx, heat.shape)
        x_del[:, :, ys, xs] = mean_val
        x_ins[:, :, ys, xs] = x[:, :, ys, xs]
        del_curve.append(prob_of(x_del))
        ins_curve.append(prob_of(x_ins))
    return {
        "deletion_auc": float(_trapz(del_curve) / (len(del_curve) - 1)),
        "insertion_auc": float(_trapz(ins_curve) / (len(ins_curve) - 1)),
        "deletion_curve": del_curve,
        "insertion_curve": ins_curve,
    }


def tumor_occlusion_drop(model, pixel_values, mask: np.ndarray, target_class: int) -> float:
    import torch
    from scipy.ndimage import zoom as _zoom

    mask = (mask > 0).astype(np.uint8)
    if mask.shape != tuple(pixel_values.shape[-2:]):
        mask = _zoom(mask.astype(np.float32), (pixel_values.shape[-2] / mask.shape[0], pixel_values.shape[-1] / mask.shape[1]), order=0) > 0.5
    with torch.no_grad():
        p0 = float(torch.softmax(model(pixel_values=pixel_values), dim=1)[0, target_class].item())
        x = pixel_values.clone()
        ys, xs = np.where(mask)
        x[:, :, ys, xs] = 0
        p1 = float(torch.softmax(model(pixel_values=x), dim=1)[0, target_class].item())
    return p0 - p1


def localization_bundle(heat, mask, threshold: float, is_tumor: bool) -> dict:
    if not is_tumor:
        return {
            "dice": None,
            "iou": None,
            "pointing_game": None,
            "note": "No-tumor images: Dice/IoU/pointing game are not defined.",
        }
    if mask is None:
        return {
            "dice": None,
            "iou": None,
            "pointing_game": None,
            "note": "MASK_UNAVAILABLE",
        }
    heat_a, mask_a = _align(heat, mask)
    binary = threshold_map(heat_a, threshold)
    mass = relevance_mass(heat_a, mask_a)
    return {
        "dice": dice(binary, mask_a),
        "iou": iou(binary, mask_a),
        "pointing_game": pointing_game(heat_a, mask_a),
        **mass,
        "threshold": threshold,
        "normalization_before_threshold": "minmax",
    }

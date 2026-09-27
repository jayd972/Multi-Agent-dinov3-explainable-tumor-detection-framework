from __future__ import annotations

import copy

import numpy as np
import torch

from xai.tokens import normalize_minmax


def _ssim_fallback(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2) + 1e-12))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    r, _ = spearmanr(a.ravel(), b.ravel())
    return float(r) if np.isfinite(r) else 0.0


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_minmax(a) if np.ptp(a) > 0 else a
    b = normalize_minmax(b) if np.ptp(b) > 0 else b
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(a, b, data_range=1.0))
    except Exception:
        return _ssim_fallback(a, b)


def topk_overlap(a: np.ndarray, b: np.ndarray, k_frac: float = 0.1) -> float:
    k = max(1, int(a.size * k_frac))
    ia = set(np.argsort(a.ravel())[-k:])
    ib = set(np.argsort(b.ravel())[-k:])
    return len(ia & ib) / k


def compare_maps(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        raise ValueError("map shape mismatch")
    ta = (normalize_minmax(a) >= 0.5).astype(np.uint8) if np.ptp(a) > 0 else np.zeros_like(a, dtype=np.uint8)
    tb = (normalize_minmax(b) >= 0.5).astype(np.uint8) if np.ptp(b) > 0 else np.zeros_like(b, dtype=np.uint8)
    inter = np.logical_and(ta, tb).sum()
    union = np.logical_or(ta, tb).sum()
    return {
        "spearman": spearman(a, b),
        "ssim": ssim(a, b),
        "iou": float(inter / union) if union else 0.0,
        "topk_overlap": topk_overlap(a, b),
    }


def is_constant(heat: np.ndarray) -> bool:
    return float(np.nanstd(heat)) < 1e-12


def has_invalid(heat: np.ndarray) -> bool:
    return bool(np.any(~np.isfinite(heat)))


def randomize_classifier_head(model):
    with torch.no_grad():
        for p in model.classifier.parameters():
            p.copy_(torch.randn_like(p) * p.std().clamp_min(1e-3))


def randomize_block(model, block_idx: int):
    from xai.methods._common import find_encoder_layers

    layer = find_encoder_layers(model.backbone)[block_idx]
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.randn_like(p) * p.std().clamp_min(1e-3))


def mark_sanity_failure(corr: dict, threshold: float = 0.9) -> bool:
    """Nearly identical maps after randomization fail the sanity test."""
    return corr["spearman"] >= threshold and corr["ssim"] >= threshold

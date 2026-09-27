from __future__ import annotations

import numpy as np
import pytest
import torch

from xai.metrics import (
    dice,
    insertion_deletion_auc,
    iou,
    localization_bundle,
    pointing_game,
    relevance_mass,
)
from xai.result import ExplanationResult
from xai.sanity import has_invalid, is_constant
from xai.tokens import (
    num_register_tokens,
    patch_grid,
    strip_special_tokens,
    token_layout,
)


class _Cfg:
    num_register_tokens = 4
    patch_size = 16
    image_size = 224


class _BB:
    config = _Cfg()


def test_class_and_register_token_removal():
    bb = _BB()
    n_reg = num_register_tokens(bb)
    assert n_reg == 4
    seq = torch.arange(201).float()[None, :, None]  # CLS + 4 reg + 196
    patches = strip_special_tokens(seq, bb)
    assert patches.shape[1] == 196
    layout = token_layout(bb, 201)
    assert layout["patch_slice"] == [5, 201]
    assert layout["n_register_tokens"] == 4
    assert layout["cls_index"] == 0


def test_patch_grid_reshape():
    gh, gw = patch_grid(196, 224, 16)
    assert (gh, gw) == (14, 14)
    heat = np.arange(196, dtype=np.float32).reshape(14, 14)
    assert heat.shape == (14, 14)
    with pytest.raises(RuntimeError):
        patch_grid(200, 224, 16)


def test_dice_iou_pointing():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    heat = np.zeros((8, 8), dtype=np.float32)
    heat[2:6, 2:6] = 1
    binary = (heat >= 0.5).astype(np.uint8)
    assert dice(binary, mask) == 1.0
    assert iou(binary, mask) == 1.0
    assert pointing_game(heat, mask) == 1.0
    mass = relevance_mass(heat, mask)
    assert mass["mass_inside"] == pytest.approx(1.0)


def test_no_tumor_skips_overlap():
    heat = np.random.rand(8, 8).astype(np.float32)
    loc = localization_bundle(heat, np.ones((8, 8)), 0.5, is_tumor=False)
    assert loc["dice"] is None and loc["iou"] is None
    loc2 = localization_bundle(heat, None, 0.5, is_tumor=True)
    assert loc2["note"] == "MASK_UNAVAILABLE"


def test_constant_and_invalid_detection():
    assert is_constant(np.ones((4, 4)))
    assert has_invalid(np.array([[np.nan, 1.0]]))
    with pytest.raises(RuntimeError):
        ExplanationResult(
            method="x",
            algorithm="x",
            original_attribution=np.ones((4, 4)),
            normalized_map=np.ones((4, 4)),
            target_class=0,
            predicted_class=0,
            class_probabilities=[1, 0, 0, 0],
            input_hw=(4, 4),
            attribution_hw=(4, 4),
            config={},
            runtime_s=0,
        ).validate()


def test_insertion_deletion_toy(tiny):
    x = torch.ones(1, 3, 16, 16)
    heat = np.ones((16, 16), dtype=np.float32)
    heat[:8, :8] = 2
    out = insertion_deletion_auc(tiny, x, heat, target_class=0, n_steps=4)
    assert np.isfinite(out["deletion_auc"])
    assert np.isfinite(out["insertion_auc"])
    assert len(out["deletion_curve"]) == 5

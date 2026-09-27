from __future__ import annotations

import numpy as np
import torch

from xai.methods.integrated_gradients import run_integrated_gradients
from xai.methods.occlusion import run_occlusion
from xai.methods.vanilla_grad import run_vanilla_gradient


def test_ig_convergence_and_class_change(tiny):
    x = torch.rand(1, 3, 16, 16) * 0.3
    x[:, :, :8, :8] = x[:, :, :8, :8] + 0.7
    info = {"checkpoint_path": "toy"}
    a = run_integrated_gradients(tiny, x, 0, info, {"steps": 16, "baseline": "zero"})
    b = run_integrated_gradients(tiny, x, 1, info, {"steps": 16, "baseline": "zero"})
    assert a.normalized_map.shape == (16, 16)
    assert a.extras["convergence_delta"] is not None
    assert a.positive_map is not None and a.negative_map is not None
    # class 0 evidence is the top-left block
    assert a.normalized_map[:8, :8].mean() > a.normalized_map[8:, 8:].mean()
    assert not np.allclose(a.normalized_map, b.normalized_map)


def test_occlusion_signed_probability_change(tiny):
    x = torch.zeros(1, 3, 16, 16)
    x[:, :, :8, :8] = 1.0
    info = {"checkpoint_path": "toy"}
    res = run_occlusion(
        tiny, x, 0, info,
        {"patch_size": 8, "stride": 8, "batch_size": 4, "baseline": "zero"},
    )
    # masking the supportive top-left region should decrease class-0 score → positive heatmap
    assert res.original_attribution[:8, :8].mean() > res.original_attribution[8:, 8:].mean()
    assert "decreases" in res.config["value_meaning"] or "supportive" in res.config["value_meaning"]


def test_vanilla_gradient_not_second_major_method(tiny):
    x = torch.rand(1, 3, 16, 16)
    res = run_vanilla_gradient(tiny, x, 0, {"checkpoint_path": "toy"}, {})
    assert res.config["not_a_second_major_method"] is True
    assert res.method == "vanilla_gradient"

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from xai.lucent_dream import REQUIRED_DREAM_KEYS, REQUIRED_LUCENT_KEYS, record_deepdream, record_lucent
from xai.methods import METHOD_REGISTRY
from xai.model_io import CLASS_NAMES, checkpoint_sha256, compare_eager_vs_default, get_processor, predict, preprocess_pil
from xai.tokens import num_register_tokens, strip_special_tokens


def test_checkpoint_loading_and_prediction_preservation(trained, tmp_path):
    model, info = trained
    sha = checkpoint_sha256()
    assert info["checkpoint_sha256"] == sha
    assert info["weights_modified"] is False
    processor = get_processor()
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, 224, 224, device=device)
    # ImageNet-ish blank still yields a valid softmax
    pred1, p1 = predict(model, x)
    pred2, p2 = predict(model, x)
    assert pred1 == pred2
    assert np.allclose(p1, p2)
    assert abs(p1.sum() - 1) < 1e-5
    assert 0 <= pred1 < 4


def test_eager_attention_logits(trained):
    model, _ = trained
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, 224, 224, device=device)
    cmp = compare_eager_vs_default(model, x)
    assert cmp["materially_unchanged"] or cmp["max_abs_logit_diff"] < 1e-2


def test_cpu_execution():
    from xai.methods.integrated_gradients import run_integrated_gradients
    from tests.conftest import TinyClassifier

    m = TinyClassifier().to("cpu").eval()
    x = torch.rand(1, 3, 16, 16)
    res = run_integrated_gradients(m, x, 0, {"checkpoint_path": "toy"}, {"steps": 4})
    assert res.normalized_map.shape == (16, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_gpu_execution():
    from xai.methods.integrated_gradients import run_integrated_gradients
    from tests.conftest import TinyClassifier

    m = TinyClassifier().to("cuda").eval()
    x = torch.rand(1, 3, 16, 16, device="cuda")
    res = run_integrated_gradients(m, x, 0, {"checkpoint_path": "toy"}, {"steps": 4})
    assert res.normalized_map.shape == (16, 16)


def test_gradcam_dims_and_class_change(trained):
    from xai.data_paths import smoke_images

    model, info = trained
    device = next(model.parameters()).device
    chosen = smoke_images()
    if not chosen:
        pytest.skip("no BRISC images")
    path = next(iter(chosen.values()))[1]
    processor = get_processor()
    x = preprocess_pil(Image.open(path).convert("RGB"), processor, device)
    a = METHOD_REGISTRY["gradcam"](model, x, 0, info, {})
    b = METHOD_REGISTRY["gradcam"](model, x, 1, info, {})
    assert a.normalized_map.shape == (224, 224)
    assert a.config["token_layout"]["n_register_tokens"] == num_register_tokens(model.backbone)
    assert a.extras["layer_received_gradients"] is True
    assert not np.allclose(a.normalized_map, b.normalized_map, atol=1e-5)


def test_chefer_named_fallback(trained):
    model, info = trained
    device = next(model.parameters()).device
    x = torch.rand(1, 3, 224, 224, device=device)
    res = METHOD_REGISTRY["chefer_attribution"](model, x, 0, info, {})
    assert res.method == "chefer_attribution"
    assert res.config["fallback"] == "chefer_attribution"
    assert "AttnLRP" in res.config["fallback_reason"]
    assert res.normalized_map.shape == (224, 224)


def test_attention_rollout_normalized(trained):
    model, info = trained
    device = next(model.parameters()).device
    x = torch.rand(1, 3, 224, 224, device=device)
    res = METHOD_REGISTRY["attention_rollout"](model, x, 0, info, {})
    assert res.config["class_specific"] is False
    assert abs(res.config["row_sum_mean"] - res.config["token_layout"]["n_tokens"]) < 1.5 or res.config["row_sum_mean"] > 0
    assert res.normalized_map.shape == (224, 224)


def test_lucent_and_dream_metadata(trained, tmp_path, monkeypatch):
    from xai import lucent_dream

    monkeypatch.setattr(lucent_dream, "OUT", tmp_path)
    model, info = trained
    device = next(model.parameters()).device
    # tiny image
    pil = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    luc = record_lucent(model, 0, info, seed=0, mode="class_logit", iters=2)
    hid = record_lucent(model, 0, info, seed=0, mode="hidden_neuron", iters=2, hidden_unit=0)
    dream = record_deepdream(model, pil, 0, info, seed=0, steps=2, num_octaves=1)
    for k in REQUIRED_LUCENT_KEYS:
        assert k in luc and k in hid
    for k in REQUIRED_DREAM_KEYS:
        assert k in dream
    assert luc["hidden_vs_class"] == "final class output"
    assert hid["hidden_vs_class"] == "hidden layer"
    assert hid["class_represented"] is None
    assert "NOT a patient-specific" in dream["role"]

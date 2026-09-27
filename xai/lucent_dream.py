"""Lucent and Deep Dream with required metadata. Global feature analysis only."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from xai.model_io import CLASS_NAMES, IMG_SIZE, MEAN, STD

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "reviewer_experiments" / "feature_viz"


def _to_display(x: torch.Tensor) -> Image.Image:
    t = x.detach().clamp(0, 1)[0].cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _tv(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()


def record_deepdream(
    model,
    pil: Image.Image,
    class_idx: int,
    info: dict,
    seed: int = 0,
    steps: int = 40,
    lr: float = 0.08,
    tv_weight: float = 0.01,
    num_octaves: int = 2,
    octave_scale: float = 1.3,
    jitter: int = 4,
    layer: str = "classifier_logit",
) -> dict:
    """Optimize a real MRI toward a class logit. Not a patient-specific explanation."""
    dest = OUT / "deepdream"
    dest.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.time()
    device = next(model.parameters()).device
    mean = MEAN.to(device)
    std = STD.to(device)
    img = pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    x0 = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    x = x0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    final_act = None
    for octave in range(num_octaves):
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            if jitter:
                ox = int(torch.randint(-jitter, jitter + 1, (1,)).item())
                oy = int(torch.randint(-jitter, jitter + 1, (1,)).item())
                xx = torch.roll(x, shifts=(oy, ox), dims=(2, 3))
            else:
                xx = x
            normed = (xx.clamp(0, 1) - mean) / std
            logits = model(pixel_values=normed)
            score = logits[0, class_idx]
            loss = -score + tv_weight * _tv(xx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([x], max_norm=0.8)
            opt.step()
            with torch.no_grad():
                x.clamp_(0, 1)
                if step % 20 == 0:
                    x.copy_(F.avg_pool2d(x, 3, 1, 1).clamp(0, 1))
            final_act = float(score.detach().item())
        if octave + 1 < num_octaves:
            new = int(x.shape[-1] * octave_scale)
            with torch.no_grad():
                up = F.interpolate(x, size=(new, new), mode="bilinear", align_corners=False)
                up = F.interpolate(up, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
            x = up.detach().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=lr)

    out_png = dest / f"class{class_idx}_{CLASS_NAMES[class_idx]}_seed{seed}.png"
    _to_display(x).save(out_png)
    meta = {
        "method": "deep_dream",
        "role": "global feature visualization, NOT a patient-specific explanation",
        "target_layer": layer,
        "target_channel_or_neuron": f"class_logit[{class_idx}] = {CLASS_NAMES[class_idx]}",
        "hidden_vs_class": "final class output",
        "objective": f"maximize classifier logit {class_idx} minus TV",
        "optimization_steps": steps,
        "steps_per_octave": steps,
        "learning_rate": lr,
        "regularization": ["total_variation", "gradient_clip_0.8", "periodic_avg_pool_blur", "output_clamp_0_1", f"jitter_{jitter}"],
        "image_parameterization": "pixel RGB in [0,1] starting from a real MRI",
        "octaves": num_octaves,
        "octave_scale": octave_scale,
        "random_seed": seed,
        "input_initialization": "real MRI resized to 224 (then optimized)",
        "final_activation_value": final_act,
        "class_represented": CLASS_NAMES[class_idx],
        "runtime_s": time.time() - t0,
        "output": str(out_png),
        "checkpoint": info.get("checkpoint_path"),
        "limitation": (
            "Blurred/abstract patterns are expected. Deep Dream is a global feature "
            "analysis method. Do not treat the output as a localization map or as "
            "confirmed medical anatomy."
        ),
    }
    (dest / f"class{class_idx}_{CLASS_NAMES[class_idx]}_seed{seed}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def record_lucent(
    model,
    class_idx: int,
    info: dict,
    seed: int = 0,
    mode: str = "class_logit",
    iters: int = 64,
    lr: float = 0.05,
    hidden_unit: int | None = None,
) -> dict:
    """FFT-parameterized feature visualization with full metadata."""
    dest = OUT / "lucent" / mode
    dest.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.time()
    device = next(model.parameters()).device
    mean = MEAN.to(device)
    std = STD.to(device)
    # FFT parameterization (Lucent-style): optimize spectral coefficients
    h = w = IMG_SIZE
    spectrum = torch.randn(1, 3, h, w, 2, device=device, dtype=torch.float32) * 0.01
    spectrum.requires_grad_(True)
    opt = torch.optim.Adam([spectrum], lr=lr)
    final_act = None
    target_desc = (
        f"class_logit[{class_idx}]={CLASS_NAMES[class_idx]}"
        if mode == "class_logit"
        else f"hidden classifier.0 channel {hidden_unit}"
    )

    def _image_from_spectrum(spec):
        comp = torch.complex(spec[..., 0], spec[..., 1])
        img = torch.fft.ifft2(comp, s=(h, w)).real
        img = img / (img.std() + 1e-6) * 0.2 + 0.5
        return img.clamp(0, 1)

    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        img = _image_from_spectrum(spectrum)
        # cheap jitter
        ox = int(torch.randint(-8, 9, (1,)).item())
        oy = int(torch.randint(-8, 9, (1,)).item())
        img_j = torch.roll(img, shifts=(oy, ox), dims=(2, 3))
        normed = (img_j - mean) / std
        if mode == "class_logit":
            score = model(pixel_values=normed)[0, class_idx]
        else:
            feats = model.backbone(pixel_values=normed).last_hidden_state[:, 0]
            hidden = model.classifier[0](feats)
            score = hidden[0, int(hidden_unit if hidden_unit is not None else class_idx)]
        (-score + 0.005 * _tv(img_j)).backward()
        opt.step()
        final_act = float(score.detach().item())

    with torch.no_grad():
        img = _image_from_spectrum(spectrum)
    tag = CLASS_NAMES[class_idx] if mode == "class_logit" else f"hidden_{hidden_unit}"
    out_png = dest / f"{mode}_{tag}_seed{seed}.png"
    _to_display(img).save(out_png)
    meta = {
        "method": "lucent_style_feature_visualization",
        "mode": mode,
        "exact_target_layer": "classifier[-1] class logit" if mode == "class_logit" else "classifier[0] Linear 768->256",
        "exact_target_neuron_or_channel": target_desc,
        "hidden_vs_class": "final class output" if mode == "class_logit" else "hidden layer",
        "optimization_objective": f"maximize {target_desc} minus TV, FFT parameterization",
        "optimization_steps": iters,
        "learning_rate": lr,
        "regularization": ["total_variation", "unit-std rescale", "spatial jitter", "clamp_0_1"],
        "image_parameterization": "FFT spectral coefficients (real/imag), inverse FFT to RGB",
        "random_seed": seed,
        "input_initialization": "small random FFT spectrum, NOT a real MRI",
        "final_activation_value": final_act,
        "class_represented": CLASS_NAMES[class_idx] if mode == "class_logit" else None,
        "runtime_s": time.time() - t0,
        "output": str(out_png),
        "checkpoint": info.get("checkpoint_path"),
        "limitation": (
            "Synthetic visualization. Blur/abstraction is expected from FFT parameterization "
            "and MRI-domain shift. Do not interpret patterns as confirmed medical features. "
            "Hidden-neuron images are not class images."
        ),
    }
    (dest / f"{mode}_{tag}_seed{seed}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


REQUIRED_LUCENT_KEYS = [
    "exact_target_layer",
    "exact_target_neuron_or_channel",
    "hidden_vs_class",
    "optimization_objective",
    "optimization_steps",
    "learning_rate",
    "regularization",
    "image_parameterization",
    "random_seed",
    "input_initialization",
    "final_activation_value",
    "class_represented",
]
REQUIRED_DREAM_KEYS = [
    "target_layer",
    "target_channel_or_neuron",
    "objective",
    "optimization_steps",
    "learning_rate",
    "regularization",
    "octaves",
    "random_seed",
    "final_activation_value",
]

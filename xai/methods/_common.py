from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

from xai.model_io import CLASS_NAMES, predict
from xai.result import ExplanationResult
from xai.tokens import normalize_minmax, upsample


def finish(
    method: str,
    algorithm: str,
    raw: np.ndarray,
    target_class: int,
    pred: int,
    probs: np.ndarray,
    input_hw: tuple[int, int],
    config: dict,
    t0: float,
    warning: Optional[str] = None,
    positive_map=None,
    negative_map=None,
    extras=None,
    upsample_to: Optional[int] = None,
) -> ExplanationResult:
    raw = np.asarray(raw, dtype=np.float32)
    if upsample_to is not None and raw.shape != (upsample_to, upsample_to):
        raw_up = upsample(raw, upsample_to)
    else:
        raw_up = raw
    norm = normalize_minmax(raw_up)
    result = ExplanationResult(
        method=method,
        algorithm=algorithm,
        original_attribution=raw_up,
        normalized_map=norm,
        target_class=int(target_class),
        predicted_class=int(pred),
        class_probabilities=[float(p) for p in probs],
        input_hw=input_hw,
        attribution_hw=(int(raw_up.shape[0]), int(raw_up.shape[1])),
        config={"method": method, "algorithm": algorithm, **config},
        runtime_s=time.time() - t0,
        warning=warning,
        positive_map=positive_map,
        negative_map=negative_map,
        extras=extras or {},
    )
    result.validate()
    return result


def predict_from_pixels(model, pixel_values):
    return predict(model, pixel_values)


def class_score(model, pixel_values: torch.Tensor, target_class: int) -> torch.Tensor:
    logits = model(pixel_values=pixel_values)
    return logits[0, target_class]


def find_encoder_layers(backbone):
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        return backbone.encoder.layer
    if hasattr(backbone, "layer"):
        return backbone.layer
    if hasattr(backbone, "blocks"):
        return backbone.blocks
    raise RuntimeError("Could not locate DINOv3 transformer blocks")

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class MethodConfig:
    method: str
    algorithm: str
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "algorithm": self.algorithm, **self.extras}


@dataclass
class ExplanationResult:
    method: str
    algorithm: str
    original_attribution: np.ndarray
    normalized_map: np.ndarray
    target_class: int
    predicted_class: int
    class_probabilities: list[float]
    input_hw: tuple[int, int]
    attribution_hw: tuple[int, int]
    config: dict[str, Any]
    runtime_s: float
    warning: Optional[str] = None
    positive_map: Optional[np.ndarray] = None
    negative_map: Optional[np.ndarray] = None
    extras: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        heat = np.asarray(self.normalized_map, dtype=np.float32)
        if heat.size == 0:
            raise RuntimeError(f"{self.method}: empty attribution map")
        if not np.isfinite(heat).all():
            raise RuntimeError(f"{self.method}: NaN/Inf in attribution map")
        if float(np.nanstd(heat)) < 1e-12:
            raise RuntimeError(f"{self.method}: constant/invalid attribution map")

    def to_serializable(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("original_attribution", "normalized_map", "positive_map", "negative_map"):
            arr = d.get(key)
            if isinstance(arr, np.ndarray):
                d[key] = arr.tolist()
        return d

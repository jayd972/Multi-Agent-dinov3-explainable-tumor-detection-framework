from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xai.model_io import DEFAULT_CKPT


@pytest.fixture(scope="session")
def ckpt_available():
    return DEFAULT_CKPT.exists()


@pytest.fixture(scope="session")
def trained(ckpt_available):
    if not ckpt_available:
        pytest.skip("artifacts/models/clf.pth not found")
    from xai.model_io import load_trained_model

    return load_trained_model()


class TinyClassifier(torch.nn.Module):
    """Class-specific linear readout on pixels for IG/occlusion tests."""

    def __init__(self, c=4, h=16, w=16):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(c, 3, h, w))
        self.w.data[0, :, :8, :8] = 1.0
        self.w.data[1, :, 8:, 8:] = 1.0
        self.classifier = torch.nn.Identity()

    def forward(self, pixel_values=None, **kwargs):
        x = pixel_values if pixel_values is not None else kwargs.get("pixel_values")
        return (x[:, None] * self.w[None]).sum(dim=(2, 3, 4))


@pytest.fixture
def tiny():
    return TinyClassifier().eval()

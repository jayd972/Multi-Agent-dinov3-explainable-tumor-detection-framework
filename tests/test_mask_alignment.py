from __future__ import annotations

import numpy as np

from xai.masks import find_mask_for_image, mask_status
from xai.metrics import _align, localization_bundle


def test_mask_status_does_not_invent():
    st = mask_status()
    assert "available" in st
    if not st["available"]:
        assert st["mask_dir"] is None


def test_align_heatmap_to_mask():
    heat = np.arange(16, dtype=np.float32).reshape(4, 4)
    mask = np.ones((8, 8), dtype=np.uint8)
    h2, m2 = _align(heat, mask)
    assert h2.shape == m2.shape == (8, 8)


def test_find_mask_missing(tmp_path):
    img = tmp_path / "no_such_case.jpg"
    img.write_bytes(b"x")
    assert find_mask_for_image(img) is None
    loc = localization_bundle(np.ones((4, 4)), None, 0.5, True)
    assert loc["note"] == "MASK_UNAVAILABLE"

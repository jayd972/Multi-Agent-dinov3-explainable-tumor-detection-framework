"""Build consolidated test CSVs/summary from checkpoint (or partial run)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from xai.data_paths import official_test_images
from xai.evaluate import OUT, images_in_checkpoint, write_metric_summaries
from xai.masks import mask_status
from xai.methods import PRIMARY_METHODS
from xai.model_io import load_trained_model


def main():
    ckpt = OUT / "test_checkpoint.csv"
    per_image = OUT / "test_per_image.csv"
    if not ckpt.exists():
        raise SystemExit(f"No checkpoint at {ckpt}")

    df = pd.read_csv(ckpt).drop_duplicates(subset=["image_id", "method"], keep="last")
    df.to_csv(per_image, index=False)

    thresh_path = OUT / "threshold_selection.json"
    threshold = float(json.loads(thresh_path.read_text())["selected_threshold"])
    _, info = load_trained_model()
    summary = write_metric_summaries(df, OUT, "test", threshold, mask_status(), PRIMARY_METHODS, info)

    n = df["image_id"].nunique()
    total = len(official_test_images())
    print(f"Consolidated {n}/{total} images ({len(df)} rows)")
    print(f"  {per_image}")
    print(f"  {OUT / 'test_overall.csv'}")
    print(f"  {OUT / 'test_per_class.csv'}")
    print(f"  {OUT / 'test_summary.json'}")
    return 0 if n >= total else 1


if __name__ == "__main__":
    raise SystemExit(main())

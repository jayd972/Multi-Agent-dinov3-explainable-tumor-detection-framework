"""Resume full official-test XAI + quant on GPU. Skips images with cached maps and checkpointed metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from xai.data_paths import official_test_images
from xai.evaluate import OUT, evaluate_images, image_maps_complete, images_in_checkpoint
from xai.methods import PRIMARY_METHODS


def main():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this run.")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    thresh_path = OUT / "threshold_selection.json"
    threshold = float(json.loads(thresh_path.read_text())["selected_threshold"])
    print(f"Frozen threshold: {threshold}", flush=True)

    items = official_test_images()
    maps_root = OUT / "maps"
    ckpt = OUT / "test_checkpoint.csv"
    cached = sum(1 for _, p in items if image_maps_complete(maps_root / p.stem, PRIMARY_METHODS))
    done = images_in_checkpoint(ckpt, PRIMARY_METHODS)
    print(f"Test images: {len(items)} | maps cached (all 4 methods): {cached} | metrics checkpointed: {len(done)}", flush=True)

    # Phase 1: metrics for images that already have all XAI maps (fast — no map recompute)
    print("\n=== Phase 1: metrics for images with cached maps ===", flush=True)
    cached_items = [
        (lab, p) for lab, p in items
        if image_maps_complete(maps_root / p.stem, PRIMARY_METHODS)
    ]
    evaluate_images(
        cached_items,
        PRIMARY_METHODS,
        threshold,
        OUT,
        split_name="test",
        save_visuals=False,
        only_missing_maps=False,
        checkpoint_csv=ckpt,
        idel_steps=10,
    )

    # Phase 2: compute missing XAI maps + metrics
    print("\n=== Phase 2: remaining XAI maps (skip cached) ===", flush=True)
    evaluate_images(
        items,
        PRIMARY_METHODS,
        threshold,
        OUT,
        split_name="test",
        save_visuals=False,
        only_missing_maps=True,
        checkpoint_csv=ckpt,
        idel_steps=10,
    )

    # Phase 3: finalize summaries for all 1000 images
    print("\n=== Phase 3: finalize consolidated outputs ===", flush=True)
    import pandas as pd
    from xai.evaluate import write_metric_summaries, mask_status
    from xai.model_io import load_trained_model

    df = pd.read_csv(OUT / "test_per_image.csv")
    _, info = load_trained_model()
    summary = write_metric_summaries(
        df, OUT, "test", threshold, mask_status(), PRIMARY_METHODS, info
    )
    n_unique = df["image_id"].nunique()
    print(f"\nDONE: {n_unique} images, {len(df)} rows", flush=True)
    print(f"Outputs:\n  {OUT / 'test_per_image.csv'}\n  {OUT / 'test_overall.csv'}\n  {OUT / 'test_per_class.csv'}", flush=True)
    if "iou" in df.columns:
        tumor = df[df["is_tumor"] & df["iou"].notna()]
        print("\nMean IoU by method (tumor images):", flush=True)
        print(tumor.groupby("method")["iou"].mean().sort_values(ascending=False).to_string(), flush=True)


if __name__ == "__main__":
    main()

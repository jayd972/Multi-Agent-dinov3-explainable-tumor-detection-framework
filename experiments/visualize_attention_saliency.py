#!/usr/bin/env python3
"""
Generate attention-rollout and gradient-saliency overlays for test images.

Fixes the previous "same diffuse map on every image" failure:
  - DINOv3 register tokens are excluded before reshaping to 14x14
  - Rollout starts at mid-depth so early uniform layers do not wash out the map
  - Saliency is class-targeted SmoothGrad, not a single noisy backprop
  - Overlay uses percentile clip + gamma > 1 so peaks are visible

Output: artifacts/explain/by_image/<image_stem>/
  attn_rollout.png
  attn_last_layer.png
  grad_saliency.png
  xai_panel.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from servers.pytorch_model_loader import load_pytorch_model
from servers.xai_maps import (
    compute_attention_rollout,
    compute_gradient_saliency,
    compute_last_layer_cls_attention,
    overlay_heatmap,
    upsample_map,
)

MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "clf.pth"
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_TEST = Path(r"C:\Users\darji\Downloads\BT_Images\Testing")
OUT_ROOT = PROJECT_ROOT / "artifacts" / "explain" / "by_image"
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def list_images(root: Path) -> list[tuple[str, Path]]:
    items = []
    if not root.exists():
        return items
    class_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if class_dirs:
        for cls_dir in class_dirs:
            for p in sorted(cls_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in VALID_EXT:
                    items.append((cls_dir.name, p))
    else:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in VALID_EXT:
                items.append(("unknown", p))
    return items


def make_panel(original: Image.Image, overlays: list[tuple[str, Image.Image]], out_path: Path) -> None:
    original = original.convert("RGB")
    w, h = original.size
    cells = [("MRI", original)] + overlays
    pad, title_h = 8, 28
    panel_w = pad + len(cells) * (w + pad)
    panel_h = title_h + h + pad
    panel = Image.new("RGB", (panel_w, panel_h), (18, 18, 18))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    x = pad
    for title, img in cells:
        panel.paste(img.resize((w, h), Image.BICUBIC), (x, title_h))
        draw.text((x + 4, 6), title, fill=(240, 240, 240), font=font)
        x += w + pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--limit", type=int, default=0, help="0 = all test images")
    parser.add_argument("--smooth", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {MODEL_PATH}")
    model, class_names, _ = load_pytorch_model(str(MODEL_PATH), model_id=MODEL_ID)
    model.eval()
    backbone = model.backbone
    n_reg = int(getattr(backbone.config, "num_register_tokens", 0) or 0)
    print(f"DINOv3 register tokens: {n_reg}")

    from transformers import AutoImageProcessor
    from huggingface_hub import HfFolder
    token = HfFolder.get_token()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=token)

    images = list_images(args.test_dir)
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found under {args.test_dir}")
    print(f"Visualizing {len(images)} images from {args.test_dir}")

    rows = []
    failures = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for true_label, path in tqdm(images, desc="XAI maps"):
        stem = path.stem
        out_dir = args.out_dir / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            pil = Image.open(path).convert("RGB")
            inputs = processor(images=pil, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            with torch.no_grad():
                logits = model(pixel_values=pixel_values)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

            attn = compute_attention_rollout(backbone, pixel_values)
            last = compute_last_layer_cls_attention(backbone, pixel_values)
            sal = compute_gradient_saliency(model, pixel_values, pred_idx, n_smooth=args.smooth)

            attn_up = upsample_map(attn, size=max(pil.size))
            last_up = upsample_map(last, size=max(pil.size))

            attn_png = out_dir / "attn_rollout.png"
            last_png = out_dir / "attn_last_layer.png"
            sal_png = out_dir / "grad_saliency.png"
            overlay_heatmap(pil, attn_up, attn_png)
            overlay_heatmap(pil, last_up, last_png)
            overlay_heatmap(pil, sal, sal_png)

            make_panel(
                pil,
                [
                    ("Attention rollout", Image.open(attn_png)),
                    ("Last-layer attention", Image.open(last_png)),
                    ("SmoothGrad saliency", Image.open(sal_png)),
                ],
                out_dir / "xai_panel.png",
            )

            rows.append({
                "image_id": stem,
                "path": str(path),
                "true_label": true_label,
                "predicted_class": pred_name,
                "confidence": float(probs[pred_idx]),
                "register_tokens": n_reg,
                "attn_rollout": str(attn_png),
                "attn_last_layer": str(last_png),
                "grad_saliency": str(sal_png),
                "ok": True,
            })
        except Exception as e:
            failures.append({"image_id": stem, "path": str(path), "error": f"{type(e).__name__}: {e}"})
            rows.append({
                "image_id": stem,
                "path": str(path),
                "true_label": true_label,
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            })

    manifest = args.out_dir / "attention_saliency_manifest.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_images": len(images),
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "n_failed": len(failures),
        "register_tokens": n_reg,
        "model_path": str(MODEL_PATH),
        "test_dir": str(args.test_dir),
        "failures": failures[:20],
        "note": (
            "Maps exclude DINOv3 register tokens and start rollout at mid-depth. "
            "They are attribution visualizations, not tumor segmentations."
        ),
    }
    with open(args.out_dir / "attention_saliency_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {summary['n_ok']} / {summary['n_images']} image folders under {args.out_dir}")
    if failures:
        print(f"Failed: {len(failures)}")
        print(failures[0])


if __name__ == "__main__":
    main()

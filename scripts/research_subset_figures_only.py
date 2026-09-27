"""Regenerate paper figures from saved research-subset maps (no XAI recompute)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
if __name__ == "__main__":
    out = ROOT / "artifacts/reviewer_experiments/xai_full/research_subset"
    if not (out / "selected_images.json").exists():
        raise SystemExit("Run: python -m xai research-subset --per-class 10 first")
    from xai.data_paths import CLASS_ORDER, n_per_class, official_test_images
    from xai.figures import make_attention_figure, make_gallery, make_main_figure
    from xai.masks import load_mask
    from xai.cli import _load, _prepare_image, explain_image
    from xai.model_io import CLASS_NAMES, predict

    n = 10
    items = n_per_class(official_test_images(), n)
    model, info, processor, device = _load()
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    main_ex, attn_ex, gallery = {}, {}, []
    for cls in CLASS_ORDER:
        cls_items = [(lab, p) for lab, p in items if lab == cls]
        if not cls_items:
            continue
        lab, path = cls_items[0]
        pil, pv = _prepare_image(path, processor, device)
        pred, probs = predict(model, pv)
        mask = load_mask(path, size=(224, 224))
        maps_root = out / "maps" / path.stem

        def _map(name, root=maps_root):
            npy = root / name / "normalized.npy"
            return np.load(npy) if npy.exists() else None

        main_ex[cls] = {
            "mri": pil.resize((224, 224)),
            "mask": mask,
            "gradcam": _map("gradcam"),
            "chefer": _map("chefer_attribution"),
            "ig": _map("integrated_gradients"),
            "occlusion": _map("occlusion"),
            "true": cls,
            "pred": CLASS_NAMES[pred],
            "prob": float(probs[pred]),
        }
        res_attn = explain_image(
            model, info, pil, pv, pred,
            ["attention_rollout", "last_layer_attention"],
            maps_root.parent, path.stem, save_visuals=False,
        )
        per_head_path = maps_root / "last_layer_attention" / "per_head.npy"
        heads = np.load(per_head_path) if per_head_path.exists() else None
        if heads is None and hasattr(res_attn.get("last_layer_attention"), "extras"):
            heads = res_attn["last_layer_attention"].extras.get("per_head")
        attn_ex[cls] = {
            "mri": pil.resize((224, 224)),
            "rollout": res_attn["attention_rollout"].normalized_map,
            "last_mean": res_attn["last_layer_attention"].normalized_map,
            "heads": list(heads[:3]) if heads is not None else [],
            "true": cls,
            "pred": CLASS_NAMES[pred],
            "prob": float(probs[pred]),
        }
        for lab2, p2 in cls_items:
            pil2, pv2 = _prepare_image(p2, processor, device)
            pred2, probs2 = predict(model, pv2)
            g = np.load(out / "maps" / p2.stem / "gradcam" / "normalized.npy")
            gallery.append({
                "image": g,
                "mri": pil2.resize((224, 224)),
                "overlay": True,
                "caption": f"{cls} {p2.stem[-12:]} pred={CLASS_NAMES[pred2]} P={probs2[pred2]:.2f}",
            })

    make_main_figure(main_ex, fig_dir / "main_xai.png", fig_dir / "main_xai.pdf")
    make_attention_figure(attn_ex, fig_dir / "attention.png", fig_dir / "attention.pdf")
    for cls in CLASS_ORDER:
        cls_gallery = [g for g in gallery if g["caption"].startswith(cls + " ")]
        if cls_gallery:
            make_gallery(
                f"Grad-CAM: {cls} (first {n} official-test images)",
                cls_gallery,
                fig_dir / f"gradcam_gallery_{cls}.png",
                fig_dir / f"gradcam_gallery_{cls}.pdf",
                ncols=5,
            )
    print(f"Figures written to {fig_dir}")

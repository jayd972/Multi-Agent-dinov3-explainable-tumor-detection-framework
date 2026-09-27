from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from xai.data_paths import list_split_images, normalize_class
from xai.masks import load_mask, mask_status
from xai.methods import METHOD_REGISTRY
from xai.metrics import insertion_deletion_auc, localization_bundle, tumor_occlusion_drop
from xai.model_io import CLASS_NAMES, IMG_SIZE, get_processor, load_trained_model, preprocess_pil
from xai.sanity import compare_maps, has_invalid, is_constant, mark_sanity_failure, randomize_classifier_head

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "reviewer_experiments" / "xai_full"


def image_maps_complete(maps_dir: Path, methods: list[str]) -> bool:
    if not maps_dir.exists():
        return False
    return all((maps_dir / m / "normalized.npy").exists() for m in methods)


def images_in_checkpoint(checkpoint_csv: Path, methods: list[str]) -> set[str]:
    if not checkpoint_csv.exists():
        return set()
    import pandas as pd

    df = pd.read_csv(checkpoint_csv)
    if df.empty or "image_id" not in df.columns or "method" not in df.columns:
        return set()
    counts = df.groupby("image_id")["method"].nunique()
    return set(counts[counts >= len(methods)].index.astype(str))


def _append_checkpoint(rows: list[dict], checkpoint_csv: Path) -> None:
    import pandas as pd

    checkpoint_csv.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if checkpoint_csv.exists():
        old = pd.read_csv(checkpoint_csv)
        combined = pd.concat([old, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["image_id", "method"], keep="last")
        combined.to_csv(checkpoint_csv, index=False)
    else:
        new_df.to_csv(checkpoint_csv, index=False)


def save_result_arrays(out_dir: Path, result) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "attribution.npy", result.original_attribution)
    np.save(out_dir / "normalized.npy", result.normalized_map)
    meta = result.to_serializable()
    meta.pop("original_attribution", None)
    meta.pop("normalized_map", None)
    meta.pop("positive_map", None)
    meta.pop("negative_map", None)
    extras = dict(meta.get("extras") or {})
    per_head = extras.pop("per_head", None)
    meta["extras"] = extras
    if result.positive_map is not None:
        np.save(out_dir / "positive.npy", result.positive_map)
    if result.negative_map is not None:
        np.save(out_dir / "negative.npy", result.negative_map)
    if per_head is not None:
        np.save(out_dir / "per_head.npy", np.asarray(per_head))
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def _method_configs_for_eval():
    frozen = OUT / "occlusion_settings.json"
    occ = {"patch_size": 16, "stride": 8, "baseline": "zero", "batch_size": 16}
    if frozen.exists():
        occ.update(json.loads(frozen.read_text(encoding="utf-8")).get("selected", {}))
    return {
        "occlusion": occ,
        "integrated_gradients": {"baseline": "zero", "steps": 32},
        "attention_rollout": {"head_fusion": "mean"},
    }


def _load_saved_map(dest: Path):
    npy = dest / "normalized.npy"
    if not npy.exists():
        return None
    from types import SimpleNamespace

    cfg = {}
    if (dest / "config.json").exists():
        cfg = json.loads((dest / "config.json").read_text(encoding="utf-8"))
    heat = np.load(npy)
    return SimpleNamespace(
        normalized_map=heat,
        original_attribution=heat,
        runtime_s=cfg.get("runtime_s"),
        warning=cfg.get("warning") or "loaded_from_disk",
        method=cfg.get("method", dest.name),
    )


def explain_image(
    model,
    info,
    pil,
    pixel_values,
    target_class,
    methods,
    out_root: Path,
    image_id: str,
    method_cfgs: dict | None = None,
    save_visuals: bool = True,
):
    from servers.xai_maps import overlay_heatmap
    import matplotlib.pyplot as plt

    pred_results = {}
    method_cfgs = method_cfgs or {}
    for name in methods:
        dest = out_root / image_id / name
        cached = _load_saved_map(dest)
        if cached is not None:
            pred_results[name] = cached
            continue
        fn = METHOD_REGISTRY[name]
        try:
            res = fn(model, pixel_values, target_class, info, config=method_cfgs.get(name, {}))
        except Exception as e:
            dest.mkdir(parents=True, exist_ok=True)
            err = {"method": name, "error": f"{type(e).__name__}: {e}", "failed": True}
            (dest / "error.json").write_text(json.dumps(err, indent=2), encoding="utf-8")
            print(f"  FAIL {name}: {type(e).__name__}: {e}", flush=True)
            pred_results[name] = None
            continue
        save_result_arrays(dest, res)
        if save_visuals:
            overlay_heatmap(pil, res.normalized_map, dest / "overlay.png")
            fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
            im = ax.imshow(res.normalized_map, cmap="jet", vmin=0, vmax=1)
            ax.set_title(name, fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.savefig(dest / "heatmap.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        pred_results[name] = res
    return pred_results


def evaluate_images(
    image_items: list[tuple[str, Path]],
    methods: list[str],
    threshold: float,
    out_root: Path,
    limit: int = 0,
    split_name: str = "unknown",
    save_visuals: bool = False,
    only_missing_maps: bool = False,
    checkpoint_csv: Path | None = None,
    idel_steps: int = 20,
):
    model, info = load_trained_model()
    processor = get_processor()
    device = next(model.parameters()).device
    items = image_items[:limit] if limit else image_items
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    mstat = mask_status()
    done_metrics = images_in_checkpoint(checkpoint_csv, methods) if checkpoint_csv else set()
    maps_root = out_root / "maps"

    if only_missing_maps:
        items = [
            (lab, p) for lab, p in items
            if not image_maps_complete(maps_root / p.stem, methods)
        ]
        print(f"Resume: {len(items)} images still need XAI maps", flush=True)

    for i, (true_label, path) in enumerate(items, 1):
        if path.stem in done_metrics:
            print(f"[{split_name} {i}/{len(items)}] {path.stem} SKIP metrics (checkpoint)", flush=True)
            continue
        cached = image_maps_complete(maps_root / path.stem, methods)
        tag = "cached maps" if cached else "compute"
        print(f"[{split_name} {i}/{len(items)}] {path.stem} ({true_label}) [{tag}]", flush=True)
        pil = Image.open(path).convert("RGB")
        pv = preprocess_pil(pil, processor, device)
        from xai.model_io import predict

        pred, probs = predict(model, pv)
        is_tumor = true_label.replace(" ", "_").lower() not in {"no_tumor", "no", "notumor"}
        mask = load_mask(path, size=(IMG_SIZE, IMG_SIZE))
        method_cfgs = _method_configs_for_eval()
        results = explain_image(
            model, info, pil, pv, pred, methods, out_root / "maps", path.stem, method_cfgs,
            save_visuals=save_visuals,
        )
        for name, res in results.items():
            if res is None:
                rows.append({
                    "split": split_name,
                    "image_id": path.stem,
                    "true_label": true_label,
                    "predicted_class": CLASS_NAMES[pred],
                    "correct": _label_match(true_label, CLASS_NAMES[pred]),
                    "confidence": float(probs[pred]),
                    "method": name,
                    "is_tumor": is_tumor,
                    "mask_available": mask is not None,
                    "dice": None, "iou": None, "pointing_game": None,
                    "mass_inside": None, "mass_outside": None,
                    "deletion_auc": None, "insertion_auc": None,
                    "tumor_region_confidence_drop": None,
                    "runtime_s": None,
                    "warning": "METHOD_FAILED",
                })
                continue
            loc = localization_bundle(res.normalized_map, mask, threshold, is_tumor)
            try:
                idel = insertion_deletion_auc(model, pv, res.normalized_map, pred, n_steps=idel_steps)
            except Exception as e:
                idel = {"deletion_auc": None, "insertion_auc": None, "error": str(e)}
            drop = None
            if is_tumor and mask is not None:
                drop = tumor_occlusion_drop(model, pv, mask, pred)
            row = {
                "split": split_name,
                "image_id": path.stem,
                "true_label": true_label,
                "predicted_class": CLASS_NAMES[pred],
                "correct": _label_match(true_label, CLASS_NAMES[pred]),
                "confidence": float(probs[pred]),
                "method": name,
                "is_tumor": is_tumor,
                "mask_available": mask is not None,
                **{k: loc.get(k) for k in ("dice", "iou", "pointing_game", "mass_inside", "mass_outside")},
                "deletion_auc": idel.get("deletion_auc"),
                "insertion_auc": idel.get("insertion_auc"),
                "tumor_region_confidence_drop": drop,
                "runtime_s": res.runtime_s,
                "warning": res.warning,
            }
            rows.append(row)
        if checkpoint_csv and rows:
            _append_checkpoint(rows, checkpoint_csv)
            rows = []

    import pandas as pd

    if checkpoint_csv and checkpoint_csv.exists():
        df = pd.read_csv(checkpoint_csv)
    else:
        df = pd.DataFrame(rows)
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True) if len(df) else pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["image_id", "method"], keep="last")
    df.to_csv(out_root / f"{split_name}_per_image.csv", index=False)
    summary = write_metric_summaries(df, out_root, split_name, threshold, mstat, methods, info)
    return df, summary


def _label_match(true_label: str, pred: str) -> bool:
    from xai.data_paths import normalize_class

    return normalize_class(true_label) == normalize_class(pred)


def write_metric_summaries(df, out_root: Path, split_name: str, threshold, mstat, methods, info):
    import pandas as pd

    numeric = [
        "dice", "iou", "pointing_game", "mass_inside", "mass_outside",
        "deletion_auc", "insertion_auc", "tumor_region_confidence_drop",
    ]
    overall_rows = []
    for method, sub in df.groupby("method"):
        for subset_name, part in (
            ("all", sub),
            ("correct_tumor", sub[sub["correct"] & sub["is_tumor"]]),
            ("incorrect_tumor", sub[(~sub["correct"]) & sub["is_tumor"]]),
        ):
            rec = {"method": method, "subset": subset_name, "n": int(len(part))}
            for col in numeric:
                if col in part:
                    rec[f"{col}_mean"] = float(part[col].mean()) if part[col].notna().any() else None
            overall_rows.append(rec)
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(out_root / f"{split_name}_overall.csv", index=False)

    class_rows = []
    for (method, lab), part in df.groupby(["method", "true_label"]):
        rec = {"method": method, "true_label": lab, "n": int(len(part))}
        for col in numeric:
            if lab == "no_tumor" and col in ("dice", "iou", "pointing_game"):
                rec[f"{col}_mean"] = None
                continue
            rec[f"{col}_mean"] = float(part[col].mean()) if col in part and part[col].notna().any() else None
        class_rows.append(rec)
    by_class = pd.DataFrame(class_rows)
    by_class.to_csv(out_root / f"{split_name}_per_class.csv", index=False)

    from xai.figures import write_latex_table

    write_latex_table(
        overall,
        out_root / f"{split_name}_overall.tex",
        caption=f"XAI metrics on {split_name} (threshold frozen from validation).",
        label=f"tab:xai-{split_name}",
    )
    summary = {
        "split": split_name,
        "n_images": int(df["image_id"].nunique()) if len(df) else 0,
        "mask_status": mstat,
        "threshold": threshold,
        "threshold_selected_on": "validation" if split_name != "test" else "frozen_from_validation",
        "normalization_before_threshold": "minmax",
        "methods": methods,
        "checkpoint": info,
        "test_masks_used_for_threshold": False,
    }
    (out_root / f"{split_name}_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def select_threshold_on_validation(val_items, methods, out_root: Path, candidates=None):
    """Select heatmap threshold using validation masks only."""
    candidates = candidates or [0.3, 0.4, 0.5, 0.6, 0.7]
    model, info = load_trained_model()
    processor = get_processor()
    device = next(model.parameters()).device
    scores = {t: [] for t in candidates}
    used = 0
    for true_label, path in val_items:
        if not _label_match(true_label, "glioma_tumor") and "glioma" not in true_label.lower() \
           and "meningioma" not in true_label.lower() and "pituitary" not in true_label.lower():
            continue
        mask = load_mask(path, size=(IMG_SIZE, IMG_SIZE))
        if mask is None:
            continue
        pil = Image.open(path).convert("RGB")
        pv = preprocess_pil(pil, processor, device)
        from xai.model_io import predict
        pred, _ = predict(model, pv)
        res = METHOD_REGISTRY["gradcam"](model, pv, pred, info, {})
        for t in candidates:
            loc = localization_bundle(res.normalized_map, mask, t, True)
            if loc.get("dice") is not None:
                scores[t].append(loc["dice"])
        used += 1
        if used >= 24:
            break
    curve = {str(t): (float(np.mean(v)) if v else None) for t, v in scores.items()}
    valid = {t: np.mean(v) for t, v in scores.items() if v}
    chosen = max(valid, key=valid.get) if valid else 0.5
    payload = {
        "candidates": candidates,
        "mean_dice_by_threshold": curve,
        "selected_threshold": float(chosen),
        "n_val_images_with_masks": used,
        "note": "Threshold frozen before any official test evaluation." if used else "No validation masks found; default 0.5 used. Test masks were not used.",
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "threshold_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return float(chosen), payload


def run_sanity_suite(model, info, pixel_values, pil, out_root: Path):
    from xai.model_io import predict
    from copy import deepcopy

    pred, _ = predict(model, pixel_values)
    base = {}
    for name in ("gradcam", "chefer_attribution", "integrated_gradients"):
        base[name] = METHOD_REGISTRY[name](model, pixel_values, pred, info, {}).normalized_map

    # Target class change
    other = (pred + 1) % 4
    class_change = {}
    for name in ("gradcam", "chefer_attribution", "integrated_gradients"):
        alt = METHOD_REGISTRY[name](model, pixel_values, other, info, {}).normalized_map
        class_change[name] = compare_maps(base[name], alt)
        class_change[name]["maps_changed"] = class_change[name]["spearman"] < 0.99

    # Head randomization (copy state, scramble, restore)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    randomize_classifier_head(model)
    head_rand = {}
    for name in ("gradcam", "chefer_attribution"):
        try:
            alt = METHOD_REGISTRY[name](model, pixel_values, pred, info, {}).normalized_map
            head_rand[name] = compare_maps(base[name], alt)
            head_rand[name]["failed_sanity"] = mark_sanity_failure(head_rand[name])
        except Exception as e:
            head_rand[name] = {"error": str(e)}
    model.load_state_dict(state)

    noise = pixel_values + 0.05 * torch.randn_like(pixel_values)
    noise_stab = {}
    for name in ("gradcam",):
        alt = METHOD_REGISTRY[name](model, noise, pred, info, {}).normalized_map
        noise_stab[name] = compare_maps(base[name], alt)

    from xai.sanity import randomize_block

    block_rand = {}
    try:
        randomize_block(model, 0)
        alt = METHOD_REGISTRY["gradcam"](model, pixel_values, pred, info, {}).normalized_map
        block_rand["block0_gradcam"] = compare_maps(base["gradcam"], alt)
        block_rand["block0_gradcam"]["failed_sanity"] = mark_sanity_failure(block_rand["block0_gradcam"])
    except Exception as e:
        block_rand["block0_gradcam"] = {"error": str(e)}
    model.load_state_dict(state)

    intensity = pixel_values * 1.02
    intensity_stab = {}
    for name in ("gradcam",):
        alt = METHOD_REGISTRY[name](model, intensity, pred, info, {}).normalized_map
        intensity_stab[name] = compare_maps(base[name], alt)

    report = {
        "constant_map": {n: is_constant(m) for n, m in base.items()},
        "invalid_values": {n: has_invalid(m) for n, m in base.items()},
        "target_class_change": class_change,
        "classifier_head_randomization": head_rand,
        "progressive_block_randomization": block_rand,
        "input_noise_stability": noise_stab,
        "small_intensity_change_stability": intensity_stab,
        "reproducibility": {
            "gradcam": compare_maps(
                base["gradcam"],
                METHOD_REGISTRY["gradcam"](model, pixel_values, pred, info, {}).normalized_map,
            )
        },
        "multi_seed_stability": {
            "status": "INSUFFICIENT EVIDENCE",
            "note": "Only one trained checkpoint is loaded (clf.pth). Cross-seed map stability was not computed.",
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "sanity_tests.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def occlusion_sensitivity_analysis(val_items, out_root: Path, limit: int = 8):
    """Choose occlusion patch/stride/baseline on validation; freeze before test."""
    from xai.model_io import predict

    configs = [
        {"patch_size": 16, "stride": 8, "baseline": "zero"},
        {"patch_size": 24, "stride": 12, "baseline": "zero"},
        {"patch_size": 16, "stride": 8, "baseline": "mean"},
        {"patch_size": 16, "stride": 8, "baseline": "blur"},
    ]
    model, info = load_trained_model()
    processor = get_processor()
    device = next(model.parameters()).device
    items = val_items[:limit]
    scores = []
    for cfg in configs:
        aucs = []
        for _, path in items:
            pil = Image.open(path).convert("RGB")
            pv = preprocess_pil(pil, processor, device)
            pred, _ = predict(model, pv)
            res = METHOD_REGISTRY["occlusion"](model, pv, pred, info, {**cfg, "batch_size": 16})
            try:
                idel = insertion_deletion_auc(model, pv, res.normalized_map, pred, n_steps=10)
                aucs.append(idel["insertion_auc"])
            except Exception:
                continue
        scores.append({**cfg, "mean_insertion_auc": float(np.mean(aucs)) if aucs else None, "n": len(aucs)})
    usable = [s for s in scores if s["mean_insertion_auc"] is not None]
    selected = max(usable, key=lambda s: s["mean_insertion_auc"]) if usable else configs[0]
    payload = {
        "candidates": scores,
        "selected": {k: selected[k] for k in ("patch_size", "stride", "baseline")},
        "selected_on": "validation",
        "frozen_before_test": True,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "occlusion_settings.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

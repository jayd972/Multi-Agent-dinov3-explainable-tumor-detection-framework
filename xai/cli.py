"""Command-line interface for local XAI, figures, and grounded reports."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from xai.data_paths import (
    CLASS_ORDER,
    first_image_per_class,
    n_per_class,
    official_test_images,
    smoke_images,
    validation_images,
)
from xai.evaluate import (
    OUT,
    explain_image,
    evaluate_images,
    occlusion_sensitivity_analysis,
    run_sanity_suite,
    select_threshold_on_validation,
)
from xai.methods import METHOD_REGISTRY, PRIMARY_METHODS, SUPPLEMENTARY_METHODS
from xai.model_io import CLASS_NAMES, checkpoint_sha256, get_processor, load_trained_model, predict, preprocess_pil

ALL_LOCAL = PRIMARY_METHODS + SUPPLEMENTARY_METHODS


def _load():
    model, info = load_trained_model()
    processor = get_processor()
    device = next(model.parameters()).device
    return model, info, processor, device


def _prepare_image(path: Path, processor, device):
    pil = Image.open(path).convert("RGB")
    pv = preprocess_pil(pil, processor, device)
    return pil, pv


def cmd_explain_one(args):
    model, info, processor, device = _load()
    pil, pv = _prepare_image(Path(args.image), processor, device)
    pred, probs = predict(model, pv)
    methods = [args.method] if args.method else ALL_LOCAL
    dest = Path(args.out or OUT / "single")
    results = explain_image(model, info, pil, pv, pred if args.target is None else args.target, methods, dest, Path(args.image).stem)
    print(json.dumps({
        "predicted_class": CLASS_NAMES[pred],
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
        "methods": list(results),
        "out": str(dest),
        "checkpoint_sha256": info["checkpoint_sha256"],
    }, indent=2))


def cmd_evaluate(args):
    if args.split == "test":
        items = official_test_images()
        split = "test"
    else:
        items = validation_images()
        split = "validation"
    if not items:
        raise SystemExit("No images found. Set BRISC_DATA_ROOT.")
    if getattr(args, "per_class", 0):
        items = n_per_class(items, int(args.per_class))
        split = f"{split}_n{args.per_class}_per_class"
    out = Path(args.out or OUT)
    methods = PRIMARY_METHODS if args.primary_only else ALL_LOCAL
    if args.method:
        methods = [args.method]
    thresh_path = out / "threshold_selection.json"
    if thresh_path.exists():
        threshold = json.loads(thresh_path.read_text())["selected_threshold"]
        print(f"Using frozen validation threshold {threshold}")
    elif args.split == "test":
        threshold, _ = select_threshold_on_validation(validation_images(), methods, out)
    else:
        threshold, _ = select_threshold_on_validation(items, methods, out)
    save_visuals = bool(getattr(args, "with_visuals", False))
    evaluate_images(
        items, methods, threshold, out, limit=args.limit, split_name=split, save_visuals=save_visuals
    )


def cmd_research_subset(args):
    """10 images per class: overlays + paper figures, then quantitative metrics on the same subset."""
    from xai.figures import make_attention_figure, make_gallery, make_main_figure
    from xai.masks import load_mask

    n = int(args.per_class)
    items = n_per_class(official_test_images(), n)
    if not items:
        raise SystemExit("No test images found. Set BRISC_DATA_ROOT.")
    out = Path(args.out or OUT / "research_subset")
    out.mkdir(parents=True, exist_ok=True)
    manifest = [{"class": lab, "path": str(p)} for lab, p in items]
    (out / "selected_images.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Selected {len(items)} images ({n} per class)", flush=True)

    methods = PRIMARY_METHODS
    thresh_path = OUT / "threshold_selection.json"
    if thresh_path.exists():
        threshold = float(json.loads(thresh_path.read_text())["selected_threshold"])
    else:
        threshold, _ = select_threshold_on_validation(validation_images(), methods, OUT)

    print("Generating XAI maps + overlays...", flush=True)
    evaluate_images(
        items, methods, threshold, out,
        split_name=f"test_{n}_per_class",
        save_visuals=True,
    )

    print("Building paper figures (1 row per class in main panel)...", flush=True)
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
        import numpy as np

        def _map(name):
            npy = maps_root / name / "normalized.npy"
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
        if heads is None and res_attn.get("last_layer_attention") and hasattr(res_attn["last_layer_attention"], "extras"):
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
                f"Grad-CAM: {cls} (first {n} official-test images, sorted by filename)",
                cls_gallery,
                fig_dir / f"gradcam_gallery_{cls}.png",
                fig_dir / f"gradcam_gallery_{cls}.pdf",
                ncols=5,
            )
    print(f"Done. Figures: {fig_dir}")
    print(f"Quant tables: {out}/test_{n}_per_class_*.csv")


def cmd_lucent(args):
    from xai.lucent_dream import record_lucent

    model, info, _, _ = _load()
    metas = []
    for i in range(4):
        metas.append(record_lucent(model, i, info, seed=args.seed, mode="class_logit", iters=args.iters))
        metas.append(record_lucent(model, i, info, seed=args.seed, mode="hidden_neuron", iters=args.iters, hidden_unit=i))
    print(json.dumps(metas, indent=2, default=str))


def cmd_deepdream(args):
    from xai.lucent_dream import record_deepdream

    model, info, _, _ = _load()
    chosen = smoke_images()
    metas = []
    for lab, (_l, path) in chosen.items():
        pil = Image.open(path).convert("RGB")
        # target the predicted class
        processor = get_processor()
        device = next(model.parameters()).device
        pv = preprocess_pil(pil, processor, device)
        pred, _ = predict(model, pv)
        metas.append(record_deepdream(model, pil, pred, info, seed=args.seed, steps=args.steps))
    print(json.dumps(metas, indent=2, default=str))


def cmd_figures(args):
    from xai.figures import make_attention_figure, make_gallery, make_main_figure

    model, info, processor, device = _load()
    chosen = first_image_per_class(official_test_images() or validation_images())
    fig_dir = Path(args.out or OUT / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    main_ex, attn_ex = {}, {}
    loc_items = []
    for cls, (_l, path) in chosen.items():
        pil, pv = _prepare_image(path, processor, device)
        pred, probs = predict(model, pv)
        res = explain_image(model, info, pil, pv, pred, ALL_LOCAL, fig_dir / "maps", path.stem)
        main_ex[cls] = {
            "mri": pil.resize((224, 224)),
            "mask": None,
            "gradcam": res["gradcam"].normalized_map,
            "chefer": res["chefer_attribution"].normalized_map,
            "ig": res["integrated_gradients"].normalized_map,
            "occlusion": res["occlusion"].normalized_map,
            "true": cls,
            "pred": CLASS_NAMES[pred],
            "prob": float(probs[pred]),
        }
        heads = res["last_layer_attention"].extras.get("per_head")
        attn_ex[cls] = {
            "mri": pil.resize((224, 224)),
            "rollout": res["attention_rollout"].normalized_map,
            "last_mean": res["last_layer_attention"].normalized_map,
            "heads": list(heads) if heads is not None else [],
            "true": cls,
            "pred": CLASS_NAMES[pred],
            "prob": float(probs[pred]),
        }
        loc_items.append({
            "image": res["gradcam"].normalized_map,
            "mri": pil.resize((224, 224)),
            "overlay": True,
            "caption": f"{cls} true={cls} pred={CLASS_NAMES[pred]} P={probs[pred]:.2f} Grad-CAM",
        })
    make_main_figure(main_ex, fig_dir / "main_xai.png", fig_dir / "main_xai.pdf")
    make_attention_figure(attn_ex, fig_dir / "attention.png", fig_dir / "attention.pdf")
    make_gallery("Deterministic one-per-class Grad-CAM (not cherry-picked)", loc_items, fig_dir / "per_class_gradcam.png", fig_dir / "per_class_gradcam.pdf")
    print(f"Wrote figures to {fig_dir}")


def cmd_tables(args):
    import pandas as pd
    from xai.figures import write_latex_table

    out = Path(args.out or OUT)
    for csv in out.glob("*_overall.csv"):
        df = pd.read_csv(csv)
        write_latex_table(df, csv.with_suffix(".tex"), caption=csv.stem, label=f"tab:{csv.stem}")
        print(f"regenerated {csv.with_suffix('.tex')}")


def cmd_report(args):
    from reports.generator import build_evidence, generate_report
    from xai.masks import load_mask
    from xai.metrics import localization_bundle

    model, info, processor, device = _load()
    path = Path(args.image)
    pil, pv = _prepare_image(path, processor, device)
    pred, probs = predict(model, pv)
    dest = OUT / "maps" / path.stem
    res = explain_image(model, info, pil, pv, pred, PRIMARY_METHODS, dest.parent, path.stem)
    mask = load_mask(path, size=(224, 224))
    is_tumor = "no" not in Path(args.image).parent.name.lower()
    metrics = localization_bundle(res["gradcam"].normalized_map, mask, 0.5, is_tumor)
    xai_imgs = {n: dest.parent / path.stem / n / "overlay.png" for n in PRIMARY_METHODS}
    evidence = build_evidence(
        image_id=path.stem,
        predicted_class=CLASS_NAMES[pred],
        probabilities=list(probs),
        methods=PRIMARY_METHODS,
        metrics=metrics,
        mode=args.mode,
        mri_path=path if args.mode == "multimodal" else None,
        xai_image_paths=xai_imgs if args.mode == "multimodal" else None,
        true_class=args.true_class,
    )
    out = Path(args.out or OUT.parent / "reports_full" / args.mode / path.stem)
    result = generate_report(evidence, out, real=args.real)
    print(json.dumps({"out": result["out_dir"], "factual": result["factual"]}, indent=2))


def cmd_evaluate_reports(args):
    from reports.factual import evaluate_report

    root = Path(args.dir or OUT.parent / "reports_full")
    rows = []
    for man in root.rglob("request_manifest.json"):
        case = man.parent
        evidence = json.loads(man.read_text(encoding="utf-8"))
        ev = {
            "predicted_class": evidence.get("predicted_class"),
            "class_probabilities": evidence.get("class_probabilities"),
            "xai_methods": evidence.get("exact_xai_methods_included"),
            "metrics": evidence.get("quantitative_localization_metrics"),
        }
        report = json.loads((case / "generated_report.json").read_text(encoding="utf-8"))
        fac = evaluate_report(report, ev)
        (case / "factual_consistency.json").write_text(json.dumps(fac, indent=2), encoding="utf-8")
        rows.append({"case": str(case), **fac})
    import pandas as pd

    df = pd.DataFrame(rows)
    out = root / "report_evaluation.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} n={len(df)}")


def cmd_occlusion_sensitivity(args):
    items = validation_images()
    if not items:
        raise SystemExit("No validation images.")
    payload = occlusion_sensitivity_analysis(items, Path(args.out or OUT), limit=args.limit)
    print(json.dumps(payload, indent=2))


def cmd_smoke(args):
    """One image per BRISC class: methods, sanity, figures, offline reports."""
    from reports.generator import build_evidence, generate_report
    from xai.figures import make_attention_figure, make_gallery, make_main_figure
    from xai.lucent_dream import record_deepdream, record_lucent
    from xai.masks import load_mask, mask_status
    from xai.metrics import localization_bundle

    sha_before = checkpoint_sha256()
    model, info, processor, device = _load()
    sha_after_load = info["checkpoint_sha256"]
    assert sha_before == sha_after_load, "Checkpoint hash changed while loading"
    chosen = smoke_images()
    if len(chosen) < 4:
        raise SystemExit(f"Need one image per class, found {list(chosen)}")
    out = Path(args.out or OUT / "smoke")
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    reports_dir = out / "reports"
    pred_snap = {}
    main_ex, attn_ex = {}, {}
    smoke_methods = ["gradcam", "chefer_attribution", "integrated_gradients", "occlusion", "attention_rollout", "last_layer_attention", "vanilla_gradient"]
    occ_cfg = {"patch_size": 32, "stride": 32, "baseline": "zero", "batch_size": 8}
    ig_cfg = {"baseline": "zero", "steps": 8}

    first_pv = None
    first_pil = None
    for cls, (_l, path) in chosen.items():
        pil, pv = _prepare_image(path, processor, device)
        if first_pv is None:
            first_pv, first_pil = pv, pil
        pred, probs = predict(model, pv)
        pred_snap[cls] = {
            "image": str(path),
            "predicted_class": CLASS_NAMES[pred],
            "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
        }
        res = explain_image(
            model, info, pil, pv, pred, smoke_methods, out / "maps", path.stem,
            {"occlusion": occ_cfg, "integrated_gradients": ig_cfg},
        )
        mask = load_mask(path, size=(224, 224))
        metrics = localization_bundle(res["gradcam"].normalized_map, mask, 0.5, cls != "no_tumor")
        main_ex[cls] = {
            "mri": pil.resize((224, 224)),
            "mask": mask,
            "gradcam": res["gradcam"].normalized_map,
            "chefer": res["chefer_attribution"].normalized_map,
            "ig": res["integrated_gradients"].normalized_map,
            "occlusion": res["occlusion"].normalized_map,
            "true": cls,
            "pred": CLASS_NAMES[pred],
            "prob": float(probs[pred]),
        }
        heads = res["last_layer_attention"].extras.get("per_head")
        attn_ex[cls] = {
            "mri": pil.resize((224, 224)),
            "rollout": res["attention_rollout"].normalized_map,
            "last_mean": res["last_layer_attention"].normalized_map,
            "heads": list(heads[:3]) if heads is not None else [],
            "true": cls,
            "pred": CLASS_NAMES[pred],
            "prob": float(probs[pred]),
        }
        for mode in ("structured", "multimodal"):
            ev = build_evidence(
                image_id=path.stem,
                predicted_class=CLASS_NAMES[pred],
                probabilities=list(probs),
                methods=PRIMARY_METHODS,
                metrics=metrics,
                mode=mode,
                mri_path=path if mode == "multimodal" else None,
                xai_image_paths={n: out / "maps" / path.stem / n / "overlay.png" for n in PRIMARY_METHODS} if mode == "multimodal" else None,
                true_class=cls,
            )
            generate_report(ev, reports_dir / mode / path.stem, real=False)

    sanity = run_sanity_suite(model, info, first_pv, first_pil, out)
    make_main_figure(main_ex, fig_dir / "main_xai.png", fig_dir / "main_xai.pdf")
    make_attention_figure(attn_ex, fig_dir / "attention.png", fig_dir / "attention.pdf")
    make_gallery(
        "Smoke: deterministic one image per class (Grad-CAM)",
        [{"image": main_ex[c]["gradcam"], "mri": main_ex[c]["mri"], "overlay": True,
          "caption": f"{c} pred={main_ex[c]['pred']} P={main_ex[c]['prob']:.2f}"} for c in main_ex],
        fig_dir / "correct_or_all_gradcam.png",
        fig_dir / "correct_or_all_gradcam.pdf",
    )
    lucent_meta = []
    dream_meta = []
    if not args.skip_feature_viz:
        for i in range(4):
            lucent_meta.append(record_lucent(model, i, info, seed=0, mode="class_logit", iters=16))
            lucent_meta.append(record_lucent(model, i, info, seed=0, mode="hidden_neuron", iters=16, hidden_unit=i))
        # one Deep Dream from first smoke image
        first_path = list(chosen.values())[0][1]
        dream_meta.append(record_deepdream(model, Image.open(first_path).convert("RGB"), 0, info, seed=0, steps=8, num_octaves=1))

    sha_end = checkpoint_sha256()
    pred2 = {}
    for cls, (_l, path) in chosen.items():
        pil, pv = _prepare_image(path, processor, device)
        pred, probs = predict(model, pv)
        pred2[cls] = CLASS_NAMES[pred]
    unchanged = {c: pred_snap[c]["predicted_class"] == pred2[c] for c in pred_snap}
    summary = {
        "checkpoint_sha256_before": sha_before,
        "checkpoint_sha256_after": sha_end,
        "weights_changed": sha_before != sha_end,
        "predictions_unchanged": all(unchanged.values()),
        "predictions": pred_snap,
        "mask_status": mask_status(),
        "sanity": {k: sanity[k] for k in sanity if k != "progressive_block_randomization"},
        "figures": [str(p) for p in fig_dir.glob("*")],
        "lucent": lucent_meta,
        "deepdream": dream_meta,
        "attnlrp": "NOT IMPLEMENTED; fallback chefer_attribution",
        "salience_saliency": "consolidated as vanilla_gradient",
    }
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_checklist(out)
    print(json.dumps({k: summary[k] for k in ("weights_changed", "predictions_unchanged", "mask_status")}, indent=2))
    print(f"Smoke outputs: {out}")


def _write_checklist(out: Path):
    text = (Path(__file__).resolve().parent.parent / "artifacts" / "reviewer_experiments" / "xai_full" / "REVIEWER_EVIDENCE_CHECKLIST.md")
    # written separately at package level
    src = Path(__file__).resolve().parent.parent / "docs" / "REVIEWER_EVIDENCE_CHECKLIST.md"
    if src.exists():
        dest = out / "REVIEWER_EVIDENCE_CHECKLIST.md"
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def build_parser():
    p = argparse.ArgumentParser(prog="python -m xai", description="DINOv3 BRISC XAI and grounded reports")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("explain-one", help="Explain one image with all or one method")
    s.add_argument("--image", required=True)
    s.add_argument("--method", default=None)
    s.add_argument("--target", type=int, default=None)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_explain_one)

    s = sub.add_parser("explain-method", help="Run one XAI method on one image")
    s.add_argument("--image", required=True)
    s.add_argument("--method", required=True)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_explain_one)

    s = sub.add_parser("explain-all", help="Run every local XAI method on one image")
    s.add_argument("--image", required=True)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_explain_one, method=None)

    s = sub.add_parser("evaluate-subset", help="Evaluate XAI on a small subset")
    s.add_argument("--split", default="validation", choices=["validation", "test"])
    s.add_argument("--limit", type=int, default=8)
    s.add_argument("--per-class", type=int, default=0, help="Use first N images per class (deterministic)")
    s.add_argument("--with-visuals", action="store_true")
    s.add_argument("--method", default=None)
    s.add_argument("--primary-only", action="store_true")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("evaluate-test", help="Evaluate XAI on the official test set")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--per-class", type=int, default=0)
    s.add_argument("--with-visuals", action="store_true")
    s.add_argument("--method", default=None)
    s.add_argument("--primary-only", action="store_true")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_evaluate, split="test")

    s = sub.add_parser(
        "research-subset",
        help="10/class (default): overlays, figures, and quant metrics on the same subset",
    )
    s.add_argument("--per-class", type=int, default=10)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_research_subset)

    s = sub.add_parser("occlusion-sensitivity", help="Validation occlusion setting search")
    s.add_argument("--limit", type=int, default=4)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_occlusion_sensitivity)

    s = sub.add_parser("lucent", help="Generate Lucent-style visualizations")
    s.add_argument("--iters", type=int, default=64)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_lucent)

    s = sub.add_parser("deepdream", help="Generate Deep Dream visualizations")
    s.add_argument("--steps", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_deepdream)

    s = sub.add_parser("figures", help="Create paper figures")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_figures)

    s = sub.add_parser("tables", help="Create / regenerate paper tables from saved CSVs")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_tables)

    s = sub.add_parser("report-structured", help="LLM/offline report, structured evidence only")
    s.add_argument("--image", required=True)
    s.add_argument("--true-class", default=None)
    s.add_argument("--real", action="store_true", help="Call the real API (requires OPENAI_API_KEY)")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_report, mode="structured")

    s = sub.add_parser("report-multimodal", help="LLM/offline report with MRI + XAI panels")
    s.add_argument("--image", required=True)
    s.add_argument("--true-class", default=None)
    s.add_argument("--real", action="store_true")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_report, mode="multimodal")

    s = sub.add_parser("evaluate-reports", help="Evaluate saved reports without another API call")
    s.add_argument("--dir", default=None)
    s.set_defaults(func=cmd_evaluate_reports)

    s = sub.add_parser("regenerate-tables", help="Regenerate LaTeX from saved CSVs")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_tables)

    s = sub.add_parser("smoke", help="Local smoke: 1 image per class")
    s.add_argument("--out", default=None)
    s.add_argument("--clean", action="store_true")
    s.add_argument("--skip-feature-viz", action="store_true")
    s.set_defaults(func=cmd_smoke)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LLM Reporter Evaluation — Real GPT-4o-mini generations.

Phases:
  1. Single smoke test (1 report)
  2. All classes x both modes (8 reports)
  3. Repeatability (20 reports: 5 per class)
  4. Ablation: structured vs multimodal vs multimodal+MRI (12 reports)
  5. Aggregate summary for the paper
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reports.generator import build_evidence, generate_report
from reports.factual import evaluate_report

OUT = ROOT / "artifacts" / "reviewer_experiments" / "reports_full"
OUT.mkdir(parents=True, exist_ok=True)

SMOKE_MAPS = ROOT / "artifacts" / "reviewer_experiments" / "xai_full" / "smoke" / "maps"
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0

CASES = {
    "glioma_tumor": {
        "image_id": "brisc2025_test_00001_gl_ax_t1",
        "probabilities": [1.0, 2.96e-08, 7.59e-11, 6.66e-09],
    },
    "meningioma_tumor": {
        "image_id": "brisc2025_test_00255_me_ax_t1",
        "probabilities": [1.25e-06, 1.0, 5.29e-09, 1.53e-06],
    },
    "no_tumor": {
        "image_id": "brisc2025_test_00561_no_ax_t1",
        "probabilities": [1.02e-07, 2.56e-08, 1.0, 1.49e-07],
    },
    "pituitary_tumor": {
        "image_id": "brisc2025_test_00701_pi_ax_t1",
        "probabilities": [5.12e-07, 3.88e-06, 1.15e-08, 1.0],
    },
}
METHODS = ["gradcam", "chefer_attribution", "integrated_gradients", "occlusion"]


def load_api_key():
    global MODEL
    cfg_path = ROOT / "config" / "api_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        gemini = cfg.get("api_keys", {}).get("gemini", {})
        if gemini.get("key"):
            os.environ["GEMINI_API_KEY"] = gemini["key"]
            MODEL = gemini.get("model", "gemini-3.6-flash")
            print(f"Gemini API key loaded. Model: {MODEL}")
            return
        openai_cfg = cfg.get("api_keys", {}).get("openai", {})
        if openai_cfg.get("key"):
            os.environ["OPENAI_API_KEY"] = openai_cfg["key"]
            MODEL = openai_cfg.get("model", "gpt-4o-mini")
            print(f"OpenAI API key loaded. Model: {MODEL}")
            return
    if os.environ.get("GEMINI_API_KEY"):
        MODEL = "gemini-3.6-flash"
        print(f"Gemini key from env. Model: {MODEL}")
        return
    if os.environ.get("OPENAI_API_KEY"):
        print(f"OpenAI key from env. Model: {MODEL}")
        return
    print("ERROR: No API key found. Add gemini or openai key to config/api_config.json")
    sys.exit(1)


def xai_overlay_paths(image_id: str) -> dict[str, Path]:
    base = SMOKE_MAPS / image_id
    paths = {}
    for method in METHODS:
        overlay = base / method / "overlay.png"
        if overlay.exists():
            paths[method] = overlay
    return paths


def mri_path_for(image_id: str) -> Path | None:
    for cls, info in CASES.items():
        if info["image_id"] == image_id:
            from xai.data_paths import list_split_images
            try:
                items = list_split_images("test")
                for label, p in items:
                    if p.stem == image_id:
                        return p
            except Exception:
                pass
    return None


def run_one(cls: str, mode: str, out_dir: Path, mri_path=None, xai_paths=None) -> dict:
    info = CASES[cls]
    metrics = {"dice": None, "iou": None, "note": "MASK_UNAVAILABLE"}
    if cls == "no_tumor":
        metrics["note"] = "No-tumor images: Dice/IoU/pointing game are not defined."

    ev = build_evidence(
        image_id=info["image_id"],
        predicted_class=cls,
        probabilities=info["probabilities"],
        methods=METHODS,
        metrics=metrics,
        mode=mode,
        mri_path=mri_path if mode == "multimodal" else None,
        xai_image_paths=xai_paths if mode == "multimodal" else None,
    )
    for attempt in range(5):
        try:
            result = generate_report(ev, out_dir, real=True, model=MODEL, temperature=TEMPERATURE)
            return result
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower():
                wait = 45 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s (attempt {attempt+1}/5)...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after 5 retries for {cls}/{mode}")


def print_result(label: str, result: dict):
    fac = result["factual"]
    status = "PASS" if fac["pass_fail"] == "pass" else "FAIL"
    claims = fac["unsupported_claim_list"]
    print(f"  {label}: {status} | anatomy={fac['unsupported_anatomy_count']} pathology={fac['unsupported_pathology_count']} schema={fac['schema_compliant']} claims={claims if claims else '[]'}")


# =========================================================================
# Phase 1: Single smoke test
# =========================================================================
def phase1():
    print("\n" + "=" * 70)
    print("PHASE 1: Single Smoke Test")
    print("=" * 70)
    out = OUT / "phase1_smoke"
    result = run_one("glioma_tumor", "structured", out)
    print_result("glioma_structured", result)
    return [result]


# =========================================================================
# Phase 2: All classes x both modes
# =========================================================================
def phase2():
    print("\n" + "=" * 70)
    print("PHASE 2: All Classes x Both Modes (8 reports)")
    print("=" * 70)
    results = []
    for cls in CASES:
        image_id = CASES[cls]["image_id"]
        xai_paths = xai_overlay_paths(image_id)
        mri = mri_path_for(image_id)
        for mode in ("structured", "multimodal"):
            out = OUT / "phase2_all_classes" / mode / cls
            result = run_one(cls, mode, out, mri_path=mri, xai_paths=xai_paths)
            print_result(f"{cls}_{mode}", result)
            results.append({"class": cls, "mode": mode, **result["factual"]})
    return results


# =========================================================================
# Phase 3: Repeatability (5 per class, structured mode)
# =========================================================================
def phase3():
    print("\n" + "=" * 70)
    print("PHASE 3: Repeatability (3 x 4 classes = 12 reports)")
    print("=" * 70)
    results = []
    for cls in CASES:
        for rep in range(3):
            out = OUT / "phase3_repeatability" / cls / f"rep{rep}"
            result = run_one(cls, "structured", out)
            fac = result["factual"]
            status = "PASS" if fac["pass_fail"] == "pass" else "FAIL"
            results.append({"class": cls, "rep": rep, **fac})
            if rep == 0 or fac["pass_fail"] == "fail":
                print_result(f"{cls}_rep{rep}", result)
        n_pass = sum(1 for r in results if r["class"] == cls and r["pass_fail"] == "pass")
        print(f"  {cls}: {n_pass}/3 pass")
    return results


# =========================================================================
# Phase 4: Ablation (3 conditions x 4 classes = 12 reports)
# =========================================================================
def phase4():
    print("\n" + "=" * 70)
    print("PHASE 4: Ablation (3 conditions x 4 classes = 12 reports)")
    print("=" * 70)
    conditions = [
        ("structured_no_images", "structured", False, False),
        ("multimodal_xai_only", "multimodal", True, False),
        ("multimodal_xai_and_mri", "multimodal", True, True),
    ]
    results = []
    for cls in CASES:
        image_id = CASES[cls]["image_id"]
        xai_paths = xai_overlay_paths(image_id)
        mri = mri_path_for(image_id)
        for cond_name, mode, use_xai, use_mri in conditions:
            out = OUT / "phase4_ablation" / cond_name / cls
            result = run_one(
                cls, mode, out,
                mri_path=mri if use_mri else None,
                xai_paths=xai_paths if use_xai else None,
            )
            print_result(f"{cls}_{cond_name}", result)
            results.append({"class": cls, "condition": cond_name, **result["factual"]})
    return results


# =========================================================================
# Phase 5: Aggregate summary
# =========================================================================
def phase5(p1, p2, p3, p4):
    print("\n" + "=" * 70)
    print("PHASE 5: Aggregate Summary")
    print("=" * 70)

    all_results = []
    for r in p2:
        all_results.append(r)
    for r in p3:
        all_results.append(r)
    for r in p4:
        all_results.append(r)

    total = len(all_results)
    n_pass = sum(1 for r in all_results if r["pass_fail"] == "pass")
    n_fail = total - n_pass

    schema_ok = sum(1 for r in all_results if r.get("schema_compliant"))
    pred_ok = sum(1 for r in all_results if r.get("predicted_class_agreement"))
    prob_ok = sum(1 for r in all_results if r.get("probability_agreement"))
    limit_ok = sum(1 for r in all_results if r.get("limitation_statement_present"))
    total_anatomy = sum(r.get("unsupported_anatomy_count", 0) for r in all_results)
    total_pathology = sum(r.get("unsupported_pathology_count", 0) for r in all_results)

    # Repeatability stats
    rep_results = p3
    rep_pass = sum(1 for r in rep_results if r["pass_fail"] == "pass")
    rep_pred = sum(1 for r in rep_results if r.get("predicted_class_agreement"))
    rep_prob = sum(1 for r in rep_results if r.get("probability_agreement"))

    # Ablation stats
    abl_results = p4
    ablation_by_condition = {}
    for r in abl_results:
        cond = r["condition"]
        if cond not in ablation_by_condition:
            ablation_by_condition[cond] = {"n": 0, "pass": 0, "anatomy": 0, "pathology": 0}
        ablation_by_condition[cond]["n"] += 1
        if r["pass_fail"] == "pass":
            ablation_by_condition[cond]["pass"] += 1
        ablation_by_condition[cond]["anatomy"] += r.get("unsupported_anatomy_count", 0)
        ablation_by_condition[cond]["pathology"] += r.get("unsupported_pathology_count", 0)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "temperature": TEMPERATURE,
        "total_reports_generated": total,
        "total_pass": n_pass,
        "total_fail": n_fail,
        "pass_rate": n_pass / total if total else 0,
        "schema_compliance_rate": schema_ok / total if total else 0,
        "predicted_class_agreement_rate": pred_ok / total if total else 0,
        "probability_agreement_rate": prob_ok / total if total else 0,
        "limitation_statement_rate": limit_ok / total if total else 0,
        "total_unsupported_anatomy": total_anatomy,
        "total_unsupported_pathology": total_pathology,
        "hallucination_rate_overall": (total_anatomy + total_pathology) / max(1, total),
        "repeatability": {
            "n_reports": len(rep_results),
            "n_pass": rep_pass,
            "diagnosis_consistency": rep_pred / len(rep_results) if rep_results else 0,
            "probability_consistency": rep_prob / len(rep_results) if rep_results else 0,
        },
        "ablation": ablation_by_condition,
        "factual_checker": "programmatic_rule_based_not_llm",
        "prompt_constraints": "explicit forbidden-terms list for anatomy, pathology, size, clinical",
    }

    out_path = OUT / "llm_reporter_evaluation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  Total reports: {total}")
    print(f"  Pass rate: {n_pass}/{total} ({summary['pass_rate']:.0%})")
    print(f"  Schema compliance: {summary['schema_compliance_rate']:.0%}")
    print(f"  Predicted class agreement: {summary['predicted_class_agreement_rate']:.0%}")
    print(f"  Probability agreement: {summary['probability_agreement_rate']:.0%}")
    print(f"  Limitation statement: {summary['limitation_statement_rate']:.0%}")
    print(f"  Unsupported anatomy total: {total_anatomy}")
    print(f"  Unsupported pathology total: {total_pathology}")
    print(f"\n  Repeatability: {rep_pass}/{len(rep_results)} pass")
    print(f"  Ablation:")
    for cond, stats in ablation_by_condition.items():
        print(f"    {cond}: {stats['pass']}/{stats['n']} pass, anatomy={stats['anatomy']}, pathology={stats['pathology']}")

    print(f"\n  Summary saved to: {out_path}")
    return summary


def main():
    print("=" * 70)
    print("LLM REPORTER EVALUATION — REAL GPT-4o-mini GENERATIONS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    load_api_key()

    p1 = phase1()
    if p1[0]["factual"]["pass_fail"] == "fail":
        print("\nPhase 1 FAILED. Stopping — check the generated report before proceeding.")
        print(f"Report at: {OUT / 'phase1_smoke' / 'generated_report.json'}")
        return

    p2 = phase2()
    p3 = phase3()
    p4 = phase4()
    summary = phase5(p1, p2, p3, p4)

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

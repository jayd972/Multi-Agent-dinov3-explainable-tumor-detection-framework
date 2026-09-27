#!/usr/bin/env python3
"""
Comprehensive Report Evaluation Pipeline
=========================================
Evaluates DINOv3 LLM reports across 100 test images (25 per class) without
reference radiology reports. Employs a strict LLM-as-a-judge (default: gpt-5.6-luna)
with Response Caching, Token Cost Tracking, and Bootstrap Confidence Intervals.
"""
from __future__ import annotations

import json
import os
import sys
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Insert project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reports.generator import build_evidence, generate_report
from xai.masks import find_mask_for_image

# Outputs directory
OUT_DIR = PROJECT_ROOT / "artifacts" / "reviewer_experiments" / "reporter_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cache files
JUDGE_CACHE_PATH = OUT_DIR / "llm_judge_cache.json"

# Pricing config for token tracking (Standard gpt-4o flagship rates: $2.50/M input, $10.00/M output)
INPUT_TOKEN_RATE = 2.50 / 1_000_000
OUTPUT_TOKEN_RATE = 10.00 / 1_000_000


def load_openai_client() -> Tuple[Any, str]:
    """Load OpenAI client and resolve API keys."""
    from openai import OpenAI
    
    cfg_path = PROJECT_ROOT / "config" / "api_config.json"
    api_key = ""
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            api_key = cfg.get("api_keys", {}).get("openai", {}).get("key", "")
        except Exception:
            pass
            
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
        
    if not api_key:
        print("CRITICAL ERROR: No OpenAI API key found in config or env.")
        sys.exit(1)
        
    # Get configuration for judge model
    judge_model = os.getenv("REPORT_JUDGE_MODEL", "gpt-4o")
    return OpenAI(api_key=api_key), judge_model


class LLMJudge:
    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        
        # Load cache if exists
        if JUDGE_CACHE_PATH.exists():
            try:
                self.cache = json.loads(JUDGE_CACHE_PATH.read_text(encoding="utf-8"))
                print(f"Loaded {len(self.cache)} cached judge responses.")
            except Exception:
                print("Failed to load judge cache. Starting fresh.")

    def save_cache(self):
        JUDGE_CACHE_PATH.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")

    def query_judge(self, case_id: str, report_text: str, true_class: str, dino_pred: str, confidence: float, has_mask: bool) -> Dict[str, Any]:
        # Clean/standardize true label names
        true_label_mapped = true_class.replace("_tumor", "")
        dino_pred_mapped = dino_pred.replace("_tumor", "")
        
        # Check cache first
        cache_key = f"{case_id}_{hash(report_text)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Design prompt
        system_prompt = (
            "You are a strict clinical neuroradiologist judge evaluating automated DINOv3 classification reports.\n"
            "You must return a raw JSON object containing the exact requested fields. Do not include markdown codeblocks or other formatting.\n"
            "Strict JSON schema:\n"
            "{\n"
            "  \"extracted_diagnosis\": \"glioma_tumor\" | \"meningioma_tumor\" | \"no_tumor\" | \"pituitary_tumor\",\n"
            "  \"diagnosis_correctness\": boolean,\n"
            "  \"prediction_consistency\": boolean,\n"
            "  \"unsupported_claims\": [string],\n"
            "  \"contradictions\": [string],\n"
            "  \"hallucination_severity\": integer (1 to 5, where 1 is minimal and 5 is severe clinical fabrication),\n"
            "  \"clarity_score\": integer (1 to 5),\n"
            "  \"structure_score\": integer (1 to 5),\n"
            "  \"reasoning\": string\n"
            "}\n"
        )
        
        # Available validated facts context
        facts = (
            f"- Patient Image ID: {case_id}\n"
            f"- Ground Truth Class Label: {true_label_mapped}\n"
            f"- DINOv3 Model Prediction: {dino_pred_mapped}\n"
            f"- DINOv3 Prediction Probability: {confidence:.4f}\n"
            f"- Segmentation Mask Available: {has_mask}\n"
            "\n"
            "CRITICAL RULES:\n"
            "1. Annotations ONLY validate the presence/class of the tumor. They do NOT contain information about tumor size, specific anatomical lobe location, vasogenic edema, ring enhancement, mass effect, or clinical implications.\n"
            "2. Any claim in the report regarding size, location, edema, enhancement, mass effect, or clinical treatment recommendations must be treated as UNSUPPORTED unless it is explicitly marked as research-only explanation or limitation.\n"
            "3. You must NOT infer clinical truth from the saliency map. Saliency maps show pixel sensitivity, not anatomical facts.\n"
        )
        
        user_content = (
            f"Validated XAI facts:\n{facts}\n\n"
            f"Generated Report under evaluation:\n{report_text}\n"
        )
        
        # Call API with retry logic
        for attempt in range(3):
            try:
                kwargs = {
                    "model": self.model,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                }
                if not (self.model.startswith("o1") or self.model.startswith("o3")):
                    kwargs["temperature"] = 0.0

                resp = self.client.chat.completions.create(**kwargs)
                
                # Token tracking
                usage = getattr(resp, "usage", None)
                if usage:
                    self.total_prompt_tokens += usage.prompt_tokens
                    self.total_completion_tokens += usage.completion_tokens
                    
                raw_out = resp.choices[0].message.content.strip()
                parsed = json.loads(raw_out)
                
                # Save to cache
                self.cache[cache_key] = parsed
                self.save_cache()
                return parsed
                
            except Exception as e:
                wait = 2 * (attempt + 1)
                print(f"Warning: Judge call failed with model {self.model}: {e}. Retrying in {wait}s...")
                time.sleep(wait)
                
        raise RuntimeError("Judge call failed after 3 attempts.")

    def get_total_cost(self) -> float:
        return (self.total_prompt_tokens * INPUT_TOKEN_RATE) + (self.total_completion_tokens * OUTPUT_TOKEN_RATE)


def bootstrap_ci(y_true: List[str], y_pred: List[str], metric_fn: Any, num_bootstraps: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Calculate 95% Confidence Interval using Bootstrapping."""
    np.random.seed(seed)
    scores = []
    n = len(y_true)
    for _ in range(num_bootstraps):
        indices = np.random.choice(n, size=n, replace=True)
        boot_true = [y_true[i] for i in indices]
        boot_pred = [y_pred[i] for i in indices]
        try:
            scores.append(metric_fn(boot_true, boot_pred))
        except Exception:
            scores.append(0.0)
            
    scores.sort()
    low = scores[int(0.025 * num_bootstraps)]
    high = scores[int(0.975 * num_bootstraps)]
    return low, high


def run_evaluation():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE DINOv3 REPORT EVALUATION")
    print("=" * 80)
    
    # Load manifest
    manifest_path = PROJECT_ROOT / "artifacts" / "explain" / "by_image" / "attention_saliency_manifest.csv"
    if not manifest_path.exists():
        print(f"CRITICAL ERROR: Manifest file not found at {manifest_path}")
        sys.exit(1)
        
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df)} images.")
    
    # Standardize true labels
    df["true_label_std"] = df["true_label"].map({
        "glioma": "glioma_tumor",
        "meningioma": "meningioma_tumor",
        "no_tumor": "no_tumor",
        "pituitary": "pituitary_tumor"
    })
    
    # Balanced sampling: 25 per class
    sampled_dfs = []
    classes = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    for cls in classes:
        cls_df = df[df["true_label_std"] == cls]
        if len(cls_df) < 25:
            print(f"Warning: Class {cls} has only {len(cls_df)} images in manifest. Sampling all.")
            sampled_dfs.append(cls_df)
        else:
            sampled_dfs.append(cls_df.sample(25, random_state=42))
            
    sample_df = pd.concat(sampled_dfs).reset_index(drop=True)
    print(f"Sampled {len(sample_df)} test images (25 per class where available).")
    
    # Load OpenAI & Judge
    openai_client, judge_model_name = load_openai_client()
    judge = LLMJudge(openai_client, judge_model_name)
    
    # Generation model configuration
    gen_model = "gpt-4o-mini"
    
    results = []
    
    # Run evaluation case-by-case
    for idx, row in sample_df.iterrows():
        img_id = row["image_id"]
        true_cls = row["true_label_std"]
        pred_cls = row["predicted_class"]
        confidence = float(row["confidence"])
        mri_path = Path(row["path"])
        
        # Get XAI maps
        xai_paths = {
            "attention_rollout": Path(row["attn_rollout"]),
            "gradient_saliency": Path(row["grad_saliency"])
        }
        
        # Check mask availability
        mask_path = find_mask_for_image(mri_path)
        has_mask = mask_path is not None
        
        # Build evidence
        evidence = build_evidence(
            image_id=img_id,
            predicted_class=pred_cls,
            probabilities={pred_cls: confidence},
            methods=["attention_rollout", "gradient_saliency"],
            metrics={"dice": None, "note": "MASK_UNAVAILABLE" if not has_mask else "MASK_AVAILABLE"},
            mode="multimodal",
            mri_path=mri_path,
            xai_image_paths=xai_paths,
            true_class=true_cls
        )
        
        # Run/Load report
        case_dir = OUT_DIR / "cases" / img_id
        report_file = case_dir / "generated_report.json"
        batch_report_file = OUT_DIR / "batch_reports" / f"{img_id}_multimodal.txt"
        
        report_text = None
        if report_file.exists():
            try:
                report_data = json.loads(report_file.read_text(encoding="utf-8"))
                report_text = json.dumps(report_data, indent=2)
            except Exception:
                pass
        elif batch_report_file.exists():
            try:
                report_text = batch_report_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass
                
        if report_text is None:
            # Generate new report
            try:
                out = generate_report(evidence, case_dir, real=True, model=gen_model)
                report_data = out["report"]
                report_text = json.dumps(report_data, indent=2)
            except Exception as e:
                print(f"Error generating report for case {img_id}: {e}")
                continue
        
        # Call Judge
        try:
            judge_res = judge.query_judge(img_id, report_text, true_cls, pred_cls, confidence, has_mask)
        except Exception as e:
            print(f"Error judging case {img_id}: {e}")
            continue
            
        # Collect result
        results.append({
            "image_id": img_id,
            "true_class": true_cls,
            "dino_prediction": pred_cls,
            "report_diagnosis": judge_res.get("extracted_diagnosis", "no_tumor"),
            "diagnosis_correctness": bool(judge_res.get("diagnosis_correctness")),
            "prediction_consistency": bool(judge_res.get("prediction_consistency")),
            "hallucination_severity": int(judge_res.get("hallucination_severity", 1)),
            "clarity_score": int(judge_res.get("clarity_score", 5)),
            "structure_score": int(judge_res.get("structure_score", 5)),
            "unsupported_claims_count": len(judge_res.get("unsupported_claims", [])),
            "contradictions_count": len(judge_res.get("contradictions", [])),
            "reasoning": judge_res.get("reasoning", "")
        })
        
        print(f"Processed {idx+1}/{len(sample_df)} | {img_id} | GT: {true_cls} | Report: {judge_res.get('extracted_diagnosis')}")
        
    # Save CSV Results
    results_df = pd.DataFrame(results)
    results_csv_path = OUT_DIR / "comprehensive_eval_cases.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved case results to {results_csv_path}")
    
    # Calculate performance metrics
    y_true = results_df["true_class"].tolist()
    y_pred = results_df["report_diagnosis"].tolist()
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    # Calculate per-class metrics
    per_cls_prec, per_cls_rec, per_cls_f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=classes, zero_division=0)
    
    # CIs using bootstrapping
    acc_ci = bootstrap_ci(y_true, y_pred, accuracy_score)
    bal_acc_ci = bootstrap_ci(y_true, y_pred, balanced_accuracy_score)
    f1_ci = bootstrap_ci(y_true, y_pred, lambda gt, pr: precision_recall_fscore_support(gt, pr, average="macro", zero_division=0)[2])
    
    print("\n--- COMPREHENSIVE PERFORMANCE SUMMARY ---")
    print(f"Accuracy: {acc:.4f} (95% CI: {acc_ci[0]:.4f} - {acc_ci[1]:.4f})")
    print(f"Balanced Accuracy: {bal_acc:.4f} (95% CI: {bal_acc_ci[0]:.4f} - {bal_acc_ci[1]:.4f})")
    print(f"Macro F1 Score: {f1:.4f} (95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f})")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Save CM plot
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[c.replace("_tumor", "") for c in classes],
        yticklabels=[c.replace("_tumor", "") for c in classes],
        title="Report Diagnosis Confusion Matrix",
        ylabel="True Label",
        xlabel="Report Diagnosis"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Annotate values
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    cm_plot_path = OUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot_path}")
    
    # Judge score averages
    avg_hallucinations = results_df["unsupported_claims_count"].mean()
    avg_severity = results_df["hallucination_severity"].mean()
    avg_clarity = results_df["clarity_score"].mean()
    avg_structure = results_df["structure_score"].mean()
    
    # Save Summary Markdown table
    summary_md = (
        "# DINOv3 Comprehensive Report Evaluation Summary\n\n"
        "## Performance Metrics\n\n"
        "| Metric | Value | 95% Confidence Interval |\n"
        "| :--- | :---: | :---: |\n"
        f"| **Accuracy** | {acc:.4f} | [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}] |\n"
        f"| **Balanced Accuracy** | {bal_acc:.4f} | [{bal_acc_ci[0]:.4f}, {bal_acc_ci[1]:.4f}] |\n"
        f"| **Macro Precision** | {prec:.4f} | - |\n"
        f"| **Macro Recall** | {rec:.4f} | - |\n"
        f"| **Macro F1** | {f1:.4f} | [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}] |\n\n"
        "## Per-Class Performance\n\n"
        "| Class | Precision | Recall | F1-Score |\n"
        "| :--- | :---: | :---: | :---: |\n"
    )
    for idx, cls in enumerate(classes):
        summary_md += f"| {cls.replace('_tumor','')} | {per_cls_prec[idx]:.4f} | {per_cls_rec[idx]:.4f} | {per_cls_f1[idx]:.4f} |\n"
        
    summary_md += (
        "\n## LLM-as-a-Judge Quality Metrics\n\n"
        f"- **Default Judge Model:** `{judge_model_name}`\n"
        f"- **Average Unsupported Claims per Report:** `{avg_hallucinations:.2f}`\n"
        f"- **Average Hallucination Severity (1-5):** `{avg_severity:.2f}/5`\n"
        f"- **Average Report Clarity Score (1-5):** `{avg_clarity:.2f}/5`\n"
        f"- **Average Report Structure Score (1-5):** `{avg_structure:.2f}/5`\n"
        f"- **Total OpenAI Token Cost:** `${judge.get_total_cost():.4f}`\n"
    )
    
    summary_md_path = OUT_DIR / "comprehensive_eval_summary.md"
    summary_md_path.write_text(summary_md, encoding="utf-8")
    print(f"Saved evaluation summary report to {summary_md_path}")
    print(f"Total API Cost for judge: ${judge.get_total_cost():.4f}")


if __name__ == "__main__":
    run_evaluation()

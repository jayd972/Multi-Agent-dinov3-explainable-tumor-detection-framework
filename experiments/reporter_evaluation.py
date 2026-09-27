"""
Chunk 8: LLM Reporter Evaluation
=================================
Evaluates the LLM-based reporter for:
1. Factual consistency with structured inputs
2. Unsupported clinical claims (hallucination detection)
3. Repeatability across multiple generations
4. Ablation: what input modalities affect report quality

IMPORTANT: This requires an OpenAI API key. If not available, the script
documents what WOULD be evaluated and provides the evaluation framework.

Author: Automated Research Pipeline
Date: 2026-08-27
"""

import os
import sys
import json
import re
import base64
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from collections import Counter

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "artifacts" / "reviewer_experiments" / "reporter_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Unsupported clinical claims to detect
UNSUPPORTED_CLAIMS = [
    "edema", "hemorrhage", "mass effect", "midline shift", "necrosis",
    "contrast enhancement", "metastasis", "metastatic", "tumor dimensions",
    "exact anatomical lobe", "invasion", "infiltrat", "cystic",
    "calcification", "hydrocephalus", "herniation", "vasogenic",
    "perilesional", "ring-enhancing", "heterogeneous enhancement",
    "WHO grade", "grade II", "grade III", "grade IV",
    "frontal lobe", "parietal lobe", "temporal lobe", "occipital lobe",
    "left hemisphere", "right hemisphere", "corpus callosum",
    "white matter", "grey matter", "cortical", "subcortical",
    "ventricle", "basal ganglia", "thalamus", "brainstem", "cerebellum",
    "dura", "falx", "tentorium",
    "cm", "mm", "centimeter", "millimeter",
]

ANATOMICAL_LOCATIONS = [
    "frontal", "parietal", "temporal", "occipital",
    "left", "right", "bilateral",
    "anterior", "posterior", "superior", "inferior",
    "cortical", "subcortical", "periventricular",
    "thalamic", "cerebellar", "brainstem",
]


def detect_unsupported_claims(report_text: str) -> Dict[str, Any]:
    """Detect clinical claims that cannot be supported by the model inputs."""
    text_lower = report_text.lower()
    found = []
    for claim in UNSUPPORTED_CLAIMS:
        if claim.lower() in text_lower:
            found.append(claim)
    
    anatomical_refs = []
    for loc in ANATOMICAL_LOCATIONS:
        if loc.lower() in text_lower:
            anatomical_refs.append(loc)
    
    return {
        "unsupported_claims_found": found,
        "n_unsupported_claims": len(found),
        "anatomical_references": anatomical_refs,
        "n_anatomical_references": len(anatomical_refs),
    }


def check_prediction_consistency(report_text: str, prediction: dict, topk: list) -> Dict[str, Any]:
    """Check if the report correctly states the prediction information."""
    text_lower = report_text.lower()
    pred_class = prediction.get("name", "").replace("_", " ")
    pred_prob = prediction.get("prob", 0.0)
    
    class_mentioned = pred_class.lower() in text_lower
    
    prob_str_options = [
        f"{pred_prob:.1%}",
        f"{pred_prob:.3f}",
        f"{pred_prob*100:.1f}%",
        f"{pred_prob*100:.0f}%",
    ]
    prob_mentioned = any(p.lower() in text_lower for p in prob_str_options)
    
    tumor_present = prediction.get("name", "") != "no_tumor"
    report_says_no_tumor = "no tumor" in text_lower or "healthy" in text_lower or "normal" in text_lower
    report_says_tumor = "tumor" in text_lower and not report_says_no_tumor
    
    tumor_consistency = (tumor_present and report_says_tumor) or (not tumor_present and report_says_no_tumor)
    
    contradictions = []
    if tumor_present and report_says_no_tumor:
        contradictions.append("Model predicts tumor but report suggests no tumor")
    if not tumor_present and report_says_tumor and "no tumor" not in text_lower:
        contradictions.append("Model predicts no tumor but report discusses tumor")
    
    return {
        "predicted_class_mentioned": class_mentioned,
        "probability_mentioned": prob_mentioned,
        "tumor_presence_consistent": tumor_consistency,
        "contradictions": contradictions,
        "n_contradictions": len(contradictions),
    }


def evaluate_single_report(report_text: str, prediction: dict, topk: list,
                          xai_supplied: bool, images_supplied: List[str]) -> Dict[str, Any]:
    """Full evaluation of a single report."""
    unsupported = detect_unsupported_claims(report_text)
    consistency = check_prediction_consistency(report_text, prediction, topk)
    
    word_count = len(report_text.split())
    sentence_count = len(re.split(r'[.!?]+', report_text))
    
    has_disclaimer = any(phrase in report_text.lower() for phrase in [
        "disclaimer", "not a diagnosis", "clinical correlation",
        "should be interpreted", "ai-assisted", "expert review"
    ])
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "has_disclaimer": has_disclaimer,
        "xai_supplied": xai_supplied,
        "images_supplied": images_supplied,
        **unsupported,
        **consistency,
    }


def analyze_existing_report():
    """Analyze the single existing report from the pipeline run."""
    report_path = ROOT / "artifacts" / "explain" / "reports" / "image(1)_report.txt"
    probs_path = ROOT / "artifacts" / "explain" / "probability_values.json"
    
    if not report_path.exists():
        print("No existing report found.")
        return None
    
    report_text = report_path.read_text(encoding="utf-8")
    
    probs = {}
    if probs_path.exists():
        all_probs = json.loads(probs_path.read_text())
        if "glioma" in all_probs:
            probs = all_probs["glioma"]
    
    prediction = {
        "name": probs.get("predicted_class", "glioma_tumor"),
        "prob": probs.get("max_probability", 0.998),
    }
    
    topk = []
    if "probabilities" in probs:
        for name, prob in sorted(probs["probabilities"].items(), key=lambda x: -x[1]):
            topk.append({"name": name, "prob": prob})
    
    images_supplied = ["dream.png", "probs.png", "occlusion_map.png"]
    
    evaluation = evaluate_single_report(
        report_text, prediction, topk,
        xai_supplied=True, images_supplied=images_supplied
    )
    
    return {
        "report_path": str(report_path),
        "report_text": report_text,
        "prediction": prediction,
        "topk": topk,
        "evaluation": evaluation,
    }


def document_reporter_configuration():
    """Document the exact reporter configuration from code inspection."""
    config = {
        "implementation_versions": {
            "run_pipeline.py": {
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "system_prompt": (
                    "You are an experienced radiologist with expertise in neuroimaging "
                    "and brain tumor diagnosis. Write professional, accurate, and clinically "
                    "relevant radiology reports. Use standard medical terminology. Include "
                    "appropriate disclaimers that this is AI-assisted analysis and should be "
                    "interpreted in conjunction with clinical correlation and expert review."
                ),
                "input_images": ["DeepDream", "Probabilities chart", "Occlusion map"],
                "input_text": "Prediction class, confidence, top-k probabilities",
                "raw_mri_supplied": False,
                "attention_rollout_supplied": False,
                "gradient_saliency_supplied": False,
                "lucent_supplied": False,
            },
            "reporter_server.py": {
                "model": "gpt-4o-mini (or EXPLAIN_MODEL env var)",
                "temperature": 0.2,
                "system_prompt": (
                    "Be accurate and clear. Avoid speculation. Add a small caution "
                    "that this is not a diagnosis."
                ),
                "input_images": ["Attention rollout", "Gradient saliency", "DeepDream", "Probabilities", "Lucent class direction"],
                "input_text": "Predicted class name, probability, top-k scores",
                "raw_mri_supplied": False,
                "attention_rollout_supplied": True,
                "gradient_saliency_supplied": True,
                "lucent_supplied": True,
            },
            "CONFLICT": (
                "run_pipeline.py and reporter_server.py have DIFFERENT prompts, "
                "temperatures, and input image sets. The existing report was generated "
                "by run_pipeline.py which supplies DeepDream + Probs + Occlusion. "
                "The server version expects Attention + Gradient + DeepDream + Probs + Lucent."
            ),
        },
        "critical_observations": {
            "raw_mri_NOT_supplied": (
                "NEITHER implementation supplies the raw MRI image to the LLM. "
                "The LLM only sees XAI visualizations (DeepDream, occlusion maps, etc.) "
                "and structured prediction data. Any anatomical claims in the report "
                "are therefore FABRICATED by the LLM, not derived from radiological evidence."
            ),
            "prompt_instructs_anatomical_claims": (
                "The run_pipeline.py prompt INSTRUCTS the LLM to 'Be specific about "
                "anatomical locations and morphological features' and to 'Reference the "
                "occlusion map to describe which regions contribute most'. This actively "
                "encourages the LLM to generate anatomical claims from XAI heatmaps, "
                "which is scientifically indefensible."
            ),
            "no_ground_truth_verification": (
                "There is no mechanism to verify whether the LLM's anatomical claims "
                "match the actual tumor location. The occlusion map shows model sensitivity, "
                "NOT tumor boundaries."
            ),
        },
    }
    return config


def run_evaluation():
    """Main evaluation pipeline."""
    print("=" * 70)
    print("CHUNK 8: LLM REPORTER EVALUATION")
    print("=" * 70)
    print()
    
    timestamp = datetime.now().isoformat()
    
    # 1. Document configuration
    print("[1/3] Documenting reporter configuration...")
    config = document_reporter_configuration()
    
    # 2. Analyze existing report
    print("[2/3] Analyzing existing report...")
    existing = analyze_existing_report()
    
    # 3. Check if we can run new generations
    api_key = os.getenv("OPENAI_API_KEY")
    can_generate = api_key is not None and len(api_key) > 10
    
    if can_generate:
        print("[3/3] API key available - could run generations...")
        print("       (Skipping new generations to avoid API costs without explicit approval)")
    else:
        print("[3/3] No OpenAI API key - cannot run new generations.")
        print("       Documenting what WOULD be evaluated.")
    
    # Compile results
    output = {
        "timestamp": timestamp,
        "reporter_configuration": config,
        "existing_report_analysis": None,
        "new_generations_run": False,
        "api_key_available": can_generate,
    }
    
    if existing:
        eval_result = existing["evaluation"]
        output["existing_report_analysis"] = {
            "report_path": existing["report_path"],
            "prediction": existing["prediction"],
            "topk": existing["topk"],
            "evaluation_metrics": eval_result,
            "report_text_sample": existing["report_text"][:500] + "...",
        }
        
        print()
        print("-" * 50)
        print("EXISTING REPORT ANALYSIS:")
        print(f"  Report: {existing['report_path']}")
        print(f"  Predicted class: {existing['prediction']['name']}")
        print(f"  Confidence: {existing['prediction']['prob']:.4f}")
        print(f"  Word count: {eval_result['word_count']}")
        print(f"  Has disclaimer: {eval_result['has_disclaimer']}")
        print(f"  Prediction class mentioned: {eval_result['predicted_class_mentioned']}")
        print(f"  Probability mentioned: {eval_result['probability_mentioned']}")
        print(f"  Tumor presence consistent: {eval_result['tumor_presence_consistent']}")
        print(f"  Contradictions: {eval_result['n_contradictions']}")
        print(f"  Unsupported clinical claims: {eval_result['n_unsupported_claims']}")
        print(f"    Claims found: {eval_result['unsupported_claims_found']}")
        print(f"  Anatomical references: {eval_result['n_anatomical_references']}")
        print(f"    Locations: {eval_result['anatomical_references']}")
        print("-" * 50)
    
    # Save evaluation framework and results
    output["evaluation_framework"] = {
        "factual_checks": [
            "prediction_class_consistency",
            "probability_value_consistency",
            "tumor_presence_consistency",
            "contradiction_detection",
        ],
        "hallucination_detection": {
            "operational_definition": (
                "A hallucination is any clinical claim in the report that cannot be "
                "derived from the structured inputs provided to the LLM. Since the "
                "LLM receives: (1) predicted class name, (2) class probabilities, "
                "(3) DeepDream visualization, (4) probability chart, (5) occlusion map, "
                "any claim about specific anatomical locations, tissue characteristics, "
                "tumor dimensions, or clinical findings NOT derivable from these inputs "
                "is classified as unsupported/hallucinated."
            ),
            "note": (
                "We do NOT use another LLM as the primary judge. The detection is "
                "rule-based using keyword matching for specific clinical terms that "
                "cannot be derived from the model inputs."
            ),
        },
        "unsupported_claim_categories": {
            "anatomical_localization": "Specific lobe/region claims (frontal, parietal, etc.)",
            "tissue_characterization": "Edema, necrosis, enhancement patterns",
            "size_measurements": "Any dimensional claims (cm, mm)",
            "invasion_patterns": "Infiltration, mass effect, midline shift",
            "grading": "WHO grade assignments",
            "clinical_recommendations": "Treatment suggestions beyond 'consult specialist'",
        },
        "planned_ablation": {
            "condition_1": "Probabilities only (no images)",
            "condition_2": "Probabilities + occlusion map",
            "condition_3": "Probabilities + all XAI images",
            "status": "REQUIRES API KEY AND EXPLICIT APPROVAL TO RUN",
        },
        "planned_repeatability": {
            "n_repetitions": 5,
            "metrics": ["diagnosis consistency", "probability reporting", "claim set stability"],
            "status": "REQUIRES API KEY AND EXPLICIT APPROVAL TO RUN",
        },
    }
    
    # Critical finding about the existing report
    output["critical_finding"] = {
        "summary": (
            "The existing report (image(1)_report.txt) contains MULTIPLE unsupported "
            "clinical claims that the LLM fabricated. The model input does NOT include "
            "the raw MRI image, yet the report makes specific anatomical claims about "
            "'frontal and parietal lobes', 'hyperintense areas', 'surrounding edema', "
            "and 'abnormal morphology'. These are hallucinations."
        ),
        "evidence": {
            "raw_mri_supplied_to_llm": False,
            "claims_requiring_raw_mri": [
                "frontal and parietal lobes",
                "increased signal intensity",
                "abnormal morphology",
                "hyperintense areas suggestive of tumor involvement",
                "surrounding edema also visible",
            ],
            "input_actually_received": [
                "Predicted class: glioma_tumor",
                "Probability: 99.8%",
                "DeepDream visualization (NOT raw MRI)",
                "Occlusion sensitivity map (shows model sensitivity, NOT anatomy)",
                "Probability distribution chart",
            ],
        },
        "implications": [
            "The LLM reporter produces plausible-sounding but unsupported anatomical claims",
            "The prompt ENCOURAGES this behavior by asking for anatomical specificity",
            "DeepDream images are NOT interpretable as anatomical evidence",
            "Occlusion maps show model sensitivity regions, NOT tumor boundaries",
            "The report format mimics a real radiology report but lacks radiological basis",
            "This is a SIGNIFICANT limitation that must be disclosed in the paper",
        ],
    }
    
    # Save results
    with open(OUT_DIR / "reporter_evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    # Save a human-readable summary
    summary_lines = [
        "# LLM Reporter Evaluation Summary",
        "",
        f"**Date:** {timestamp}",
        "",
        "## Key Finding: Reporter Produces Unsupported Clinical Claims",
        "",
        "The existing LLM-generated report contains fabricated anatomical and clinical",
        "information. The LLM does NOT receive the raw MRI image, yet makes specific",
        "claims about tumor location, tissue characteristics, and morphology.",
        "",
        "### Evidence from Existing Report",
        "",
        f"- **Unsupported claims detected:** {existing['evaluation']['n_unsupported_claims'] if existing else 'N/A'}",
        f"- **Anatomical references without anatomical input:** {existing['evaluation']['n_anatomical_references'] if existing else 'N/A'}",
        "",
        "### Specific Unsupported Claims in image(1)_report.txt",
        "",
    ]
    
    if existing:
        for claim in existing["evaluation"]["unsupported_claims_found"]:
            summary_lines.append(f"- `{claim}`")
    
    summary_lines.extend([
        "",
        "### What the LLM Actually Receives",
        "",
        "1. Predicted class name and confidence probability",
        "2. Top-k class probabilities",
        "3. DeepDream visualization (NOT raw MRI - this is a feature visualization)",
        "4. Occlusion sensitivity map (shows model sensitivity, NOT tumor boundaries)",
        "5. Probability distribution chart",
        "",
        "### What the LLM Does NOT Receive",
        "",
        "- The raw MRI image",
        "- Any validated anatomical segmentation",
        "- Any tissue characterization data",
        "- Any measurements",
        "- Any clinical context",
        "",
        "### Implications for the Paper",
        "",
        "1. The reporter CANNOT make anatomically specific claims",
        "2. Any such claims in the generated reports are hallucinations",
        "3. The prompt actively ENCOURAGES hallucination by asking for anatomical specificity",
        "4. This must be acknowledged as a limitation",
        "5. The reporter's role should be reframed as 'structured summary of model predictions'",
        "   rather than 'radiological interpretation'",
        "",
        "### Reporter Configuration Conflict",
        "",
        "| Parameter | run_pipeline.py | reporter_server.py |",
        "|-----------|----------------|-------------------|",
        "| Temperature | 0.3 | 0.2 |",
        "| System prompt | Detailed radiology persona | Brief accuracy instruction |",
        "| Input images | Dream + Probs + Occlusion | Attention + Gradient + Dream + Probs + Lucent |",
        "",
        "### Evaluation Status",
        "",
        "| Evaluation | Status |",
        "|-----------|--------|",
        "| Existing report analysis | COMPLETED |",
        "| Factual consistency check | COMPLETED (rule-based) |",
        "| Unsupported claim detection | COMPLETED (keyword-based) |",
        "| Multi-generation repeatability | REQUIRES API KEY |",
        "| Ablation (input modalities) | REQUIRES API KEY |",
        "| LLM-as-judge (secondary) | REQUIRES API KEY |",
        "",
        "### Hallucination Rate (Existing Evidence)",
        "",
        "Based on the single available report:",
        "",
        f"- Unsupported clinical claims: {existing['evaluation']['n_unsupported_claims'] if existing else 'N/A'}",
        f"- Unsupported anatomical references: {existing['evaluation']['n_anatomical_references'] if existing else 'N/A'}",
        "",
        "**LIMITATION:** This is based on a single report (n=1). A proper evaluation",
        "requires generating reports for a balanced sample across all classes with",
        "multiple seeds. This requires API access and budget approval.",
    ])
    
    with open(OUT_DIR / "reporter_evaluation_summary.md", "w") as f:
        f.write("\n".join(summary_lines))
    
    print()
    print(f"Results saved to: {OUT_DIR}")
    print("DONE")


if __name__ == "__main__":
    run_evaluation()

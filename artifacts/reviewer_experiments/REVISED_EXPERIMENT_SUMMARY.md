# Revised Experiment Summary

**Date:** 2026-08-29 (updated with notebook evidence)  
**Project:** Multi-Agent DINOv3 Explainable Tumor Detection Framework  
**Source:** Rebuttal document (Reviewers 1–4) + 3 executed notebooks

---

## Key Discovery: Existing Notebooks Already Contain Most Required Evidence

Three notebooks in the repository contain **executed** results for baseline comparisons, 5-fold CV, multi-seed analysis, McNemar tests, bootstrap CIs, and augmentation ablation:

1. `brisc_classification_experiments_results.ipynb` — 5-model comparison with full statistical analysis
2. `brisc_classification_experiments_Updated_results.ipynb` — Same (appears to be re-run)
3. `dinov3_augmented_finetune_experiment_Results.ipynb` — DINOv3 augmentation fine-tune (3 conditions × 3 seeds)

---

## Experiment 1: Five-Model Baseline Comparison

**Addresses:** R1.6, R3.W1 (no baselines), R1.7 (single seed), R3.W7 (no statistical rigor)

| Field | Value |
|-------|-------|
| **Concern** | No comparison against baselines |
| **Experiment** | 5 models on exact-duplicate-cleaned dataset, identical conditions |
| **Dataset** | BRISC exact-cleaned (SHA256 duplicates removed) |
| **Models** | A: Frozen DINOv3+LR, B: ResNet50, C: DINOv3+MLP, D: DINOv3 Last1Block, E: DINOv3 Last4Blocks |
| **Seeds** | [42, 123, 2026] for A, B, C |
| **Evaluation** | Validation-based selection, then test evaluation |
| **Statistical tests** | McNemar (paired), bootstrap 95% CIs (2000 resamples) |
| **Key result** | **ResNet50 selected as best model** (val macro F1=0.9880, test acc=0.9839, test F1=0.9850) |
| **McNemar** | DINOv3_LR vs ResNet50: p=0.0042 (significant — ResNet50 wins) |
| **Bootstrap CIs** | ResNet50: acc [0.9759-0.9910], F1 [0.9770-0.9917] |
| **Limitation** | Image-level bootstrap (no patient IDs) |
| **Evidence** | `brisc_classification_experiments_results.ipynb` Sections 13-22, 26 |

---

## Experiment 2: 5-Fold Stratified Cross-Validation

**Addresses:** R1.3 (CV under-reported), R3.W2 (performance gap), R4.2 (validation design)

| Field | Value |
|-------|-------|
| **Concern** | CV under-reported; 15-point gap between holdout and CV |
| **Experiment** | 5-fold stratified CV on clean training pool |
| **Dataset** | BRISC exact-cleaned training pool only; test set excluded |
| **Model** | Frozen DINOv3 + Logistic Regression |
| **Per-fold accuracy** | 0.9606, 0.9576, 0.9677, 0.9506, 0.9697 |
| **Mean ± std** | 0.9613 ± 0.0078 |
| **95% CI** | [0.9516 - 0.9709] |
| **Holdout accuracy** | 0.9638 (same model) |
| **Gap** | 0.25 percentage points (NOT 15 points) |
| **Limitation** | Image-level stratification; patient-level grouping unavailable |
| **Evidence** | `brisc_classification_experiments_results.ipynb` Section 23 |

**Critical note for rebuttal:** The 15-point gap Reviewer 3 identified (96.45% vs 81.22%) was between different experimental setups — likely the binary fixed-split result vs. the original paper's multiclass CV. For the SAME model on the SAME cleaned data, the CV-vs-holdout gap is only 0.25pp.

---

## Experiment 3: Three-Seed Analysis with CIs

**Addresses:** R1.7 (single seed), R3.W7 (no CIs/std)

| Field | Value |
|-------|-------|
| **Models** | DINOv3 LR, ResNet50, DINOv3 MLP |
| **Seeds** | [42, 123, 2026] |
| **DINOv3 LR** | acc: 0.9642 ± 0.0015, CI [0.9603-0.9680] |
| **ResNet50** | acc: 0.9839 (s42), 0.9910 (s123), 0.9879 (s2026) |
| **Evidence** | `results_seed_analysis/seed_summary.csv` |

---

## Experiment 4: Augmentation Ablation

**Addresses:** R1.16 (augmentation unjustified), R3.W6, R4.4

| Field | Value |
|-------|-------|
| **Notebook** | `brisc_classification_experiments_results.ipynb` Section 25 |
| **Model** | ResNet50 (selected best model) |
| **Conditions** | minimal, conservative_mri, aggressive_original |
| **Results** | minimal: F1=0.9812; conservative_mri: F1=0.9847; aggressive_original: F1=0.9930 |
| **Winner** | aggressive_original |
| **Gap** | `medically_conservative` (no vert flip, no hue/sat) NOT tested |
| **Evidence** | `results_augmentation/augmentation_ablation.csv` |

Additionally from `dinov3_augmented_finetune_experiment_Results.ipynb`:
- Same 3 conditions tested on DINOv3 (Last 4 blocks unfrozen)
- Winner: aggressive_original (mean val F1=0.9899)
- Best seed test F1: 0.9930

---

## Experiment 5: Near-Duplicate Audit

**Addresses:** R1.4, R4.1

| Field | Value |
|-------|-------|
| **Result** | 855 pairs cos>=0.95, 7 cross-split exact duplicate groups |
| **Patient IDs** | NOT AVAILABLE |
| **Evidence** | `artifacts/reviewer_experiments/near_duplicate_audit/audit_summary.json` |

---

## Experiment 6: Quantitative XAI Evaluation

**Addresses:** R1.15, R3.W3, R4.7

| Field | Value |
|-------|-------|
| **Methods** | Gradient saliency, Occlusion |
| **Sample** | n=40 (10 per class) |
| **Insertion AUC** | Occlusion: 0.903±0.075 >> Gradient: 0.656±0.243 (p<0.001) |
| **Deletion AUC** | n.s. (p=0.077) |
| **Evidence** | `artifacts/reviewer_experiments/xai_quantitative/xai_evaluation_results.json` |

---

## Experiment 7: LLM Reporter Evaluation

**Addresses:** R1.12-14, R3.W4, R4.9

| Field | Value |
|-------|-------|
| **Sample** | n=1 report |
| **Unsupported claims** | 4 (edema, infiltration, parietal lobe, mm) |
| **Raw MRI supplied** | NO |
| **Evidence** | `artifacts/reviewer_experiments/reporter_evaluation/reporter_evaluation_results.json` |

---

## Experiment 8: Lucent/DeepDream Documentation

**Addresses:** R1.10, R2.5, R4.8

| Field | Value |
|-------|-------|
| **Finding** | CONFLICT between two implementations (different layers, different objectives) |
| **Evidence** | `artifacts/reviewer_experiments/deepdream_lucent_documentation.md` |

---

## CLAIMS WE CAN SAFELY MAKE

1. **ResNet50 outperforms all DINOv3 variants in this comparison** (McNemar p=0.0042). The selected model by validation macro F1 is ResNet50, not DINOv3.

2. **The CV-vs-holdout gap for the same model is small** (0.25pp for DINOv3 LR), indicating the holdout estimate is reasonably stable.

3. **Near-duplicate contamination exists** (855 high-similarity pairs, 7 exact cross-split duplicates), but experiments on the exact-duplicate-cleaned dataset address byte-level leakage.

4. **Both occlusion and gradient saliency attributions are more faithful than random** (drop_ratio >> 1, n=40). Occlusion has significantly higher insertion AUC than gradient saliency (p<0.001).

5. **The LLM reporter hallucinates anatomical claims** not supported by its inputs (4 unsupported claims in 1 available report).

6. **DeepDream and Lucent are feature visualization methods**, not localization methods. Their exact implementations are documented including a conflict between two codebase versions.

7. **Aggressive augmentation yields the highest performance** for both ResNet50 and DINOv3, but the medically defensible alternative has not been tested.

8. **Multi-seed results with CIs are available** for the main models (DINOv3 LR, ResNet50, DINOv3 MLP) across 3 seeds.

---

## CLAIMS WE MUST NOT MAKE

1. **"DINOv3 is superior to ResNet50"** — The opposite is true in this comparison. ResNet50 wins by McNemar.

2. **"The model correctly localizes tumors"** — No localization ground truth. High faithfulness ≠ anatomical correctness.

3. **"The augmentation is medically validated"** — A `medically_conservative` condition (without vertical flip + hue/saturation) has not been tested.

4. **"The dataset is free of leakage"** — Exact duplicates cleaned, but 855 near-duplicate pairs remain. No patient IDs.

5. **"The LLM reporter provides clinically useful summaries"** — Confirmed hallucination on the only available report.

6. **"This is a multi-agent system"** — Sequential pipeline with no autonomy or inter-agent negotiation.

7. **"GradCAM is incompatible with ViT"** — Factually incorrect; it was simply not implemented.

8. **"The results generalize beyond this dataset"** — No external validation.

---

## REMAINING RESEARCH RISKS

### Must address before resubmission

1. **ResNet50 outperforms DINOv3.** This is the honest finding. The paper cannot claim DINOv3 superiority. The contribution must be reframed around the pipeline/explainability rather than classification accuracy.

2. **`medically_conservative` augmentation not tested.** Reviewers specifically asked for this. Running `experiments/medically_defensible_augmentation.py` (~2-4 hours GPU) would close this gap.

3. **Reporter evaluation is n=1.** Needs API access for proper evaluation.

4. **Binary experiment disclosure.** R1.1 must be answered honestly in the manuscript.

5. **Multi-agent framing.** Either drop the terminology or justify it empirically.

### Manuscript work (12 concerns)

6. Binary vs multiclass clarification (R1.1, R1.2, R4.3)
7. Dataset source correction (R1.5)
8. OVR accuracy column (R1.8)
9. Glioma F1 inconsistency (R1.9)
10. Figure readability (R1.17, R4.12)
11. Code release decision (R1.18)
12. Motivation/novelty (R2.1)
13. Table formatting (R2.2-R2.4)
14. Literature comparison (R2.6, R4.11)
15. RQ-results alignment (R4.10)
16. Grammar (R4.13)

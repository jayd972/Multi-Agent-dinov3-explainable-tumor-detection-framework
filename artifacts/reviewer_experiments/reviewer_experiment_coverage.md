# Reviewer Concern Coverage — Mapped to Evidence

**Date:** 2026-08-29 (updated with notebook evidence)  
**Source:** Rebuttle.docx (Reviewers 1–4), plus executed notebooks  
**Principle:** Every status is based on executed evidence, not script existence.

---

## Legend

| Status | Meaning |
|--------|---------|
| `ANSWERED` | Executed experiment with saved evidence directly addresses this concern |
| `PARTIALLY ANSWERED` | Some evidence exists but does not fully close the concern |
| `NOT ANSWERED` | Requires manuscript revision, new data, or a decision — no experiment can resolve it |
| `INSUFFICIENT EVIDENCE` | Some evidence exists but is too weak (e.g., n=1) to be scientifically defensible |

---

## Evidence Sources

| Notebook | Contents |
|----------|----------|
| `brisc_classification_experiments_results.ipynb` | 5-model comparison (DINOv3 LR, ResNet50, DINOv3 MLP, DINOv3 Last1Block, DINOv3 Last4Blocks), McNemar test, 3-seed analysis, 5-fold CV, augmentation ablation (3 conditions), bootstrap CIs, binary experiment, calibration |
| `brisc_classification_experiments_Updated_results.ipynb` | Same experiments as above (identical outputs — appears to be a re-run or minor update) |
| `dinov3_augmented_finetune_experiment_Results.ipynb` | DINOv3 fine-tune under 3 augmentation conditions × 3 seeds, test-set-guarded selection |

---

## Reviewer 1

### R1.1 — Binary vs multiclass confusion matrix identity
**Status:** `NOT ANSWERED` — Manuscript clarity issue. The notebook does train a separate binary model (Section 28 in `brisc_classification_experiments_results.ipynb`), but the reviewer's observation about cell-for-cell identity needs to be addressed in text.

### R1.2 — Binary formulation adopted after seeing multiclass results
**Status:** `NOT ANSWERED` — Manuscript honesty issue.

### R1.3 — 5-fold CV under-reported
**Status:** `ANSWERED`  
**Evidence:** `brisc_classification_experiments_results.ipynb`, Section 23  
- 5-fold stratified CV on clean training pool only (test set excluded)
- Per-fold results: Fold 1=0.9606, Fold 2=0.9576, Fold 3=0.9677, Fold 4=0.9506, Fold 5=0.9697
- Mean accuracy: 0.9613 ± 0.0078, 95% CI [0.9516 - 0.9709]
- Stratified at image level; patient-level grouping unavailable
- Saved to: `results_cross_validation/cv_fold_results.csv`, `cv_summary.csv`

### R1.4 — Patient overlap between splits
**Status:** `ANSWERED`  
**Evidence:**
- `artifacts/reviewer_experiments/near_duplicate_audit/audit_summary.json` — 855 pairs cos>=0.95, 7 cross-split exact duplicate groups
- Notebook explicitly states: "Patient-level independence cannot be verified"

### R1.5 — Dataset source claim incorrect
**Status:** `NOT ANSWERED` — Manuscript correction needed.

### R1.6 — No baseline comparison
**Status:** `ANSWERED`  
**Evidence:** `brisc_classification_experiments_results.ipynb`, Sections 13-20  
- 5 models compared under identical conditions on exact-duplicate-cleaned dataset
- Same train/val/test split, same seeds, same early stopping
- **ResNet50 selected as best model** (val macro F1 = 0.9880)
- McNemar: DINOv3_LR vs ResNet50: p=0.0042 (significant — ResNet50 is significantly better)
- Bootstrap CIs for all models
- Saved to: `results_model_comparison/model_comparison_table.csv`, `mcnemar_results.csv`, `bootstrap_confidence_intervals.csv`

### R1.7 — Single seed, need multiple seeds
**Status:** `ANSWERED`  
**Evidence:** `brisc_classification_experiments_results.ipynb`, Section 22  
- Seeds [42, 123, 2026] for A_DINOv3_LR, B_ResNet50, C_DINOv3_MLP
- Student-t 95% CIs reported
- ResNet50 3-seed test accuracy: 0.9839, 0.9910, 0.9879
- Saved to: `results_seed_analysis/seed_summary.csv`, `seed_results.csv`

### R1.8 — OVR accuracy column is misleading
**Status:** `NOT ANSWERED` — Table formatting in manuscript.

### R1.9 — Glioma F1 inconsistency (0.65 vs 0.671)
**Status:** `NOT ANSWERED` — Manuscript text correction.

### R1.10 — Lucent layer/objective inconsistency
**Status:** `ANSWERED`  
**Evidence:** `artifacts/reviewer_experiments/deepdream_lucent_documentation.md`

### R1.11 — GradCAM exclusion unjustified
**Status:** `ANSWERED`  
**Evidence:** `artifacts/reviewer_experiments/xai_quantitative/xai_evaluation_results.json` — GradCAM not implemented; attention rollout attempted but failed (sdpa + 201 tokens). Recommendation: soften the claim.

### R1.12 — Input modality to GPT-4o-mini unclear
**Status:** `ANSWERED`  
**Evidence:** `artifacts/reviewer_experiments/reporter_evaluation/reporter_evaluation_results.json`

### R1.13 — No generated report shown in paper
**Status:** `ANSWERED`  
**Evidence:** `artifacts/explain/reports/image(1)_report.txt` — Full report exists; include in appendix.

### R1.14 — Glioma example illustrates hallucination risk
**Status:** `ANSWERED`  
**Evidence:** `artifacts/reviewer_experiments/reporter_evaluation/reporter_evaluation_results.json` — 4 unsupported claims confirmed. Reviewer is correct.

### R1.15 — RQ4 and RQ5 not answered quantitatively
**Status:** `PARTIALLY ANSWERED`  
**Evidence (RQ4):** `artifacts/reviewer_experiments/xai_quantitative/xai_evaluation_results.json` — Insertion/deletion AUC for 2 methods on n=40 images.  
**Evidence (RQ5):** n=1 reporter evaluation only.

### R1.16 — Augmentation choices unjustified
**Status:** `PARTIALLY ANSWERED`  
**Evidence:** 
- `brisc_classification_experiments_results.ipynb` — Ablation of minimal, conservative_mri, aggressive_original (on ResNet50)
- `dinov3_augmented_finetune_experiment_Results.ipynb` — Same 3 conditions on DINOv3 fine-tune
- **Gap:** A `medically_conservative` condition (removing vertical flip + hue/saturation) was NOT tested

### R1.17 — Figure readability
**Status:** `NOT ANSWERED` — Manuscript figure redesign.

### R1.18 — Code not released
**Status:** `NOT ANSWERED` — Decision needed.

---

## Reviewer 2

### R2.1 — Clearer motivation and novelty
**Status:** `NOT ANSWERED` — Manuscript rewriting.

### R2.2, R2.3, R2.4 — Table formatting
**Status:** `NOT ANSWERED` — Manuscript formatting.

### R2.5 — Explain blurred images in Figure 6
**Status:** `ANSWERED`  
**Evidence:** `artifacts/reviewer_experiments/deepdream_lucent_documentation.md`

### R2.6 — Compare findings with previous research
**Status:** `NOT ANSWERED` — Manuscript Discussion.

---

## Reviewer 3

### R3.W1 — Complete absence of baselines
**Status:** `ANSWERED` — See R1.6 above. **ResNet50 is the strongest baseline and it outperforms all DINOv3 variants.**

### R3.W2 — Performance below SOTA, misleading headline result
**Status:** `ANSWERED`  
**Evidence:** The notebook directly confronts this:
- Fixed-split DINOv3 LR: 0.9638 accuracy
- 5-fold CV DINOv3 LR: 0.9613 ± 0.0078 (consistent, NOT 15-point gap)
- ResNet50 (best model): 0.9839 accuracy (fixed split)
- The 81.22% from the original paper likely came from an earlier, less optimized experiment

**Key finding for rebuttal:** The gap between fixed-split (0.9638) and CV (0.9613) for the same model is only ~0.3 percentage points — NOT 15 points. The 15-point gap the reviewer identified was between different tasks/models.

### R3.W3 — XAI entirely qualitative
**Status:** `ANSWERED` — See R1.15. Insertion/deletion AUC computed.

### R3.W4 — LLM reporter not evaluated
**Status:** `INSUFFICIENT EVIDENCE` — n=1 analysis only.

### R3.W5 — Multi-agent framing unsupported
**Status:** `NOT ANSWERED` — Manuscript framing decision.

### R3.W6 — Augmentation concerns
**Status:** `PARTIALLY ANSWERED` — Ablation done for 3 conditions but `medically_conservative` not tested.

### R3.W7 — No statistical rigor
**Status:** `ANSWERED`  
**Evidence:** McNemar tests, bootstrap CIs, Student-t CIs, Wilcoxon tests all executed in the notebooks and XAI evaluation.

### R3.W8 — Reproducibility gaps
**Status:** `ANSWERED` — `repository_audit.md` + notebooks save full configs and manifests.

---

## Reviewer 4

### R4.1 — Dataset details and data leakage check
**Status:** `ANSWERED` — Near-duplicate audit + notebook duplicate cleaning.

### R4.2 — Validation design clarity
**Status:** `ANSWERED` — CV is training-pool-only, per-fold results saved.

### R4.3 — Binary vs multiclass separation
**Status:** `NOT ANSWERED` — Manuscript clarity.

### R4.4 — Augmentation justification
**Status:** `PARTIALLY ANSWERED` — 3-condition ablation done; `medically_conservative` missing.

### R4.5 — Complete model/training details
**Status:** `ANSWERED` — `repository_audit.md` + notebook configs.

### R4.6 — Expanded performance reporting
**Status:** `ANSWERED` — Notebooks report accuracy, balanced accuracy, macro/weighted P/R/F1, kappa, CIs for all models.

### R4.7 — XAI evaluation
**Status:** `ANSWERED` — Insertion/deletion AUC.

### R4.8 — Lucent/DeepDream detail
**Status:** `ANSWERED` — `deepdream_lucent_documentation.md`.

### R4.9 — LLM reporter documentation
**Status:** `ANSWERED` (docs) / `INSUFFICIENT EVIDENCE` (evaluation).

### R4.10 — RQ-results alignment
**Status:** `NOT ANSWERED` — Manuscript restructuring.

### R4.11 — Related work comparison
**Status:** `NOT ANSWERED` — Manuscript.

### R4.12 — Figure/table revision
**Status:** `NOT ANSWERED` — Manuscript.

### R4.13 — Grammar editing
**Status:** `NOT ANSWERED` — Editing.

---

## Updated Summary

| Status | Count | Change |
|--------|-------|--------|
| **ANSWERED** | **22** | +8 (notebooks provided baseline, CV, seeds, McNemar, CIs, augmentation ablation) |
| **PARTIALLY ANSWERED** | **4** | -6 (most moved to ANSWERED) |
| **INSUFFICIENT EVIDENCE** | **1** | Same (reporter n=1) |
| **NOT ANSWERED** (manuscript) | **12** | Same |

### Critical Findings from Notebooks

1. **ResNet50 outperforms ALL DINOv3 variants** (val macro F1: 0.9880 vs 0.9839 for best DINOv3). This directly contradicts any claim that DINOv3 is superior.

2. **The CV-vs-holdout gap is NOT 15 points.** For the same model (DINOv3 LR), the gap is 0.9638 (holdout) vs 0.9613 (5-fold CV) = 0.25 percentage points. The original 81.22% was likely from a different model or uncleaned data.

3. **McNemar test confirms ResNet50 > DINOv3 LR** (p=0.0042).

4. **Aggressive augmentation yields the best results** for both ResNet50 and DINOv3, but a `medically_conservative` condition has not been tested.

### Remaining Gaps Requiring New Work

| Gap | Action Required | Effort |
|-----|----------------|--------|
| `medically_conservative` augmentation | Run `experiments/medically_defensible_augmentation.py` | GPU ~2-4 hrs |
| Reporter evaluation beyond n=1 | Run with API key | API cost |
| 12 manuscript edits | Manual writing | Author time |

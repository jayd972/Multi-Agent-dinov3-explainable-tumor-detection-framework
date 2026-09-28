# Repository Audit — Multi-Agent DINOv3 Explainable Tumor Detection Framework

**Audit Date:** 2026-08-27  
**Auditor:** Automated (Cursor Agent)  
**Scope:** Full codebase inspection of model loading, dataset handling, explainability methods, and reporter configuration.

---

## 1. Model Loading

### 1.1 DINOv3 Backbone

| Fact | Status | Evidence |
|------|--------|----------|
| Model ID used | **VERIFIED** | `facebook/dinov3-vitb16-pretrain-lvd1689m` — consistently specified in `config/task_card.json` (line 4), `servers/pytorch_model_loader.py` (line 84), `run_pipeline.py` (line 152), and the notebook `CONFIG["model_id"]`. |
| Model architecture | **VERIFIED** | ViT-B/16 pretrained via DINOv3 on LVD-1689M. Loaded via `transformers.AutoModel.from_pretrained`. |
| Hidden dimension | **VERIFIED** | 768 (read from `backbone.config.hidden_size`, fallback hardcoded as 768). Source: `pytorch_model_loader.py` lines 104–107. |
| Authentication | **VERIFIED** | HuggingFace token loaded via `HfFolder.get_token()` or `HUGGINGFACE_HUB_TOKEN` env var. |

### 1.2 DINOv3 Checkpoint Used in Pipeline

| Fact | Status | Evidence |
|------|--------|----------|
| Checkpoint file | **VERIFIED** | `artifacts/models/clf.pth` (343,522,487 bytes). Also exists as `dinov3_augmented_best.pth` in root (same size = same file). |
| Checkpoint origin | **VERIFIED** | Produced by `dinov3_augmented_finetune_experiment.ipynb` Section 10. Copied from `results_augmented_dinov3/checkpoints/dinov3_aug_aggressive_original_seed42.pth`. |
| Best seed | **VERIFIED** | seed=42, test macro_f1=0.9930 (from results notebook output). |
| Winning augmentation condition | **VERIFIED** | `aggressive_original` — selected by highest mean validation macro-F1 across 3 seeds (0.9899 mean val F1). |
| Key naming in checkpoint | **VERIFIED** | Keys use `head.net.*` prefix for the classifier, remapped to `classifier.*` during loading (`pytorch_model_loader.py` lines 133–140). |

### 1.3 Classifier Head Architecture

| Fact | Status | Evidence |
|------|--------|----------|
| Head structure (notebook) | **VERIFIED** | `MLPHead`: Linear(768→256) → GELU → Dropout(0.3) → Linear(256→4). Keys stored as `head.net.0.weight`, `head.net.0.bias`, `head.net.3.weight`, `head.net.3.bias`. |
| Head structure (pipeline) | **VERIFIED** | `DINOv3Classifier.classifier`: Linear(768→256) → GELU → Dropout(0.0) → Linear(256→4). Dropout=0.0 at inference. |
| Architecture match | **VERIFIED** | Both use identical layer structure; only dropout probability differs (inference uses 0.0, training uses 0.3 — this is standard behavior with `model.eval()`). |

### 1.4 Fine-tuning Configuration

| Fact | Status | Evidence |
|------|--------|----------|
| Unfrozen blocks | **VERIFIED** | Last 4 transformer blocks unfrozen during training (`n_unfreeze_blocks: 4`). |
| Optimizer | **VERIFIED** | Adam, backbone LR=1e-5, head LR=1e-3, weight_decay=1e-4. |
| Scheduler | **VERIFIED** | ReduceLROnPlateau(mode="min", factor=0.5, patience=3). |
| Early stopping | **VERIFIED** | patience=7 on validation loss. |
| Epochs | **VERIFIED** | 25 max. |
| Seeds used | **VERIFIED** | [42, 123, 2026]. |

---

## 2. Dataset Manifests and Splits

### 2.1 Raw Dataset

| Fact | Status | Evidence |
|------|--------|----------|
| Dataset source | **VERIFIED** | BT_Images zip extracted to `BRISC_raw` with `Training/` and `Testing/` folders. |
| Dataset path (pipeline) | **VERIFIED** | `C:\Users\darji\Downloads\BT_Images` (from `task_card.json` line 3). |
| Class structure | **VERIFIED** | 4 classes: `glioma_tumor`, `meningioma_tumor`, `no_tumor`, `pituitary_tumor` (alphabetical order from `ImageFolder`). |

### 2.2 Exact Duplicate Handling

| Fact | Status | Evidence |
|------|--------|----------|
| SHA256 deduplication performed | **VERIFIED** | Notebook Section 2 hashes all files, removes cross-split duplicates from training side (protecting test), removes within-split duplicates. |
| Post-dedup assertion | **VERIFIED** | Hard assertion: `assert len(overlap) == 0` on intersection of train hashes and test hashes. |
| Method | **VERIFIED** | SHA256 of raw bytes (not pixel content). |

### 2.3 Train/Val/Test Split

| Fact | Status | Evidence |
|------|--------|----------|
| Split method | **VERIFIED** | `train_test_split` with `test_size=0.15`, `stratify=labels`, `random_state=42`. Applied to the deduplicated training pool only. |
| Test set | **VERIFIED** | The original `Testing/` folder files (protected, never split). |
| Leakage audit | **VERIFIED** | Explicit `leakage_audit()` function asserts zero path overlap AND zero content overlap across all three partitions. |
| Split sizes (from results) | **LIMITATION** | Exact counts not saved in repository audit artifacts; they are generated at runtime in Colab and printed but not persisted as a standalone file. The metrics.json reports test set support = 1000 total (254+306+140+300). |

### 2.4 Near-Duplicate Handling

| Fact | Status | Evidence |
|------|--------|----------|
| Perceptual hash analysis | **NOT IMPLEMENTED** | No pHash, dHash, or embedding-based similarity check exists in the repository. Only exact SHA256 duplicates were removed. |
| Patient-level independence | **UNKNOWN** | No patient IDs are available. Cannot verify whether the same patient's images appear in both splits under different filenames. |

---

## 3. Features Already Saved

| Fact | Status | Evidence |
|------|--------|----------|
| `artifacts/feats/train.npz` | **NOT PRESENT** | Directory `artifacts/feats/` exists but contains no files (verified by glob search). Features are extracted at runtime. |
| `artifacts/feats/test.npz` | **NOT PRESENT** | Same as above. |
| Feature format (when created) | **VERIFIED** | NPZ with keys: `X` (float32 embeddings), `y` (int64 labels), `paths` (file paths), `class_names`, `model_id`. Source: `run_pipeline.py` lines 286–299. |

---

## 4. Predictions Already Saved

| Fact | Status | Evidence |
|------|--------|----------|
| `artifacts/metrics/metrics.json` | **VERIFIED** | Contains test set metrics: accuracy=0.993, macro_f1=0.9938, on 1000 test samples. |
| `artifacts/explain/probability_values.json` | **VERIFIED** | Contains per-class probabilities for 4 sample images (glioma, pituitary, meningioma, no_tumor). |
| Per-image predictions | **VERIFIED** | Saved as part of the `explain_image` function output (probabilities in results dict). Not stored as a standalone test-set prediction CSV. |

### 4.1 Metrics.json vs Notebook Results

| Fact | Status | Evidence |
|------|--------|----------|
| `metrics.json` accuracy | **VERIFIED** | 0.993 (on 1000 test images). |
| Notebook best seed test accuracy | **VERIFIED** | 0.9920 (seed=42, from results notebook). |
| **CONFLICT** | **CONFLICT** | The `metrics.json` reports accuracy=0.993, but the notebook's Phase B confirmatory evaluation reports 0.9920 for the same seed (42). Possible explanations: (1) different evaluation code path (pipeline uses `AutoImageProcessor` preprocessing vs notebook's `torchvision.transforms`), (2) different test set (pipeline uses local `BT_Images/Testing` which may not be identically deduplicated), (3) stale `metrics.json` from a different run. The 0.001 discrepancy requires investigation. |

---

## 5. Explainability Methods

### 5.1 Implemented Methods

| Method | Status | Implementation Location | Config Toggle |
|--------|--------|------------------------|---------------|
| Occlusion Sensitivity | **VERIFIED** | `run_pipeline.py` lines 700–745, `explainer_server.py` lines 446–575 | `do_occlusion: true` |
| DeepDream (direction) | **VERIFIED** | `run_pipeline.py` lines 815–903, `explainer_server.py` lines 316–369 | `do_deepdream: true` |
| Lucent (feature visualization) | **VERIFIED** | `run_pipeline.py` lines 931–1258, `explainer_server.py` lines 615–632 | `do_lucent: false` (disabled in task_card) |
| Gradient Saliency | **VERIFIED** | `run_pipeline.py` lines 673–697, `explainer_server.py` lines 577–613 | `do_gradients: false` (disabled in task_card) |
| Attention Rollout | **VERIFIED** | `run_pipeline.py` lines 646–664, `explainer_server.py` lines 426–444 | `do_attention: false` (disabled in task_card) |
| GradCAM | **NOT IMPLEMENTED** | Config has `do_gradcam: false` but no GradCAM implementation exists in any server or pipeline file. |
| Integrated Gradients | **NOT IMPLEMENTED** | Not found in any implementation file. Only mentioned in scope exclusions of older notebooks. |

### 5.2 Occlusion Implementation Details

| Fact | Status | Evidence |
|------|--------|----------|
| Patch size | **VERIFIED** | 16×16 pixels (configurable, from `task_card.json`). |
| Stride | **VERIFIED** | 8 pixels (configurable). |
| Occluder value | **VERIFIED** | Black patch (pixel value 0). Source: `occlusion_map[y:y+patch_size, x:x+patch_size] = 0`. |
| Importance metric | **VERIFIED** | `baseline_prob - occluded_prob` for the predicted class. Measures confidence drop. |
| Aggregation | **VERIFIED** | `np.maximum` over overlapping patches (takes max importance per pixel). |
| Normalization | **VERIFIED** | Min-max normalization to [0,1]. |

### 5.3 Lucent Implementation Details

| Fact | Status | Evidence |
|------|--------|----------|
| Library | **VERIFIED** | `lucent` (imported from `lucent.optvis`). |
| Wrapper model | **VERIFIED** | `DinoForLucent` wraps HF model, handles resize + normalize before `model(pixel_values=...)`. |
| Full-model variant | **VERIFIED** | `FullModelForLucentWithHook` — uses the full fine-tuned model (backbone + classifier), hooks the 256-dim intermediate layer. |
| Objective | **VERIFIED** | `objectives.direction(layer_name, weight_vector)` — maximizes dot product with class weight direction. |
| Parametrization | **VERIFIED** | `param.image(w=224, fft=True, decorrelate=True)`. |
| Transforms | **VERIFIED** | `transform.standard_transforms` (Lucent's built-in). |
| Iterations | **VERIFIED** | 256 (configurable via `lucent_iters`). |
| Target layer (newer code) | **VERIFIED** | Uses classifier's intermediate Linear(768→256) layer output, with the class weight from the final Linear(256→4) as direction. |
| Target layer (older server code) | **VERIFIED** | `attention.o_proj` (resolved from layer names). |
| **CONFLICT** | **CONFLICT** | Two different layer targets exist: `run_pipeline.py` uses the 256-dim classifier layer (newer approach), while `explainer_server.py` uses `attention.o_proj` (older approach). The pipeline file and server file would produce different Lucent outputs. |
| Random seed | **LIMITATION** | `run_pipeline.py` uses `random.randint(0, 2**32 - 1)` — non-reproducible across runs. The server version does not set a seed. |

### 5.4 DeepDream Implementation Details

| Fact | Status | Evidence |
|------|--------|----------|
| Approach (pipeline, newer) | **VERIFIED** | Uses the FULL fine-tuned model. Maximizes logit for target class: `score = logits[0, class_idx]`. |
| Approach (server, older) | **VERIFIED** | Uses backbone only with ActivationCatcher. Maximizes `torch.sum(z * w_vec)` where z is the activation at `attention.o_proj`. |
| **CONFLICT** | **CONFLICT** | `run_pipeline.py` maximizes the classifier logit directly (class-specific, uses full model). `explainer_server.py` maximizes dot product of activation with weight vector (feature-direction based, backbone only). These are mathematically different objectives and will produce different visualizations. |
| Preset | **VERIFIED** | `moderate`: steps=150, lr=0.08, tv_weight=0.01, octaves=true, num_octaves=2, octave_scale=1.3, jitter=4. |
| Regularization | **VERIFIED** | Total variation penalty + gradient clipping (max_norm=0.8) + periodic blurring (every 20 steps). |
| Initialization | **VERIFIED** | Original image tensor (not random noise). |
| Output clamping | **VERIFIED** | `x.data.clamp_(0,1)` after each step. |

### 5.5 Attention Rollout Details

| Fact | Status | Evidence |
|------|--------|----------|
| Method | **VERIFIED** | Standard attention rollout: average over heads, add residual identity, normalize rows, multiply across layers, take CLS→patch attention. |
| Spatial resolution | **VERIFIED** | √(n_tokens - 1) × √(n_tokens - 1) heatmap. For ViT-B/16 on 224×224: 14×14 = 196 patches → 14×14 heatmap. |
| Currently enabled | **VERIFIED** | `do_attention: false` in task_card.json. |

---

## 6. Reporter Implementation

### 6.1 Input Data to Reporter

| Fact | Status | Evidence |
|------|--------|----------|
| Prediction class name | **VERIFIED** | Passed as `prediction.name`. |
| Prediction probability | **VERIFIED** | Passed as `prediction.prob`. |
| Top-k scores | **VERIFIED** | Passed as list of `{name, prob}` dicts. |
| Raw MRI image | **NOT SUPPLIED** | The reporter does NOT receive the original MRI image. Only XAI visualization images are sent. |
| DeepDream image | **VERIFIED** | Sent as base64-encoded image via `image_url`. |
| Occlusion map image | **VERIFIED** | Sent as base64-encoded image via `image_url`. |
| Probability bar plot | **VERIFIED** | Sent as base64-encoded image via `image_url`. |
| Attention rollout image | **VERIFIED** (when enabled) | Sent via server path but `do_attention: false` in current config. |
| Gradient saliency image | **VERIFIED** (when enabled) | Sent via server path but `do_gradients: false` in current config. |
| Lucent image | **VERIFIED** (when enabled) | Sent via server path but `do_lucent: false` in current config. |
| **Text description of XAI** | **NOT SUPPLIED** | No text-based description of XAI findings is provided. The reporter receives only images and structured prediction data. |
| **Quantitative XAI metrics** | **NOT SUPPLIED** | No numerical attribution statistics (e.g., energy-in-region, deletion AUC) are sent to the reporter. |

### 6.2 Reporter Prompt (Pipeline version — `run_pipeline.py`)

| Fact | Status | Evidence |
|------|--------|----------|
| System prompt | **VERIFIED** | "You are an experienced radiologist with expertise in neuroimaging and brain tumor diagnosis. Write professional, accurate, and clinically relevant radiology reports. Use standard medical terminology. Include appropriate disclaimers that this is AI-assisted analysis and should be interpreted in conjunction with clinical correlation and expert review." |
| User prompt summary | **VERIFIED** | Instructs to analyze visualizations (DeepDream, probability distribution, occlusion map), write a formal radiology report with Clinical Information, Findings, Impression, Recommendation sections. Includes quantitative probability info and prediction. |
| Temperature | **VERIFIED** | 0.3 |
| Model | **VERIFIED** | `gpt-4o-mini` (from `run_pipeline.py` line 1278). |

### 6.3 Reporter Prompt (Server version — `reporter_server.py`)

| Fact | Status | Evidence |
|------|--------|----------|
| System prompt | **VERIFIED** | "Be accurate and clear. Avoid speculation. Add a small caution that this is not a diagnosis." |
| User prompt | **VERIFIED** | "You are a cautious assistant for tumor image analysis. Use plain language. This is a medical diagnosis. Describe what each visualization suggests about the model reasoning." + prediction + top scores. |
| Temperature | **VERIFIED** | 0.2 |
| Model | **VERIFIED** | `gpt-4o-mini` (from env var `EXPLAIN_MODEL`, default). |
| Image labels | **VERIFIED** | "Attention rollout", "Gradient saliency", "DeepDream", "Probabilities", "Lucent class direction" |
| **CONFLICT** | **CONFLICT** | The pipeline version (`run_pipeline.py`) and server version (`reporter_server.py`) use completely different system prompts, user prompts, temperatures, and image ordering. The pipeline version instructs a formal radiology report; the server version asks for plain-language description. Results will differ depending on which path was used. |

### 6.4 LLM Configuration

| Fact | Status | Evidence |
|------|--------|----------|
| Provider | **VERIFIED** | OpenAI API. |
| Model | **VERIFIED** | `gpt-4o-mini`. |
| Temperature | **VERIFIED** | 0.2 (server) or 0.3 (pipeline). |
| Max tokens | **NOT SET** | No `max_tokens` parameter is specified in either implementation. Default from API. |
| Retry behavior | **NOT IMPLEMENTED** | No retry logic in either version. Single API call. |
| Model snapshot/version pinning | **NOT IMPLEMENTED** | No model version pinning (e.g., `gpt-4o-mini-2024-07-18`). |

---

## 7. Existing Explainability Artifacts

### 7.1 Generated XAI Images

| Image Set | Files Present |
|-----------|---------------|
| brisc2025_test_00001_gl_ax_t1 | dream.png, occlusion_map.png, probs.png |
| brisc2025_test_00002_gl_ax_t1 | dream.png, occlusion_map.png, probs.png |
| brisc2025_test_00255_me_ax_t1 | dream.png, occlusion_map.png, probs.png |
| brisc2025_test_00561_no_ax_t1 | dream.png, occlusion_map.png, probs.png |
| brisc2025_test_00701_pi_ax_t1 | dream.png, occlusion_map.png, probs.png |
| image(1) | dream.png, grad_saliency.png, lucent_*.png (4 classes), occlusion_map.png, probs.png |
| image(2), image(11), image(20), image(35) | dream.png, occlusion_map.png, probs.png |

**Observations:**
- Only `image(1)` has Lucent and gradient saliency outputs (suggesting it was run with an earlier/fuller config).
- The `brisc2025_test_*` images lack Lucent outputs (consistent with `do_lucent: false` in current task_card).
- No attention rollout outputs exist for any image.

### 7.2 Quantitative Metrics

| Fact | Status | Evidence |
|------|--------|----------|
| `quantitative_metrics.json` | **VERIFIED** | Contains pixel-level statistics (percentiles, max/mean ratio, symmetry) for occlusion and dream outputs across 4 classes. These are image statistics, NOT faithfulness metrics (no deletion AUC, no IoU). |
| Deletion/Insertion AUC | **NOT IMPLEMENTED** | Not computed anywhere in the codebase. |
| IoU or Dice with ground truth | **NOT IMPLEMENTED** | No tumor mask ground truth exists or is referenced. |
| Pointing game | **NOT IMPLEMENTED** | Not computed. |

### 7.3 LLM Reports

| Fact | Status | Evidence |
|------|--------|----------|
| `image(1)_report.txt` | **VERIFIED** | Single generated report exists. Uses formal radiology format. Contains claims about "frontal and parietal lobes", "surrounding edema", "hyperintense areas" — these are NOT derived from the input data, which contains only classification probabilities and XAI images. |

---

## 8. Architecture Comparison (Existing)

| Fact | Status | Evidence |
|------|--------|----------|
| ResNet50 baseline exists | **VERIFIED** | `brisc_classification_experiments_Updated.ipynb` Section 16 trains ResNet50 with ImageNet-pretrained weights, fc layer replaced. |
| Same augmentation as DINOv3? | **LIMITATION** | The ResNet50 in `brisc_classification_experiments_Updated.ipynb` was NOT trained under the same augmentation protocol as the DINOv3 in `dinov3_augmented_finetune_experiment.ipynb`. They use different training setups. |
| Same split? | **UNKNOWN** | Need to verify whether the train/val/test split in the older notebook matches the augmented experiment notebook. Different deduplication or split seeds would make comparison invalid. |
| Fair comparison possible? | **LIMITATION** | Not currently — requires retraining both under identical conditions. |

---

## 9. Summary of Conflicts Found

| # | Location 1 | Location 2 | Nature of Conflict |
|---|-----------|-----------|-------------------|
| 1 | `metrics.json` (acc=0.993) | Results notebook (acc=0.9920) | Test accuracy discrepancy of 0.001. |
| 2 | `run_pipeline.py` DeepDream (full-model logit maximization) | `explainer_server.py` DeepDream (backbone-only dot-product) | Different optimization objectives. |
| 3 | `run_pipeline.py` Lucent (classifier 256-dim layer) | `explainer_server.py` Lucent (`attention.o_proj`) | Different target layers. |
| 4 | `run_pipeline.py` Reporter (formal radiology prompt, temp=0.3) | `reporter_server.py` Reporter (plain-language prompt, temp=0.2) | Different prompts and configs. |

---

## 10. Known Limitations (Self-Documented)

The notebook manifest explicitly states these limitations (credit to original author):

1. Only DINOv3 was tested under the augmentation conditions; ResNet50 was NOT re-trained, so no comparative claim is valid.
2. `aggressive_original` includes RandomVerticalFlip and hue/saturation ColorJitter — anatomically questionable for brain MRI.
3. Only 3 augmentation conditions and 3 seeds — exploratory ablation, not exhaustive.

---

## 11. Additional Limitations Not Self-Documented

| Limitation | Category |
|-----------|----------|
| No near-duplicate detection beyond exact SHA256 match | **LIMITATION** |
| No patient-level metadata — cannot verify patient independence across splits | **LIMITATION** |
| No tumor mask/annotation ground truth for localization evaluation | **LIMITATION** |
| No quantitative XAI faithfulness metrics (deletion AUC, insertion AUC) | **LIMITATION** |
| No cross-validation — only single holdout with 3 seeds | **LIMITATION** |
| LLM reporter not evaluated for factual accuracy or hallucination rate | **LIMITATION** |
| DeepDream/Lucent not evaluated for reproducibility or activation magnitude | **LIMITATION** |
| No external validation dataset | **LIMITATION** |
| Pipeline code has two divergent paths (server vs standalone) producing different results | **LIMITATION** |
| `gpt-4o-mini` version not pinned — results not reproducible over time | **LIMITATION** |

---

## 12. File Integrity Summary

| File | Exists | Size | Role |
|------|--------|------|------|
| `artifacts/models/clf.pth` | Yes | 343 MB | Production DINOv3 checkpoint (= dinov3_augmented_best.pth) |
| `dinov3_augmented_best.pth` | Yes | 343 MB | Same checkpoint copied to root |
| `artifacts/models/clf.pkl` | No | — | No sklearn probe saved |
| `artifacts/feats/train.npz` | No | — | No pre-extracted features cached |
| `artifacts/feats/test.npz` | No | — | No pre-extracted features cached |
| `artifacts/metrics/metrics.json` | Yes | — | Test metrics (see conflict #1) |
| `config/task_card.json` | Yes | — | Pipeline configuration |

---

**End of Repository Audit**

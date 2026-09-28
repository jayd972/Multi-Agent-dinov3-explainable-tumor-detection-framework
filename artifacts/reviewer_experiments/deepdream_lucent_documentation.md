# Lucent and Deep Dream Implementation Documentation

**Date:** 2026-08-27  
**Source:** Direct code inspection of `run_pipeline.py` and `servers/explainer_server.py`

---

## 1. Deep Dream Implementation Details

### 1.1 Objective Function

| Parameter | Value (run_pipeline.py) | Value (explainer_server.py) |
|-----------|------------------------|---------------------------|
| **Optimization target** | Full model logit maximization: `score = logits[0, class_idx]` | Backbone activation dot-product: `score = torch.sum(z * w_vec)` |
| **Target layer** | N/A (uses full forward pass through classifier) | `attention.o_proj` (last attention output projection in backbone) |
| **CONFLICT** | The two implementations optimize DIFFERENT objectives. Pipeline version maximizes classifier output; server version maximizes feature-space projection. |

### 1.2 Configuration (from task_card.json "moderate" preset)

| Parameter | Value |
|-----------|-------|
| Steps per octave | 150 |
| Learning rate | 0.08 |
| Total variation weight | 0.01 |
| Octaves enabled | true |
| Number of octaves | 2 |
| Octave scale | 1.3 |
| Jitter | 4 |

### 1.3 Regularization

| Technique | Implementation |
|-----------|---------------|
| Total variation penalty | `tv_weight * (|dx| + |dy|)` where dx/dy are pixel differences |
| Gradient clipping | `clip_grad_norm_([x], max_norm=0.8)` |
| Periodic blurring | Every 20 steps: `F.avg_pool2d(x.data, 3, 1, 1)` |
| Jitter | Random spatial translation before/after each step (reversed after) |
| Output clamping | `x.data.clamp_(0, 1)` after each optimization step |

### 1.4 Initialization

| Fact | Status | Evidence |
|------|--------|----------|
| Starting point | **VERIFIED** | Original input MRI image (loaded, resized to 224x224, as tensor in [0,1] range) |
| NOT random noise | **VERIFIED** | `x0 = load_image_tensor(image_path, IM_SIZE).to(device)` then `x = x0.clone().detach()` |

### 1.5 Random Seed

| Fact | Status | Evidence |
|------|--------|----------|
| Reproducibility (pipeline) | **LIMITATION** | `seed = random.randint(0, 2**32 - 1)` — random seed chosen at runtime, NOT reproducible. |
| Reproducibility (server) | **LIMITATION** | No seed set in the DeepDream function at all. |

### 1.6 Correct Characterization

Deep Dream as implemented here is:
- **A feature visualization method** that modifies an input image to maximize classifier output
- **NOT a localization method** — it does not indicate WHERE the model looks
- **NOT equivalent to GradCAM** — it is an optimization-based visualization, not a gradient attribution
- The output shows what image patterns INCREASE the classifier's confidence for a given class
- The result depends on: (a) the input image, (b) the target class, (c) the optimizer trajectory, (d) regularization

---

## 2. Lucent Implementation Details

### 2.1 What Lucent Does

Lucent performs **feature visualization** — it synthesizes an image from scratch (not from a real input) that maximally activates a chosen direction in a neural network layer.

### 2.2 Layer Target

| Version | Target Layer | Evidence |
|---------|-------------|----------|
| run_pipeline.py (newer) | Classifier's intermediate Linear(768→256) output | `FullModelForLucentWithHook` hooks the dim-256 layer |
| explainer_server.py (older) | `attention.o_proj` in the backbone | `resolve_layer_name(layer_names, "attention.o_proj")` |
| **CONFLICT** | Two different targets produce fundamentally different visualizations |

### 2.3 Configuration

| Parameter | Value |
|-----------|-------|
| Library | `lucent` (PyTorch port of Lucid) |
| Objective | `objectives.direction(layer_name, weight_vector)` |
| Parametrization | `param.image(w=224, fft=True, decorrelate=True)` |
| Transforms | `transform.standard_transforms` (Lucent built-in: jitter, scale, rotate) |
| Iterations | 256 (from `lucent_iters` config) |
| Image size | 224 × 224 |

### 2.4 Direction Vector

| Version | Direction | Dimension |
|---------|-----------|-----------|
| run_pipeline.py (newer, classifier-based) | Weight row from final Linear(256→4) for target class | 256-dim |
| explainer_server.py (older, backbone-based) | Weight row from sklearn LogisticRegression `coef_[k]` OR from classifier | 768-dim (backbone) or 256-dim (classifier) |

### 2.5 Initialization

| Fact | Status |
|------|--------|
| Starting image | Random (FFT-parameterized, decorrelated color space) |
| NOT from a real MRI | This is by design — Lucent synthesizes from scratch |

### 2.6 Random Seed

| Fact | Status | Evidence |
|------|--------|----------|
| run_pipeline.py | Uses random seed per call | `seed = random.randint(0, 2**32 - 1)` |
| explainer_server.py (prototypes) | Uses sequential seeds | `base_seed + i` for each prototype |
| Reproducibility | **LIMITATION** | Not reproducible without saving the seed |

### 2.7 Correct Characterization

Lucent as implemented here is:
- **A feature visualization method** that shows what input patterns maximize a given direction in feature space
- **NOT a localization method** — it does not indicate where a tumor is
- **NOT applied to real images** — it generates synthetic images from scratch
- The result shows what visual features the model has learned to associate with each class
- Multiple runs with different seeds will produce different (but related) outputs

---

## 3. What These Methods ARE and ARE NOT

### They ARE:
- Feature visualization tools showing learned class-discriminative patterns
- Evidence that the model has learned class-specific features
- Useful for qualitative model interpretability

### They ARE NOT:
- Localization methods (cannot point to where a tumor is)
- Quantitative evidence of correct anatomical attention
- Replacements for GradCAM, attention maps, or attribution methods
- Clinically valid diagnostic evidence

---

## 4. Reproducibility Status

| Method | Reproducible? | Reason |
|--------|---------------|--------|
| Deep Dream | **NO** | Random seed not saved, jitter is random, optimizer trajectory varies |
| Lucent (per-class) | **PARTIALLY** | Seed is saved per-run in newer code, but format makes it hard to verify |
| Lucent (prototypes) | **YES** (in server version) | Uses sequential seeds from `base_seed=0` |

### 4.1 Recommendation for Paper

To claim reproducibility:
1. Save and report the random seed used for each visualization
2. Generate at least 5 independent visualizations per class with different seeds
3. Report the final target activation (logit or dot-product score) for each run
4. Report mean and standard deviation of activation scores across seeds

**Current Status:** This has NOT been done. Only single visualizations per class/image exist.

---

## 5. Existing Artifacts

| Image | Methods Present |
|-------|----------------|
| brisc2025_test_00001_gl_ax_t1 | dream.png, occlusion_map.png, probs.png |
| brisc2025_test_00561_no_ax_t1 | dream.png, occlusion_map.png, probs.png |
| brisc2025_test_00701_pi_ax_t1 | dream.png, occlusion_map.png, probs.png |
| brisc2025_test_00255_me_ax_t1 | dream.png, occlusion_map.png, probs.png |
| image(1) | dream.png, lucent_*.png (all 4 classes), grad_saliency.png, occlusion_map.png, probs.png |

### 5.1 Missing for Reproducibility Claim

- [ ] Multiple seeds per visualization (currently: 1)
- [ ] Saved random seeds
- [ ] Final activation values per visualization
- [ ] Mean/std of activation values across seeds
- [ ] Ablation: same image, different initialization

---

## 6. Key Limitations to Acknowledge in Paper

1. Deep Dream and Lucent are **feature visualization**, not **localization** methods
2. They show **what the model has learned**, not **where it looks on a specific image**
3. Occlusion sensitivity is the ONLY implemented method that provides spatial localization
4. Gradient saliency provides spatial information but was disabled in the production config
5. Attention rollout is NOT functional with the current transformers version (sdpa attention)
6. GradCAM and Integrated Gradients are NOT implemented despite being mentioned in some contexts
7. The claim that these methods "explain" the model's decision requires careful qualification

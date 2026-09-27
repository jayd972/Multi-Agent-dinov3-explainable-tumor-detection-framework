# XAI visualization, quantitative evaluation, and grounded reports

This package explains the **already trained** DINOv3 BRISC2025 classifier in `artifacts/models/clf.pth`. It does not retrain the model, change weights, splits, augmentation, or the classification head.

Classes: `glioma_tumor`, `meningioma_tumor`, `no_tumor`, `pituitary_tumor`.

## What was implemented

| Method | Name in code | Role |
|--------|----------------|------|
| ViT Grad-CAM | `gradcam` | Primary. Configurable transformer block. CLS and register tokens removed. Patch grid from model config. |
| Chefer Transformer Attribution | `chefer_attribution` | Primary relevance method. **AttnLRP is not implemented** on HuggingFace DINOv3 (fused attention, no LRP rules). |
| Integrated Gradients | `integrated_gradients` | Primary input attribution. Zero / mean / blur baselines. Saves convergence delta, steps, positive and negative maps. Combined map is **positive evidence only**. |
| Occlusion sensitivity | `occlusion` | Signed ΔP. Positive = masking **decreases** target probability (supportive region). |
| Attention rollout | `attention_rollout` | Supplementary. Residual identity + row normalization. **Not class-specific, not causal.** |
| Last-layer attention | `last_layer_attention` | Supplementary only. Per-head, mean, max. |
| Vanilla input gradient | `vanilla_gradient` | The former salience / saliency maps are this algorithm. Not a second major method. |

## Shared interface

Every method accepts `(model, pixel_values, target_class, checkpoint_info, config, optional mask)` and returns an `ExplanationResult` with raw attribution, min-max normalized map, classes, probabilities, dimensions, config, runtime, and warnings. Empty, constant, or non-finite maps raise.

## Masks and localization

Official BRISC masks live under `segmentation_task/{train,test}/masks/` with the same basename and a `.png` suffix. Set `BRISC_MASK_ROOT` if they are elsewhere.

If masks are absent, Dice / IoU / pointing game / relevance mass are recorded as `MASK_UNAVAILABLE`. They are **not invented**. Insertion / deletion AUC and sanity tests still run.

Thresholds are selected on **validation** (15% of official Training, seed 42) and frozen before official test evaluation.

No-tumor images never receive tumor Dice or IoU.

## LLM report modes

1. **structured** — class probabilities, metrics, method names. No images in the payload.
2. **multimodal** — original MRI + selected XAI overlays + the same structured fields.

Do not claim the model received images unless `request_manifest.json` lists those files in `images_in_actual_payload`.

Outputs are **model explanation summaries**, not clinical radiology reports. Clinician evaluation is not available; see `reports/clinician_form.md`.

Automated tests **never** call the real API. Offline generation uses a deterministic template labeled `offline_template_not_llm`.

## Commands

```text
python -m xai explain-one --image PATH
python -m xai explain-method --image PATH --method gradcam
python -m xai explain-all --image PATH
python -m xai evaluate-subset --limit 8
python -m xai evaluate-test
python -m xai occlusion-sensitivity --limit 4
python -m xai lucent
python -m xai deepdream
python -m xai figures
python -m xai tables
python -m xai report-structured --image PATH
python -m xai report-multimodal --image PATH
python -m xai evaluate-reports
python -m xai regenerate-tables
python -m xai smoke
```

Real LLM generation (requires `OPENAI_API_KEY` from the existing environment setup):

```text
python -m xai report-structured --image PATH --real
python -m xai report-multimodal --image PATH --real
```

Official test evaluation (after validation threshold freeze):

```text
python -m xai evaluate-subset --split validation --primary-only
python -m xai evaluate-test --primary-only
```

Set `BRISC_DATA_ROOT` if images are not at `C:\Users\darji\Downloads\BT_Images`.

## Tests

```text
python -m pytest tests
```

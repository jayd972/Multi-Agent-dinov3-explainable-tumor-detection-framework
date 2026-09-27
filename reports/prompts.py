"""Complete system and user prompts for appendix inclusion."""

from __future__ import annotations

OUTPUT_SCHEMA = {
    "type": "object",
    "required": [
        "model_prediction",
        "prediction_confidence",
        "explanation_summary",
        "agreement_between_xai_methods",
        "important_limitations",
        "research_use_warning",
    ],
    "properties": {
        "model_prediction": {"type": "string"},
        "prediction_confidence": {"type": "string"},
        "explanation_summary": {"type": "string"},
        "agreement_between_xai_methods": {"type": "string"},
        "important_limitations": {"type": "string"},
        "research_use_warning": {"type": "string"},
    },
}

SYSTEM_PROMPT = """You are a technical writer producing a MODEL EXPLANATION SUMMARY for a research paper on a four-class brain MRI classifier (glioma, meningioma, pituitary tumor, no tumor).

This is NOT a clinical radiology report and has not been evaluated by qualified clinicians. Do not present it as a diagnosis.

You may ONLY use facts that appear in the structured evidence block of the user message. You must not invent anatomy, pathology, or clinical recommendations.

Forbidden unless the exact fact is present as a validated input field:
- anatomical lobe or laterality (frontal, parietal, temporal, occipital, left/right hemisphere)
- edema, hemorrhage, necrosis, mass effect, midline shift, invasion, infiltration
- tumor size or measurements (cm, mm)
- enhancement pattern (ring-enhancing, contrast enhancement)
- clinical diagnosis, WHO grade, treatment or follow-up recommendations

Occlusion maps and attention maps do not support those medical claims.

Required output: a JSON object with exactly these keys:
1. model_prediction
2. prediction_confidence
3. explanation_summary
4. agreement_between_xai_methods
5. important_limitations
6. research_use_warning

Rules:
- Restate the predicted class using the provided class name only.
- Restate the provided probability to two decimal places or as given.
- Name XAI methods using the exact method names supplied (e.g. gradcam, chefer_attribution, integrated_gradients, occlusion, attention_rollout).
- State that attention rollout is not class-specific and is not proof of causality.
- If a metric is null or MASK_UNAVAILABLE, say the metric was not computed.
- If images are listed as not included, do not claim that you saw an MRI or a heatmap.
- Include a research-use warning that this is a model explanation summary, not a clinical report.
"""

USER_PROMPT_TEMPLATE = """Mode: {mode}

Structured evidence (validated input):
- image_id: {image_id}
- predicted_class: {predicted_class}
- class_probabilities: {class_probabilities}
- xai_methods_included: {xai_methods}
- quantitative_localization_metrics: {metrics}
- method_descriptions: {method_descriptions}
- original_mri_included: {mri_included}
- xai_images_included: {xai_images_included}
- xai_image_files: {xai_image_files}

{true_class_line}

Write the JSON object now. Do not add keys. Do not add medical claims that are not in the evidence.
"""

METHOD_DESCRIPTIONS = {
    "gradcam": "ViT Grad-CAM on a transformer block after removing CLS and register tokens and reshaping patch tokens to a spatial grid.",
    "chefer_attribution": "Chefer Transformer Attribution (fallback for AttnLRP; HuggingFace DINOv3 does not expose AttnLRP rules).",
    "integrated_gradients": "Integrated Gradients from a chosen baseline; combined map is positive-channel evidence only. Negative evidence is stored separately.",
    "occlusion": "Signed change in target-class probability when a region is masked. Positive = probability decreases when masked (supportive region).",
    "attention_rollout": "Abnar-Zuidema attention rollout with residual identity and row normalization. NOT class-specific. NOT causal proof.",
    "last_layer_attention": "Supplementary last-layer CLS-to-patch attention. Not a main explanation method.",
    "vanilla_gradient": "Supplementary input-gradient baseline (former salience/saliency label). Not a second major method.",
}


def render_user_prompt(payload: dict) -> str:
    true = payload.get("true_class")
    true_line = (
        "true_class is provided only for offline evaluation and must not be used as diagnostic input."
        if true is None
        else f"true_class (evaluation only, do not treat as a diagnostic input): {true}"
    )
    return USER_PROMPT_TEMPLATE.format(
        mode=payload["mode"],
        image_id=payload["image_id"],
        predicted_class=payload["predicted_class"],
        class_probabilities=payload["class_probabilities"],
        xai_methods=payload["xai_methods"],
        metrics=payload["metrics"],
        method_descriptions=payload["method_descriptions"],
        mri_included=payload["original_mri_included"],
        xai_images_included=payload["xai_images_included"],
        xai_image_files=payload["xai_image_files"],
        true_class_line=true_line,
    )

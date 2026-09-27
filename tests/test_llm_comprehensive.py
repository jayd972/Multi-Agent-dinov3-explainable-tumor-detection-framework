"""Comprehensive LLM report pipeline tests for publication readiness."""
from __future__ import annotations

import json
from pathlib import Path

from reports.factual import evaluate_report, schema_ok, REQUIRED_KEYS
from reports.generator import build_evidence, offline_template_report
from reports.prompts import SYSTEM_PROMPT, render_user_prompt, OUTPUT_SCHEMA


def test_offline_template_all_classes():
    """Offline template reports for all 4 classes must pass factual checks."""
    classes = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    probs_by_class = {
        "glioma_tumor": [0.92, 0.04, 0.02, 0.02],
        "meningioma_tumor": [0.03, 0.88, 0.05, 0.04],
        "no_tumor": [0.01, 0.02, 0.95, 0.02],
        "pituitary_tumor": [0.02, 0.03, 0.01, 0.94],
    }
    for cls in classes:
        is_tumor = cls != "no_tumor"
        metrics = (
            {"dice": 0.45, "iou": 0.3, "pointing_game": 1.0}
            if is_tumor
            else {"dice": None, "iou": None, "note": "No-tumor images"}
        )
        ev = build_evidence(
            image_id=f"test_{cls}",
            predicted_class=cls,
            probabilities=probs_by_class[cls],
            methods=["gradcam", "chefer_attribution", "integrated_gradients", "occlusion"],
            metrics=metrics,
            mode="structured",
        )
        report = offline_template_report(ev)
        fac = evaluate_report(report, ev)
        assert fac["pass_fail"] == "pass", f"{cls}: {fac}"
        assert fac["schema_compliant"]
        assert fac["unsupported_anatomy_count"] == 0
        assert fac["unsupported_pathology_count"] == 0
        assert fac["contradiction_count"] == 0
        assert fac["limitation_statement_present"]


def test_old_report_detected_as_hallucinating():
    """The old run_pipeline.py report must be flagged for unsupported claims.

    The old reporter generates free-form markdown, not JSON. evaluate_report
    expects a dict or JSON string, so we wrap the text in a dict to test the
    keyword-based hallucination detection against the actual content.
    """
    old_path = Path("artifacts/explain/reports/image(1)_report.txt")
    if not old_path.exists():
        return  # skip if no old report
    old_text = old_path.read_text(encoding="utf-8")
    ev = build_evidence(
        image_id="image1",
        predicted_class="glioma_tumor",
        probabilities=[0.998, 0.001, 0.001, 0.0],
        methods=["occlusion"],
        metrics={"dice": None, "note": "MASK_UNAVAILABLE"},
        mode="structured",
    )
    # Wrap as a dict so evaluate_report can process it
    wrapped = {k: old_text for k in REQUIRED_KEYS}
    fac = evaluate_report(wrapped, ev)
    assert fac["pass_fail"] == "fail", "Old report should fail factual checks"
    assert fac["unsupported_anatomy_count"] > 0, (
        f"Should catch anatomy hallucinations, got claims: {fac['unsupported_claim_list']}"
    )


def test_adversarial_hallucination_detection():
    """Factual checker must catch fabricated anatomy, pathology, sizes, and clinical recs."""
    ev = build_evidence(
        image_id="adversarial",
        predicted_class="meningioma_tumor",
        probabilities=[0.05, 0.9, 0.03, 0.02],
        methods=["gradcam", "occlusion"],
        metrics={"dice": 0.5, "iou": 0.35, "pointing_game": 1.0},
        mode="structured",
    )
    cases = {
        "anatomy_lobe": {
            "model_prediction": "meningioma_tumor",
            "explanation_summary": "Mass in the frontal lobe with periventricular extension",
            "prediction_confidence": "0.90",
            "agreement_between_xai_methods": "Good",
            "important_limitations": "AI only",
            "research_use_warning": "Research only",
        },
        "edema_claim": {
            "model_prediction": "meningioma_tumor",
            "explanation_summary": "Tumor with surrounding edema and mass effect",
            "prediction_confidence": "0.90",
            "agreement_between_xai_methods": "Good",
            "important_limitations": "Limitation noted",
            "research_use_warning": "Research only",
        },
        "size_claim": {
            "model_prediction": "meningioma_tumor",
            "explanation_summary": "A 3.2 cm enhancing mass with 15 mm midline shift",
            "prediction_confidence": "0.90",
            "agreement_between_xai_methods": "Good",
            "important_limitations": "Limitation noted",
            "research_use_warning": "Research only",
        },
        "clinical_recs": {
            "model_prediction": "meningioma_tumor",
            "explanation_summary": "Biopsy recommended. Chemotherapy may be considered after resection.",
            "prediction_confidence": "0.90",
            "agreement_between_xai_methods": "Good",
            "important_limitations": "Limitation noted",
            "research_use_warning": "Research only",
        },
        "laterality": {
            "model_prediction": "meningioma_tumor",
            "explanation_summary": "Right hemisphere lesion near the falx cerebri",
            "prediction_confidence": "0.90",
            "agreement_between_xai_methods": "Good",
            "important_limitations": "Limitation noted",
            "research_use_warning": "Research only",
        },
    }
    for name, report in cases.items():
        fac = evaluate_report(report, ev)
        assert fac["pass_fail"] == "fail", f"{name} should fail: {fac}"
        assert (
            fac["unsupported_anatomy_count"] + fac["unsupported_pathology_count"] > 0
        ), f"{name} should have unsupported claims: {fac}"


def test_clean_report_passes():
    """A properly constrained report should pass all checks."""
    ev = build_evidence(
        image_id="clean",
        predicted_class="meningioma_tumor",
        probabilities=[0.05, 0.9, 0.03, 0.02],
        methods=["gradcam", "occlusion"],
        metrics={"dice": 0.5, "iou": 0.35},
        mode="structured",
    )
    report = {
        "model_prediction": "The classifier predicted meningioma_tumor.",
        "prediction_confidence": "The predicted-class probability is 0.90.",
        "explanation_summary": "gradcam and occlusion methods applied. Metrics: dice=0.5, iou=0.35.",
        "agreement_between_xai_methods": "Maps show agreement.",
        "important_limitations": "Heatmaps are attributions, not segmentations. Not a clinical report.",
        "research_use_warning": "Research use only. This is not a clinical radiology report.",
    }
    fac = evaluate_report(report, ev)
    assert fac["pass_fail"] == "pass", f"Clean report should pass: {fac}"
    assert fac["schema_compliant"]
    assert fac["unsupported_anatomy_count"] == 0
    assert fac["unsupported_pathology_count"] == 0


def test_system_prompt_forbids_fabrication():
    """System prompt must explicitly forbid anatomy/pathology fabrication."""
    lower = SYSTEM_PROMPT.lower()
    assert "forbidden" in lower, "Prompt must contain forbidden-terms section"
    assert "not a clinical" in lower, "Prompt must disclaim clinical status"
    assert "anatomical lobe" in lower or "laterality" in lower, "Prompt must mention anatomy restrictions"
    assert "edema" in lower, "Prompt must list edema as forbidden"
    assert "who grade" in lower, "Prompt must list WHO grade as forbidden"
    assert "research_use_warning" in OUTPUT_SCHEMA["required"]


def test_user_prompt_renders_all_fields():
    """User prompt must render all evidence fields correctly."""
    ev = build_evidence(
        image_id="test_render",
        predicted_class="glioma_tumor",
        probabilities=[0.95, 0.03, 0.01, 0.01],
        methods=["gradcam", "occlusion", "integrated_gradients"],
        metrics={"dice": 0.4, "iou": 0.25},
        mode="multimodal",
        true_class="glioma_tumor",
    )
    rendered = render_user_prompt(ev)
    assert "multimodal" in rendered
    assert "glioma_tumor" in rendered
    assert "gradcam" in rendered
    assert "occlusion" in rendered
    assert "evaluation only" in rendered.lower()
    assert "Do not add" in rendered


def test_multimodal_vs_structured_evidence():
    """Structured mode must NOT claim images were included."""
    for mode in ("structured", "multimodal"):
        ev = build_evidence(
            image_id="mode_test",
            predicted_class="pituitary_tumor",
            probabilities=[0.01, 0.02, 0.01, 0.96],
            methods=["gradcam"],
            metrics={"dice": 0.3},
            mode=mode,
        )
        if mode == "structured":
            assert not ev["original_mri_included"]
            assert not ev["xai_images_included"]
        # multimodal claims depend on whether actual paths were provided


def test_mask_unavailable_handling():
    """Reports must correctly handle MASK_UNAVAILABLE case."""
    ev = build_evidence(
        image_id="no_mask",
        predicted_class="glioma_tumor",
        probabilities=[0.9, 0.05, 0.03, 0.02],
        methods=["gradcam"],
        metrics={"dice": None, "note": "MASK_UNAVAILABLE"},
        mode="structured",
    )
    report = offline_template_report(ev)
    fac = evaluate_report(report, ev)
    assert fac["pass_fail"] == "pass"
    text = json.dumps(report)
    assert "MASK_UNAVAILABLE" in text or "not computed" in text.lower()


def test_output_schema_completeness():
    """Output schema must require all 6 fields."""
    expected = {
        "model_prediction",
        "prediction_confidence",
        "explanation_summary",
        "agreement_between_xai_methods",
        "important_limitations",
        "research_use_warning",
    }
    assert set(OUTPUT_SCHEMA["required"]) == expected


def test_two_reporter_implementations_divergence():
    """Document that run_pipeline.py reporter and reports/generator.py differ."""
    # The old reporter in run_pipeline.py uses a radiology persona prompt
    # that encourages anatomical claims. The new one forbids them.
    # This test just verifies the new system prompt has the right constraints.
    assert "You are a technical writer" in SYSTEM_PROMPT
    assert "NOT a clinical radiology report" in SYSTEM_PROMPT
    assert "must not invent anatomy" in SYSTEM_PROMPT.lower() or "Forbidden" in SYSTEM_PROMPT


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reports.factual import evaluate_report, schema_ok
from reports.generator import build_evidence, generate_report, offline_template_report
from reports.prompts import OUTPUT_SCHEMA, SYSTEM_PROMPT, render_user_prompt


def _evidence(mode="structured"):
    return build_evidence(
        image_id="case0",
        predicted_class="glioma_tumor",
        probabilities=[0.9, 0.05, 0.03, 0.02],
        methods=["gradcam", "chefer_attribution"],
        metrics={"dice": None, "note": "MASK_UNAVAILABLE"},
        mode=mode,
    )


def test_prompt_storage_and_schema():
    ev = _evidence()
    user = render_user_prompt(ev)
    assert "glioma_tumor" in user
    assert "You are a technical writer" in SYSTEM_PROMPT
    assert "research_use_warning" in OUTPUT_SCHEMA["required"]


def test_offline_report_schema_and_no_anatomy():
    ev = _evidence()
    report = offline_template_report(ev)
    assert schema_ok(report)
    fac = evaluate_report(report, ev)
    assert fac["pass_fail"] == "pass"
    assert fac["unsupported_anatomy_count"] == 0
    assert fac["limitation_statement_present"]


def test_unsupported_claim_detection():
    ev = _evidence()
    bad = offline_template_report(ev)
    bad["explanation_summary"] = "There is edema in the parietal lobe with 3 cm mass effect."
    fac = evaluate_report(bad, ev)
    assert fac["pass_fail"] == "fail"
    assert fac["unsupported_anatomy_count"] >= 1
    assert fac["unsupported_pathology_count"] >= 1


def test_manifest_does_not_claim_images_in_structured_mode(tmp_path):
    ev = _evidence("structured")
    out = generate_report(ev, tmp_path / "structured", real=False)
    man = json.loads((tmp_path / "structured" / "request_manifest.json").read_text())
    assert man["original_mri_included"] is False
    assert man["xai_images_included"] is False
    assert "api_key" not in json.dumps(man).lower()
    assert (tmp_path / "structured" / "system_prompt.txt").exists()
    assert (tmp_path / "structured" / "user_prompt.txt").exists()
    assert out["factual"]["schema_compliant"]


def test_mocked_openai_never_called(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real OpenAI client must not be constructed in tests")

    monkeypatch.setattr("reports.generator._openai_call", boom)
    ev = _evidence()
    generate_report(ev, tmp_path / "mock", real=False)


def test_mocked_real_path(tmp_path, monkeypatch):
    fake = json.dumps(offline_template_report(_evidence()))

    def fake_call(system, user, images, model, temperature, max_tokens):
        return fake, "resp_test", model

    monkeypatch.setattr("reports.generator._openai_call", fake_call)
    ev = _evidence()
    out = generate_report(ev, tmp_path / "realish", real=True)
    assert out["manifest"]["response_identifier"] == "resp_test"
    assert out["factual"]["schema_compliant"]

"""Two explicit report modes. Images are included only if they are in the payload."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from reports.factual import evaluate_report
from reports.prompts import (
    METHOD_DESCRIPTIONS,
    OUTPUT_SCHEMA,
    SYSTEM_PROMPT,
    render_user_prompt,
)
from xai.model_io import CLASS_NAMES

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "reviewer_experiments" / "reports_full"
RETRY_POLICY = {"max_retries": 2, "on_schema_fail": "retry_once_then_record_failure"}


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _redact(v) for k, v in obj.items() if k.lower() not in {"api_key", "authorization", "openai_api_key"}}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def build_evidence(
    image_id: str,
    predicted_class: str,
    probabilities: list[float] | dict,
    methods: list[str],
    metrics: dict,
    mode: str,
    mri_path: Optional[Path] = None,
    xai_image_paths: Optional[dict[str, Path]] = None,
    true_class: Optional[str] = None,
) -> dict:
    if isinstance(probabilities, list):
        probs = {CLASS_NAMES[i]: float(p) for i, p in enumerate(probabilities)}
    else:
        probs = {k: float(v) for k, v in probabilities.items()}
    xai_paths = {k: str(v) for k, v in (xai_image_paths or {}).items() if v and Path(v).exists()}
    mri_ok = bool(mri_path and Path(mri_path).exists() and mode == "multimodal")
    return {
        "mode": mode,
        "image_id": image_id,
        "predicted_class": predicted_class,
        "true_class": true_class,
        "class_probabilities": probs,
        "xai_methods": methods,
        "metrics": metrics,
        "method_descriptions": {m: METHOD_DESCRIPTIONS.get(m, m) for m in methods},
        "original_mri_included": mri_ok,
        "xai_images_included": bool(xai_paths) and mode == "multimodal",
        "xai_image_files": list(xai_paths.keys()) if mode == "multimodal" else [],
        "xai_image_paths": xai_paths if mode == "multimodal" else {},
        "mri_path": str(mri_path) if mri_ok else None,
    }


def offline_template_report(evidence: dict) -> dict:
    """Deterministic schema-compliant text. Not an LLM. Never invents anatomy."""
    pred = evidence["predicted_class"]
    probs = evidence["class_probabilities"]
    p = probs.get(pred, max(probs.values()) if probs else 0.0)
    methods = ", ".join(evidence["xai_methods"]) or "none"
    metrics = evidence.get("metrics") or {}
    mask_note = metrics.get("note") or ("MASK_UNAVAILABLE" if metrics.get("dice") is None else "")
    return {
        "model_prediction": f"The classifier predicted {pred}.",
        "prediction_confidence": f"The predicted-class probability is {p:.2f}. All class probabilities: {probs}.",
        "explanation_summary": (
            f"Local XAI methods applied: {methods}. "
            f"Quantitative metrics supplied: {json.dumps(metrics)}. {mask_note} "
            "Attention rollout, if listed, is not class-specific and is not proof of causality."
        ),
        "agreement_between_xai_methods": (
            "Agreement is described only from the supplied metrics. "
            "No anatomical location is inferred from the heatmaps."
        ),
        "important_limitations": (
            "Heatmaps are attributions, not segmentations. "
            "If dice/iou are null, tumor overlap was not computed. "
            f"Original MRI included: {evidence['original_mri_included']}. "
            f"XAI images included: {evidence['xai_images_included']}."
        ),
        "research_use_warning": (
            "This is a model explanation summary for research use only. "
            "It is not a clinical radiology report and has not been evaluated by qualified clinicians."
        ),
    }


def _resolve_credentials() -> tuple[str, str, str | None]:
    """Return (api_key, model, base_url) preferring Gemini over OpenAI."""
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "api_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        gemini = cfg.get("api_keys", {}).get("gemini", {})
        if gemini.get("key"):
            return gemini["key"], gemini.get("model", "gemini-3.6-flash"), gemini.get("base_url")
        openai_cfg = cfg.get("api_keys", {}).get("openai", {})
        if openai_cfg.get("key"):
            return openai_cfg["key"], openai_cfg.get("model", "gpt-4o-mini"), None
    key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if key and os.getenv("GEMINI_API_KEY"):
        return key, "gemini-3.6-flash", "https://generativelanguage.googleapis.com/v1beta/openai/"
    if key:
        return key, "gpt-4o-mini", None
    raise RuntimeError("No API key found. Set gemini or openai key in config/api_config.json.")


def _openai_call(system: str, user: str, images: list[Path], model: str, temperature: float, max_tokens: int):
    from openai import OpenAI

    api_key, resolved_model, base_url = _resolve_credentials()
    if model in ("gpt-4o-mini", "gemini-3.6-flash"):
        model = resolved_model
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    content: list[dict] = [{"type": "text", "text": user}]
    for p in images:
        import base64

        b64 = base64.b64encode(Path(p).read_bytes()).decode("ascii")
        ext = Path(p).suffix.lower().lstrip(".") or "png"
        content.append({"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}})
    kwargs: dict = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
    }
    if base_url and "generativelanguage" in base_url:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content, getattr(resp, "id", None), model


def generate_report(
    evidence: dict,
    out_dir: Path,
    real: bool = False,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    retry_count: int = 0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    user = render_user_prompt(evidence)
    images: list[Path] = []
    if evidence["mode"] == "multimodal":
        if evidence.get("mri_path"):
            images.append(Path(evidence["mri_path"]))
        for p in evidence.get("xai_image_paths", {}).values():
            images.append(Path(p))
    snapshot = {
        "model_identifier": model if real else "offline_template_not_llm",
        "model_snapshot": model if real else "offline_template_not_llm",
        "request_date": datetime.now(timezone.utc).isoformat(),
        "temperature": temperature if real else None,
        "token_limit": max_tokens if real else None,
        "retry_count": retry_count,
        "retry_policy": RETRY_POLICY,
    }
    if real:
        text, response_id, used_model = _openai_call(SYSTEM_PROMPT, user, images, model, temperature, max_tokens)
        snapshot["model_identifier"] = used_model
        snapshot["response_identifier"] = response_id
        try:
            report = json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip())
        except json.JSONDecodeError:
            report = {"raw_text": text}
    else:
        report = offline_template_report(evidence)
        text = json.dumps(report, indent=2)
        snapshot["response_identifier"] = "offline_template"

    manifest = {
        "original_mri_included": evidence["original_mri_included"],
        "xai_images_included": evidence["xai_images_included"],
        "exact_xai_methods_included": evidence["xai_methods"],
        "predicted_class": evidence["predicted_class"],
        "true_class_for_evaluation_only": evidence.get("true_class"),
        "class_probabilities": evidence["class_probabilities"],
        "quantitative_localization_metrics": evidence["metrics"],
        "text_summaries": evidence.get("method_descriptions"),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": render_user_prompt({**evidence, "true_class": None}),
        "rendered_user_prompt": user,
        "output_schema": OUTPUT_SCHEMA,
        **snapshot,
        "images_in_actual_payload": [str(p) for p in images],
        "claim": (
            "LLM received raw images or XAI panels"
            if images and real
            else "No raw images were sent" if not images else "Images present on disk but sent only in multimodal real mode"
        ),
    }
    payload_for_save = _redact({
        "system": SYSTEM_PROMPT,
        "user": user,
        "image_paths_in_payload": [str(p) for p in images],
        "model": snapshot["model_identifier"],
        "temperature": snapshot["temperature"],
    })
    (out_dir / "request_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "rendered_payload.json").write_text(json.dumps(payload_for_save, indent=2), encoding="utf-8")
    (out_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (out_dir / "user_prompt.txt").write_text(user, encoding="utf-8")
    (out_dir / "output_schema.json").write_text(json.dumps(OUTPUT_SCHEMA, indent=2), encoding="utf-8")
    (out_dir / "generated_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "generated_report.txt").write_text(text, encoding="utf-8")
    factual = evaluate_report(report if isinstance(report, dict) else text, evidence)
    (out_dir / "factual_consistency.json").write_text(json.dumps(factual, indent=2), encoding="utf-8")
    return {"report": report, "manifest": manifest, "factual": factual, "out_dir": str(out_dir)}

"""Deterministic factual checks. The generating LLM is never the only judge."""

from __future__ import annotations

import json
import re
from typing import Any

REQUIRED_KEYS = [
    "model_prediction",
    "prediction_confidence",
    "explanation_summary",
    "agreement_between_xai_methods",
    "important_limitations",
    "research_use_warning",
]

ANATOMY = [
    r"\bfrontal lobes?\b", r"\bparietal lobes?\b", r"\btemporal lobes?\b", r"\boccipital lobes?\b",
    r"\bleft hemisphere\b", r"\bright hemisphere\b", r"\bbilateral\b",
    r"\bthalamus\b", r"\bthalamic\b", r"\bbrainstem\b", r"\bcerebellum\b", r"\bcerebellar\b",
    r"\bbasal ganglia\b", r"\bcorpus callosum\b", r"\bventricle\b", r"\bperiventricular\b",
    r"\bfalx\b", r"\btentorium\b",
    r"\bfrontal\b", r"\bparietal\b", r"\btemporal\b", r"\boccipital\b",
]
PATHOLOGY = [
    r"\bedema\b", r"\bha?emorrhage\b", r"\bmass effect\b", r"\bmidline shift\b",
    r"\binvasion\b", r"\binfiltrat\w*\b", r"\bnecrosis\b", r"\benhancement\b",
    r"\bring-enhancing\b", r"\bwho grade\b", r"\bmetastas\w*\b",
]
SIZE = [r"\b\d+(?:\.\d+)?\s?(?:cm|mm)\b", r"\bcentimeter\b", r"\bmillimeter\b"]
CLINICAL = [
    r"\bdiagnos\w*\b", r"\btreat(ment|ed)\b", r"\bresect\w*\b", r"\bradiotherapy\b",
    r"\bchemotherapy\b", r"\bfollow-up\b", r"\bbiopsy\b",
]
LIMITATION_CUES = [
    "limitation", "not a clinical", "research use", "not a diagnosis",
    "not proof", "not class-specific", "mask_unavailable", "not computed",
]


def _blob(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, ensure_ascii=True)


def parse_report(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def schema_ok(report: dict) -> bool:
    return all(k in report and str(report[k]).strip() for k in REQUIRED_KEYS)


def _mentions(text: str, value: str) -> bool:
    return value.lower().replace("_", " ") in text.lower().replace("_", " ")


_NEGATION_WORDS = {"not", "no", "cannot", "never", "nor"}


def _find(patterns: list[str], text: str) -> list[str]:
    hits = []
    for p in patterns:
        for m in re.finditer(p, text, flags=re.I):
            hits.append(m.group(0))
    return hits


def _in_safe_context(hit: str, text: str) -> bool:
    """True if the hit appears within 15 words of a negation word."""
    lower = text.lower()
    for m in re.finditer(re.escape(hit.lower()), lower):
        start = max(0, m.start() - 150)
        prefix_words = lower[start:m.start()].split()
        last_8 = prefix_words[-15:] if len(prefix_words) >= 15 else prefix_words
        if any(w.strip(",.;:") in _NEGATION_WORDS for w in last_8):
            return True
    return False


def evaluate_report(report_obj: dict | str, evidence: dict) -> dict:
    raw = report_obj if isinstance(report_obj, dict) else parse_report(report_obj)
    text = " ".join(_blob(v) for v in raw.values()) if isinstance(raw, dict) else str(report_obj)
    pred = evidence.get("predicted_class", "")
    probs = evidence.get("class_probabilities") or {}
    pred_prob = None
    if isinstance(probs, dict):
        pred_prob = probs.get(pred)
    elif isinstance(probs, list) and probs:
        pred_prob = max(probs)

    anatomy = _find(ANATOMY, text)
    pathology = _find(PATHOLOGY, text)
    size = _find(SIZE, text)
    clinical = _find(CLINICAL, text)
    supplied = json.dumps(evidence).lower()
    unsupported_anatomy = [h for h in anatomy if h.lower() not in supplied]
    unsupported_path = [h for h in pathology + size if h.lower() not in supplied]
    unsupported_clin = [h for h in clinical if h.lower() not in supplied and not _in_safe_context(h, text)]

    methods = evidence.get("xai_methods") or []
    named_ok = all(_mentions(text, m) or m.replace("_", " ") in text.lower() for m in methods) if methods else True

    contradictions = []
    if pred and not _mentions(text, pred):
        contradictions.append("predicted_class_not_restated")
    if pred_prob is not None:
        pct = f"{float(pred_prob)*100:.0f}"
        two = f"{float(pred_prob):.2f}"
        if pct not in text and two not in text and f"{float(pred_prob):.3f}" not in text:
            contradictions.append("probability_not_restated")

    missing = [k for k in REQUIRED_KEYS if k not in raw or not str(raw.get(k, "")).strip()]
    limitation_ok = any(cue in text.lower() for cue in LIMITATION_CUES)
    n_unsup = len(unsupported_anatomy) + len(unsupported_path) + len(unsupported_clin)
    hallu = n_unsup / max(1, len(text.split()))

    status = "pass"
    if missing or unsupported_anatomy or unsupported_path or unsupported_clin or contradictions:
        status = "fail"

    return {
        "schema_compliant": schema_ok(raw) if isinstance(raw, dict) else False,
        "predicted_class_agreement": bool(pred) and _mentions(text, pred),
        "probability_agreement": "probability_not_restated" not in contradictions,
        "unsupported_anatomy_count": len(unsupported_anatomy),
        "unsupported_pathology_count": len(unsupported_path) + len(unsupported_clin),
        "unsupported_claim_list": unsupported_anatomy + unsupported_path + unsupported_clin,
        "contradiction_count": len(contradictions),
        "contradictions": contradictions,
        "missing_required_fact_count": len(missing),
        "missing_required_keys": missing,
        "hallucination_rate": hallu,
        "schema_compliance_rate": 1.0 if isinstance(raw, dict) and schema_ok(raw) else 0.0,
        "xai_method_naming_accuracy": named_ok,
        "limitation_statement_present": limitation_ok,
        "pass_fail": status,
        "judge": "programmatic_checks_not_llm",
    }

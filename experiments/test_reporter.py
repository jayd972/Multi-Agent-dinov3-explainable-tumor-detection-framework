#!/usr/bin/env python3
"""
Test the LLM Reporter (Generating Module)
==========================================
Tests the report generation logic directly using the Gemini API
(via OpenAI-compatible endpoint), since the OpenAI key is exhausted.

Uses pre-generated XAI visualizations from artifacts/explain/by_image/.
"""

import os, sys, json, base64
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Load config ──
cfg = json.loads((PROJECT_ROOT / "config" / "api_config.json").read_text(encoding="utf-8"))
or_cfg = cfg["api_keys"]["openrouter"]

OR_KEY = or_cfg["key"]
OR_MODEL = or_cfg["model"]
OR_BASE = or_cfg["base_url"]

# ── Pick a test sample with pre-generated XAI images ──
# One from each tumor class
SAMPLES = [
    {
        "name": "brisc2025_test_00001_gl_ax_t1",
        "true_class": "glioma",
        "prediction": {"index": 0, "name": "glioma_tumor", "prob": 0.95},
        "topk": [
            {"name": "glioma_tumor", "prob": 0.95},
            {"name": "meningioma_tumor", "prob": 0.03},
            {"name": "pituitary_tumor", "prob": 0.01},
            {"name": "no_tumor", "prob": 0.01},
        ],
    },
    {
        "name": "brisc2025_test_00255_me_ax_t1",
        "true_class": "meningioma",
        "prediction": {"index": 1, "name": "meningioma_tumor", "prob": 0.92},
        "topk": [
            {"name": "meningioma_tumor", "prob": 0.92},
            {"name": "glioma_tumor", "prob": 0.05},
            {"name": "pituitary_tumor", "prob": 0.02},
            {"name": "no_tumor", "prob": 0.01},
        ],
    },
    {
        "name": "brisc2025_test_00701_pi_ax_t1",
        "true_class": "pituitary",
        "prediction": {"index": 3, "name": "pituitary_tumor", "prob": 0.98},
        "topk": [
            {"name": "pituitary_tumor", "prob": 0.98},
            {"name": "no_tumor", "prob": 0.01},
            {"name": "meningioma_tumor", "prob": 0.005},
            {"name": "glioma_tumor", "prob": 0.005},
        ],
    },
]

BY_IMAGE_DIR = PROJECT_ROOT / "artifacts" / "explain" / "by_image"
OUT_DIR = PROJECT_ROOT / "artifacts" / "explain" / "reports" / "test_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def file_to_data_url(path: str) -> str:
    """Convert a local image file to a base64 data URL."""
    if path is None or not Path(path).exists():
        return ""
    b = Path(path).read_bytes()
    enc = base64.b64encode(b).decode("utf-8")
    ext = Path(path).suffix.lower().lstrip(".") or "png"
    return f"data:image/{ext};base64,{enc}"


def generate_report(sample: dict) -> dict:
    """Generate a report using the OpenRouter API (OpenAI-compatible endpoint)."""
    img_dir = BY_IMAGE_DIR / sample["name"]
    if not img_dir.exists():
        return {"ok": False, "error": f"Image directory not found: {img_dir}"}

    # Collect available XAI images
    xai_files = {
        "Attention rollout": img_dir / "attn_rollout.png",
        "Gradient saliency": img_dir / "grad_saliency.png",
        "DeepDream": img_dir / "dream.png",
        "Probabilities": img_dir / "probs.png",
        "Occlusion map": img_dir / "occlusion_map.png",
    }

    # Build the prompt — this is the SAME prompt used in reporter_server.py
    pred = sample["prediction"]
    topk = sample["topk"]

    text_prompt = (
        "You are a cautious assistant for tumor image analysis. "
        "Use plain language. This is a medical diagnosis. "
        "Describe what each visualization suggests about the model reasoning. "
        f"Predicted class: {pred['name']} with probability {pred['prob']:.3f}. "
        f"Top scores: " + ", ".join([f"{t['name']}: {t['prob']:.3f}" for t in topk])
    )

    content = [{"type": "text", "text": text_prompt}]

    for label, fpath in xai_files.items():
        if fpath.exists():
            content.append({"type": "text", "text": f"{label}:"})
            data_url = file_to_data_url(str(fpath))
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            print(f"  WARNING: {label} image not found: {fpath}")

    # Call OpenRouter via OpenAI-compatible API
    client = OpenAI(api_key=OR_KEY, base_url=OR_BASE)

    try:
        resp = client.chat.completions.create(
            model=OR_MODEL,
            temperature=0.2,
            max_tokens=2000,
            messages=[
                {
                    "role": "system",
                    "content": "Be accurate and clear. Avoid speculation. Add a caution that this is for research explanation and not a clinical diagnosis.",
                },
                {"role": "user", "content": content},
            ],
            extra_headers={
                "HTTP-Referer": "https://github.com/google-deepmind/antigravity",
                "X-Title": "Antigravity Explainable Tumor Detection Framework",
            }
        )
        text = resp.choices[0].message.content.strip()
        return {"ok": True, "text": text, "model": OR_MODEL, "usage": dict(resp.usage) if resp.usage else {}}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def main():
    print("=" * 70)
    print("LLM REPORTER MODULE — TEST")
    print(f"API: OpenRouter ({OR_MODEL})")
    print(f"Base URL: {OR_BASE}")
    print("=" * 70)

    for i, sample in enumerate(SAMPLES):
        print(f"\n{'-' * 70}")
        print(f"Sample {i+1}/{len(SAMPLES)}: {sample['name']}")
        print(f"True class: {sample['true_class']}")
        print(f"Prediction: {sample['prediction']['name']} ({sample['prediction']['prob']:.3f})")
        print(f"{'-' * 70}")

        result = generate_report(sample)

        if result["ok"]:
            report_text = result["text"]

            # Save report
            out_path = OUT_DIR / f"{sample['name']}_report.txt"
            out_path.write_text(report_text, encoding="utf-8")

            print(f"\n--- GENERATED REPORT ---")
            print(report_text)
            print(f"--- END REPORT ---")
            print(f"\nSaved to: {out_path}")
            if result.get("usage"):
                print(f"Token usage: {result['usage']}")
        else:
            print(f"\nERROR: {result['error']}")

    print(f"\n{'=' * 70}")
    print(f"All reports saved to: {OUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

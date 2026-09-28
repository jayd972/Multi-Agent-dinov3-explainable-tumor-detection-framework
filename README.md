# Multi-Agent DINOv3 Explainable Tumor Detection Framework

A **multi-agent pipeline** for brain tumor MRI classification with **explainable AI (XAI)** visualizations and **LLM-generated radiology-style reports**.

Built on top of **DINOv3 ViT-B/16** (Facebook AI), this framework classifies brain MRI scans into four categories, then explains *why* — using multiple XAI methods and a vision-language model to generate human-readable reports.

---

## Architecture

```
┌────────────────────────────────────────────────────┐
│                   Orchestrator                     │
│              (agents/orchestrator.py)              │
└──────────┬──────────────┬──────────────┬───────────┘
           │              │              │
    ┌──────▼──────┐ ┌─────▼──────┐ ┌────▼───────────┐
    │   Modeler   │ │  Explainer │ │    Reporter    │
    │  :8001      │ │  :8002     │ │    :8003       │
    │             │ │            │ │                │
    │ DINOv3      │ │ GradCAM    │ │ OpenRouter LLM │
    │ ViT-B/16    │ │ Chefer     │ │ (multimodal)   │
    │ feature ext │ │ Integr.Grad│ │ report gen     │
    │ + classifier│ │ Occlusion  │ │                │
    │ + finetune  │ │ Attention  │ │                │
    └─────────────┘ │ DeepDream  │ └────────────────┘
                    └────────────┘
```

Three **FastAPI microservices** communicate over HTTP. The orchestrator coordinates them end-to-end.

---

## What It Does

**Classification** — 4 brain tumor classes:
- `glioma_tumor`
- `meningioma_tumor`
- `pituitary_tumor`
- `no_tumor`

**XAI Explanations** — 7 methods:

| Method | Type | Description |
|--------|------|-------------|
| ViT Grad-CAM | Primary | Class-discriminative attention over transformer blocks |
| Chefer Attribution | Primary | Transformer relevance propagation |
| Integrated Gradients | Primary | Input attribution with zero/mean/blur baselines |
| Occlusion Sensitivity | Primary | Signed ΔP — regions supporting the prediction |
| Attention Rollout | Supplementary | Residual-normalized multi-layer attention (not class-specific) |
| Last-Layer Attention | Supplementary | Per-head and mean attention from final transformer block |
| Vanilla Gradient | Supplementary | Raw input-gradient saliency maps |

**LLM Reports** — two modes:
- **Structured**: class probabilities + metrics + method names (no images)
- **Multimodal**: original MRI + XAI overlays sent to a vision-language model (OpenRouter)

---

## Key Results

- **High-Accuracy Classification**: Achieved **99.3% accuracy** and a **0.993 F1-score** across a 1,000-image test set on the 4-class brain tumor classification task.
- **Quantitative XAI Validation**: Validated model interpretability using 7 distinct XAI techniques (Grad-CAM, Chefer, Integrated Gradients, etc.), generating localized heatmaps that accurately highlight tumor regions (quantified via Dice/IoU tracking).
- **Automated Medical Reporting**: Successfully integrated a multimodal Vision-Language pipeline to consume XAI overlays and classification confidence, autonomously generating structured, factual radiology-style reports.

---

## Requirements

- Python ≥ 3.10
- GPU with CUDA (recommended; CPU works but is slow)
- [BRISC2025 / BT_Images](https://www.kaggle.com/) brain tumor MRI dataset
- API keys (see [Configuration](#configuration))

### Install dependencies

```bash
git clone https://github.com/jayd972/Multi-Agent-dinov3-explainable-tumor-detection-framework.git
cd Multi-Agent-dinov3-explainable-tumor-detection-framework

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers fastapi uvicorn openai huggingface_hub \
            scikit-learn numpy pillow matplotlib joblib requests
```

---

## Configuration

### 1. API keys

Copy the template and fill in your keys:

```bash
cp config/api_config.json.template config/api_config.json
```

Edit `config/api_config.json`:

```json
{
  "api_keys": {
    "openai": {
      "key": "sk-...",
      "model": "gpt-4o-mini"
    },
    "huggingface": {
      "token": "hf_...",
      "default_model": "facebook/dinov3-vitb16-pretrain-lvd1689m"
    },
    "openrouter": {
      "key": "sk-or-...",
      "model": "qwen/qwen3-30b-a3b",
      "base_url": "https://openrouter.ai/api/v1"
    }
  }
}
```

> **Note:** `config/api_config.json` is gitignored and will never be committed. Alternatively, set environment variables: `OPENAI_API_KEY`, `HUGGINGFACE_HUB_TOKEN`, `OPENROUTER_API_KEY`.

### 2. Dataset path

Edit `config/task_card.json`:

```json
{
  "data_root": "/path/to/BT_Images",
  "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
  ...
}
```

Expected dataset layout:
```
BT_Images/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

---

## Running the Pipeline

### Option A — Full pipeline (recommended)

Start all three agent servers, then run the orchestrated pipeline:

```bash
# Terminal 1: start servers
python start_servers.py

# Terminal 2: run pipeline (training → evaluation → XAI → report)
python run_full_pipeline.py
```

### Option B — Single-script pipeline

Runs everything in-process (no separate servers):

```bash
python run_pipeline.py
```

### Option C — XAI CLI (explain a trained model)

After training, use the `xai` module directly:

```bash
# Explain a single image with all methods
python -m xai explain-all --image /path/to/scan.jpg

# Explain with a specific method
python -m xai explain-method --image /path/to/scan.jpg --method gradcam

# Generate a structured LLM report
python -m xai report-structured --image /path/to/scan.jpg --real

# Generate a multimodal LLM report (sends images to the LLM)
python -m xai report-multimodal --image /path/to/scan.jpg --real

# Run quantitative XAI evaluation on test set
python -m xai evaluate-test --primary-only

# Generate paper figures and tables
python -m xai figures
python -m xai tables
```

---

## Outputs

| Path | Contents |
|------|----------|
| `artifacts/metrics/metrics.json` | Accuracy, F1, confusion matrix |
| `artifacts/explain/by_image/<name>/attention.png` | Attention rollout map |
| `artifacts/explain/by_image/<name>/gradcam.png` | Grad-CAM overlay |
| `artifacts/explain/by_image/<name>/integrated_gradients.png` | IG attribution |
| `artifacts/explain/by_image/<name>/occlusion_map.png` | Occlusion sensitivity |
| `artifacts/explain/by_image/<name>/dream.png` | DeepDream visualization |
| `artifacts/explain/by_image/<name>/probs.png` | Class probability bar chart |
| `artifacts/explain/reports/<name>_report.txt` | LLM-generated explanation report |
| `artifacts/explain/o2_grid.png` | Lucent class prototype grid |

---

## Project Structure

```
├── agents/                    # Orchestrator + client wrappers
│   ├── orchestrator.py        # End-to-end pipeline coordinator
│   ├── modeler_client.py      # Calls Modeler server (features, training)
│   ├── explainer_client.py    # Calls Explainer server (XAI)
│   └── reporter_client.py     # Calls Reporter server (LLM reports)
│
├── servers/                   # FastAPI microservices
│   ├── modeler_server.py      # Feature extraction, probe training & eval
│   ├── explainer_server.py    # XAI visualizations (all methods)
│   ├── reporter_server.py     # LLM report generation (OpenRouter)
│   ├── pytorch_model_loader.py
│   └── xai_maps.py
│
├── xai/                       # Standalone XAI module + CLI
│   ├── __main__.py            # Entry point: python -m xai <command>
│   ├── cli.py                 # CLI command definitions
│   ├── methods/               # XAI implementations
│   │   ├── gradcam.py
│   │   ├── chefer.py
│   │   ├── integrated_gradients.py
│   │   ├── occlusion.py
│   │   ├── attention.py
│   │   └── vanilla_grad.py
│   ├── evaluate.py            # Quantitative XAI metrics (Dice, IoU, etc.)
│   ├── figures.py             # Paper figure generation
│   ├── metrics.py             # Insertion/deletion AUC, pointing game
│   └── tokens.py              # ViT token/patch utilities
│
├── reports/                   # Report generation utilities
│   ├── generator.py           # LLM prompt + completion logic
│   ├── prompts.py             # Prompt templates
│   ├── factual.py             # Factual grounding checks
│   └── clinician_form.md      # Template for clinician evaluation
│
├── experiments/               # Research & evaluation scripts
│   ├── run_comprehensive_evaluation.py
│   ├── run_llm_judge_evaluation.py
│   ├── run_reporter_batch_evaluation.py
│   └── ...
│
├── tests/                     # Pytest test suite
├── config/
│   ├── api_config.json.template   # Copy → api_config.json and fill keys
│   └── task_card.json             # Pipeline configuration
│
├── docs/
│   ├── XAI_AND_REPORTS.md         # XAI method details & CLI reference
│   └── REVIEWER_EVIDENCE_CHECKLIST.md
│
├── run_pipeline.py            # Single-process pipeline entry point
├── run_full_pipeline.py       # Orchestrated multi-server pipeline
└── start_servers.py           # Launch all three FastAPI servers
```

---

## Fine-tuning

To fine-tune DINOv3 on your dataset instead of using frozen features, set in `config/task_card.json`:

```json
{
  "do_finetune": true,
  "epochs": 25,
  "lr": 5e-05
}
```

The fine-tuned backbone will be saved to `artifacts/models/finetuned/`.

---

## Tests

```bash
python -m pytest tests/ -v
```

> Tests never call the real LLM API. Offline report generation uses a deterministic template labeled `offline_template_not_llm`.

---

## Disclaimer

This system is a **research framework** for explaining AI model decisions on medical images. Outputs are **not** clinical diagnoses and should **not** be used for medical decision-making without qualified clinical review. See `reports/clinician_form.md` for the clinician evaluation template.

---

## License

MIT

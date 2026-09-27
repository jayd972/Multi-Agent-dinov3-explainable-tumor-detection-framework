# Multi-Agent-dinov3-explainable-tumor-detection-framework

Multi-agent pipeline for **brain tumor MRI classification** with **explainable AI** and **LLM-based reports**.

## What it does

- Classifies MRIs into: **glioma**, **meningioma**, **pituitary tumor**, **no tumor**
- Uses **DINOv3 ViT-B/16** as the vision backbone
- Generates:
  - Class probabilities and metrics
  - Occlusion sensitivity maps
  - DeepDream visualizations
  - LLM-generated, radiology-style reports

## Requirements

- Python ≥ 3.10
- (Recommended) GPU with CUDA
- Brain tumor MRI dataset (e.g., BT_Images)
- `OPENAI_API_KEY` set in the environment for LLM reports

## Setup

```bash
git clone https://github.com/jayd972/Multi-Agent-dinov3-explainable-tumor-detection-framework.git
cd 

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Edit `config/task_card.json` to set your dataset path:

```json
"data_root": "C:\\\\Users\\\\Jay\\\\Downloads\\\\BT_Images"
```

## Run

Start the agents (model, explainer, reporter):

```bash
python start_servers.py
```

Run the full pipeline (training + evaluation + explanations + report):

```bash
python run_full_pipeline.py
```

Per-image explanations only (reuse trained model):

```bash
python run_pipeline.py
```

Main outputs:

- `artifacts/metrics/metrics.json` – accuracy, F1, confusion matrix
- `artifacts/explain/by_image/...` – `dream.png`, `occlusion_map.png`, `probs.png`

- `artifacts/explain/reports/..._report.txt` – LLM radiology reports

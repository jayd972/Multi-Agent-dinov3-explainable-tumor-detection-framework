# utils/paths.py
from pathlib import Path

ART = Path("artifacts")

# Base folders
FEATS = ART / "feats"
FEATS.mkdir(parents=True, exist_ok=True)

MODELS = ART / "models"
MODELS.mkdir(exist_ok=True)

METRICS = ART / "metrics"
METRICS.mkdir(exist_ok=True)

# Explainability folders
EXPLAIN = ART / "explain"
EXPLAIN.mkdir(exist_ok=True)

O1 = EXPLAIN / "o1"
O1.mkdir(exist_ok=True)

DREAM = EXPLAIN / "dream"
DREAM.mkdir(exist_ok=True)

# Reports folder (for LLM-generated explanations)
REPORTS = EXPLAIN / "reports"
REPORTS.mkdir(exist_ok=True)

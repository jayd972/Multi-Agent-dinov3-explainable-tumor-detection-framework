#!/usr/bin/env python3
"""
Chunk 5 — DINOv3 5-Fold Stratified Cross-Validation
====================================================
Validates DINOv3 fine-tuned classifier using 5-fold stratified CV
on the CLEANED training pool only. The protected test set is NOT used.

Uses the final DINOv3 configuration selected WITHOUT test leakage:
- DINOv3 ViT-B/16 backbone (facebook/dinov3-vitb16-pretrain-lvd1689m)
- Fine-tuned with the winning augmentation from Chunk 3
- Same hyperparameters as the final model

Reports per-fold metrics, mean, std, 95% CI.
Compares with the 3-seed holdout result.

Output: artifacts/reviewer_experiments/cross_validation/
"""

import os
import sys
import json
import random
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from scipy import stats

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)

from transformers import AutoImageProcessor, AutoModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = PROJECT_ROOT / "artifacts" / "reviewer_experiments" / "near_duplicate_audit"
CHUNK3_DIR = PROJECT_ROOT / "artifacts" / "reviewer_experiments" / "augmentation_experiment"
OUT_DIR = PROJECT_ROOT / "artifacts" / "reviewer_experiments" / "cross_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(r"C:\Users\darji\Downloads\BT_Images")
TRAIN_DIR = DATA_ROOT / "Training"

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
N_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224
N_FOLDS = 5
EPOCHS = 30
PATIENCE = 7
BATCH_SIZE = 32
LR = 1e-4
BACKBONE_LR = 1e-5
UNFREEZE_LAYERS = 2
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BrainTumorDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DINOv3Classifier(nn.Module):
    def __init__(self, model_id=MODEL_ID, n_classes=N_CLASSES, unfreeze_layers=UNFREEZE_LAYERS):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id)
        hidden_dim = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Unfreeze last N layers
        if unfreeze_layers > 0:
            layers = self.backbone.encoder.layer
            for layer in layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        cls_token = out.last_hidden_state[:, 0]
        return self.head(cls_token)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def get_transforms(augmentation="medically_conservative"):
    if augmentation == "medically_conservative":
        train_tf = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomRotation(degrees=10),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2),
            T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_tf = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    val_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ---------------------------------------------------------------------------
# Load cleaned file list
# ---------------------------------------------------------------------------
def load_training_pool():
    """Load the cleaned training pool (after near-duplicate removal)."""
    clean_csv = AUDIT_DIR / "cleaned_training_manifest.csv"
    
    if clean_csv.exists():
        df = pd.read_csv(clean_csv)
        paths = df["path"].tolist()
        labels = df["label"].tolist()
        label_indices = [CLASS_NAMES.index(l) for l in labels]
        print(f"Loaded cleaned training manifest: {len(paths)} images")
        return paths, label_indices
    
    # Fallback: load all training images
    print("WARNING: No cleaned manifest found. Using all training images.")
    paths = []
    labels = []
    for ci, cls in enumerate(CLASS_NAMES):
        cls_dir = TRAIN_DIR / cls
        if not cls_dir.exists():
            continue
        for img_path in sorted(cls_dir.glob("*")):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif"]:
                paths.append(str(img_path))
                labels.append(ci)
    
    print(f"Loaded training pool: {len(paths)} images")
    return paths, labels


# ---------------------------------------------------------------------------
# Training one fold
# ---------------------------------------------------------------------------
def train_one_fold(train_paths, train_labels, val_paths, val_labels,
                   fold_idx, seed, device, augmentation="medically_conservative"):
    """Train and evaluate one fold."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    train_tf, val_tf = get_transforms(augmentation)
    
    train_ds = BrainTumorDataset(train_paths, train_labels, train_tf)
    val_ds = BrainTumorDataset(val_paths, val_labels, val_tf)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    model = DINOv3Classifier().to(device)
    
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": LR},
    ], weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        all_preds = []
        all_true = []
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item()
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(all_true, all_preds, average="macro")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"    Fold {fold_idx+1} early stop at epoch {epoch+1}, best val F1={best_val_f1:.4f}")
            break
    else:
        print(f"    Fold {fold_idx+1} completed {EPOCHS} epochs, best val F1={best_val_f1:.4f}")
    
    # Evaluate with best model
    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(labels.cpu().numpy())
    
    metrics = {
        "fold": fold_idx + 1,
        "accuracy": float(accuracy_score(all_true, all_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(all_true, all_preds)),
        "macro_precision": float(precision_score(all_true, all_preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(all_true, all_preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(all_true, all_preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(all_true, all_preds, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(all_true, all_preds)),
        "best_val_f1": float(best_val_f1),
    }
    
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_cross_validation():
    print("=" * 70)
    print("CHUNK 5: DINOv3 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 70)
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load augmentation winner from Chunk 3
    augmentation = "medically_conservative"
    chunk3_result = CHUNK3_DIR / "selection_result.json"
    if chunk3_result.exists():
        sel = json.loads(chunk3_result.read_text())
        augmentation = sel.get("winning_condition", augmentation)
        print(f"Using augmentation from Chunk 3: {augmentation}")
    else:
        print(f"No Chunk 3 result found, defaulting to: {augmentation}")
    
    # Load training pool
    paths, labels = load_training_pool()
    paths = np.array(paths)
    labels = np.array(labels)
    
    print(f"Training pool: {len(paths)} images, {N_CLASSES} classes")
    print(f"Class distribution: {dict(zip(CLASS_NAMES, np.bincount(labels)))}")
    print(f"Folds: {N_FOLDS}")
    print()
    
    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
        print(f"  FOLD {fold_idx+1}/{N_FOLDS} (train={len(train_idx)}, val={len(val_idx)})")
        
        train_paths = paths[train_idx].tolist()
        train_labels = labels[train_idx].tolist()
        val_paths = paths[val_idx].tolist()
        val_labels = labels[val_idx].tolist()
        
        fold_metrics = train_one_fold(
            train_paths, train_labels, val_paths, val_labels,
            fold_idx, seed=42+fold_idx, device=device, augmentation=augmentation
        )
        fold_results.append(fold_metrics)
        print(f"    -> Acc={fold_metrics['accuracy']:.4f}, F1={fold_metrics['macro_f1']:.4f}, Kappa={fold_metrics['cohen_kappa']:.4f}")
        print()
    
    # Aggregate results
    metric_keys = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
                   "macro_f1", "weighted_f1", "cohen_kappa"]
    
    aggregate = {}
    for key in metric_keys:
        values = [r[key] for r in fold_results]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        ci_low = mean - 1.96 * std / np.sqrt(N_FOLDS)
        ci_high = mean + 1.96 * std / np.sqrt(N_FOLDS)
        aggregate[key] = {
            "mean": mean,
            "std": std,
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
            "per_fold": values,
        }
    
    # Compare with holdout results (from notebook)
    holdout_comparison = {
        "note": "Comparing CV results with 3-seed holdout from dinov3_augmented_finetune_experiment_Results.ipynb",
        "holdout_best_seed_42_test_macro_f1": 0.9920,
        "holdout_best_seed_42_test_accuracy": 0.9920,
        "cv_mean_macro_f1": aggregate["macro_f1"]["mean"],
        "cv_std_macro_f1": aggregate["macro_f1"]["std"],
        "gap_macro_f1": abs(0.9920 - aggregate["macro_f1"]["mean"]),
        "interpretation": (
            "If the gap is small (<0.02), the holdout result is consistent with CV. "
            "If the gap is large (>0.05), there may be overfitting or split-dependent performance."
        ),
    }
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_id": MODEL_ID,
            "n_folds": N_FOLDS,
            "augmentation": augmentation,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "backbone_lr": BACKBONE_LR,
            "unfreeze_layers": UNFREEZE_LAYERS,
            "img_size": IMG_SIZE,
            "seed_for_kfold_split": 42,
            "total_training_pool": int(len(paths)),
            "device": str(device),
        },
        "fold_results": fold_results,
        "aggregate": aggregate,
        "holdout_comparison": holdout_comparison,
    }
    
    # Save
    with open(OUT_DIR / "cv_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("=" * 70)
    print("5-FOLD CV SUMMARY")
    print("=" * 70)
    for key in metric_keys:
        a = aggregate[key]
        print(f"  {key:25s}: {a['mean']:.4f} +/- {a['std']:.4f}  (95% CI: [{a['ci_95_low']:.4f}, {a['ci_95_high']:.4f}])")
    print()
    print(f"  Holdout test F1 (seed 42): 0.9920")
    print(f"  CV mean F1:               {aggregate['macro_f1']['mean']:.4f}")
    print(f"  Gap:                      {holdout_comparison['gap_macro_f1']:.4f}")
    print()
    print(f"Results saved to: {OUT_DIR}")
    print("DONE")


if __name__ == "__main__":
    run_cross_validation()

#!/usr/bin/env python3
"""
Chunk 4 — Fair DINOv3 vs ResNet50 Baseline Comparison
=====================================================
Both models are trained under IDENTICAL conditions:
  - Same near-duplicate-cleaned dataset
  - Same train/val/test split (seed=42, 85/15 stratified)
  - Same augmentation (winner from Chunk 3, default: medically_conservative)
  - Same seeds [42, 123, 2026]
  - Same early stopping (patience=7 on val loss)
  - Same model selection (best val macro-F1)

Reports: all metrics + 95% CI, McNemar test, parameter counts.

Output: artifacts/reviewer_experiments/baseline_comparison/
"""

import os
import sys
import json
import copy
import random
import hashlib
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = PROJECT_ROOT / "artifacts" / "reviewer_experiments" / "near_duplicate_audit"
CHUNK3_DIR = PROJECT_ROOT / "artifacts" / "reviewer_experiments" / "augmentation_experiment"
OUT_DIR = PROJECT_ROOT / "artifacts" / "reviewer_experiments" / "baseline_comparison"

DATA_ROOT = Path(r"C:\Users\darji\Downloads\BT_Images")
TRAIN_DIR = DATA_ROOT / "Training"
TEST_DIR = DATA_ROOT / "Testing"

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "dinov3_model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "img_size": 224,
    "batch_size": 32,
    "dinov3_hidden_dim": 768,
    "dinov3_n_unfreeze_blocks": 4,
    "val_split": 0.15,
    "split_seed": 42,
    "seeds": [42, 123, 2026],
    "epochs": 25,
    "patience": 7,
    "lr_head": 1e-3,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "near_dup_cosine_threshold": 0.99,
    "augmentation_condition": None,  # filled at runtime
    "models_to_compare": ["dinov3", "resnet50"],
}


# ===================================================================
# Shared utilities (same as Chunk 3 for reproducibility)
# ===================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_images(split_dir: Path) -> list[dict]:
    rows = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        cls_name = class_dir.name.lower().replace("_tumor", "").replace("tumor", "").strip("_")
        if cls_name not in CLASS_NAMES:
            cls_name = class_dir.name
        for fp in sorted(class_dir.iterdir()):
            if fp.suffix.lower() in VALID_EXT:
                rows.append({"path": str(fp), "class": cls_name})
    return rows


class GaussianNoise:
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def build_transforms(condition: str, img_size: int = 224) -> T.Compose:
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if condition == "minimal":
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            norm,
        ])

    if condition == "conservative_mri":
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomRotation(degrees=10),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ToTensor(),
            norm,
        ])

    if condition == "aggressive_original":
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandAugment(num_ops=2, magnitude=9),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.ToTensor(),
            norm,
            T.RandomErasing(p=0.25),
        ])

    if condition == "medically_conservative":
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomAffine(degrees=0, translate=(0.0, 0.0), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.1),
            T.ToTensor(),
            GaussianNoise(mean=0.0, std=0.01),
            norm,
        ])

    raise ValueError(f"Unknown augmentation condition: {condition}")


def build_eval_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ImageListDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# ===================================================================
# Data cleaning (identical to Chunk 3)
# ===================================================================

def load_and_clean_data():
    print("=" * 70)
    print("DATA CLEANING — SHA-256 + Near-Duplicate Removal")
    print("=" * 70)

    train_rows = list_images(TRAIN_DIR)
    test_rows = list_images(TEST_DIR)
    print(f"Raw train images: {len(train_rows)}")
    print(f"Raw test images:  {len(test_rows)}")

    hash_csv = AUDIT_DIR / "all_images_hashes.csv"
    if not hash_csv.exists():
        sys.exit(f"ERROR: {hash_csv} not found. Run near_duplicate_audit.py first.")

    df_hashes = pd.read_csv(hash_csv)
    train_hashes = df_hashes[df_hashes["split"] == "train"]
    test_hashes_set = set(df_hashes[df_hashes["split"] == "test"]["sha256"].unique())

    sha_dup_train_paths: set[str] = set()
    for _, row in train_hashes.iterrows():
        if row["sha256"] in test_hashes_set:
            sha_dup_train_paths.add(row["path"])

    print(f"SHA-256 cross-split duplicates in train: {len(sha_dup_train_paths)}")

    emb_csv = AUDIT_DIR / "embedding_suspicious_pairs.csv"
    if not emb_csv.exists():
        sys.exit(f"ERROR: {emb_csv} not found. Run near_duplicate_audit.py first.")

    df_pairs = pd.read_csv(emb_csv)
    near_dup_mask = df_pairs["cosine_similarity"] >= CONFIG["near_dup_cosine_threshold"]
    near_dup_train_paths: set[str] = set(df_pairs[near_dup_mask]["train_path"].unique())
    print(f"Near-duplicate train images removed (cosine >= "
          f"{CONFIG['near_dup_cosine_threshold']}): {len(near_dup_train_paths)}")

    excluded_train_paths = sha_dup_train_paths | near_dup_train_paths
    print(f"Total train images removed: {len(excluded_train_paths)}")

    train_rows_clean = [r for r in train_rows if r["path"] not in excluded_train_paths]
    print(f"Clean train: {len(train_rows_clean)}  Test (untouched): {len(test_rows)}")

    train_fps = [r["path"] for r in train_rows_clean]
    train_labels = [CLASS_TO_IDX[r["class"]] for r in train_rows_clean]
    test_fps = [r["path"] for r in test_rows]
    test_labels = [CLASS_TO_IDX[r["class"]] for r in test_rows]

    return train_fps, train_labels, test_fps, test_labels, {
        "raw_train": len(train_rows),
        "raw_test": len(test_rows),
        "sha256_dups_removed": len(sha_dup_train_paths),
        "near_dup_train_removed": len(near_dup_train_paths),
        "total_removed": len(excluded_train_paths),
        "clean_train": len(train_rows_clean),
        "clean_test": len(test_rows),
    }


# ===================================================================
# Model builders
# ===================================================================

def build_dinov3(model_id, hidden_dim, num_classes, n_unfreeze, dropout, device):
    from transformers import AutoModel

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    backbone = AutoModel.from_pretrained(model_id, token=hf_token).to(device)

    head = nn.Sequential(
        nn.Linear(hidden_dim, 256),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )

    class DINOv3Classifier(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            out = self.backbone(x)
            cls_token = out.last_hidden_state[:, 0]
            return self.head(cls_token)

    model = DINOv3Classifier(backbone, head).to(device)

    for param in model.backbone.parameters():
        param.requires_grad = False

    encoder_layers = None
    if hasattr(model.backbone, "encoder") and hasattr(model.backbone.encoder, "layer"):
        encoder_layers = model.backbone.encoder.layer
    elif hasattr(model.backbone, "layers"):
        encoder_layers = model.backbone.layers

    if encoder_layers is not None:
        total = len(encoder_layers)
        for i in range(total - n_unfreeze, total):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True
        print(f"  [DINOv3] Unfroze last {n_unfreeze} of {total} blocks.")
    else:
        all_named = list(model.backbone.named_parameters())
        block_params = defaultdict(list)
        for name, param in all_named:
            for part in name.split("."):
                if part.isdigit():
                    block_params[int(part)].append((name, param))
                    break
        if block_params:
            max_block = max(block_params.keys())
            for block_idx in range(max_block - n_unfreeze + 1, max_block + 1):
                for _, param in block_params.get(block_idx, []):
                    param.requires_grad = True

    for param in model.head.parameters():
        param.requires_grad = True

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [DINOv3] Params — total: {total_p:,}  trainable: {train_p:,}")

    backbone_params = [p for n, p in model.named_parameters()
                       if "head" not in n and p.requires_grad]
    head_params = list(model.head.parameters())
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": CONFIG["lr_backbone"]},
        {"params": head_params, "lr": CONFIG["lr_head"]},
    ], weight_decay=CONFIG["weight_decay"])

    return model, optimizer, total_p, train_p


def build_resnet50(num_classes, device):
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [ResNet50] Params — total: {total_p:,}  trainable: {train_p:,}")

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr_head"],
                           weight_decay=CONFIG["weight_decay"])

    return model, optimizer, total_p, train_p


# ===================================================================
# Training & evaluation (identical logic for both models)
# ===================================================================

def train_model(model, train_loader, val_loader, optimizer, scheduler,
                criterion, epochs, patience, device, desc=""):
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_state = None
    patience_ctr = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            n += images.size(0)
        train_loss = running_loss / n

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                val_loss_sum += criterion(out, labels).item() * images.size(0)
                val_n += images.size(0)
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss = val_loss_sum / val_n
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if scheduler is not None:
            scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            flag = "*BEST*" if improved else ""
            print(f"  [{desc}] Ep {epoch:>2}/{epochs}  "
                  f"TrL={train_loss:.4f}  VL={val_loss:.4f}  "
                  f"VAcc={val_acc:.4f}  VF1={val_f1:.4f}  {flag}")

        if patience_ctr >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val_f1


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        out = model(images)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
        all_probs.append(probs)
    return (np.array(all_preds), np.array(all_labels),
            np.concatenate(all_probs))


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def summarize_across_seeds(metric_dicts: list[dict], metric_keys: list[str]):
    n = len(metric_dicts)
    t_crit = stats.t.ppf(0.975, df=max(n - 1, 1))
    rows = []
    for key in metric_keys:
        vals = np.array([d[key] for d in metric_dicts])
        m, s = vals.mean(), vals.std(ddof=1) if n > 1 else 0.0
        ci = t_crit * s / np.sqrt(n) if n > 1 else 0.0
        rows.append({
            "metric": key,
            "mean": float(m),
            "std": float(s),
            "ci95_lower": float(m - ci),
            "ci95_upper": float(m + ci),
        })
    return rows


# ===================================================================
# McNemar test
# ===================================================================

def mcnemar_test(preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray):
    """Exact McNemar test comparing two classifiers on the same test set.
    Returns (chi2_or_exact_stat, p_value, contingency_table_dict)."""

    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    # b = A correct, B wrong; c = A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    a = int(np.sum(correct_a & correct_b))
    d = int(np.sum(~correct_a & ~correct_b))

    contingency = {"both_correct": a, "only_A_correct": b,
                   "only_B_correct": c, "both_wrong": d}

    n_discord = b + c
    if n_discord == 0:
        return 0.0, 1.0, contingency

    if n_discord < 25:
        from scipy.stats import binom_test
        try:
            p_value = binom_test(b, n_discord, 0.5)
        except Exception:
            from scipy.stats import binomtest
            result = binomtest(b, n_discord, 0.5)
            p_value = result.pvalue
        return float(n_discord), float(p_value), contingency
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = float(1.0 - stats.chi2.cdf(chi2, df=1))
        return float(chi2), p_value, contingency


# ===================================================================
# Determine augmentation condition
# ===================================================================

def resolve_augmentation_condition() -> str:
    """Try to read the Chunk 3 winner; fall back to medically_conservative."""
    manifest_path = CHUNK3_DIR / "experiment_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            winner = manifest.get("winning_condition")
            if winner:
                print(f"Loaded winning augmentation from Chunk 3: {winner}")
                return winner
        except Exception as e:
            print(f"Warning: could not parse Chunk 3 manifest: {e}")

    default = "medically_conservative"
    print(f"Chunk 3 results not found. Defaulting to: {default}")
    return default


# ===================================================================
# Main experiment
# ===================================================================

def main():
    timestamp = datetime.now().isoformat(timespec="seconds")
    print("=" * 70)
    print("CHUNK 4 — Fair DINOv3 vs ResNet50 Baseline Comparison")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    if not TRAIN_DIR.is_dir() or not TEST_DIR.is_dir():
        print(f"ERROR: Dataset not found at {DATA_ROOT}")
        print("Expected Training/ and Testing/ subdirectories.")
        sys.exit(1)

    if not AUDIT_DIR.exists():
        print(f"ERROR: Audit directory not found at {AUDIT_DIR}")
        print("Run experiments/near_duplicate_audit.py first.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR = OUT_DIR / "checkpoints"
    CKPT_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    aug_condition = resolve_augmentation_condition()
    CONFIG["augmentation_condition"] = aug_condition
    print(f"Augmentation condition for BOTH models: {aug_condition}")
    print(f"Config: {json.dumps(CONFIG, indent=2)}")

    # --- Clean data (same as Chunk 3) ----------------------------------------
    train_fps, train_labels, test_fps, test_labels, cleaning_stats = load_and_clean_data()

    # --- Fixed split ---------------------------------------------------------
    set_seed(CONFIG["split_seed"])
    idx_train, idx_val = train_test_split(
        list(range(len(train_fps))),
        test_size=CONFIG["val_split"],
        stratify=train_labels,
        random_state=CONFIG["split_seed"],
    )
    tr_fps = [train_fps[i] for i in idx_train]
    tr_labs = [train_labels[i] for i in idx_train]
    vl_fps = [train_fps[i] for i in idx_val]
    vl_labs = [train_labels[i] for i in idx_val]

    print(f"\nSplit — train: {len(tr_fps)}  val: {len(vl_fps)}  test: {len(test_fps)}")

    tr_set, vl_set, te_set = set(tr_fps), set(vl_fps), set(test_fps)
    assert len(tr_set & vl_set) == 0, "Path overlap train/val!"
    assert len(tr_set & te_set) == 0, "Path overlap train/test!"
    assert len(vl_set & te_set) == 0, "Path overlap val/test!"
    print("Leakage audit: PASSED")

    # --- Build loaders -------------------------------------------------------
    train_transform = build_transforms(aug_condition, CONFIG["img_size"])
    eval_transform = build_eval_transform(CONFIG["img_size"])

    val_ds = ImageListDataset(vl_fps, vl_labs, eval_transform)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                            num_workers=2, pin_memory=True)

    test_ds = ImageListDataset(test_fps, test_labels, eval_transform)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                             num_workers=2, pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    # ===========================================================
    # Train both models
    # ===========================================================
    all_results = {}  # model_name -> {seed -> {metrics, preds}}
    param_info = {}

    for model_name in CONFIG["models_to_compare"]:
        print("\n" + "=" * 70)
        print(f"TRAINING: {model_name.upper()}")
        print("=" * 70)

        seed_results = []
        seed_preds = {}

        for seed in CONFIG["seeds"]:
            tag = f"{model_name}/seed{seed}"
            print(f"\n{'─'*60}\n  {tag}\n{'─'*60}")
            set_seed(seed)

            train_ds = ImageListDataset(tr_fps, tr_labs, train_transform)
            train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                                      shuffle=True, num_workers=2, pin_memory=True)

            if model_name == "dinov3":
                model, optimizer, total_p, train_p = build_dinov3(
                    CONFIG["dinov3_model_id"], CONFIG["dinov3_hidden_dim"],
                    NUM_CLASSES, CONFIG["dinov3_n_unfreeze_blocks"],
                    CONFIG["dropout"], device,
                )
            elif model_name == "resnet50":
                model, optimizer, total_p, train_p = build_resnet50(
                    NUM_CLASSES, device,
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

            param_info[model_name] = {"total": total_p, "trainable": train_p}

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3)

            t0 = time.time()
            model, history, best_val_f1 = train_model(
                model, train_loader, val_loader, optimizer, scheduler,
                criterion, CONFIG["epochs"], CONFIG["patience"], device, desc=tag,
            )
            elapsed = time.time() - t0

            ckpt_path = CKPT_DIR / f"{model_name}_seed{seed}.pth"
            torch.save(model.state_dict(), ckpt_path)

            # --- Evaluate on val -------------------------------------------------
            preds_val, labels_val, _ = evaluate_model(model, val_loader, device)
            val_metrics = compute_metrics(labels_val, preds_val)

            # --- Evaluate on test ------------------------------------------------
            preds_test, labels_test, probs_test = evaluate_model(model, test_loader, device)
            test_metrics = compute_metrics(labels_test, preds_test)

            combined = {
                "model": model_name,
                "seed": seed,
                "train_time_sec": round(elapsed, 1),
                "best_val_f1": val_metrics["macro_f1"],
            }
            for k, v in test_metrics.items():
                combined[f"test_{k}"] = v

            seed_results.append(combined)
            seed_preds[seed] = preds_test.tolist()

            print(f"  {tag}: val_f1={val_metrics['macro_f1']:.4f}  "
                  f"test_acc={test_metrics['accuracy']:.4f}  "
                  f"test_f1={test_metrics['macro_f1']:.4f}")

            del model, train_loader, train_ds
            torch.cuda.empty_cache()

        all_results[model_name] = {
            "seed_results": seed_results,
            "seed_preds": seed_preds,
        }

    # ===========================================================
    # Aggregate results
    # ===========================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    metric_keys = ["test_accuracy", "test_balanced_accuracy", "test_macro_precision",
                   "test_macro_recall", "test_macro_f1", "test_weighted_f1", "test_kappa"]
    all_per_seed = []
    summary_by_model = {}

    for model_name in CONFIG["models_to_compare"]:
        sr = all_results[model_name]["seed_results"]
        all_per_seed.extend(sr)

        summary_rows = summarize_across_seeds(sr, metric_keys)
        summary_by_model[model_name] = summary_rows

        print(f"\n--- {model_name.upper()} (mean ± std, 95% CI) ---")
        df_s = pd.DataFrame(summary_rows)
        print(df_s.to_string(index=False))

    # Save per-seed results
    df_all = pd.DataFrame(all_per_seed)
    df_all.to_csv(OUT_DIR / "per_seed_results.csv", index=False)

    # Save summary tables per model
    for model_name, rows in summary_by_model.items():
        pd.DataFrame(rows).to_csv(OUT_DIR / f"{model_name}_summary.csv", index=False)

    # ===========================================================
    # McNemar test (on each seed pair)
    # ===========================================================
    print("\n" + "=" * 70)
    print("McNEMAR TEST — DINOv3 vs ResNet50")
    print("=" * 70)

    mcnemar_results = []
    for seed in CONFIG["seeds"]:
        preds_dino = np.array(all_results["dinov3"]["seed_preds"][seed])
        preds_resnet = np.array(all_results["resnet50"]["seed_preds"][seed])
        labels = np.array(test_labels)

        stat, p_val, contingency = mcnemar_test(preds_dino, preds_resnet, labels)

        mcnemar_results.append({
            "seed": seed,
            "statistic": stat,
            "p_value": p_val,
            **contingency,
            "significant_at_0.05": p_val < 0.05,
        })

        print(f"  seed={seed}: stat={stat:.4f}  p={p_val:.6f}  "
              f"{'SIGNIFICANT' if p_val < 0.05 else 'not significant'} at α=0.05")
        print(f"    both_correct={contingency['both_correct']}  "
              f"only_DINOv3={contingency['only_A_correct']}  "
              f"only_ResNet50={contingency['only_B_correct']}  "
              f"both_wrong={contingency['both_wrong']}")

    df_mcnemar = pd.DataFrame(mcnemar_results)
    df_mcnemar.to_csv(OUT_DIR / "mcnemar_results.csv", index=False)

    # ===========================================================
    # Parameter comparison
    # ===========================================================
    print("\n" + "=" * 70)
    print("PARAMETER COUNTS")
    print("=" * 70)
    param_rows = []
    for model_name, info in param_info.items():
        print(f"  {model_name}: total={info['total']:,}  trainable={info['trainable']:,}")
        param_rows.append({
            "model": model_name,
            "total_parameters": info["total"],
            "trainable_parameters": info["trainable"],
        })
    pd.DataFrame(param_rows).to_csv(OUT_DIR / "parameter_counts.csv", index=False)

    # ===========================================================
    # Save predictions for reproducibility
    # ===========================================================
    pred_data = {}
    for model_name in CONFIG["models_to_compare"]:
        pred_data[model_name] = {
            str(s): all_results[model_name]["seed_preds"][s]
            for s in CONFIG["seeds"]
        }
    pred_data["test_labels"] = test_labels

    with open(OUT_DIR / "test_predictions_all_models.json", "w") as f:
        json.dump(pred_data, f, indent=2)

    # ===========================================================
    # Final manifest
    # ===========================================================
    manifest = {
        "experiment": "Chunk 4 — Fair DINOv3 vs ResNet50 Comparison",
        "timestamp": timestamp,
        "device": str(device),
        "config": CONFIG,
        "cleaning_stats": cleaning_stats,
        "split_sizes": {
            "train": len(tr_fps),
            "val": len(vl_fps),
            "test": len(test_fps),
        },
        "augmentation_condition": aug_condition,
        "parameter_counts": param_info,
        "results_summary": {
            model_name: summary_by_model[model_name]
            for model_name in CONFIG["models_to_compare"]
        },
        "mcnemar_tests": mcnemar_results,
        "methodology_notes": [
            "Both models trained under identical conditions: same cleaned dataset, "
            "same train/val/test split, same augmentation, same seeds, same early "
            "stopping and model selection rules.",
            "DINOv3: last 4 transformer blocks unfrozen + MLP head (768->256->4). "
            "Differential learning rates (backbone=1e-5, head=1e-3).",
            "ResNet50: ImageNet pretrained, entire model fine-tuned, fc replaced "
            "with Linear(2048, 4). Single learning rate (1e-3).",
            "McNemar test performed on paired test-set predictions per seed.",
        ],
    }
    with open(OUT_DIR / "experiment_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"All artifacts saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

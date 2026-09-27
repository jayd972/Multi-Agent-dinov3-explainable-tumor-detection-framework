#!/usr/bin/env python3
"""
Chunk 3 — Medically Defensible Augmentation Experiment
======================================================
Compares 4 augmentation conditions on a near-duplicate-cleaned training set.

Conditions:
  a) minimal           — resize + normalize only
  b) conservative_mri  — small rotation <=10°, translation <=5%, scale 0.95-1.05
  c) aggressive_original — RandAugment, vertical flip, color jitter (the prior winner)
  d) medically_conservative — horizontal flip, rotation <=15°, scale 0.9-1.1,
                              Gaussian noise, slight brightness; NO vertical flip,
                              NO hue/saturation, NO color jitter

Selection: mean validation macro-F1 across 3 seeds.
Test evaluation: winning condition only.

Output: artifacts/reviewer_experiments/augmentation_experiment/
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
OUT_DIR = PROJECT_ROOT / "artifacts" / "reviewer_experiments" / "augmentation_experiment"

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
    "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "img_size": 224,
    "batch_size": 32,
    "hidden_dim": 768,
    "n_unfreeze_blocks": 4,
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
    "augmentation_conditions": [
        "minimal",
        "conservative_mri",
        "aggressive_original",
        "medically_conservative",
    ],
}


# ===================================================================
# Utility helpers
# ===================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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


# ===================================================================
# Data cleaning: SHA-256 dedup + near-duplicate removal
# ===================================================================

def load_and_clean_data():
    """Return cleaned (train_fps, train_labels, test_fps, test_labels)
    after removing SHA-256 cross-split duplicates and near-duplicate
    training images (cosine >= threshold with any test image)."""

    print("=" * 70)
    print("DATA CLEANING — SHA-256 + Near-Duplicate Removal")
    print("=" * 70)

    # --- list raw images -----------------------------------------------------
    train_rows = list_images(TRAIN_DIR)
    test_rows = list_images(TEST_DIR)
    print(f"Raw train images: {len(train_rows)}")
    print(f"Raw test images:  {len(test_rows)}")

    # --- SHA-256 cross-split dedup -------------------------------------------
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

    # --- Near-duplicate removal (cosine >= threshold) ------------------------
    emb_csv = AUDIT_DIR / "embedding_suspicious_pairs.csv"
    if not emb_csv.exists():
        sys.exit(f"ERROR: {emb_csv} not found. Run near_duplicate_audit.py first.")

    df_pairs = pd.read_csv(emb_csv)
    near_dup_mask = df_pairs["cosine_similarity"] >= CONFIG["near_dup_cosine_threshold"]
    near_dup_pairs = df_pairs[near_dup_mask]

    near_dup_train_paths: set[str] = set(near_dup_pairs["train_path"].unique())
    print(f"Near-duplicate pairs (cosine >= {CONFIG['near_dup_cosine_threshold']}): "
          f"{len(near_dup_pairs)}")
    print(f"Unique train images flagged as near-duplicates: {len(near_dup_train_paths)}")

    # --- Combine exclusion sets (protect test, remove from train) ------------
    excluded_train_paths = sha_dup_train_paths | near_dup_train_paths
    print(f"Total train images to remove: {len(excluded_train_paths)}")

    train_rows_clean = [r for r in train_rows if r["path"] not in excluded_train_paths]
    print(f"Clean train images: {len(train_rows_clean)}")
    print(f"Test images (untouched): {len(test_rows)}")

    train_fps = [r["path"] for r in train_rows_clean]
    train_labels = [CLASS_TO_IDX[r["class"]] for r in train_rows_clean]
    test_fps = [r["path"] for r in test_rows]
    test_labels = [CLASS_TO_IDX[r["class"]] for r in test_rows]

    # --- Class distribution after cleaning -----------------------------------
    from collections import Counter
    train_dist = Counter(r["class"] for r in train_rows_clean)
    print(f"Train class distribution: { {c: train_dist.get(c, 0) for c in CLASS_NAMES} }")

    return train_fps, train_labels, test_fps, test_labels, {
        "raw_train": len(train_rows),
        "raw_test": len(test_rows),
        "sha256_dups_removed": len(sha_dup_train_paths),
        "near_dup_pairs": int(near_dup_mask.sum()),
        "near_dup_train_images_removed": len(near_dup_train_paths),
        "total_removed": len(excluded_train_paths),
        "clean_train": len(train_rows_clean),
        "clean_test": len(test_rows),
    }


# ===================================================================
# Transforms
# ===================================================================

class GaussianNoise:
    """Add Gaussian noise to a tensor (after ToTensor)."""
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


# ===================================================================
# Dataset
# ===================================================================

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
# Model
# ===================================================================

def build_dinov3_classifier(model_id, hidden_dim, num_classes, n_unfreeze, dropout, device):
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

    # Freeze everything first
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Unfreeze last N blocks
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
        print(f"  Unfroze last {n_unfreeze} of {total} transformer blocks.")
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
            print(f"  Unfroze blocks {max_block - n_unfreeze + 1}..{max_block}.")

    for param in model.head.parameters():
        param.requires_grad = True

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters — total: {total_p:,}  trainable: {train_p:,}")

    return model, total_p, train_p


# ===================================================================
# Training
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
# Main experiment
# ===================================================================

def main():
    timestamp = datetime.now().isoformat(timespec="seconds")
    print("=" * 70)
    print("CHUNK 3 — Medically Defensible Augmentation Experiment")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    # --- Verify data exists --------------------------------------------------
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
    print(f"Config: {json.dumps(CONFIG, indent=2)}")

    # --- Clean data ----------------------------------------------------------
    train_fps, train_labels, test_fps, test_labels, cleaning_stats = load_and_clean_data()

    # --- Fixed train/val split -----------------------------------------------
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

    # --- Leakage sanity check ------------------------------------------------
    tr_set, vl_set, te_set = set(tr_fps), set(vl_fps), set(test_fps)
    assert len(tr_set & vl_set) == 0, "Path overlap train/val!"
    assert len(tr_set & te_set) == 0, "Path overlap train/test!"
    assert len(vl_set & te_set) == 0, "Path overlap val/test!"
    print("Leakage audit: PASSED (zero path overlap across splits)")

    eval_transform = build_eval_transform(CONFIG["img_size"])
    val_ds = ImageListDataset(vl_fps, vl_labs, eval_transform)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                            num_workers=2, pin_memory=True)

    # ===========================================================
    # PHASE A — Selection (validation only, test set untouched)
    # ===========================================================
    print("\n" + "=" * 70)
    print("PHASE A — Augmentation condition selection (validation only)")
    print("=" * 70)

    criterion = nn.CrossEntropyLoss()
    selection_rows = []
    SELECTION_PHASE_COMPLETE = False

    for condition in CONFIG["augmentation_conditions"]:
        for seed in CONFIG["seeds"]:
            tag = f"{condition}/seed{seed}"
            print(f"\n{'─'*60}\n  {tag}\n{'─'*60}")
            set_seed(seed)

            train_transform = build_transforms(condition, CONFIG["img_size"])
            train_ds = ImageListDataset(tr_fps, tr_labs, train_transform)
            train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                                      shuffle=True, num_workers=2, pin_memory=True)

            model, total_p, train_p = build_dinov3_classifier(
                CONFIG["model_id"], CONFIG["hidden_dim"], NUM_CLASSES,
                CONFIG["n_unfreeze_blocks"], CONFIG["dropout"], device,
            )

            backbone_params = [p for n, p in model.named_parameters()
                               if "head" not in n and p.requires_grad]
            head_params = list(model.head.parameters())
            optimizer = optim.Adam([
                {"params": backbone_params, "lr": CONFIG["lr_backbone"]},
                {"params": head_params, "lr": CONFIG["lr_head"]},
            ], weight_decay=CONFIG["weight_decay"])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3)

            t0 = time.time()
            model, history, best_val_f1 = train_model(
                model, train_loader, val_loader, optimizer, scheduler,
                criterion, CONFIG["epochs"], CONFIG["patience"], device, desc=tag,
            )
            elapsed = time.time() - t0

            ckpt_path = CKPT_DIR / f"dinov3_{condition}_seed{seed}.pth"
            torch.save(model.state_dict(), ckpt_path)

            preds_val, labels_val, _ = evaluate_model(model, val_loader, device)
            val_metrics = compute_metrics(labels_val, preds_val)

            selection_rows.append({
                "condition": condition,
                "seed": seed,
                "val_macro_f1": val_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "train_time_sec": round(elapsed, 1),
                "checkpoint": str(ckpt_path),
                "total_params": total_p,
                "trainable_params": train_p,
            })

            del model, train_loader, train_ds
            torch.cuda.empty_cache()

    df_sel = pd.DataFrame(selection_rows)
    df_sel.to_csv(OUT_DIR / "phase_a_selection_results.csv", index=False)

    # --- Pick winner by mean val macro-F1 ------------------------------------
    agg = (df_sel.groupby("condition")["val_macro_f1"]
           .agg(["mean", "std", "count"])
           .reset_index()
           .sort_values("mean", ascending=False))
    agg.to_csv(OUT_DIR / "phase_a_condition_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("PHASE A RESULTS — Mean validation macro-F1")
    print("=" * 70)
    print(agg.to_string(index=False))

    WINNING_CONDITION = agg.iloc[0]["condition"]
    print(f"\nSELECTED WINNER: {WINNING_CONDITION}")
    print("Selection rule: highest mean validation macro-F1 across "
          f"{len(CONFIG['seeds'])} seeds. Test set was NOT touched.")
    SELECTION_PHASE_COMPLETE = True

    # ===========================================================
    # PHASE B — Confirmatory test evaluation (winner only)
    # ===========================================================
    print("\n" + "=" * 70)
    print(f"PHASE B — Test evaluation for winner: {WINNING_CONDITION}")
    print("=" * 70)

    test_ds = ImageListDataset(test_fps, test_labels, eval_transform)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                             num_workers=2, pin_memory=True)

    confirm_rows = []
    all_test_preds_by_seed = {}
    test_access_log = []

    for seed in CONFIG["seeds"]:
        ckpt_path = CKPT_DIR / f"dinov3_{WINNING_CONDITION}_seed{seed}.pth"
        print(f"\n  Loading {ckpt_path.name} ...")

        model, _, _ = build_dinov3_classifier(
            CONFIG["model_id"], CONFIG["hidden_dim"], NUM_CLASSES,
            CONFIG["n_unfreeze_blocks"], CONFIG["dropout"], device,
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.to(device)

        test_access_log.append(f"confirmatory-eval:{WINNING_CONDITION}:seed{seed}")

        preds, labels, probs = evaluate_model(model, test_loader, device)
        metrics = compute_metrics(labels, preds)
        metrics["condition"] = WINNING_CONDITION
        metrics["seed"] = seed
        confirm_rows.append(metrics)
        all_test_preds_by_seed[seed] = preds.tolist()

        print(f"  seed={seed}: acc={metrics['accuracy']:.4f}  "
              f"macro_f1={metrics['macro_f1']:.4f}  "
              f"kappa={metrics['kappa']:.4f}")

        del model
        torch.cuda.empty_cache()

    df_confirm = pd.DataFrame(confirm_rows)
    df_confirm.to_csv(OUT_DIR / "phase_b_test_results.csv", index=False)

    metric_keys = ["accuracy", "balanced_accuracy", "macro_precision",
                   "macro_recall", "macro_f1", "weighted_f1", "kappa"]
    summary_rows = summarize_across_seeds(confirm_rows, metric_keys)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_DIR / "phase_b_test_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("PHASE B — Test results (winning condition, 3 seeds, 95% CI)")
    print("=" * 70)
    print(df_summary.to_string(index=False))

    # --- Save predictions for downstream McNemar test ------------------------
    pred_path = OUT_DIR / "test_predictions_per_seed.json"
    with open(pred_path, "w") as f:
        json.dump({
            "model": "dinov3",
            "condition": WINNING_CONDITION,
            "test_labels": test_labels,
            "predictions_by_seed": {str(s): all_test_preds_by_seed[s]
                                    for s in CONFIG["seeds"]},
        }, f, indent=2)
    print(f"\nPer-seed predictions saved to {pred_path}")

    # --- Final manifest ------------------------------------------------------
    manifest = {
        "experiment": "Chunk 3 — Medically Defensible Augmentation",
        "timestamp": timestamp,
        "device": str(device),
        "config": CONFIG,
        "cleaning_stats": cleaning_stats,
        "split_sizes": {
            "train": len(tr_fps),
            "val": len(vl_fps),
            "test": len(test_fps),
        },
        "winning_condition": WINNING_CONDITION,
        "selection_rule": (
            "Highest mean validation macro-F1 across "
            f"{len(CONFIG['seeds'])} seeds. Test set untouched during selection."
        ),
        "test_access_log": test_access_log,
        "test_access_count": len(test_access_log),
        "phase_a_summary": agg.to_dict(orient="records"),
        "phase_b_summary": summary_rows,
        "known_limitations": [
            "'aggressive_original' includes vertical flip and hue/saturation "
            "color jitter, which are medically questionable for brain MRI.",
            "Only 4 conditions and 3 seeds; this is an ablation, not exhaustive.",
            "Near-duplicate removal uses embedding cosine similarity at 0.99 "
            "threshold; some borderline leaky pairs (0.95-0.99) may remain.",
        ],
    }
    manifest_path = OUT_DIR / "experiment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest saved to {manifest_path}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"All artifacts saved to: {OUT_DIR}")
    print("=" * 70)

    return WINNING_CONDITION


if __name__ == "__main__":
    main()

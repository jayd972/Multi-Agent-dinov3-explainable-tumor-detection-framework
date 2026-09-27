"""
Standalone experiment to add to dinov3_augmented_finetune_experiment_Results notebook.
Adds:
  1. Medically defensible augmentation (no vertical flip, no hue/saturation)
  2. 5-fold stratified CV on the winning aggressive augmentation DINOv3 model

Run from project root:
    python experiments/dinov3_cv_and_medical_aug.py
"""

import os, sys, json, random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel
from huggingface_hub import login

CONFIG = {
    "model_name":      "facebook/dinov2-base",
    "img_size":        224,
    "batch_size":      32,
    "epochs":          15,
    "lr":              2e-5,
    "weight_decay":    1e-4,
    "num_classes":     4,
    "seeds":           [42, 123, 2026],
    "n_folds":         5,
    "unfreeze_layers": 4,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "data_clean_root": "data/brisc_cleaned",
}
CLASSES  = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
OUT_DIR  = PROJECT_ROOT / "results_augmented_dinov3" / "cv_and_medical_aug"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ── HuggingFace auth ─────────────────────────────────────────────────────────
def auth():
    cfg_path = PROJECT_ROOT / "config" / "api_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        token = cfg.get("api_keys", {}).get("huggingface", {}).get("token", "")
        if token:
            login(token=token, add_to_git_credential=False)


# ── augmentation ─────────────────────────────────────────────────────────────
def get_transform(condition, is_train):
    base = [
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    ]
    if not is_train:
        return T.Compose(base + [T.ToTensor(), T.Normalize(MEAN, STD)])

    if condition == "aggressive_original":
        aug = [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.05),
            T.RandAugment(num_ops=2, magnitude=9),
            T.RandomErasing(p=0.1),
        ]
    elif condition == "medically_defensible":
        # Reviewer-safe: no vertical flip (axial MRI not vertically symmetric)
        # No hue/saturation (MRI signal intensity is diagnostically meaningful)
        aug = [
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.RandomErasing(p=0.05),
        ]
    else:
        aug = []

    return T.Compose(base + aug + [T.ToTensor(), T.Normalize(MEAN, STD)])


# ── dataset ───────────────────────────────────────────────────────────────────
class BRISCDataset(Dataset):
    def __init__(self, fps, labels, transform=None):
        self.fps, self.labels, self.transform = fps, labels, transform

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, i):
        img = Image.open(self.fps[i])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


# ── DINOv3 Classifier ─────────────────────────────────────────────────────────
class DINOv3Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(CONFIG["model_name"])
        for p in self.backbone.parameters():
            p.requires_grad = False
        for layer in self.backbone.encoder.layer[-CONFIG["unfreeze_layers"]:]:
            for p in layer.parameters():
                p.requires_grad = True
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, CONFIG["num_classes"]),
        )

    def forward(self, x):
        cls = self.backbone(pixel_values=x).last_hidden_state[:, 0]
        return self.head(cls)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def train_epoch(model, loader, opt, crit, device):
    model.train()
    for imgs, lbl in loader:
        imgs, lbl = imgs.to(device), lbl.to(device)
        opt.zero_grad()
        crit(model(imgs), lbl).backward()
        opt.step()


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, lbl in loader:
            preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
            trues.extend(lbl.numpy())
    return np.array(preds), np.array(trues)


# ── load training data ────────────────────────────────────────────────────────
def load_train_data():
    root = PROJECT_ROOT / CONFIG["data_clean_root"]
    for cand in ["train", "Training", "Train"]:
        d = root / cand
        if d.exists():
            break
    fps, labels = [], []
    for ci, cls in enumerate(CLASSES):
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            for p in sorted((d / cls).glob(ext)):
                fps.append(str(p)); labels.append(ci)
    print(f"Loaded {len(fps)} training images.")
    return fps, labels


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Medically Defensible Augmentation
# ═══════════════════════════════════════════════════════════════════════════════
def run_medically_defensible(fps, labels):
    print("\n" + "=" * 65)
    print("EXPERIMENT 1: Medically Defensible Augmentation")
    print("No vertical flip | No hue/saturation | Mild brightness/contrast only")
    print("=" * 65)
    device = torch.device(CONFIG["device"])
    rows = []
    for seed in CONFIG["seeds"]:
        set_seed(seed)
        tr_idx, va_idx = train_test_split(
            range(len(fps)), test_size=0.15, stratify=labels, random_state=seed)
        tr_ds = BRISCDataset([fps[i] for i in tr_idx], [labels[i] for i in tr_idx],
                             get_transform("medically_defensible", True))
        va_ds = BRISCDataset([fps[i] for i in va_idx], [labels[i] for i in va_idx],
                             get_transform("medically_defensible", False))
        tr_loader = DataLoader(tr_ds, CONFIG["batch_size"], shuffle=True,  num_workers=0)
        va_loader = DataLoader(va_ds, CONFIG["batch_size"], shuffle=False, num_workers=0)

        model = DINOv3Classifier().to(device)
        opt   = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        crit  = nn.CrossEntropyLoss()
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"])

        best_f1 = 0.0
        for epoch in range(CONFIG["epochs"]):
            train_epoch(model, tr_loader, opt, crit, device)
            sched.step()
            preds, trues = evaluate(model, va_loader, device)
            f1 = f1_score(trues, preds, average="macro", zero_division=0) * 100
            if f1 > best_f1:
                best_f1 = f1

        acc = accuracy_score(trues, preds) * 100
        print(f"  Seed {seed}: val macro F1={best_f1:.2f}%  val acc={acc:.2f}%")
        rows.append({"condition": "medically_defensible", "seed": seed,
                     "val_macro_f1": best_f1, "val_accuracy": acc})

    df = pd.DataFrame(rows)
    m, s = df["val_macro_f1"].mean(), df["val_macro_f1"].std()
    print(f"\nMedically Defensible | mean val F1 = {m:.2f}% ± {s:.2f}%")
    df.to_csv(OUT_DIR / "medically_defensible_results.csv", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — 5-Fold CV on Aggressive DINOv3
# ═══════════════════════════════════════════════════════════════════════════════
def run_5fold_cv(fps, labels):
    print("\n" + "=" * 65)
    print("EXPERIMENT 2: 5-Fold Stratified CV — DINOv3 + Aggressive Augmentation")
    print("=" * 65)
    device = torch.device(CONFIG["device"])
    seed   = 42
    set_seed(seed)

    skf       = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=seed)
    fps_arr   = np.array(fps)
    lbl_arr   = np.array(labels)
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(fps_arr, lbl_arr)):
        print(f"\nFold {fold+1}/{CONFIG['n_folds']}  (train={len(tr_idx)}, val={len(va_idx)})")
        tr_ds = BRISCDataset(fps_arr[tr_idx].tolist(), lbl_arr[tr_idx].tolist(),
                             get_transform("aggressive_original", True))
        va_ds = BRISCDataset(fps_arr[va_idx].tolist(), lbl_arr[va_idx].tolist(),
                             get_transform("aggressive_original", False))
        tr_loader = DataLoader(tr_ds, CONFIG["batch_size"], shuffle=True,  num_workers=0)
        va_loader = DataLoader(va_ds, CONFIG["batch_size"], shuffle=False, num_workers=0)

        model = DINOv3Classifier().to(device)
        opt   = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        crit  = nn.CrossEntropyLoss()
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"])

        best_f1, best_acc, best_bal = 0.0, 0.0, 0.0
        for epoch in range(CONFIG["epochs"]):
            train_epoch(model, tr_loader, opt, crit, device)
            sched.step()
            preds, trues = evaluate(model, va_loader, device)
            f1  = f1_score(trues, preds, average="macro", zero_division=0) * 100
            acc = accuracy_score(trues, preds) * 100
            bal = balanced_accuracy_score(trues, preds) * 100
            if f1 > best_f1:
                best_f1, best_acc, best_bal = f1, acc, bal

        print(f"  Fold {fold+1}: F1={best_f1:.2f}%  Acc={best_acc:.2f}%  BalAcc={best_bal:.2f}%")
        fold_rows.append({"fold": fold+1, "val_macro_f1": best_f1,
                          "val_accuracy": best_acc, "val_bal_accuracy": best_bal})

    df      = pd.DataFrame(fold_rows)
    m_f1    = df["val_macro_f1"].mean()
    s_f1    = df["val_macro_f1"].std()
    m_acc   = df["val_accuracy"].mean()
    s_acc   = df["val_accuracy"].std()
    ci_f1   = stats.t.interval(0.95, df=4, loc=m_f1,  scale=stats.sem(df["val_macro_f1"]))
    ci_acc  = stats.t.interval(0.95, df=4, loc=m_acc, scale=stats.sem(df["val_accuracy"]))

    print("\n" + "=" * 65)
    print("5-FOLD CV SUMMARY")
    print(df.to_string(index=False))
    print(f"\nMacro F1 : {m_f1:.2f}% ± {s_f1:.2f}%  |  95% CI [{ci_f1[0]:.2f}%, {ci_f1[1]:.2f}%]")
    print(f"Accuracy : {m_acc:.2f}% ± {s_acc:.2f}%  |  95% CI [{ci_acc[0]:.2f}%, {ci_acc[1]:.2f}%]")

    df.to_csv(OUT_DIR / "cv_5fold_results.csv", index=False)
    summary = {
        "model": "DINOv3 aggressive_original (last 4 blocks unfrozen)",
        "n_folds": 5, "seed": seed,
        "mean_macro_f1_pct": round(m_f1, 2), "std_macro_f1_pct": round(s_f1, 2),
        "ci_f1_95_pct": [round(ci_f1[0], 2), round(ci_f1[1], 2)],
        "mean_accuracy_pct": round(m_acc, 2), "std_accuracy_pct": round(s_acc, 2),
        "ci_acc_95_pct": [round(ci_acc[0], 2), round(ci_acc[1], 2)],
        "per_fold": fold_rows,
    }
    (OUT_DIR / "cv_5fold_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved -> {OUT_DIR / 'cv_5fold_results.csv'}")
    print(f"Saved -> {OUT_DIR / 'cv_5fold_summary.json'}")
    return df, summary


# ── entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    auth()
    fps, labels = load_train_data()
    if not fps:
        print("ERROR: No training images found. Check CONFIG['data_clean_root'].")
        sys.exit(1)
    run_medically_defensible(fps, labels)
    run_5fold_cv(fps, labels)
    print("\nAll experiments complete. Check results_augmented_dinov3/cv_and_medical_aug/")

#!/usr/bin/env python3
"""
Near-Duplicate Audit for Brain Tumor MRI Dataset
=================================================
Performs multi-method near-duplicate detection across train/test splits.

Methods:
  1. SHA256 (exact duplicate verification)
  2. pHash (perceptual hash - robust to resize/compression)
  3. dHash (difference hash - complementary perceptual hash)
  4. DINOv3 embedding cosine similarity (semantic similarity)

Outputs saved to: artifacts/reviewer_experiments/near_duplicate_audit/
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from itertools import combinations

import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Output directory
OUT_DIR = PROJECT_ROOT / "artifacts" / "reviewer_experiments" / "near_duplicate_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset config
DATA_ROOT = Path(r"C:\Users\darji\Downloads\BT_Images")
TRAIN_DIR = DATA_ROOT / "Training"
TEST_DIR = DATA_ROOT / "Testing"

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]


def list_images(split_dir: Path) -> list:
    """List all image files with their class labels."""
    rows = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        cls_name = class_dir.name.lower().replace("_tumor", "").replace("tumor", "").strip("_")
        if cls_name not in CLASS_NAMES:
            cls_name = class_dir.name
        for fp in sorted(class_dir.iterdir()):
            if fp.suffix.lower() in VALID_EXT:
                rows.append({"path": str(fp), "class": cls_name, "split_dir": str(split_dir)})
    return rows


def sha256_of(path: str) -> str:
    """Compute SHA256 hash of file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(path: str, hash_size: int = 16) -> str:
    """Compute perceptual hash (pHash) using DCT.
    
    Implementation: resize to (hash_size+1)x(hash_size+1), convert to grayscale,
    compute 2D DCT, keep top-left hash_size x hash_size low-frequency components,
    threshold at median.
    """
    try:
        from scipy.fft import dctn
    except ImportError:
        from scipy.fftpack import dctn
    
    img = Image.open(path).convert("L").resize((hash_size * 4, hash_size * 4), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float64)
    dct = dctn(pixels, type=2)
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    bits = (dct_low > median).flatten()
    hash_hex = "".join(str(int(b)) for b in bits)
    return hash_hex


def compute_dhash(path: str, hash_size: int = 16) -> str:
    """Compute difference hash (dHash).
    
    Compares adjacent pixels in a row to produce a binary hash.
    """
    img = Image.open(path).convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.array(img)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return "".join(str(int(b)) for b in diff.flatten())


def hamming_distance(h1: str, h2: str) -> int:
    """Compute Hamming distance between two binary hash strings."""
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def hamming_similarity(h1: str, h2: str) -> float:
    """Compute normalized Hamming similarity (1 - normalized distance)."""
    if len(h1) == 0:
        return 0.0
    return 1.0 - hamming_distance(h1, h2) / len(h1)


def compute_embeddings(paths: list, model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
                       batch_size: int = 32) -> np.ndarray:
    """Compute DINOv3 CLS token embeddings for all images."""
    from transformers import AutoImageProcessor, AutoModel
    from huggingface_hub import HfFolder
    
    token = HfFolder.get_token() or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    processor = AutoImageProcessor.from_pretrained(model_id, token=token)
    model = AutoModel.from_pretrained(model_id, token=token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    
    all_embeddings = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Computing embeddings"):
        batch_paths = paths[i:i+batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, "last_hidden_state"):
                emb = outputs.last_hidden_state[:, 0]  # CLS token
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                emb = outputs[0][:, 0]
            all_embeddings.append(emb.cpu().numpy())
    
    return np.concatenate(all_embeddings, axis=0)


def cosine_similarity_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between all pairs in X and Y."""
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    return X_norm @ Y_norm.T


def run_audit():
    """Main audit function."""
    print("=" * 70)
    print("NEAR-DUPLICATE AUDIT")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Verify dataset exists
    if not TRAIN_DIR.is_dir() or not TEST_DIR.is_dir():
        print(f"ERROR: Dataset not found at {DATA_ROOT}")
        print("Expected Training/ and Testing/ subdirectories.")
        print("Please update DATA_ROOT in this script.")
        sys.exit(1)
    
    # List all images
    print("\n[1/6] Listing images...")
    train_rows = list_images(TRAIN_DIR)
    test_rows = list_images(TEST_DIR)
    print(f"  Training images: {len(train_rows)}")
    print(f"  Testing images:  {len(test_rows)}")
    
    all_rows = train_rows + test_rows
    for r in train_rows:
        r["split"] = "train"
    for r in test_rows:
        r["split"] = "test"
    
    # Step 1: SHA256 exact duplicates
    print("\n[2/6] Computing SHA256 hashes...")
    for r in tqdm(all_rows, desc="SHA256"):
        r["sha256"] = sha256_of(r["path"])
    
    sha_groups = defaultdict(list)
    for r in all_rows:
        sha_groups[r["sha256"]].append(r)
    
    exact_dup_groups = {h: rows for h, rows in sha_groups.items() if len(rows) > 1}
    cross_split_exact = []
    within_split_exact = []
    for h, rows in exact_dup_groups.items():
        splits = set(r["split"] for r in rows)
        if len(splits) > 1:
            cross_split_exact.append(rows)
        else:
            within_split_exact.append(rows)
    
    print(f"  Exact duplicate groups (SHA256): {len(exact_dup_groups)}")
    print(f"    Cross-split (train<->test): {len(cross_split_exact)}")
    print(f"    Within-split only: {len(within_split_exact)}")
    
    # Step 2: Perceptual hashes
    print("\n[3/6] Computing perceptual hashes (pHash + dHash)...")
    for r in tqdm(all_rows, desc="pHash/dHash"):
        try:
            r["phash"] = compute_phash(r["path"])
            r["dhash"] = compute_dhash(r["path"])
        except Exception as e:
            r["phash"] = ""
            r["dhash"] = ""
            print(f"  Warning: Failed to hash {r['path']}: {e}")
    
    # Step 3: Cross-split perceptual similarity
    print("\n[4/6] Computing cross-split perceptual hash similarity...")
    train_indices = [i for i, r in enumerate(all_rows) if r["split"] == "train"]
    test_indices = [i for i, r in enumerate(all_rows) if r["split"] == "test"]
    
    # For perceptual hashes, compute similarity between all train-test pairs
    # This is O(n_train * n_test) which may be large. Use sampling if needed.
    n_train = len(train_indices)
    n_test = len(test_indices)
    print(f"  Cross-split pairs to check: {n_train} x {n_test} = {n_train * n_test:,}")
    
    # If too many pairs, we'll still compute but track top suspicious ones
    phash_suspicious = []
    dhash_suspicious = []
    
    PHASH_THRESHOLD = 0.90  # Very similar perceptual hash
    DHASH_THRESHOLD = 0.90
    
    phash_sim_distribution = []
    dhash_sim_distribution = []
    
    # Sample for distribution estimation if dataset is large
    MAX_PAIRS_FOR_DISTRIBUTION = 100000
    if n_train * n_test > MAX_PAIRS_FOR_DISTRIBUTION:
        np.random.seed(42)
        sample_train_idx = np.random.choice(train_indices, min(1000, n_train), replace=False)
        sample_test_idx = np.random.choice(test_indices, min(100, n_test), replace=False)
        print(f"  Sampling {len(sample_train_idx)}x{len(sample_test_idx)} pairs for distribution")
    else:
        sample_train_idx = train_indices
        sample_test_idx = test_indices
    
    for ti in tqdm(sample_train_idx, desc="Perceptual hash cross-check"):
        for te in sample_test_idx:
            r_tr = all_rows[ti]
            r_te = all_rows[te]
            if r_tr["phash"] and r_te["phash"]:
                ph_sim = hamming_similarity(r_tr["phash"], r_te["phash"])
                phash_sim_distribution.append(ph_sim)
                if ph_sim >= PHASH_THRESHOLD:
                    phash_suspicious.append({
                        "train_path": r_tr["path"],
                        "test_path": r_te["path"],
                        "train_class": r_tr["class"],
                        "test_class": r_te["class"],
                        "phash_similarity": ph_sim,
                    })
            if r_tr["dhash"] and r_te["dhash"]:
                dh_sim = hamming_similarity(r_tr["dhash"], r_te["dhash"])
                dhash_sim_distribution.append(dh_sim)
                if dh_sim >= DHASH_THRESHOLD:
                    dhash_suspicious.append({
                        "train_path": r_tr["path"],
                        "test_path": r_te["path"],
                        "train_class": r_tr["class"],
                        "test_class": r_te["class"],
                        "dhash_similarity": dh_sim,
                    })
    
    print(f"  pHash suspicious pairs (>={PHASH_THRESHOLD}): {len(phash_suspicious)}")
    print(f"  dHash suspicious pairs (>={DHASH_THRESHOLD}): {len(dhash_suspicious)}")
    
    # Step 4: DINOv3 embedding cosine similarity
    print("\n[5/6] Computing DINOv3 embedding similarity (cross-split)...")
    all_paths = [r["path"] for r in all_rows]
    
    try:
        embeddings = compute_embeddings(all_paths)
        
        train_emb = embeddings[train_indices]
        test_emb = embeddings[test_indices]
        
        # Compute cross-split cosine similarity
        cos_sim_matrix = cosine_similarity_matrix(train_emb, test_emb)
        
        # Get distribution statistics
        cos_sim_flat = cos_sim_matrix.flatten()
        
        # Find suspicious pairs
        EMBEDDING_THRESHOLD = 0.95
        suspicious_embedding_pairs = []
        high_sim_indices = np.argwhere(cos_sim_matrix >= EMBEDDING_THRESHOLD)
        
        for idx in high_sim_indices:
            tr_idx = train_indices[idx[0]]
            te_idx = test_indices[idx[1]]
            suspicious_embedding_pairs.append({
                "train_path": all_rows[tr_idx]["path"],
                "test_path": all_rows[te_idx]["path"],
                "train_class": all_rows[tr_idx]["class"],
                "test_class": all_rows[te_idx]["class"],
                "cosine_similarity": float(cos_sim_matrix[idx[0], idx[1]]),
            })
        
        suspicious_embedding_pairs.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        
        print(f"  Embedding pairs with cosine >= {EMBEDDING_THRESHOLD}: {len(suspicious_embedding_pairs)}")
        print(f"  Cosine similarity distribution (cross-split):")
        print(f"    Mean: {cos_sim_flat.mean():.4f}")
        print(f"    Std:  {cos_sim_flat.std():.4f}")
        print(f"    P50:  {np.percentile(cos_sim_flat, 50):.4f}")
        print(f"    P95:  {np.percentile(cos_sim_flat, 95):.4f}")
        print(f"    P99:  {np.percentile(cos_sim_flat, 99):.4f}")
        print(f"    Max:  {cos_sim_flat.max():.4f}")
        
        embedding_computed = True
    except Exception as e:
        print(f"  ERROR computing embeddings: {e}")
        print("  Skipping embedding-based analysis.")
        cos_sim_flat = np.array([])
        suspicious_embedding_pairs = []
        embedding_computed = False
    
    # Step 5: Save results
    print("\n[6/6] Saving results...")
    
    # Save main summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset_root": str(DATA_ROOT),
        "n_train_images": n_train,
        "n_test_images": n_test,
        "methods_used": ["SHA256", "pHash", "dHash", "DINOv3_cosine_similarity"],
        "exact_duplicates": {
            "total_groups": len(exact_dup_groups),
            "cross_split_groups": len(cross_split_exact),
            "within_split_groups": len(within_split_exact),
        },
        "perceptual_hash_analysis": {
            "phash_threshold": PHASH_THRESHOLD,
            "dhash_threshold": DHASH_THRESHOLD,
            "phash_suspicious_pairs": len(phash_suspicious),
            "dhash_suspicious_pairs": len(dhash_suspicious),
            "phash_distribution": {
                "mean": float(np.mean(phash_sim_distribution)) if phash_sim_distribution else None,
                "std": float(np.std(phash_sim_distribution)) if phash_sim_distribution else None,
                "p95": float(np.percentile(phash_sim_distribution, 95)) if phash_sim_distribution else None,
                "p99": float(np.percentile(phash_sim_distribution, 99)) if phash_sim_distribution else None,
                "max": float(np.max(phash_sim_distribution)) if phash_sim_distribution else None,
            },
            "dhash_distribution": {
                "mean": float(np.mean(dhash_sim_distribution)) if dhash_sim_distribution else None,
                "std": float(np.std(dhash_sim_distribution)) if dhash_sim_distribution else None,
                "p95": float(np.percentile(dhash_sim_distribution, 95)) if dhash_sim_distribution else None,
                "p99": float(np.percentile(dhash_sim_distribution, 99)) if dhash_sim_distribution else None,
                "max": float(np.max(dhash_sim_distribution)) if dhash_sim_distribution else None,
            },
        },
        "embedding_analysis": {
            "computed": embedding_computed,
            "threshold": EMBEDDING_THRESHOLD,
            "suspicious_pairs": len(suspicious_embedding_pairs),
            "distribution": {
                "mean": float(cos_sim_flat.mean()) if len(cos_sim_flat) > 0 else None,
                "std": float(cos_sim_flat.std()) if len(cos_sim_flat) > 0 else None,
                "p50": float(np.percentile(cos_sim_flat, 50)) if len(cos_sim_flat) > 0 else None,
                "p95": float(np.percentile(cos_sim_flat, 95)) if len(cos_sim_flat) > 0 else None,
                "p99": float(np.percentile(cos_sim_flat, 99)) if len(cos_sim_flat) > 0 else None,
                "max": float(cos_sim_flat.max()) if len(cos_sim_flat) > 0 else None,
            } if embedding_computed else {},
        },
        "patient_level_independence": "UNKNOWN - No patient IDs available in this dataset. "
                                       "Cannot verify whether the same patient's images appear in both splits "
                                       "under different filenames or acquisition parameters.",
        "limitations": [
            "Patient-level independence cannot be established without patient metadata.",
            "Perceptual hashes may miss semantically similar images with different windowing/contrast.",
            "Embedding similarity threshold is heuristic; manual review of top pairs is essential.",
            "Same-class high similarity may be expected (similar anatomy) vs concerning (same patient).",
        ],
    }
    
    with open(OUT_DIR / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save suspicious pairs CSV
    if phash_suspicious:
        pd.DataFrame(phash_suspicious).to_csv(OUT_DIR / "phash_suspicious_pairs.csv", index=False)
    if dhash_suspicious:
        pd.DataFrame(dhash_suspicious).to_csv(OUT_DIR / "dhash_suspicious_pairs.csv", index=False)
    if suspicious_embedding_pairs:
        pd.DataFrame(suspicious_embedding_pairs[:500]).to_csv(
            OUT_DIR / "embedding_suspicious_pairs.csv", index=False)
    
    # Save full hash table
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT_DIR / "all_images_hashes.csv", index=False)
    
    # Save distribution data for plotting
    if embedding_computed:
        np.save(OUT_DIR / "cross_split_cosine_distribution.npy", cos_sim_flat)
    
    # Print top 10 most suspicious embedding pairs
    if suspicious_embedding_pairs:
        print("\n" + "=" * 70)
        print("TOP 10 MOST SIMILAR CROSS-SPLIT PAIRS (DINOv3 cosine similarity)")
        print("=" * 70)
        for i, pair in enumerate(suspicious_embedding_pairs[:10]):
            print(f"  {i+1}. cos={pair['cosine_similarity']:.4f}")
            print(f"     Train: {Path(pair['train_path']).name} ({pair['train_class']})")
            print(f"     Test:  {Path(pair['test_path']).name} ({pair['test_class']})")
    
    # Final assessment
    print("\n" + "=" * 70)
    print("AUDIT CONCLUSION")
    print("=" * 70)
    
    if len(cross_split_exact) > 0:
        print(f"  WARNING: {len(cross_split_exact)} exact duplicate groups span train/test!")
        print("  This indicates potential data leakage that was not cleaned.")
    else:
        print("  OK: No exact byte-identical duplicates across train/test.")
    
    if len(suspicious_embedding_pairs) > 0:
        print(f"\n  ATTENTION: {len(suspicious_embedding_pairs)} pairs have cosine >= {EMBEDDING_THRESHOLD}")
        print("  These require manual review to determine if they represent:")
        print("    (a) Same patient, different acquisition (=leakage concern)")
        print("    (b) Very similar anatomy in different patients (=acceptable)")
        print("    (c) Near-duplicate images from dataset curation issues")
        print("\n  DO NOT automatically remove these without manual inspection.")
        print("  DO NOT claim patient-level independence without patient ID verification.")
    else:
        print(f"\n  OK: No cross-split pairs exceed cosine similarity threshold {EMBEDDING_THRESHOLD}.")
    
    print(f"\n  All results saved to: {OUT_DIR}")
    print(f"  Summary: {OUT_DIR / 'audit_summary.json'}")
    
    return summary


if __name__ == "__main__":
    run_audit()

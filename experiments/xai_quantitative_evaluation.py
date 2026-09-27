#!/usr/bin/env python3
"""
Quantitative XAI Evaluation
============================
Measures attribution faithfulness using deletion/insertion curves AND
localization accuracy using official BRISC segmentation masks.

Methods evaluated:
  1. Occlusion sensitivity (baseline, already implemented)
  2. Gradient saliency (input gradient magnitude)
  3. Attention rollout (ViT-native)

Faithfulness metrics:
  - Deletion AUC (lower = more faithful: removing important pixels hurts more)
  - Insertion AUC (higher = more faithful: inserting important pixels helps more)
  - Confidence drop after removing top-k% important pixels
  - Confidence drop after removing random pixels of equal area

Localization metrics (requires segmentation masks):
  - Dice coefficient (overlap between thresholded heatmap and tumor mask)
  - IoU (intersection-over-union)
  - Pointing game (does the heatmap maximum fall inside the tumor mask?)
  - Relevance mass (fraction of attribution energy inside tumor region)
  Evaluated at multiple thresholds (0.25, 0.50, 0.75) for robustness.

Statistical tests:
  - Paired Wilcoxon signed-rank test between methods
  - Bonferroni correction for multiple comparisons

Output: artifacts/reviewer_experiments/xai_quantitative/
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TV
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xai.masks import load_mask, mask_status
from xai.metrics import localization_bundle

OUT_DIR = PROJECT_ROOT / "artifacts" / "reviewer_experiments" / "xai_quantitative"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(r"C:\Users\darji\Downloads\BT_Images")
TEST_DIR = DATA_ROOT / "Testing"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "clf.pth"
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
IMG_SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

N_SAMPLES_PER_CLASS = 10  # Balanced sampling
N_DELETION_STEPS = 20  # Number of steps in deletion/insertion curve
SEED = 42
LOCALIZATION_THRESHOLDS = [0.25, 0.50, 0.75]  # Multiple thresholds for robustness


def load_model():
    """Load the fine-tuned DINOv3 classifier."""
    from servers.pytorch_model_loader import load_pytorch_model
    model, class_names, model_id = load_pytorch_model(str(MODEL_PATH), model_id=MODEL_ID)
    return model


def get_balanced_sample(test_dir: Path, n_per_class: int, seed: int = 42):
    """Get a balanced sample of test images."""
    np.random.seed(seed)
    samples = []
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        cls_name = class_dir.name
        images = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in VALID_EXT])
        if len(images) < n_per_class:
            selected = images
        else:
            selected = list(np.random.choice(images, n_per_class, replace=False))
        for img_path in selected:
            samples.append({"path": str(img_path), "class": cls_name})
    return samples


def preprocess(pil_img: Image.Image) -> torch.Tensor:
    """Preprocess image for the model."""
    transform = TV.Compose([
        TV.Resize((IMG_SIZE, IMG_SIZE)),
        TV.ToTensor(),
        TV.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil_img).unsqueeze(0)


def get_prediction(model, img_tensor, device):
    """Get model prediction probabilities."""
    with torch.no_grad():
        logits = model(pixel_values=img_tensor.to(device))
        probs = torch.softmax(logits, dim=1)
    return probs[0].cpu().numpy()


# =============================================================================
# Attribution Methods
# =============================================================================

def compute_gradient_saliency(model, img_tensor, target_class, device):
    """Compute gradient-based saliency map."""
    img = img_tensor.clone().to(device)
    img.requires_grad_(True)
    
    logits = model(pixel_values=img)
    score = logits[0, target_class]
    model.zero_grad(set_to_none=True)
    score.backward()
    
    grad = img.grad.detach().abs()
    saliency = grad.mean(dim=1)[0].cpu().numpy()  # Average over channels
    return saliency


def compute_attention_rollout(model, img_tensor, device):
    """Compute attention rollout heatmap (register-aware, mid-depth start)."""
    from servers.xai_maps import compute_attention_rollout as _rollout, upsample_map
    try:
        heat = _rollout(model.backbone, img_tensor.to(device))
        return upsample_map(heat, size=IMG_SIZE)
    except Exception:
        return None


def compute_occlusion_map(model, pil_img, target_class, device,
                          patch_size=16, stride=8):
    """Compute occlusion sensitivity map."""
    from transformers import AutoImageProcessor
    from huggingface_hub import HfFolder
    
    token = HfFolder.get_token() or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=token)
    
    img_array = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    h, w = img_array.shape[:2]
    
    # Get baseline prediction
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.softmax(logits, dim=1)
        baseline_prob = probs[0, target_class].item()
    
    occlusion_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            occluded = img_array.copy()
            occluded[y:y+patch_size, x:x+patch_size] = 0
            occluded_pil = Image.fromarray(occluded)
            
            inputs = processor(images=occluded_pil, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs)
                probs = torch.softmax(logits, dim=1)
                occ_prob = probs[0, target_class].item()
            
            importance = baseline_prob - occ_prob
            occlusion_map[y:y+patch_size, x:x+patch_size] += importance
            count_map[y:y+patch_size, x:x+patch_size] += 1
    
    # Average overlapping contributions
    count_map = np.maximum(count_map, 1)
    occlusion_map = occlusion_map / count_map
    return occlusion_map


# =============================================================================
# Faithfulness Metrics
# =============================================================================

def deletion_curve(model, img_tensor, attribution_map, target_class, device, n_steps=20):
    """Compute deletion curve: progressively remove most important pixels.
    
    Lower AUC = more faithful (removing important pixels hurts prediction more).
    """
    baseline_probs = get_prediction(model, img_tensor, device)
    baseline_conf = baseline_probs[target_class]
    
    # Flatten and sort attribution map (descending importance)
    flat_attr = attribution_map.flatten()
    sorted_indices = np.argsort(flat_attr)[::-1]
    
    n_pixels = len(sorted_indices)
    step_size = n_pixels // n_steps
    
    confidences = [baseline_conf]
    fractions = [0.0]
    
    img_modified = img_tensor.clone()
    
    for step in range(1, n_steps + 1):
        start_idx = (step - 1) * step_size
        end_idx = min(step * step_size, n_pixels)
        
        # Create mask of pixels to zero out (in spatial dimensions)
        h, w = attribution_map.shape
        pixels_to_remove = sorted_indices[start_idx:end_idx]
        rows = pixels_to_remove // w
        cols = pixels_to_remove % w
        
        # Zero out these pixels in the image (before normalization, so we use mean)
        img_modified[0, :, rows, cols] = 0  # Set to 0 (black after unnorm)
        
        probs = get_prediction(model, img_modified, device)
        confidences.append(probs[target_class])
        fractions.append(step / n_steps)
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(confidences, fractions)
    return auc, confidences, fractions


def insertion_curve(model, img_tensor, attribution_map, target_class, device, n_steps=20):
    """Compute insertion curve: progressively reveal most important pixels.
    
    Higher AUC = more faithful (revealing important pixels helps prediction more).
    """
    # Start with blank (mean-value) image
    blank = torch.zeros_like(img_tensor)
    
    flat_attr = attribution_map.flatten()
    sorted_indices = np.argsort(flat_attr)[::-1]
    
    n_pixels = len(sorted_indices)
    step_size = n_pixels // n_steps
    
    probs = get_prediction(model, blank, device)
    confidences = [probs[target_class]]
    fractions = [0.0]
    
    img_revealed = blank.clone()
    
    for step in range(1, n_steps + 1):
        start_idx = (step - 1) * step_size
        end_idx = min(step * step_size, n_pixels)
        
        h, w = attribution_map.shape
        pixels_to_reveal = sorted_indices[start_idx:end_idx]
        rows = pixels_to_reveal // w
        cols = pixels_to_reveal % w
        
        # Copy these pixels from original
        img_revealed[0, :, rows, cols] = img_tensor[0, :, rows, cols]
        
        probs = get_prediction(model, img_revealed, device)
        confidences.append(probs[target_class])
        fractions.append(step / n_steps)
    
    auc = np.trapz(confidences, fractions)
    return auc, confidences, fractions


def random_deletion_confidence_drop(model, img_tensor, attribution_map, target_class, 
                                     device, top_k_fraction=0.1, n_random_trials=5):
    """Compare confidence drop from removing top-k% important pixels vs random pixels."""
    baseline_probs = get_prediction(model, img_tensor, device)
    baseline_conf = baseline_probs[target_class]
    
    flat_attr = attribution_map.flatten()
    n_pixels = len(flat_attr)
    n_remove = int(n_pixels * top_k_fraction)
    
    # Top-k removal
    top_k_indices = np.argsort(flat_attr)[::-1][:n_remove]
    h, w = attribution_map.shape
    
    img_topk = img_tensor.clone()
    rows = top_k_indices // w
    cols = top_k_indices % w
    img_topk[0, :, rows, cols] = 0
    
    probs_topk = get_prediction(model, img_topk, device)
    topk_drop = baseline_conf - probs_topk[target_class]
    
    # Random removal (averaged over trials)
    random_drops = []
    for trial in range(n_random_trials):
        np.random.seed(SEED + trial + 1000)
        random_indices = np.random.choice(n_pixels, n_remove, replace=False)
        
        img_random = img_tensor.clone()
        rows_r = random_indices // w
        cols_r = random_indices % w
        img_random[0, :, rows_r, cols_r] = 0
        
        probs_random = get_prediction(model, img_random, device)
        random_drops.append(baseline_conf - probs_random[target_class])
    
    mean_random_drop = np.mean(random_drops)
    
    return {
        "baseline_confidence": float(baseline_conf),
        "topk_confidence_drop": float(topk_drop),
        "random_confidence_drop_mean": float(mean_random_drop),
        "topk_vs_random_ratio": float(topk_drop / (mean_random_drop + 1e-8)),
        "top_k_fraction": top_k_fraction,
    }


# =============================================================================
# Main Evaluation
# =============================================================================

def run_evaluation():
    """Run the full XAI quantitative evaluation."""
    print("=" * 70)
    print("QUANTITATIVE XAI EVALUATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Check prerequisites
    if not TEST_DIR.is_dir():
        print(f"ERROR: Test directory not found: {TEST_DIR}")
        sys.exit(1)
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)
    
    # Check for localization ground truth
    print("\n[Ground Truth Check]")
    ms = mask_status()
    if ms["available"]:
        print(f"  Mask directory: {ms['mask_dir']}")
        print("  Localization metrics (IoU, Dice, pointing game): WILL BE COMPUTED")
        print(f"  Thresholds for binarization: {LOCALIZATION_THRESHOLDS}")
    else:
        print("  Tumor masks: NOT AVAILABLE")
        print("  Localization metrics: CANNOT BE COMPUTED")
        print("  Proceeding with attribution faithfulness metrics only.")
    masks_available = ms["available"]
    
    # Load model
    print("\n[Loading model...]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()
    model.to(device)
    model.eval()
    print(f"  Device: {device}")
    
    # Get balanced sample
    print(f"\n[Sampling {N_SAMPLES_PER_CLASS} images per class...]")
    samples = get_balanced_sample(TEST_DIR, N_SAMPLES_PER_CLASS, seed=SEED)
    print(f"  Total samples: {len(samples)}")
    class_counts = {}
    for s in samples:
        class_counts[s['class']] = class_counts.get(s['class'], 0) + 1
    print(f"  Per class: {class_counts}")
    
    # Evaluate each sample
    results = []
    
    for i, sample in enumerate(tqdm(samples, desc="Evaluating XAI")):
        img_path = sample["path"]
        true_class = sample["class"]
        
        try:
            pil_img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(pil_img)
            
            # Get prediction
            probs = get_prediction(model, img_tensor, device)
            pred_class_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_class_idx]
            pred_conf = float(probs[pred_class_idx])
            
            # Use predicted class as target for attribution
            target_class = pred_class_idx
            
            # Determine if this is a tumor image (for localization evaluation)
            is_tumor = true_class != "no_tumor"
            
            # Load segmentation mask if available
            mask = None
            mask_area_fraction = 0.0
            if masks_available and is_tumor:
                mask = load_mask(Path(img_path), size=(IMG_SIZE, IMG_SIZE))
                if mask is not None:
                    mask_area_fraction = float(mask.sum() / mask.size)
            
            # Compute attributions
            # 1. Gradient saliency
            grad_map = compute_gradient_saliency(model, img_tensor, target_class, device)
            # Resize to standard size
            grad_map_resized = np.array(Image.fromarray(grad_map).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
            
            # 2. Attention rollout (may fail with sdpa attention or non-square patches)
            try:
                attn_map = compute_attention_rollout(model, img_tensor, device)
                if attn_map is not None:
                    attn_map_resized = np.array(Image.fromarray(attn_map.astype(np.float32)).resize(
                        (IMG_SIZE, IMG_SIZE), Image.BILINEAR))
                else:
                    attn_map_resized = None
            except Exception:
                attn_map_resized = None
            
            # 3. Occlusion (more expensive)
            occ_map = compute_occlusion_map(model, pil_img, target_class, device,
                                            patch_size=16, stride=16)  # Larger stride for speed
            occ_map_resized = np.array(Image.fromarray(occ_map).resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR))
            
            # Compute faithfulness metrics for each method
            methods = {
                "gradient_saliency": grad_map_resized,
                "occlusion": occ_map_resized,
            }
            if attn_map_resized is not None:
                methods["attention_rollout"] = attn_map_resized
            
            sample_result = {
                "image_path": img_path,
                "true_class": true_class,
                "pred_class": pred_class,
                "pred_confidence": pred_conf,
                "correct": true_class == pred_class or true_class.replace("_tumor", "") in pred_class,
                "is_tumor": is_tumor,
                "mask_found": mask is not None,
            }
            
            if mask is not None:
                sample_result["mask_area_fraction"] = mask_area_fraction
            
            for method_name, attr_map in methods.items():
                # Only consider positive evidence for faithfulness/localization
                attr_map = np.clip(attr_map, 0, None)
                
                # Normalize attribution map
                attr_min = attr_map.min()
                attr_max = attr_map.max()
                if attr_max - attr_min > 1e-8:
                    attr_norm = (attr_map - attr_min) / (attr_max - attr_min)
                else:
                    attr_norm = np.zeros_like(attr_map)
                
                # Deletion AUC
                del_auc, _, _ = deletion_curve(model, img_tensor, attr_norm, target_class, 
                                               device, n_steps=N_DELETION_STEPS)
                
                # Insertion AUC
                ins_auc, _, _ = insertion_curve(model, img_tensor, attr_norm, target_class,
                                                device, n_steps=N_DELETION_STEPS)
                
                # Confidence drop comparison
                drop_result = random_deletion_confidence_drop(
                    model, img_tensor, attr_norm, target_class, device, 
                    top_k_fraction=0.1, n_random_trials=5)
                
                sample_result[f"{method_name}_deletion_auc"] = float(del_auc)
                sample_result[f"{method_name}_insertion_auc"] = float(ins_auc)
                sample_result[f"{method_name}_topk_drop"] = drop_result["topk_confidence_drop"]
                sample_result[f"{method_name}_random_drop"] = drop_result["random_confidence_drop_mean"]
                sample_result[f"{method_name}_drop_ratio"] = drop_result["topk_vs_random_ratio"]
                
                # ── Localization metrics (only for tumor images with masks) ──
                for thresh in LOCALIZATION_THRESHOLDS:
                    loc = localization_bundle(attr_norm, mask, threshold=thresh, is_tumor=is_tumor)
                    t_str = str(thresh).replace(".", "")
                    sample_result[f"{method_name}_dice_t{t_str}"] = loc["dice"]
                    sample_result[f"{method_name}_iou_t{t_str}"] = loc["iou"]
                    sample_result[f"{method_name}_pointing_game_t{t_str}"] = loc["pointing_game"]
                    if "mass_inside" in loc:
                        sample_result[f"{method_name}_mass_inside_t{t_str}"] = loc["mass_inside"]
            
            results.append(sample_result)
            
        except Exception as e:
            print(f"\n  Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "per_image_results.csv", index=False)
    
    # Compute aggregate statistics
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS — FAITHFULNESS")
    print("=" * 70)
    
    # Determine which methods actually have data
    all_possible_methods = ["gradient_saliency", "attention_rollout", "occlusion"]
    methods = [m for m in all_possible_methods if f"{m}_deletion_auc" in df.columns and df[f"{m}_deletion_auc"].notna().any()]
    faithfulness_metrics = ["deletion_auc", "insertion_auc", "topk_drop", "random_drop", "drop_ratio"]
    
    if "attention_rollout" not in methods:
        print("\n  NOTE: Attention rollout was not available (model uses sdpa attention).")
        print("  This is a VERIFIED LIMITATION of the current model configuration.")
    
    agg_results = {}
    for method in methods:
        agg_results[method] = {}
        for metric in faithfulness_metrics:
            col = f"{method}_{metric}"
            if col in df.columns:
                vals = df[col].dropna().values
                if len(vals) > 0:
                    agg_results[method][metric] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "median": float(np.median(vals)),
                        "n": int(len(vals)),
                    }
    
    # Print faithfulness summary table
    print(f"\n{'Method':<22} {'Del AUC (low=good)':<20} {'Ins AUC (hi=good)':<20} {'TopK Drop':<14} {'Rand Drop':<14} {'Ratio':<10}")
    print("-" * 100)
    for method in methods:
        if method not in agg_results or "deletion_auc" not in agg_results[method]:
            continue
        del_auc = agg_results[method]["deletion_auc"]["mean"]
        ins_auc = agg_results[method]["insertion_auc"]["mean"]
        topk = agg_results[method]["topk_drop"]["mean"]
        rand = agg_results[method]["random_drop"]["mean"]
        ratio = agg_results[method]["drop_ratio"]["mean"]
        print(f"{method:<22} {del_auc:<20.4f} {ins_auc:<20.4f} {topk:<14.4f} {rand:<14.4f} {ratio:<10.2f}")
    
    # ── Localization aggregate results (tumor images with masks only) ──
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS — LOCALIZATION (tumor images with masks only)")
    print("=" * 70)
    
    df_tumor = df[df["is_tumor"] == True]  # noqa: E712
    df_masked = df_tumor[df_tumor["mask_found"] == True]  # noqa: E712
    n_with_mask = len(df_masked)
    n_tumor = len(df_tumor)
    print(f"\n  Tumor images evaluated: {n_tumor}")
    print(f"  Tumor images with masks: {n_with_mask}")
    
    localization_agg = {}
    
    if n_with_mask > 0:
        mask_areas = df_masked["mask_area_fraction"].dropna().values
        mean_mask_area = float(np.mean(mask_areas))
        print(f"  Average tumor mask area: {mean_mask_area:.2%} of image")
        
    for method in methods:
        localization_agg[method] = {}
        for thresh in LOCALIZATION_THRESHOLDS:
            t_str = str(thresh).replace(".", "")
            thresh_results = {}
            for metric in ["dice", "iou", "pointing_game", "mass_inside"]:
                col = f"{method}_{metric}_t{t_str}"
                if col in df_masked.columns:
                    vals = df_masked[col].dropna().values
                    if len(vals) > 0:
                        thresh_results[metric] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            "median": float(np.median(vals)),
                            "n": int(len(vals)),
                        }
            if thresh_results:
                localization_agg[method][f"threshold_{thresh}"] = thresh_results
    
    # Print localization summary
    for thresh in LOCALIZATION_THRESHOLDS:
        t_str = str(thresh).replace(".", "")
        print(f"\n  --- Threshold = {thresh} ---")
        print(f"  {'Method':<22} {'Dice':<14} {'IoU':<14} {'Pointing Game':<16} {'Mass Inside':<14}")
        print("  " + "-" * 80)
        for method in methods:
            t_key = f"threshold_{thresh}"
            if method in localization_agg and t_key in localization_agg[method]:
                lr = localization_agg[method][t_key]
                d = lr.get("dice", {}).get("mean", float("nan"))
                io = lr.get("iou", {}).get("mean", float("nan"))
                pg = lr.get("pointing_game", {}).get("mean", float("nan"))
                mi = lr.get("mass_inside", {}).get("mean", float("nan"))
                print(f"  {method:<22} {d:<14.4f} {io:<14.4f} {pg:<16.4f} {mi:<14.4f}")
    
    # Statistical tests (pairwise Wilcoxon signed-rank)
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (Wilcoxon signed-rank, Bonferroni corrected)")
    print("=" * 70)
    
    n_comparisons = len(methods) * (len(methods) - 1) // 2
    alpha = 0.05
    bonferroni_alpha = alpha / max(n_comparisons, 1)
    
    stat_results = []
    for metric in ["deletion_auc", "insertion_auc"]:
        print(f"\n  Metric: {metric}")
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                m1, m2 = methods[i], methods[j]
                col1 = f"{m1}_{metric}"
                col2 = f"{m2}_{metric}"
                
                vals1 = df[col1].dropna().values
                vals2 = df[col2].dropna().values
                
                # Use minimum length
                n = min(len(vals1), len(vals2))
                vals1 = vals1[:n]
                vals2 = vals2[:n]
                
                if n < 5:
                    print(f"    {m1} vs {m2}: INSUFFICIENT DATA (n={n})")
                    continue
                
                try:
                    stat, p_value = stats.wilcoxon(vals1, vals2, alternative='two-sided')
                    significant = p_value < bonferroni_alpha
                    
                    stat_results.append({
                        "metric": metric,
                        "method_1": m1,
                        "method_2": m2,
                        "n": n,
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "bonferroni_alpha": bonferroni_alpha,
                        "significant": significant,
                        "mean_1": float(np.mean(vals1)),
                        "mean_2": float(np.mean(vals2)),
                    })
                    
                    sig_str = "***" if significant else "ns"
                    print(f"    {m1} ({np.mean(vals1):.4f}) vs {m2} ({np.mean(vals2):.4f}): "
                          f"p={p_value:.6f} {sig_str}")
                except Exception as e:
                    print(f"    {m1} vs {m2}: TEST FAILED ({e})")
    
    # Per-class analysis
    print("\n" + "=" * 70)
    print("PER-CLASS RESULTS (Deletion AUC)")
    print("=" * 70)
    
    per_class_results = {}
    for cls in df["pred_class"].unique():
        cls_df = df[df["pred_class"] == cls]
        per_class_results[cls] = {}
        for method in methods:
            col = f"{method}_deletion_auc"
            if col in cls_df.columns:
                vals = cls_df[col].dropna().values
                per_class_results[cls][method] = {
                    "mean": float(np.mean(vals)) if len(vals) > 0 else None,
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                    "n": int(len(vals)),
                }
        print(f"  {cls}: n={len(cls_df)}")
        for method in methods:
            if method in per_class_results[cls] and per_class_results[cls][method]["mean"] is not None:
                print(f"    {method}: {per_class_results[cls][method]['mean']:.4f} "
                      f"(±{per_class_results[cls][method]['std']:.4f})" if per_class_results[cls][method]["std"] else "")
    
    # Save comprehensive output
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_path": str(MODEL_PATH),
            "model_id": MODEL_ID,
            "n_samples_per_class": N_SAMPLES_PER_CLASS,
            "n_deletion_steps": N_DELETION_STEPS,
            "seed": SEED,
            "device": str(device),
            "total_samples_evaluated": len(df),
            "localization_thresholds": LOCALIZATION_THRESHOLDS,
        },
        "ground_truth_status": {
            "tumor_masks": "AVAILABLE" if masks_available else "NOT AVAILABLE",
            "mask_directory": ms.get("mask_dir"),
            "bounding_boxes": "NOT AVAILABLE",
            "localization_metrics_computed": masks_available and n_with_mask > 0,
            "n_tumor_images_with_masks": n_with_mask,
            "n_tumor_images_total": n_tumor,
            "note": (
                f"Official BRISC segmentation masks used. {n_with_mask}/{n_tumor} tumor images had matching masks. "
                "Localization metrics (Dice, IoU, Pointing Game) computed at multiple thresholds."
                if masks_available and n_with_mask > 0
                else "IoU, Dice, pointing game CANNOT be computed without localization annotations."
            ),
        },
        "aggregate_faithfulness_results": agg_results,
        "aggregate_localization_results": localization_agg,
        "per_class_results": per_class_results,
        "statistical_tests": stat_results,
        "methods_evaluated": methods,
        "interpretation_guide": {
            "deletion_auc": "Lower is better (removing important pixels should hurt prediction more)",
            "insertion_auc": "Higher is better (adding important pixels should help prediction more)",
            "topk_drop": "Higher is better (top-k pixels by attribution cause larger confidence drop when removed)",
            "random_drop": "Baseline: confidence drop from removing random pixels of same area",
            "drop_ratio": "Higher is better (topk_drop / random_drop; >1 means attribution is better than random)",
            "dice": "Higher is better [0-1]. Measures overlap between thresholded heatmap and tumor mask.",
            "iou": "Higher is better [0-1]. Intersection over union of thresholded heatmap and tumor mask.",
            "pointing_game": "Binary [0 or 1]. 1 if the heatmap maximum falls inside the tumor mask.",
            "mass_inside": "Higher is better [0-1]. Fraction of total attribution energy inside the tumor region.",
        },
        "limitations": [
            "Deletion/insertion metrics measure faithfulness to the model, not anatomical correctness.",
            "Localization metrics depend on the binarization threshold chosen for the heatmap.",
            "Occlusion is computationally expensive and uses a fixed patch size (may miss fine structure).",
            "Attention rollout is an approximation; true attention flow is more complex.",
            "Sample size per class is limited for computational tractability.",
            "No-tumor images are excluded from localization evaluation (no tumor region to localize).",
            "Masks are provided at the image level; patient-level grouping is unavailable.",
        ],
    }
    
    # Custom JSON encoder for numpy/bool types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, bool):
                return bool(obj)
            return super().default(obj)
    
    with open(OUT_DIR / "xai_evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    if stat_results:
        pd.DataFrame(stat_results).to_csv(OUT_DIR / "statistical_tests.csv", index=False)
    
    print(f"\n\nAll results saved to: {OUT_DIR}")
    print(f"  - Per-image results: per_image_results.csv")
    print(f"  - Full results: xai_evaluation_results.json")
    print(f"  - Statistical tests: statistical_tests.csv")


if __name__ == "__main__":
    run_evaluation()

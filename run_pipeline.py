#!/usr/bin/env python3
"""
Unified Brain Tumor Classification Pipeline
Consolidates Modeler, Explainer, and Reporter agents into a single file.
No MCP servers - direct function calls only.
"""
import os
import json
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as TV
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
from tqdm import tqdm
import math
import base64
import random

# Sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Transformers & HF
from transformers import AutoImageProcessor, AutoModel
from huggingface_hub import HfFolder

# Lucent
from lucent.optvis import render, param, objectives, transform
from lucent.model_utils import get_model_layers

# Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# OpenAI
from openai import OpenAI

# ============================================================================
# Configuration & Paths
# ============================================================================

def get_hf_token():
    """Get Hugging Face token from environment or HfFolder."""
    token = HfFolder.get_token() or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "HUGGINGFACE_HUB_TOKEN not found. "
            "Please set it in config/api_config.json under api_keys.huggingface.token"
        )
    return token

def load_config():
    """Load configuration from files."""
    cfg_path = Path("config/api_config.json")
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        api = cfg.get("api_keys", {})
        openai_key = api.get("openai", {}).get("key")
        hf_token = api.get("huggingface", {}).get("token")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if hf_token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
            # Also set it in HfFolder for transformers library
            HfFolder.save_token(hf_token)
    
    task_cfg_path = Path("config/task_card.json")
    if task_cfg_path.exists():
        task_cfg = json.loads(task_cfg_path.read_text(encoding="utf-8"))
    else:
        task_cfg = {}
    
    return task_cfg

# Artifact paths
ART = Path("artifacts")
FEATS = ART / "feats"
MODELS = ART / "models"
METRICS = ART / "metrics"
EXPLAIN = ART / "explain"
O1 = EXPLAIN / "o1"
REPORTS = EXPLAIN / "reports"

for p in [FEATS, MODELS, METRICS, EXPLAIN, O1, REPORTS]:
    p.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PyTorch Model Loader
# ============================================================================

class DINOv3Classifier(nn.Module):
    """DINOv3 Classifier with configurable unfreezing and dropout."""
    def __init__(self, backbone: nn.Module, hidden_dim: int, num_classes: int,
                 unfreeze_last_n: int = 0, dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        if unfreeze_last_n > 0:
            if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
                layers = self.backbone.encoder.layer
            elif hasattr(self.backbone, 'layer'):
                layers = self.backbone.layer
            elif hasattr(self.backbone, 'blocks'):
                layers = self.backbone.blocks
            else:
                for param in self.backbone.parameters():
                    param.requires_grad = True
                layers = None
            
            if layers is not None:
                num_layers = len(layers)
                start_layer = max(0, num_layers - unfreeze_last_n)
                for i in range(start_layer, num_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True
        
        # Matches the head trained in dinov3_augmented_finetune_experiment.ipynb:
        # Linear(hidden_dim -> 256) -> GELU -> Dropout -> Linear(256 -> num_classes).
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x=None, **kwargs):
        if x is not None:
            if isinstance(x, dict):
                outputs = self.backbone(**x)
            else:
                outputs = self.backbone(x)
        else:
            outputs = self.backbone(**kwargs)
        
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state[:, 0]
        elif hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output
        else:
            features = outputs[0][:, 0]
        return self.classifier(features)

def load_pytorch_model(model_path: str, model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m", num_classes: int = 4):
    """Load a PyTorch model from clf.pth file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_token = get_hf_token()
    backbone = AutoModel.from_pretrained(model_id, token=hf_token)
    
    if hasattr(backbone.config, 'hidden_size'):
        hidden_dim = backbone.config.hidden_size
    else:
        hidden_dim = 768
    
    model = DINOv3Classifier(backbone, hidden_dim, num_classes, unfreeze_last_n=0, dropout=0.0)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    # Two known key-naming mismatches are remapped here:
    # 1) Checkpoints from dinov3_augmented_finetune_experiment.ipynb store the head
    #    under "head.net.*" (a DINOv3Classifier.head that is an MLPHead with a "net"
    #    submodule); remap those keys onto this module's "classifier.*" attribute.
    # 2) transformers-version drift: some transformers versions nest DINOv3's
    #    transformer blocks under "backbone.model.layer.*", others expose them
    #    directly as "backbone.layer.*" (embeddings.* and norm.* are unaffected).
    #    Detect which naming the *current* backbone actually uses and remap the
    #    checkpoint to match, so this keeps working across transformers versions.
    current_backbone_keys = set(model.backbone.state_dict().keys())
    uses_nested_model_layer = any(k.startswith("layer.") for k in current_backbone_keys) is False and \
        any(k.startswith("model.layer.") for k in current_backbone_keys)

    remapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("head.net."):
            key = "classifier." + key[len("head.net."):]
        elif key.startswith("backbone.model.layer.") and not uses_nested_model_layer:
            key = "backbone.layer." + key[len("backbone.model.layer."):]
        elif key.startswith("backbone.layer.") and uses_nested_model_layer:
            key = "backbone.model.layer." + key[len("backbone.layer."):]
        remapped_state_dict[key] = value
    model.load_state_dict(remapped_state_dict)
    model.eval()
    model.to(device)
    
    class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    return model, class_names, model_id

# ============================================================================
# Modeler Agent - Feature Extraction, Training, Evaluation
# ============================================================================

class ModelerAgent:
    """Handles feature extraction, model training, and evaluation."""
    
    @staticmethod
    def load_hf_backbone(model_id: str, train_mode=False):
        hf_token = get_hf_token()
        processor = AutoImageProcessor.from_pretrained(model_id, token=hf_token)
        model = AutoModel.from_pretrained(model_id, token=hf_token)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        if train_mode:
            model.train()
            for param in model.parameters():
                param.requires_grad = True
        else:
            model.eval()
        return model, processor
    
    @staticmethod
    class PathImageFolder(tv.datasets.ImageFolder):
        def __getitem__(self, idx):
            path, target = self.samples[idx]
            img = self.loader(path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, target, path
    
    @staticmethod
    def make_loader(root_dir: str, size: int, batch: int = 64, shuffle: bool = False, num_workers: int = 0):
        MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        tfm = tv.transforms.Compose([
            tv.transforms.Resize((size, size), antialias=True),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(MEAN, STD),
        ])
        ds = ModelerAgent.PathImageFolder(root_dir, transform=tfm)
        ld = DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=num_workers, pin_memory=False)
        return ld, ds
    
    @staticmethod
    @torch.no_grad()
    def embed_paths(backbone, processor, paths: List[str], device: str):
        pil_batch = [Image.open(p).convert("RGB") for p in paths]
        inputs = processor(images=pil_batch, return_tensors="pt").to(device)
        out = backbone(**inputs)
        z = getattr(out, "pooler_output", None)
        if z is None:
            z = out.last_hidden_state[:, 0]
        return z
    
    @staticmethod
    @torch.no_grad()
    def embed_loader(backbone, processor, loader, device: str):
        feats, labels, paths_all = [], [], []
        for _xb, yb, pb in tqdm(loader, desc="Embedding"):
            z = ModelerAgent.embed_paths(backbone, processor, list(pb), device)
            feats.append(z.detach().cpu().numpy())
            labels.append(yb.numpy())
            paths_all += list(pb)
        return np.concatenate(feats), np.concatenate(labels), paths_all
    
    @staticmethod
    def extract_features(data_root: str, model_id: str, size: int = 224,
                        out_train_npz: str = None, out_test_npz: str = None):
        """Extract features from training and test images."""
        data_root = Path(data_root)
        train_dir = data_root / "Training"
        test_dir = data_root / "Testing"
        assert train_dir.is_dir() and test_dir.is_dir(), "Expected Training/ and Testing/ folders"
        
        if out_train_npz is None:
            out_train_npz = str(FEATS / "train.npz")
        if out_test_npz is None:
            out_test_npz = str(FEATS / "test.npz")
        
        backbone, processor = ModelerAgent.load_hf_backbone(model_id)
        device = next(backbone.parameters()).device.type
        
        tr_ld, tr_ds = ModelerAgent.make_loader(str(train_dir), size=size, shuffle=False)
        te_ld, te_ds = ModelerAgent.make_loader(str(test_dir), size=size, shuffle=False)
        
        idx_to_class = {v: k for k, v in tr_ds.class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        
        X_tr, y_tr, tr_paths = ModelerAgent.embed_loader(backbone, processor, tr_ld, device)
        X_te, y_te, te_paths = ModelerAgent.embed_loader(backbone, processor, te_ld, device)
        
        Path(out_train_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_train_npz,
                 X=X_tr.astype("float32"),
                 y=y_tr.astype("int64"),
                 paths=np.array(tr_paths, dtype="U"),
                 class_names=np.array(class_names, dtype="U"),
                 model_id=np.array([model_id], dtype="U"))
        
        Path(out_test_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_test_npz,
                 X=X_te.astype("float32"),
                 y=y_te.astype("int64"),
                 paths=np.array(te_paths, dtype="U"),
                 class_names=np.array(class_names, dtype="U"),
                 model_id=np.array([model_id], dtype="U"))
        
        return {"ok": True, "train": out_train_npz, "test": out_test_npz, "classes": class_names}
    
    @staticmethod
    def train_probe(train_npz: str, out_pkl: str = None):
        """Train a linear probe classifier."""
        if out_pkl is None:
            out_pkl = str(MODELS / "clf.pkl")
        
        d = np.load(train_npz, allow_pickle=True)
        X_tr, y_tr = d["X"], d["y"]
        class_names = d["class_names"].tolist()
        model_id = d["model_id"].tolist()[0]
        
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 solver="lbfgs", multi_class="multinomial", n_jobs=-1)
        clf.fit(X_tr, y_tr)
        
        Path(out_pkl).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": clf, "class_names": class_names, "model_id": model_id}, out_pkl)
        return {"ok": True, "pkl": out_pkl, "classes": class_names}
    
    @staticmethod
    def eval_probe(test_npz: str, pkl: str, out_json: str = None):
        """Evaluate classifier on test set."""
        if out_json is None:
            out_json = str(METRICS / "metrics.json")
        
        d = np.load(test_npz, allow_pickle=True)
        X_te, y_te = d["X"], d["y"]
        paths = d["paths"].tolist() if "paths" in d else None
        
        pkl_path = Path(pkl)
        
        # Check if it's a PyTorch model (.pth)
        if pkl_path.suffix == '.pth' or 'pth' in str(pkl_path).lower():
            test_model_id = d["model_id"].tolist()[0] if "model_id" in d else "facebook/dinov3-vitb16-pretrain-lvd1689m"
            test_class_names = d["class_names"].tolist() if "class_names" in d else ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
            num_classes = len(test_class_names)
            
            model, class_names, model_id = load_pytorch_model(str(pkl_path), model_id=test_model_id, num_classes=num_classes)
            device = next(model.parameters()).device
            
            hf_token = get_hf_token()
            processor = AutoImageProcessor.from_pretrained(model_id, token=hf_token)
            
            y_hat = []
            if paths:
                for path in tqdm(paths, desc="Evaluating"):
                    img = Image.open(path).convert('RGB')
                    inputs = processor(images=img, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        logits = model(**inputs)
                        pred = logits.argmax(dim=1).cpu().item()
                        y_hat.append(pred)
            else:
                raise ValueError("PyTorch model requires image paths, but paths not found in test.npz")
            
            y_hat = np.array(y_hat)
        else:
            meta = joblib.load(pkl)
            clf = meta["clf"]
            class_names = meta["class_names"]
            y_hat = clf.predict(X_te)
        
        acc = float(accuracy_score(y_te, y_hat))
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_te, y_hat, average="weighted", zero_division=0)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_te, y_hat, average="macro", zero_division=0)
        cm = confusion_matrix(y_te, y_hat).tolist()
        report = classification_report(y_te, y_hat, target_names=class_names, zero_division=0, output_dict=True)
        
        metrics = {
            "accuracy": acc,
            "precision_weighted": float(prec_w),
            "recall_weighted": float(rec_w),
            "f1_weighted": float(f1_w),
            "precision_macro": float(prec_m),
            "recall_macro": float(rec_m),
            "f1_macro": float(f1_m),
            "confusion_matrix": cm,
            "classification_report": report,
            "classes": class_names
        }
        
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(metrics, indent=2))
        return {"ok": True, "metrics": out_json, "summary": {"acc": acc, "f1_macro": f1_m}}

# ============================================================================
# Explainer Agent - XAI Visualizations
# ============================================================================

class ExplainerAgent:
    """Handles explainability visualizations."""
    
    # ... (continuing in next part due to length)
    
    @staticmethod
    def load_hf_backbone(model_id: str):
        token = get_hf_token()
        processor = AutoImageProcessor.from_pretrained(model_id, token=token)
        model = AutoModel.from_pretrained(model_id, token=token).eval().to("cuda" if torch.cuda.is_available() else "cpu")
        return model, processor
    
    @staticmethod
    def _load_probe(pkl_path: str):
        """Load classifier - supports both .pkl (sklearn) and .pth (PyTorch) formats."""
        pkl_path_obj = Path(pkl_path)
        
        if pkl_path_obj.suffix == '.pth' or 'pth' in pkl_path.lower():
            model, class_names, model_id = load_pytorch_model(str(pkl_path_obj))
            class PyTorchClassifierWrapper:
                def __init__(self, model, class_names, model_id):
                    self.model = model
                    self.class_names = class_names
                    self.model_id = model_id
                    self.device = next(model.parameters()).device
            clf = PyTorchClassifierWrapper(model, class_names, model_id)
            return clf, class_names, model_id
        else:
            meta = joblib.load(pkl_path)
            return meta["clf"], meta["class_names"], meta["model_id"]
    
    @staticmethod
    def lucent_prototypes(pkl: str, layer: str, n_prototypes: int = 3, iters: int = 512, out_dir: str = None):
        """Generate Lucent prototypes for each class."""
        if out_dir is None:
            out_dir = str(O1)
        
        clf, class_names, model_id = ExplainerAgent._load_probe(pkl)
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        
        # Simplified - would need full Lucent implementation
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return {"ok": True, "dir": out_dir, "classes": class_names}
    
    @staticmethod
    def explain_image(pkl: str, image_path: str, out_dir: str, **kwargs):
        """Generate comprehensive explanations for a single image."""
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        clf, class_names, model_id = ExplainerAgent._load_probe(pkl)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pil = Image.open(image_path).convert("RGB")
        
        # Get config from kwargs
        do_attention = kwargs.get("do_attention", False)
        do_gradients = kwargs.get("do_gradients", True)
        do_deepdream = kwargs.get("do_deepdream", True)
        do_probs = kwargs.get("do_probs", True)
        do_lucent = kwargs.get("do_lucent", True)
        do_occlusion = kwargs.get("do_occlusion", True)
        layer_suffix = kwargs.get("layer_suffix", "attention.o_proj")
        steps = kwargs.get("steps", 120)
        lr = kwargs.get("lr", 0.05)
        tv_weight = kwargs.get("tv_weight", 0.001)
        octaves = kwargs.get("octaves", True)
        num_octaves = kwargs.get("num_octaves", 2)
        octave_scale = kwargs.get("octave_scale", 1.35)
        jitter = kwargs.get("jitter", 6)
        lucent_iters = kwargs.get("lucent_iters", 256)
        occlusion_patch_size = kwargs.get("occlusion_patch_size", 16)
        occlusion_stride = kwargs.get("occlusion_stride", 8)
        
        # Prediction and get weight vector
        if hasattr(clf, 'model'):
            pytorch_model = clf.model
            hf_token = get_hf_token()
            processor = AutoImageProcessor.from_pretrained(model_id, token=hf_token)
            inputs = processor(images=pil, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = pytorch_model(**inputs)
                probs_tensor = torch.softmax(logits, dim=1)
                probs = probs_tensor[0].cpu().numpy()
            k = int(np.argmax(probs))
            pred_name = class_names[k]
            # Extract classifier weights
            classifier_layers = [m for m in pytorch_model.classifier.modules() if isinstance(m, nn.Linear)]
            if classifier_layers:
                last_layer = classifier_layers[-1]
                w_vec = last_layer.weight[k].detach().clone().to(device)
            else:
                raise ValueError("Could not extract classifier weights")
            backbone = pytorch_model.backbone
        else:
            backbone, processor = ExplainerAgent.load_hf_backbone(model_id)
            z = ExplainerAgent._embed_single(backbone, processor, pil, device)
            probs = clf.predict_proba(z.detach().cpu().numpy())[0]
            k = int(np.argmax(probs))
            pred_name = class_names[k]
            w_vec = torch.tensor(clf.coef_[k], dtype=torch.float32, device=device)
        
        results = {
            "prediction": {"index": k, "name": pred_name, "prob": float(probs[k])},
            "probs": {class_names[i]: float(p) for i, p in enumerate(probs)},
            "topk": [{"name": class_names[i], "prob": float(probs[i])} 
                     for i in np.argsort(probs)[::-1][:5]],
        }
        
        # Attention rollout
        attn_png = None
        if do_attention:
            try:
                heat = ExplainerAgent._attention_rollout(backbone, processor, pil, device)
                attn_png = str(Path(out_dir) / "attn_rollout.png")
                ExplainerAgent._overlay_heatmap(pil, heat, attn_png, alpha=0.45)
            except Exception as e:
                results["attention_error"] = f"{type(e).__name__}: {e}"
        
        # Gradient saliency
        grad_png = None
        if do_gradients:
            try:
                if hasattr(clf, 'model'):
                    g = ExplainerAgent._gradient_saliency(backbone, None, pil, pytorch_model=clf.model, class_idx=k)
                else:
                    g = ExplainerAgent._gradient_saliency(backbone, w_vec, pil)
                grad_png = str(Path(out_dir) / "grad_saliency.png")
                ExplainerAgent._overlay_heatmap(pil, g, grad_png, alpha=0.45)
            except Exception as e:
                results["grad_error"] = f"{type(e).__name__}: {e}"
        
        # DeepDream - Use fine-tuned classifier for tumor-relevant visualizations
        dream_png = None
        if do_deepdream:
            try:
                # Pass the full fine-tuned model to maximize classifier output
                full_model_for_dream = pytorch_model if hasattr(clf, 'model') else None
                img = ExplainerAgent.deepdream_direction_octaves(
                    image_path, backbone, w_vec, layer_suffix, steps, lr, jitter,
                    tv_weight, num_octaves if octaves else 1, octave_scale,
                    full_model=full_model_for_dream, class_idx=k
                )
                dream_png = str(Path(out_dir) / "dream.png")
                img.save(dream_png)
            except Exception as e:
                results["dream_error"] = f"{type(e).__name__}: {e}"
        
        # Lucent class direction - generate for ALL classes
        lucent_pngs = {}
        lucent_png = None  # Initialize for backward compatibility
        if do_lucent:
            try:
                # Get the classifier's last layer weights for all classes
                if hasattr(clf, 'model'):
                    classifier_layers = [m for m in pytorch_model.classifier.modules() if isinstance(m, nn.Linear)]
                    if classifier_layers:
                        last_layer = classifier_layers[-1]
                        # Get weights for all classes
                        all_class_weights = last_layer.weight.detach().clone().to(device)  # Shape: [4, 256]
                    else:
                        # Fallback: use w_vec for predicted class only
                        all_class_weights = w_vec.unsqueeze(0)  # [1, 768]
                else:
                    # Sklearn model - get all class weights
                    all_class_weights = torch.tensor(clf.coef_, dtype=torch.float32, device=device)  # [4, feature_dim]
                
                # Generate Lucent visualization for each class
                pytorch_model_for_lucent = pytorch_model if hasattr(clf, 'model') else None
                for class_idx, class_name in enumerate(class_names):
                    try:
                        w_vec_class = all_class_weights[class_idx]
                        lucent_png = str(Path(out_dir) / f"lucent_{class_name}.png")
                        ExplainerAgent._lucent_direction_for_class(
                            backbone, model_id, w_vec_class, lucent_iters, lucent_png, 
                            layer_name="classifier", pytorch_model=pytorch_model_for_lucent
                        )
                        lucent_pngs[class_name] = lucent_png
                        print(f"  Generated Lucent for {class_name}")
                    except Exception as e:
                        print(f"  Failed to generate Lucent for {class_name}: {e}")
                        lucent_pngs[class_name] = None
                
                # Also keep the original single lucent_png for backward compatibility
                lucent_png = lucent_pngs.get(pred_name, None)
            except Exception as e:
                results["lucent_error"] = f"{type(e).__name__}: {e}"
                import traceback
                print(f"Lucent error: {traceback.format_exc()}")
                lucent_png = None
        
        # Probability plot
        probs_png = None
        if do_probs:
            try:
                probs_png = str(Path(out_dir) / "probs.png")
                ExplainerAgent._save_prob_plot(class_names, np.array(probs), probs_png, topk=5)
            except Exception as e:
                results["probs_plot_error"] = f"{type(e).__name__}: {e}"
        
        # Occlusion map
        occlusion_png = None
        if do_occlusion:
            try:
                pytorch_model_for_occ = clf.model if hasattr(clf, 'model') else None
                occ_map = ExplainerAgent._occlusion_map(
                    backbone, clf, pil, processor, model_id,
                    occlusion_patch_size, occlusion_stride, pytorch_model_for_occ
                )
                occlusion_png = str(Path(out_dir) / "occlusion_map.png")
                ExplainerAgent._overlay_heatmap(pil, occ_map, occlusion_png, alpha=0.45)
            except Exception as e:
                results["occlusion_error"] = f"{type(e).__name__}: {e}"
        
        results.update({
            "ok": True,
            "attn_rollout_png": attn_png,
            "grad_saliency_png": grad_png,
            "dream_png": dream_png,
            "lucent_png": lucent_png,  # For backward compatibility (predicted class)
            "lucent_all_classes": lucent_pngs if do_lucent else {},  # All classes
            "probs_png": probs_png,
            "occlusion_map_png": occlusion_png
        })
        
        return results
    
    @staticmethod
    def _embed_single(backbone, processor, pil: Image.Image, device: str) -> torch.Tensor:
        inputs = processor(images=pil, return_tensors="pt").to(device)
        out = backbone(**inputs)
        z = getattr(out, "pooler_output", None)
        if z is None:
            z = out.last_hidden_state[:, 0]
        return z
    
    # Constants for DeepDream
    IM_SIZE = 224
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    
    @staticmethod
    def _overlay_heatmap(pil: Image.Image, heat: np.ndarray, out_png: str, alpha: float = 0.5, sharpen: bool = True):
        """Overlay heatmap on image."""
        from servers.xai_maps import overlay_heatmap
        overlay_heatmap(pil, heat, out_png, alpha=alpha, sharpen=sharpen)
    
    @staticmethod
    def _attention_rollout(backbone, processor, pil: Image.Image, device: str) -> np.ndarray:
        """Generate attention rollout heatmap (register-aware, mid-depth start)."""
        from servers.xai_maps import compute_attention_rollout, upsample_map
        inputs = processor(images=pil, return_tensors="pt").to(device)
        heat = compute_attention_rollout(backbone, inputs["pixel_values"])
        return upsample_map(heat, size=max(pil.size))
    
    @staticmethod
    def preprocess_for_vit(x: torch.Tensor, size: int = 224) -> torch.Tensor:
        """Preprocess tensor for ViT."""
        x = F.interpolate(x, size=(size,size), mode="bilinear", align_corners=False)
        return (x - ExplainerAgent.MEAN.to(x.device)) / ExplainerAgent.STD.to(x.device)
    
    @staticmethod
    def _gradient_saliency(backbone, w_vec: torch.Tensor, pil: Image.Image, pytorch_model=None, class_idx=None) -> np.ndarray:
        """Compute gradient saliency map."""
        device = next(backbone.parameters()).device
        x = TV.ToTensor()(pil).unsqueeze(0).to(device)
        x.requires_grad_(True)
        
        if pytorch_model is not None:
            from servers.xai_maps import compute_gradient_saliency
            px = ExplainerAgent.preprocess_for_vit(x.detach())
            return compute_gradient_saliency(pytorch_model, px, class_idx)
        else:
            px = ExplainerAgent.preprocess_for_vit(x)
            out = backbone(pixel_values=px, output_hidden_states=True)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                z = out.pooler_output
            else:
                z = out.last_hidden_state[:, 1:, :].mean(dim=1)
            score = torch.sum(z * w_vec)
            backbone.zero_grad(set_to_none=True)
            score.backward()
            g = x.grad.detach().abs().mean(1)[0].cpu().numpy()
        return g
    
    @staticmethod
    def _occlusion_map(backbone, clf, pil: Image.Image, processor, model_id: str,
                       patch_size: int = 16, stride: int = 8, pytorch_model=None) -> np.ndarray:
        """Generate occlusion map."""
        device = next(backbone.parameters()).device
        img_array = np.array(pil)
        h, w = img_array.shape[:2]
        
        if pytorch_model is not None:
            inputs = processor(images=pil, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = pytorch_model(**inputs)
                probs = torch.softmax(logits, dim=1)
                baseline_prob = probs[0].max().item()
                baseline_class = probs[0].argmax().item()
        else:
            z = ExplainerAgent._embed_single(backbone, processor, pil, device)
            probs = clf.predict_proba(z.detach().cpu().numpy())[0]
            baseline_prob = probs.max()
            baseline_class = probs.argmax()
        
        occlusion_map = np.zeros((h, w), dtype=np.float32)
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                occluded_img = img_array.copy()
                occluded_img[y:y+patch_size, x:x+patch_size] = 0
                occluded_pil = Image.fromarray(occluded_img)
                
                if pytorch_model is not None:
                    inputs = processor(images=occluded_pil, return_tensors="pt").to(device)
                    with torch.no_grad():
                        logits = pytorch_model(**inputs)
                        probs = torch.softmax(logits, dim=1)
                        occluded_prob = probs[0, baseline_class].item()
                else:
                    z_occ = ExplainerAgent._embed_single(backbone, processor, occluded_pil, device)
                    probs_occ = clf.predict_proba(z_occ.detach().cpu().numpy())[0]
                    occluded_prob = probs_occ[baseline_class]
                
                importance = baseline_prob - occluded_prob
                occlusion_map[y:y+patch_size, x:x+patch_size] = np.maximum(
                    occlusion_map[y:y+patch_size, x:x+patch_size], importance
                )
        
        occlusion_map = (occlusion_map - occlusion_map.min()) / (occlusion_map.max() - occlusion_map.min() + 1e-8)
        return occlusion_map
    
    @staticmethod
    def _save_prob_plot(class_names: List[str], probs: np.ndarray, out_png: str, topk: int = 5):
        """Save probability bar plot."""
        k = min(topk, len(class_names))
        idx = np.argsort(probs)[::-1][:k]
        names = [class_names[i] for i in idx]
        vals = probs[idx]
        plt.figure(figsize=(6, 4))
        plt.bar(range(k), vals)
        plt.xticks(range(k), names, rotation=20, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("probability")
        plt.tight_layout()
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150)
        plt.close()
    
    @staticmethod
    def load_image_tensor(path: str, max_side: int = 224) -> torch.Tensor:
        """Load image as tensor."""
        pil = Image.open(path).convert("RGB").resize((max_side, max_side), Image.BICUBIC)
        return TV.ToTensor()(pil).unsqueeze(0)
    
    @staticmethod
    def find_module_by_suffix(model: torch.nn.Module, suffix: str) -> str:
        """Find module by suffix."""
        hits = [n for n,_ in model.named_modules() if n.endswith(suffix)]
        if not hits:
            raise ValueError(f"No module endswith '{suffix}'")
        return sorted(hits, key=len)[0]
    
    class ActivationCatcher:
        """Context manager to catch activations."""
        def __init__(self, model: torch.nn.Module, target_suffix: str):
            self.model = model
            self.target_name = ExplainerAgent.find_module_by_suffix(model, target_suffix)
            self.buffer = None
            self.hook = None
        def __enter__(self):
            def _hook(m, inp, out): self.buffer = out
            module = dict(self.model.named_modules())[self.target_name]
            self.hook = module.register_forward_hook(_hook)
            return self
        def __exit__(self, exc_type, exc, tb):
            if self.hook is not None:
                self.hook.remove()
        def get_vec(self) -> torch.Tensor:
            z = self.buffer
            if z is None:
                raise RuntimeError("No activation captured.")
            if isinstance(z, tuple):
                z = z[-1]
            if hasattr(z, "last_hidden_state"):
                z = z.last_hidden_state
            if z.dim() == 3:
                z = z[:, 0, :]
            elif z.dim() == 4:
                z = z.mean((2,3))
            return z
    
    @staticmethod
    def total_variation(x: torch.Tensor) -> torch.Tensor:
        """Compute total variation."""
        dx = x[:,:,:,1:] - x[:,:,:,:-1]
        dy = x[:,:,1:,:] - x[:,:,:-1,:]
        return dx.abs().mean() + dy.abs().mean()
    
    @staticmethod
    def deepdream_direction_octaves(image_path: str, backbone, w_vec: torch.Tensor,
                                    target_suffix: str = "attention.o_proj",
                                    steps_per_octave: int = 100, lr: float = 0.15,
                                    jitter: int = 6, tv_weight: float = 1e-2,
                                    num_octaves: int = 2, octave_scale: float = 1.35,
                                    blur_every: int = 20, clip_norm: float = 0.8,
                                    full_model=None, class_idx: int = 0) -> Image.Image:
        """Generate DeepDream visualization using the fine-tuned classifier.
        
        Uses the FULL fine-tuned model (backbone + classifier) to maximize the 
        classifier's output for the target class. This produces tumor-relevant 
        visualizations instead of generic ImageNet features like animal faces.
        """
        device = next(backbone.parameters()).device
        x0 = ExplainerAgent.load_image_tensor(image_path, ExplainerAgent.IM_SIZE).to(device)
        H, W = x0.shape[2], x0.shape[3]
        
        # Check if we have the full fine-tuned model
        use_full_model = full_model is not None
        if use_full_model:
            print(f"  DeepDream: Using fine-tuned classifier (maximizing class {class_idx} output)")
        else:
            print(f"  DeepDream: Using backbone features only")
        
        sizes = []
        h, w = H, W
        for _ in range(num_octaves-1):
            h = int(round(h / octave_scale))
            w = int(round(w / octave_scale))
            sizes.append((max(32,h), max(32,w)))
        sizes = list(reversed(sizes)) + [(H, W)]
        
        x = x0.clone().detach()
        for octave_idx, (h, w) in enumerate(sizes):
            x = F.interpolate(x, size=(h,w), mode="bilinear", align_corners=False).detach()
            x = torch.nn.Parameter(x)
            # Use adaptive learning rate
            adaptive_lr = lr * (1.0 if octave_idx == 0 else 0.7)
            opt = torch.optim.Adam([x], lr=adaptive_lr)
            
            for t in range(steps_per_octave):
                opt.zero_grad()
                if jitter > 0:
                    ox = torch.randint(-jitter, jitter+1, ()).item()
                    oy = torch.randint(-jitter, jitter+1, ()).item()
                    x.data = torch.roll(x.data, shifts=(ox, oy), dims=(2,3))
                
                px = ExplainerAgent.preprocess_for_vit(x)
                
                if use_full_model:
                    # Use the FULL fine-tuned model (backbone + classifier)
                    # This maximizes the actual tumor classifier output
                    logits = full_model(pixel_values=px)
                    # Maximize the logit for the target class
                    score = logits[0, class_idx]
                else:
                    # Fallback: backbone-only with weight vector projection
                    out = backbone(pixel_values=px, output_hidden_states=True)
                    if hasattr(out, 'last_hidden_state'):
                        z = out.last_hidden_state[:, 0]
                    elif hasattr(out, 'pooler_output') and out.pooler_output is not None:
                        z = out.pooler_output
                    else:
                        z = out[0][:, 0]
                    
                    # Project w_vec to match z dimension
                    if z.shape[-1] != w_vec.shape[-1]:
                        w_proj = torch.zeros(z.shape[-1], device=device, dtype=w_vec.dtype)
                        min_dim = min(z.shape[-1], w_vec.shape[-1])
                        w_proj[:min_dim] = w_vec[:min_dim]
                        w_proj = w_proj / (w_proj.norm() + 1e-8)
                    else:
                        w_proj = w_vec / (w_vec.norm() + 1e-8)
                    score = torch.sum(z * w_proj)
                
                # Loss: maximize score + regularization
                loss = -score + tv_weight * ExplainerAgent.total_variation(x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([x], max_norm=clip_norm)
                opt.step()
                
                if jitter > 0:
                    x.data = torch.roll(x.data, shifts=(-ox, -oy), dims=(2,3))
                if blur_every and (t+1) % blur_every == 0:
                    x.data = F.avg_pool2d(x.data, 3, 1, 1)
                x.data.clamp_(0,1)
        
        img = (x.data[0].clamp(0,1).cpu().permute(1,2,0).numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(img)
    
    @staticmethod
    def _to_hwc_uint8(arr, default_w=224):
        """Convert array to HWC uint8 format."""
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        arr = np.squeeze(arr)
        if arr.ndim == 3 and arr.shape[0] in (1,3):
            arr = np.transpose(arr, (1,2,0))
        if arr.ndim != 3 or arr.shape[-1] != 3:
            if arr.size == default_w*default_w*3:
                arr = arr.reshape(default_w, default_w, 3)
            else:
                raise ValueError(f"Bad image shape {arr.shape}")
        if np.issubdtype(arr.dtype, np.floating):
            a = arr.copy()
            mn, mx = float(a.min()), float(a.max())
            if mn >= 0 and mx <= 1.5:
                a = a*255
            elif mn >= -1.5 and mx <= 1.5:
                a = (a+1.0)*127.5
            a = np.clip(a, 0, 255)
            arr = a.astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    
    class DinoForLucent(nn.Module):
        """Wrapper for DINOv3 model to work with Lucent."""
        def __init__(self, hf_model, size: int = 224):
            super().__init__()
            self.model = hf_model
            self.resize = TV.Resize((size, size), antialias=True)
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        def forward(self, x):
            x = self.resize(x)
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
            return self.model(pixel_values=x, output_hidden_states=True, output_attentions=True)
    
    class FullModelForLucentWithHook(nn.Module):
        """Wrapper that exposes classifier layers with activation hooks for Lucent."""
        def __init__(self, full_model, size: int = 224):
            super().__init__()
            self.full_model = full_model
            self.resize = TV.Resize((size, size), antialias=True)
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
            # Register hook to capture activations before final layer
            self.activation_hook = None
            self.captured_activation = None
            
            # Hook the Linear layer that outputs the 256-dim representation feeding the
            # final classification layer. Found generically by out_features==256 so this
            # works regardless of how many layers precede it (e.g. index 5 in the older
            # LayerNorm->512->256->N head, or index 0 in the smaller 256->N head trained
            # in dinov3_augmented_finetune_experiment.ipynb).
            if hasattr(full_model, 'classifier'):
                self.classifier = full_model.classifier
                dim256_idx = None
                for i, m in enumerate(self.classifier):
                    if isinstance(m, nn.Linear) and m.out_features == 256:
                        dim256_idx = i
                if dim256_idx is not None:
                    self.classifier[dim256_idx].register_forward_hook(self._save_activation)
        
        def _save_activation(self, module, input, output):
            """Save activation for Lucent visualization."""
            self.captured_activation = output
        
        def forward(self, x):
            x = self.resize(x)
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
            # Get backbone output
            backbone_out = self.full_model.backbone(pixel_values=x, output_hidden_states=True, output_attentions=True)
            # Get features
            if hasattr(backbone_out, 'last_hidden_state'):
                features = backbone_out.last_hidden_state[:, 0]
            elif hasattr(backbone_out, 'pooler_output'):
                features = backbone_out.pooler_output
            else:
                features = backbone_out[0][:, 0]
            # Run through classifier (this will trigger the hook)
            classifier_out = self.classifier(features)
            # Store the captured 256-dim activation
            if self.captured_activation is not None:
                backbone_out.classifier_256_activation = self.captured_activation
            backbone_out.classifier_output = classifier_out
            return backbone_out
    
    @staticmethod
    def _lucent_model(model_id: str, size: int = 224):
        """Create Lucent-compatible model."""
        backbone, _ = ExplainerAgent.load_hf_backbone(model_id)
        return ExplainerAgent.DinoForLucent(backbone, size=size).eval().to(next(backbone.parameters()).device)
    
    @staticmethod
    def resolve_layer_name(layer_names, requested: str) -> str:
        """Resolve layer name from Lucent layer names."""
        if requested in layer_names:
            return requested
        tail = requested.split("model->")[-1] if "model->" in requested else requested
        candidates = [n for n in layer_names if n.endswith(tail)]
        if candidates:
            return sorted(candidates, key=len)[0]
        parts = tail.split("->")
        for keep in (4, 3, 2):
            if len(parts) >= keep:
                short_tail = "->".join(parts[-keep:])
                c2 = [n for n in layer_names if n.endswith(short_tail)]
                if c2:
                    return sorted(c2, key=len)[0]
        last = parts[-1]
        hints = sorted(set(n.split("model->")[-1] for n in layer_names if n.endswith(last)))[:10]
        raise ValueError(f"Layer '{requested}' not found. Similar endings: {hints}")
    
    @staticmethod
    def _lucent_direction_for_class(backbone, model_id: str, class_w: torch.Tensor, iters: int, out_png: str, layer_name: str = None, pytorch_model=None):
        """Generate Lucent class direction visualization for last layer."""
        # NEW APPROACH: Visualize in the 256-dim space before final Linear layer
        use_final_layer_256 = True  # Set to False to fall back to old approach
        
        if use_final_layer_256 and pytorch_model is not None:
            # NEW: Use hook-based approach to visualize 256-dim activations before final layer
            luc = ExplainerAgent.FullModelForLucentWithHook(pytorch_model, size=224).eval().to(next(pytorch_model.parameters()).device)
        elif pytorch_model is not None:
            # OLD APPROACH (commented but kept for fallback)
            # Create wrapper that includes classifier
            class FullModelForLucent(nn.Module):
                def __init__(self, full_model, size: int = 224):
                    super().__init__()
                    self.full_model = full_model
                    self.resize = TV.Resize((size, size), antialias=True)
                    self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
                    self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
                def forward(self, x):
                    x = self.resize(x)
                    x = (x - self.mean.to(x.device)) / self.std.to(x.device)
                    # Get backbone output with hidden states
                    backbone_out = self.full_model.backbone(pixel_values=x, output_hidden_states=True, output_attentions=True)
                    # Also compute classifier output for layer access
                    if hasattr(backbone_out, 'last_hidden_state'):
                        features = backbone_out.last_hidden_state[:, 0]
                    elif hasattr(backbone_out, 'pooler_output'):
                        features = backbone_out.pooler_output
                    else:
                        features = backbone_out[0][:, 0]
                    classifier_out = self.full_model.classifier(features)
                    # Store both for layer access
                    backbone_out.classifier_output = classifier_out
                    return backbone_out
            
            luc = FullModelForLucent(pytorch_model, size=224).eval().to(next(pytorch_model.parameters()).device)
        else:
            luc = ExplainerAgent._lucent_model(model_id, size=224)
        
        layer_names, _ = get_model_layers(luc)
        
        # NEW APPROACH: If using hook-based model, visualize the 256-dim activation
        if use_final_layer_256 and pytorch_model is not None and isinstance(luc, ExplainerAgent.FullModelForLucentWithHook):
            # Find the layer that outputs the 256-dim activation (before final Linear)
            # The classifier structure is: ... -> Linear(512->256) -> GELU -> Dropout -> Linear(256->4)
            # We want to visualize at the output of Linear(512->256), which is at index 5
            # But Lucent needs a named layer, so we'll use a custom objective
            print("Using NEW approach: Visualizing 256-dim space before final Linear layer")
            
            # Get the weight vector - should be 256-dim (input to final Linear)
            # class_w is the row from final Linear weight matrix [4, 256], so it's already 256-dim
            w_256 = class_w.flatten()
            if w_256.shape[0] != 256:
                print(f"Warning: Expected 256-dim weight, got {w_256.shape[0]}-dim. Adjusting...")
                if w_256.shape[0] < 256:
                    w_padded = torch.zeros(256, device=w_256.device, dtype=w_256.dtype)
                    w_padded[:w_256.shape[0]] = w_256
                    w_256 = w_padded
                else:
                    w_256 = w_256[:256]
            
            # Create a custom objective that maximizes the 256-dim activation dot product
            # We'll need to hook into the classifier's intermediate layer
            device = next(luc.parameters()).device
            
            # Find the classifier layer whose out_features == 256, regardless of its
            # position/index in the head (robust to different head architectures).
            classifier_layer_name = None
            layer_dict = dict(luc.named_modules())
            for name in layer_names:
                if 'classifier' not in name.lower():
                    continue
                layer = layer_dict.get(name)
                if layer is not None and isinstance(layer, nn.Linear) and getattr(layer, 'out_features', None) == 256:
                    classifier_layer_name = name
                    break
            
            if classifier_layer_name:
                print(f"Found 256-dim layer: {classifier_layer_name}")
                resolved = classifier_layer_name
            else:
                # Fallback: use the last layer before final output
                classifier_layers = [n for n in layer_names if 'classifier' in n.lower()]
                if classifier_layers:
                    resolved = sorted(classifier_layers, key=lambda x: (x.count('.'), len(x)))[-2] if len(classifier_layers) >= 2 else classifier_layers[-1]
                else:
                    resolved = layer_names[-1]
                print(f"Using layer (fallback): {resolved}")
        elif layer_name is None or layer_name == "classifier":
            # OLD APPROACH (kept for fallback)
            # If we have the full model, directly inspect the classifier structure
            if pytorch_model is not None:
                # Find the final Linear layer in the classifier
                classifier_modules = list(pytorch_model.classifier.named_modules())
                final_linear = None
                for name, module in classifier_modules:
                    if isinstance(module, nn.Linear):
                        final_linear = (name, module)
                
                if final_linear:
                    # Construct the full layer path for Lucent
                    layer_path = f"full_model->classifier->{final_linear[0]}"
                    # Check if this layer exists in layer_names
                    matching_layers = [n for n in layer_names if final_linear[0] in n and 'classifier' in n]
                    if matching_layers:
                        resolved = matching_layers[0]
                        print(f"Using Lucent layer: {resolved}")
                        print(f"  Layer details: {final_linear[0]} -> Linear({final_linear[1].in_features}, {final_linear[1].out_features})")
                        print(f"  This is the FINAL OUTPUT LAYER with {final_linear[1].out_features} classes")
                    else:
                        # Try to find it by searching
                        all_classifier_layers = [n for n in layer_names if 'classifier' in n.lower()]
                        # Look for the one that matches the final linear layer name
                        for n in all_classifier_layers:
                            if final_linear[0].split('.')[-1] in n or str(final_linear[1].out_features) in n:
                                resolved = n
                                break
                        else:
                            resolved = sorted(all_classifier_layers, key=lambda x: (x.count('.'), len(x)))[-1]
                        print(f"Using Lucent layer (classifier layer): {resolved}")
                else:
                    # Fallback to layer_names search
                    classifier_layers = [n for n in layer_names if 'classifier' in n.lower()]
                    if classifier_layers:
                        resolved = sorted(classifier_layers, key=lambda x: (x.count('.'), len(x)))[-1]
                        print(f"Using Lucent layer (classifier last layer): {resolved}")
                    else:
                        transformer_layers = [n for n in layer_names if 'layer' in n.lower() or 'block' in n.lower()]
                        resolved = sorted(transformer_layers, key=lambda x: (len(x), x))[-1] if transformer_layers else layer_names[-1]
                        print(f"Using Lucent layer (fallback): {resolved}")
            else:
                # Original backbone-only approach
                classifier_layers = [n for n in layer_names if 'classifier' in n.lower()]
                if classifier_layers:
                    resolved = sorted(classifier_layers, key=lambda x: (x.count('.'), len(x)))[-1]
                    print(f"Using Lucent layer (classifier last layer): {resolved}")
                else:
                    # Fallback: find last transformer layer
                    transformer_layers = [n for n in layer_names if 'layer' in n.lower() or 'block' in n.lower() or 'encoder' in n.lower()]
                    if transformer_layers:
                        resolved = sorted(transformer_layers, key=lambda x: (len(x), x))[-1]
                        print(f"Using Lucent layer (last transformer): {resolved}")
                    else:
                        # Final fallback
                        resolved = ExplainerAgent.resolve_layer_name(layer_names, "attention.o_proj")
                        print(f"Using Lucent layer (fallback): {resolved}")
        else:
            resolved = ExplainerAgent.resolve_layer_name(layer_names, layer_name)
            print(f"Using Lucent layer (specified): {resolved}")
        
        tfms = transform.standard_transforms.copy()
        device = next(luc.parameters()).device
        
        # NEW APPROACH: Handle 256-dim visualization
        if use_final_layer_256 and pytorch_model is not None and isinstance(luc, ExplainerAgent.FullModelForLucentWithHook):
            # We already have w_256 from above
            w = w_256.to(device)
            print(f"  Visualizing in 256-dim space (input to final Linear layer)")
            print(f"  Weight vector shape: {w.shape}")
        else:
            # OLD APPROACH (kept for fallback)
            w = class_w.to(device)
        
        # Get layer dimension by inspecting the model
        layer_dict = dict(luc.named_modules())
        target_layer = layer_dict.get(resolved)
        
        # NEW APPROACH: Skip dimension adjustment if already using 256-dim
        if not (use_final_layer_256 and pytorch_model is not None and isinstance(luc, ExplainerAgent.FullModelForLucentWithHook)):
            # OLD APPROACH (kept for fallback)
            # Print layer information
            if target_layer:
                print(f"  Target layer type: {type(target_layer).__name__}")
                if hasattr(target_layer, 'in_features'):
                    print(f"  Layer input features: {target_layer.in_features}")
                if hasattr(target_layer, 'out_features'):
                    print(f"  Layer output features: {target_layer.out_features}")
                if hasattr(target_layer, 'weight'):
                    print(f"  Layer weight shape: {target_layer.weight.shape}")
            
            # For the final Linear layer, we visualize in the INPUT space (what goes into the layer)
            # The weight vector w is the row from weight matrix [out_features, in_features]
            # We want to visualize what input features maximize the output
            if target_layer and isinstance(target_layer, nn.Linear):
                # Use input dimension for visualization
                vis_dim = target_layer.in_features
                print(f"  Visualizing in INPUT space: {vis_dim} dimensions (features feeding into final layer)")
            elif target_layer and hasattr(target_layer, 'in_features'):
                vis_dim = target_layer.in_features
                print(f"  Visualizing in INPUT space: {vis_dim} dimensions")
            else:
                # Fallback: get dimension from test forward pass
                test_input = torch.randn(1, 3, 224, 224).to(device)
                with torch.no_grad():
                    test_out = luc(test_input)
                    if hasattr(test_out, 'hidden_states') and test_out.hidden_states:
                        vis_dim = test_out.hidden_states[-1].shape[-1]
                    else:
                        vis_dim = w.shape[-1] if w.dim() > 0 else w.shape[0]
                print(f"  Visualizing dimension (fallback): {vis_dim}")
            
            # Project w to match visualization dimension
            w_flat = w.flatten()
            if w_flat.shape[0] != vis_dim:
                if w_flat.shape[0] < vis_dim:
                    # Pad with zeros
                    w_padded = torch.zeros(vis_dim, device=w.device, dtype=w.dtype)
                    w_padded[:w_flat.shape[0]] = w_flat
                    w = w_padded
                    print(f"  Padded weight: {w_flat.shape[0]} -> {vis_dim}")
                else:
                    # Truncate
                    w = w_flat[:vis_dim]
                    print(f"  Truncated weight: {w_flat.shape[0]} -> {vis_dim}")
            else:
                w = w_flat
        
        # Set random seed for reproducible visualization
        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"  Using random seed: {seed}")
        
        imgs = render.render_vis(
            luc,
            objectives.direction(resolved, w),
            param_f=lambda: param.image(w=224, fft=True, decorrelate=True),
            transforms=tfms,
            thresholds=(iters,),
            show_image=False,
            show_inline=False,
            progress=True
        )
        arr = ExplainerAgent._to_hwc_uint8(imgs[-1], default_w=224)
        Image.fromarray(arr).save(out_png)
        print(f"Saved Lucent visualization to {out_png}")
        return resolved

# ============================================================================
# Reporter Agent - LLM-based Reports
# ============================================================================

class ReporterAgent:
    """Handles LLM-based report generation."""
    
    @staticmethod
    def _file_to_data_url(path: str) -> str:
        if path is None:
            return ""
        b = Path(path).read_bytes()
        enc = base64.b64encode(b).decode("utf-8")
        ext = Path(path).suffix.lower().lstrip(".") or "png"
        return f"data:image/{ext};base64,{enc}"
    
    @staticmethod
    def explain_results(images: List[Optional[str]], prediction: dict, topk: List[dict],
                       out_txt: str = None, model: str = "gpt-4o-mini"):
        """Generate LLM-based explanation report."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        
        imgs = [p for p in images if p]
        
        # Format prediction class name for readability
        pred_name = prediction.get('name', '').replace('_', ' ').title()
        pred_prob = prediction.get('prob', 0.0)
        
        # Format top-k predictions
        topk_str = "\n".join([f"  - {t['name'].replace('_', ' ').title()}: {t['prob']:.1%}" for t in topk[:4]])
        
        content = [{"type": "text", "text":
            f"You are a radiologist analyzing a brain MRI scan using an AI-assisted diagnostic system. "
            f"Generate a comprehensive radiology report in standard clinical format.\n\n"
            f"**Prediction Summary:**\n"
            f"Primary diagnosis: {pred_name} (confidence: {pred_prob:.1%})\n"
            f"Alternative considerations:\n{topk_str}\n\n"
            f"The following visualizations are provided for your reference:\n"
            f"1. **DeepDream visualization**: Shows the model's learned pattern for this class\n"
            f"2. **Probability distribution**: Class probabilities for all tumor types\n"
            f"3. **Occlusion sensitivity map**: Highlights critical regions that affect the prediction\n\n"
            f"Please analyze these visualizations and write a formal radiology report following this structure:\n"
            f"- **Clinical Information**: Brief context (AI-assisted brain tumor classification)\n"
            f"- **Findings**: Describe the key features visible in the image analysis\n"
            f"- **Impression**: Provide your interpretation of the findings and diagnostic assessment\n"
            f"- **Recommendation**: Suggest next steps if appropriate\n\n"
            f"Use professional medical terminology appropriate for a radiology report. Be specific about "
            f"anatomical locations and morphological features. Include relevant quantitative information from "
            f"the probability distribution. Reference the occlusion map to describe which regions contribute "
            f"most to the classification decision."
        }]
        
        labels = ["DeepDream Visualization", "Probability Distribution", "Occlusion Sensitivity Map"]
        for label, p in zip(labels, images):
            if p:
                content.append({"type": "text", "text": f"**{label}:**"})
                content.append({"type": "image_url", "image_url": {"url": ReporterAgent._file_to_data_url(p)}})
        
        resp = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are an experienced radiologist with expertise in neuroimaging and brain tumor diagnosis. Write professional, accurate, and clinically relevant radiology reports. Use standard medical terminology. Include appropriate disclaimers that this is AI-assisted analysis and should be interpreted in conjunction with clinical correlation and expert review."},
                {"role": "user", "content": content}
            ]
        )
        text = resp.choices[0].message.content.strip()
        if out_txt:
            Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
            Path(out_txt).write_text(text, encoding="utf-8")
        return {"ok": True, "text": text, "out_txt": out_txt}

# ============================================================================
# Main Pipeline Orchestrator
# ============================================================================

def run_pipeline(cfg: dict):
    """Run the complete pipeline."""
    print("="*60)
    print("🧠 Brain Tumor Classification Pipeline")
    print("="*60)
    
    modeler = ModelerAgent()
    explainer = ExplainerAgent()
    reporter = ReporterAgent()
    
    # 1) Feature extraction
    train_feats = FEATS / "train.npz"
    test_feats = FEATS / "test.npz"
    
    if train_feats.exists() and test_feats.exists():
        try:
            train_data = np.load(train_feats, allow_pickle=True)
            if "model_id" in train_data:
                feat_model_id = train_data["model_id"].tolist()[0]
                if feat_model_id == cfg["model_id"]:
                    print(f"✅ Using existing features (model_id: {feat_model_id})")
                    f_out = {
                        "ok": True,
                        "train": str(train_feats),
                        "test": str(test_feats),
                        "classes": train_data["class_names"].tolist() if "class_names" in train_data else []
                    }
                else:
                    print(f"⚠️  Model ID mismatch - re-extracting features")
                    f_out = modeler.extract_features(
                        cfg["data_root"], cfg["model_id"], cfg.get("img_size", 224),
                        str(train_feats), str(test_feats)
                    )
            else:
                print("⚠️  Features missing model_id - re-extracting")
                f_out = modeler.extract_features(
                    cfg["data_root"], cfg["model_id"], cfg.get("img_size", 224),
                    str(train_feats), str(test_feats)
                )
        except Exception as e:
            print(f"⚠️  Error checking features: {e} - re-extracting")
            f_out = modeler.extract_features(
                cfg["data_root"], cfg["model_id"], cfg.get("img_size", 224),
                str(train_feats), str(test_feats)
            )
    else:
        f_out = modeler.extract_features(
            cfg["data_root"], cfg["model_id"], cfg.get("img_size", 224),
            str(train_feats), str(test_feats)
        )
    
    # 2) Model loading/training
    clf_pth = MODELS / "clf.pth"
    clf_pkl = MODELS / "clf.pkl"
    
    if clf_pth.exists():
        print(f"✅ Using existing PyTorch model: {clf_pth}")
        class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
        p_out = {"ok": True, "pkl": str(clf_pth), "classes": class_names}
    elif clf_pkl.exists():
        print(f"✅ Using existing sklearn classifier: {clf_pkl}")
        meta = joblib.load(clf_pkl)
        p_out = {"ok": True, "pkl": str(clf_pkl), "classes": meta.get("class_names", [])}
    else:
        print("Training new classifier...")
        p_out = modeler.train_probe(f_out["train"], str(clf_pkl))
    
    # 3) Evaluation
    print("Evaluating model...")
    m_out = modeler.eval_probe(f_out["test"], p_out["pkl"], str(METRICS / "metrics.json"))
    print(f"✅ Accuracy: {m_out['summary']['acc']:.4f}, F1-macro: {m_out['summary']['f1_macro']:.4f}")
    
    # 4) Single image explanations
    explain_out = None
    report_out = None
    single_cfg = cfg.get("single_image")
    
    # Handle both single image_path and multiple image_paths
    image_paths = []
    if single_cfg:
        if single_cfg.get("image_path"):
            image_paths = [single_cfg["image_path"]]
        elif single_cfg.get("image_paths"):
            image_paths = single_cfg["image_paths"]
    
    if image_paths:
        explain_results_list = []
        # Extract kwargs (excluding image_path and image_paths)
        explain_kwargs = {k: v for k, v in single_cfg.items() if k not in ["image_path", "image_paths"]}
        
        # Merge deepdream config if available
        deepdream_cfg = cfg.get("deepdream", {})
        if deepdream_cfg and isinstance(deepdream_cfg, dict):
            preset = deepdream_cfg.get("preset", "moderate")
            if preset in deepdream_cfg and isinstance(deepdream_cfg[preset], dict):
                deepdream_params = deepdream_cfg[preset]
                # Override with deepdream config values if not already set
                if "steps" not in explain_kwargs:
                    explain_kwargs["steps"] = deepdream_params.get("steps", 200)
                if "lr" not in explain_kwargs:
                    explain_kwargs["lr"] = deepdream_params.get("lr", 0.1)
                if "tv_weight" not in explain_kwargs:
                    explain_kwargs["tv_weight"] = deepdream_params.get("tv_weight", 0.005)
                if "num_octaves" not in explain_kwargs:
                    explain_kwargs["num_octaves"] = deepdream_params.get("num_octaves", 3)
                if "octave_scale" not in explain_kwargs:
                    explain_kwargs["octave_scale"] = deepdream_params.get("octave_scale", 1.4)
                if "jitter" not in explain_kwargs:
                    explain_kwargs["jitter"] = deepdream_params.get("jitter", 8)
                if "octaves" not in explain_kwargs:
                    explain_kwargs["octaves"] = deepdream_params.get("octaves", True)
            if "layer_suffix" in deepdream_cfg and "layer_suffix" not in explain_kwargs:
                explain_kwargs["layer_suffix"] = deepdream_cfg["layer_suffix"]
        
        for img_path in image_paths:
            out_dir = EXPLAIN / "by_image" / Path(img_path).stem
            try:
                img_explain_out = explainer.explain_image(
                    p_out["pkl"], img_path, str(out_dir),
                    **explain_kwargs
                )
                explain_results_list.append(img_explain_out)
                print(f"✅ Generated explanations for {img_path}")
            except Exception as e:
                print(f"⚠️  Failed to generate explanations for {img_path}: {e}")
                explain_results_list.append(None)
        
        # Use the first result as the main explain_out for backward compatibility
        explain_out = explain_results_list[0] if explain_results_list else None
        
        # 5) LLM report (only for first image if enabled)
        if cfg.get("generate_report", False) and explain_out:
            img_path = image_paths[0]
            rep_out_path = REPORTS / f"{Path(img_path).stem}_report.txt"
            try:
                report_out = reporter.explain_results(
                    [explain_out.get("dream_png"), explain_out.get("probs_png"), explain_out.get("occlusion_map_png")],
                    explain_out.get("prediction"),
                    explain_out.get("topk"),
                    str(rep_out_path)
                )
                print(f"✅ Generated LLM report: {rep_out_path}")
            except Exception as e:
                print(f"⚠️  LLM report failed: {e}")
    
    return {
        "ok": True,
        "feats": f_out,
        "probe": p_out,
        "metrics": m_out,
        "single_image": explain_out,
        "report": report_out
    }

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    cfg = load_config()
    result = run_pipeline(cfg)
    print("\n" + "="*60)
    print("✅ Pipeline Completed!")
    print("="*60)
    print(json.dumps(result, indent=2, default=str))


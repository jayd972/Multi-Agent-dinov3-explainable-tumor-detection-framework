# servers/explainer_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from pathlib import Path
import os, io, numpy as np, joblib, torch, math
from PIL import Image

# Lucent
from lucent.optvis import render, param, objectives, transform
from lucent.model_utils import get_model_layers

# Torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TV
from transformers import AutoImageProcessor, AutoModel
from huggingface_hub import HfFolder

# Matplotlib for plots and color maps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.secrets import load_secrets
load_secrets()


app = FastAPI(title="explainer")

# ====== Request Schemas ======
class ProtosReq(BaseModel):
    pkl: str
    layer: str
    n_prototypes: int = 3
    iters: int = 512
    out_dir: str

class GridReq(BaseModel):
    pkl: str
    layer: str
    classes: List[str]
    n_prototypes: int = 3
    iters: int = 512
    out_png: str

class DeepDreamReq(BaseModel):
    pkl: str
    image_path: str
    target_class: str
    layer_suffix: str = "attention.o_proj"
    steps: int = 100
    lr: float = 0.15
    tv_weight: float = 1e-2
    octaves: bool = True
    num_octaves: int = 2
    octave_scale: float = 1.35
    jitter: int = 6
    out_png: str

class ExplainImageReq(BaseModel):
    pkl: str
    image_path: str
    out_dir: str
    layer_suffix: str = "attention.o_proj"
    steps: int = 80
    lr: float = 0.15
    tv_weight: float = 1e-2
    octaves: bool = True
    num_octaves: int = 2
    octave_scale: float = 1.35
    jitter: int = 6
    do_attention: bool = True
    do_gradients: bool = True
    do_deepdream: bool = True
    do_probs: bool = True
    do_lucent: bool = True
    do_occlusion: bool = True
    lucent_iters: int = 256
    occlusion_patch_size: int = 16
    occlusion_stride: int = 8

# ====== HF backbone + Lucent wrapper ======
def load_hf_backbone(model_id: str):
    token = HfFolder.get_token() or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    processor = AutoImageProcessor.from_pretrained(model_id, token=token)
    model = AutoModel.from_pretrained(model_id, token=token).eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return model, processor

class DinoForLucent(nn.Module):
    def __init__(self, hf_model, size: int = 224):
        super().__init__()
        self.model = hf_model
        self.resize = TV.Resize((size, size), antialias=True)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    def forward(self, x):
        x = self.resize(x)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return self.model(pixel_values=x, output_hidden_states=True, output_attentions=True)

def _to_hwc_uint8(arr, default_w=224):
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
        a = arr.copy(); mn, mx = float(a.min()), float(a.max())
        if mn >= 0 and mx <= 1.5: a = a*255
        elif mn >= -1.5 and mx <= 1.5: a = (a+1.0)*127.5
        a = np.clip(a, 0, 255); arr = a.astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def _load_probe(pkl_path: str):
    """Load classifier - supports both .pkl (sklearn) and .pth (PyTorch) formats."""
    pkl_path_obj = Path(pkl_path)
    
    # Check if it's a PyTorch model (.pth)
    if pkl_path_obj.suffix == '.pth' or 'pth' in pkl_path.lower():
        from servers.pytorch_model_loader import load_pytorch_model
        model, class_names, model_id = load_pytorch_model(str(pkl_path_obj))
        # Create a wrapper to make it sklearn-like
        class PyTorchClassifierWrapper:
            def __init__(self, model, class_names, model_id):
                self.model = model
                self.class_names = class_names
                self.model_id = model_id
                self.device = next(model.parameters()).device
                
            def predict_proba(self, X):
                """Predict probabilities.
                
                X can be:
                - torch.Tensor (features from backbone) - will use backbone + classifier
                - PIL Image or list of images - will process through full model
                - numpy array (features) - not supported, need to use backbone
                """
                import torch
                from transformers import AutoImageProcessor
                from huggingface_hub import HfFolder
                
                # If X is a torch tensor (features from _embed_single)
                if isinstance(X, torch.Tensor):
                    # X is features from backbone, need to pass through classifier head
                    with torch.no_grad():
                        logits = self.model.classifier(X.to(self.device))
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                    return probs
                # If X is images (PIL or paths)
                elif isinstance(X, (list, tuple)) and len(X) > 0:
                    from PIL import Image
                    # Assume images
                    if isinstance(X[0], (str, Path)):
                        images = [Image.open(str(x)).convert('RGB') for x in X]
                    elif isinstance(X[0], Image.Image):
                        images = X
                    else:
                        images = X
                    
                    # Process images
                    hf_token = HfFolder.get_token() or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                    processor = AutoImageProcessor.from_pretrained(self.model_id, token=hf_token)
                    inputs = processor(images=images, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        logits = self.model(**inputs)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                    return probs
                else:
                    # numpy array - try to convert to tensor
                    if isinstance(X, np.ndarray):
                        X_tensor = torch.from_numpy(X).float().to(self.device)
                        with torch.no_grad():
                            logits = self.model.classifier(X_tensor)
                            probs = torch.softmax(logits, dim=1).cpu().numpy()
                        return probs
                    raise ValueError(f"Unsupported input type: {type(X)}")
            
            def predict(self, X):
                probs = self.predict_proba(X)
                return probs.argmax(axis=1)
        
        clf = PyTorchClassifierWrapper(model, class_names, model_id)
        return clf, class_names, model_id
    else:
        # Original sklearn format
        meta = joblib.load(pkl_path)
        return meta["clf"], meta["class_names"], meta["model_id"]

def _lucent_model(model_id: str, size: int = 224):
    backbone, _ = load_hf_backbone(model_id)
    return DinoForLucent(backbone, size=size).eval().to(next(backbone.parameters()).device)

def resolve_layer_name(layer_names, requested: str) -> str:
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

# ====== O1 prototypes ======
def o1_generate_class_prototypes_simple(lucent_model, clf, class_names, class_to_idx, layer_name,
                                        n_prototypes=3, iters=512, out_dir="o1_prototypes", base_seed=0):
    os.makedirs(out_dir, exist_ok=True)
    tfms = transform.standard_transforms.copy()
    device = next(lucent_model.parameters()).device

    def render_one(w_vec, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        imgs = render.render_vis(
            lucent_model,
            objectives.direction(layer_name, w_vec),
            param_f=lambda: param.image(w=224, fft=True, decorrelate=True),
            transforms=tfms,
            thresholds=(iters,),
            show_image=False, show_inline=False, progress=True
        )
        return imgs[-1] if isinstance(imgs[-1], list) else imgs[-1]

    for cname in class_names:
        k = class_to_idx[cname]
        # Handle both sklearn and PyTorch models
        if hasattr(clf, 'coef_'):
            # Sklearn model
            w = torch.tensor(clf.coef_[k], dtype=torch.float32, device=device)
        elif hasattr(clf, 'model'):
            # PyTorch model - extract classifier head weights
            pytorch_model = clf.model
            classifier_layers = [m for m in pytorch_model.classifier.modules() if isinstance(m, nn.Linear)]
            if classifier_layers:
                last_layer = classifier_layers[-1]
                w = last_layer.weight[k].detach().clone().to(device)
            else:
                raise ValueError("Could not extract classifier weights from PyTorch model")
        else:
            raise ValueError("Unknown classifier type")
        
        protos = []
        for i in range(n_prototypes):
            im = render_one(w, base_seed + i)
            protos.append(_to_hwc_uint8(im, default_w=224))
        for i, arr in enumerate(protos, 1):
            Image.fromarray(arr).save(os.path.join(out_dir, f"{cname}_proto_{i}.png"))
        h, w0, _ = protos[0].shape
        tile = Image.new("RGB", (w0 * n_prototypes, h), (255,255,255))
        x = 0
        for arr in protos:
            tile.paste(Image.fromarray(arr), (x, 0)); x += w0
        tag = layer_name.replace("->","_")
        tile.save(os.path.join(out_dir, f"{cname}_tile_{tag}.png"))

# ====== DeepDream helpers ======
IM_SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def load_image_tensor(path: str, max_side: int = IM_SIZE) -> torch.Tensor:
    pil = Image.open(path).convert("RGB").resize((max_side, max_side), Image.BICUBIC)
    return TV.ToTensor()(pil).unsqueeze(0)

def preprocess_for_vit(x: torch.Tensor, size: int = IM_SIZE) -> torch.Tensor:
    x = F.interpolate(x, size=(size,size), mode="bilinear", align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)

def find_module_by_suffix(model: torch.nn.Module, suffix: str) -> str:
    hits = [n for n,_ in model.named_modules() if n.endswith(suffix)]
    if not hits:
        raise ValueError(f"No module endswith '{suffix}'. Inspect model.named_modules().")
    return sorted(hits, key=len)[0]

class ActivationCatcher:
    def __init__(self, model: torch.nn.Module, target_suffix: str):
        self.model = model
        self.target_name = find_module_by_suffix(model, target_suffix)
        self.buffer = None
        self.hook = None
    def __enter__(self):
        def _hook(m, inp, out): self.buffer = out
        module = dict(self.model.named_modules())[self.target_name]
        self.hook = module.register_forward_hook(_hook); return self
    def __exit__(self, exc_type, exc, tb):
        if self.hook is not None: self.hook.remove()
    def get_vec(self) -> torch.Tensor:
        z = self.buffer
        if z is None: raise RuntimeError("No activation captured.")
        if isinstance(z, tuple): z = z[-1]
        if hasattr(z, "last_hidden_state"): z = z.last_hidden_state
        if z.dim() == 3: z = z[:, 0, :]
        elif z.dim() == 4: z = z.mean((2,3))
        return z

def total_variation(x: torch.Tensor) -> torch.Tensor:
    dx = x[:,:,:,1:] - x[:,:,:,:-1]
    dy = x[:,:,1:,:] - x[:,:,:-1,:]
    return dx.abs().mean() + dy.abs().mean()

def deepdream_direction_octaves(
    image_path: str,
    backbone,
    w_vec: torch.Tensor,
    target_suffix: str = "attention.o_proj",
    steps_per_octave: int = 100,
    lr: float = 0.15,
    jitter: int = 6,
    tv_weight: float = 1e-2,
    num_octaves: int = 2,
    octave_scale: float = 1.35,
    blur_every: int = 20,
    clip_norm: float = 0.8
) -> Image.Image:
    device = next(backbone.parameters()).device
    x0 = load_image_tensor(image_path, IM_SIZE).to(device)
    H, W = x0.shape[2], x0.shape[3]

    sizes = []
    h, w = H, W
    for _ in range(num_octaves-1):
        h = int(round(h / octave_scale)); w = int(round(w / octave_scale))
        sizes.append((max(32,h), max(32,w)))
    sizes = list(reversed(sizes)) + [(H, W)]

    x = x0.clone().detach()
    for (h, w) in sizes:
        x = F.interpolate(x, size=(h,w), mode="bilinear", align_corners=False).detach()
        x = torch.nn.Parameter(x)
        opt = torch.optim.Adam([x], lr=lr)

        with ActivationCatcher(backbone, target_suffix=target_suffix) as catcher:
            for t in range(steps_per_octave):
                opt.zero_grad()
                if jitter > 0:
                    ox = torch.randint(-jitter, jitter+1, ()).item()
                    oy = torch.randint(-jitter, jitter+1, ()).item()
                    x.data = torch.roll(x.data, shifts=(ox, oy), dims=(2,3))
                px = preprocess_for_vit(x)
                _ = backbone(pixel_values=px, output_hidden_states=True)
                z = catcher.get_vec()
                score = torch.sum(z * w_vec)
                loss = -score + tv_weight * total_variation(x)
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

# ====== Helpers for single image explain ======
def _embed_single(backbone, processor, pil: Image.Image, device: str) -> torch.Tensor:
    inputs = processor(images=pil, return_tensors="pt").to(device)
    out = backbone(**inputs)
    z = getattr(out, "pooler_output", None)
    if z is None:
        z = out.last_hidden_state[:, 0]
    return z

def _save_prob_plot(class_names: List[str], probs: np.ndarray, out_png: str, topk: int = 5):
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

# def _overlay_heatmap(pil: Image.Image, heat: np.ndarray, out_png: str, alpha: float = 0.45):
#     heat = heat.astype(np.float32)
#     heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
#     cmap = plt.get_cmap("hot")
#     colored = (cmap(heat)[..., :3] * 255).astype(np.uint8)
#     hm = Image.fromarray(colored).resize(pil.size, Image.BICUBIC)
#     out = Image.blend(pil.convert("RGB"), hm.convert("RGB"), alpha=alpha)
#     Path(out_png).parent.mkdir(parents=True, exist_ok=True)
#     out.save(out_png)

def _overlay_heatmap(pil: Image.Image, heat: np.ndarray, out_png: str, alpha: float = 0.5, sharpen: bool = True):
    from servers.xai_maps import overlay_heatmap
    overlay_heatmap(pil, heat, out_png, alpha=alpha, sharpen=sharpen)


def _attention_rollout(backbone, processor, pil: Image.Image, device: str) -> np.ndarray:
    from servers.xai_maps import compute_attention_rollout, upsample_map
    inputs = processor(images=pil, return_tensors="pt").to(device)
    heat = compute_attention_rollout(backbone, inputs["pixel_values"])
    return upsample_map(heat, size=max(pil.size))

def _occlusion_map(backbone, clf, pil: Image.Image, processor, model_id: str, 
                   patch_size: int = 16, stride: int = 8, pytorch_model=None) -> np.ndarray:
    """
    Generate occlusion map by systematically occluding patches and measuring prediction changes.
    
    Args:
        backbone: Backbone model (for sklearn) or full model (for PyTorch)
        clf: Classifier (sklearn or PyTorch wrapper)
        pil: Input image
        processor: Image processor
        model_id: Model ID
        patch_size: Size of occlusion patch
        stride: Stride for sliding window
        pytorch_model: Full PyTorch model (if using PyTorch classifier)
    
    Returns:
        Occlusion map as numpy array
    """
    device = next(backbone.parameters()).device
    img_array = np.array(pil)
    h, w = img_array.shape[:2]
    
    # Get baseline prediction
    if pytorch_model is not None:
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = pytorch_model(**inputs)
            probs = torch.softmax(logits, dim=1)
            baseline_prob = probs[0].max().item()
            baseline_class = probs[0].argmax().item()
    else:
        z = _embed_single(backbone, processor, pil, device)
        probs = clf.predict_proba(z.detach().cpu().numpy())[0]
        baseline_prob = probs.max()
        baseline_class = probs.argmax()
    
    # Create occlusion map
    occlusion_map = np.zeros((h, w), dtype=np.float32)
    
    # Slide window over image
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Create occluded image
            occluded_img = img_array.copy()
            occluded_img[y:y+patch_size, x:x+patch_size] = 0  # Black patch
            occluded_pil = Image.fromarray(occluded_img)
            
            # Get prediction on occluded image
            if pytorch_model is not None:
                inputs = processor(images=occluded_pil, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = pytorch_model(**inputs)
                    probs = torch.softmax(logits, dim=1)
                    occluded_prob = probs[0, baseline_class].item()
            else:
                z_occ = _embed_single(backbone, processor, occluded_pil, device)
                probs_occ = clf.predict_proba(z_occ.detach().cpu().numpy())[0]
                occluded_prob = probs_occ[baseline_class]
            
            # Measure change in prediction
            importance = baseline_prob - occluded_prob
            
            # Fill occlusion map
            occlusion_map[y:y+patch_size, x:x+patch_size] = np.maximum(
                occlusion_map[y:y+patch_size, x:x+patch_size], 
                importance
            )
    
    # Normalize
    occlusion_map = (occlusion_map - occlusion_map.min()) / (occlusion_map.max() - occlusion_map.min() + 1e-8)
    return occlusion_map

def _occlusion_map(backbone, clf, pil: Image.Image, processor, model_id: str, 
                   patch_size: int = 16, stride: int = 8, pytorch_model=None) -> np.ndarray:
    """
    Generate occlusion map by systematically occluding patches and measuring prediction changes.
    """
    device = next(backbone.parameters()).device
    img_array = np.array(pil)
    h, w = img_array.shape[:2]
    
    # Get baseline prediction
    if pytorch_model is not None:
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = pytorch_model(**inputs)
            probs = torch.softmax(logits, dim=1)
            baseline_prob = probs[0].max().item()
            baseline_class = probs[0].argmax().item()
    else:
        z = _embed_single(backbone, processor, pil, device)
        probs = clf.predict_proba(z.detach().cpu().numpy())[0]
        baseline_prob = probs.max()
        baseline_class = probs.argmax()
    
    # Create occlusion map
    occlusion_map = np.zeros((h, w), dtype=np.float32)
    
    # Slide window over image
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Create occluded image
            occluded_img = img_array.copy()
            occluded_img[y:y+patch_size, x:x+patch_size] = 0  # Black patch
            occluded_pil = Image.fromarray(occluded_img)
            
            # Get prediction on occluded image
            if pytorch_model is not None:
                inputs = processor(images=occluded_pil, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = pytorch_model(**inputs)
                    probs = torch.softmax(logits, dim=1)
                    occluded_prob = probs[0, baseline_class].item()
            else:
                z_occ = _embed_single(backbone, processor, occluded_pil, device)
                probs_occ = clf.predict_proba(z_occ.detach().cpu().numpy())[0]
                occluded_prob = probs_occ[baseline_class]
            
            # Measure change in prediction
            importance = baseline_prob - occluded_prob
            
            # Fill occlusion map
            occlusion_map[y:y+patch_size, x:x+patch_size] = np.maximum(
                occlusion_map[y:y+patch_size, x:x+patch_size], 
                importance
            )
    
    # Normalize
    occlusion_map = (occlusion_map - occlusion_map.min()) / (occlusion_map.max() - occlusion_map.min() + 1e-8)
    return occlusion_map

def _gradient_saliency(backbone, w_vec: torch.Tensor, pil: Image.Image, pytorch_model=None, class_idx=None) -> np.ndarray:
    """
    Compute gradient saliency map.
    
    Args:
        backbone: Backbone model (for sklearn) or full model (for PyTorch)
        w_vec: Weight vector for class (for sklearn) or None (for PyTorch)
        pil: Input image
        pytorch_model: Full PyTorch model (if using PyTorch classifier)
        class_idx: Class index for PyTorch model
    """
    device = next(backbone.parameters()).device
    x = TV.ToTensor()(pil).unsqueeze(0).to(device)
    x.requires_grad_(True)
    
    if pytorch_model is not None:
        from servers.xai_maps import compute_gradient_saliency
        px = preprocess_for_vit(x.detach())
        return compute_gradient_saliency(pytorch_model, px, class_idx)
    else:
        # Sklearn model - use feature space gradient
        px = preprocess_for_vit(x)
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

def _lucent_direction_for_class(backbone, model_id: str, class_w: torch.Tensor, iters: int, out_png: str):
    luc = _lucent_model(model_id, size=224)
    layer_names, _ = get_model_layers(luc)
    resolved = resolve_layer_name(layer_names, "attention.o_proj")
    tfms = transform.standard_transforms.copy()
    device = next(luc.parameters()).device
    w = class_w.to(device)
    imgs = render.render_vis(
        luc,
        objectives.direction(resolved, w),
        param_f=lambda: param.image(w=224, fft=True, decorrelate=True),
        transforms=tfms,
        thresholds=(iters,),
        show_image=False, show_inline=False, progress=True
    )
    arr = _to_hwc_uint8(imgs[-1], default_w=224)
    Image.fromarray(arr).save(out_png)
    return resolved

# ====== API: lucent_prototypes ======
@app.post("/lucent_prototypes")
def lucent_prototypes(req: ProtosReq):
    try:
        clf, class_names, model_id = _load_probe(req.pkl)
        class_to_idx = {c:i for i,c in enumerate(class_names)}
        luc = _lucent_model(model_id, size=224)
        layer_names, _ = get_model_layers(luc)
        resolved = resolve_layer_name(layer_names, req.layer)
        Path(req.out_dir).mkdir(parents=True, exist_ok=True)
        o1_generate_class_prototypes_simple(
            luc, clf, class_names, class_to_idx,
            layer_name=resolved,
            n_prototypes=req.n_prototypes,
            iters=req.iters,
            out_dir=req.out_dir,
            base_seed=0
        )
        return {"ok": True, "dir": req.out_dir, "classes": class_names, "resolved_layer": resolved}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"lucent_prototypes failed: {type(e).__name__}: {e}")

# ====== API: o2_grid ======
@app.post("/o2_grid")
def o2_grid(req: GridReq):
    try:
        rows = []
        for cname in req.classes:
            row = []
            o1_dir = Path(req.out_png).parent / "o1"   # artifacts/explain/o1
            for i in range(1, req.n_prototypes+1):
                cand = list((o1_dir).glob(f"{cname}_proto_{i}.png"))
                if not cand:
                    raise FileNotFoundError(f"Missing prototype for {cname} #{i}")
                row.append(Image.open(cand[0]).convert("RGB"))
            rows.append(row)
        tile_h = max(img.height for r in rows for img in r)
        tile_w = max(img.width  for r in rows for img in r)
        rows = [[img.resize((tile_w, tile_h), Image.BICUBIC) for img in r] for r in rows]
        H = tile_h * len(rows); W = tile_w * req.n_prototypes
        canvas = Image.new("RGB", (W, H), (255,255,255))
        y = 0
        for r in rows:
            x = 0
            for img in r:
                canvas.paste(img, (x, y)); x += tile_w
            y += tile_h
        Path(req.out_png).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(req.out_png)
        return {"ok": True, "grid": req.out_png}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"o2_grid failed: {type(e).__name__}: {e}")

# ====== API: deepdream ======
@app.post("/deepdream")
def deepdream(req: DeepDreamReq):
    try:
        clf, class_names, model_id = _load_probe(req.pkl)
        class_to_idx = {c:i for i,c in enumerate(class_names)}
        if req.target_class not in class_to_idx:
            raise ValueError(f"Unknown class '{req.target_class}'. Available: {class_names}")
        k = class_to_idx[req.target_class]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Handle both sklearn and PyTorch models
        if hasattr(clf, 'coef_'):
            # Sklearn model
            w_vec = torch.tensor(clf.coef_[k], dtype=torch.float32, device=device)
            backbone, _ = load_hf_backbone(model_id)
        elif hasattr(clf, 'model'):
            # PyTorch model wrapper - extract classifier head weights
            pytorch_model = clf.model
            # Get the last linear layer weights (classifier output layer)
            classifier_layers = [m for m in pytorch_model.classifier.modules() if isinstance(m, nn.Linear)]
            if classifier_layers:
                last_layer = classifier_layers[-1]
                w_vec = last_layer.weight[k].detach().clone().to(device)
            else:
                raise ValueError("Could not extract classifier weights from PyTorch model")
            # Use model's backbone
            backbone = pytorch_model.backbone
        else:
            raise ValueError("Unknown classifier type - cannot extract weights")
        img = deepdream_direction_octaves(
            image_path=req.image_path,
            backbone=backbone,
            w_vec=w_vec,
            target_suffix=req.layer_suffix,
            steps_per_octave=req.steps,
            lr=req.lr,
            jitter=req.jitter,
            tv_weight=req.tv_weight,
            num_octaves=(req.num_octaves if req.octaves else 1),
            octave_scale=req.octave_scale
        )
        Path(req.out_png).parent.mkdir(parents=True, exist_ok=True)
        img.save(req.out_png)
        return {"ok": True, "png": req.out_png, "class": req.target_class, "layer_suffix": req.layer_suffix}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"deepdream failed: {type(e).__name__}: {e}")

# ====== API: explain_image ======
@app.post("/explain_image")
def explain_image(req: ExplainImageReq):
    try:
        Path(req.out_dir).mkdir(parents=True, exist_ok=True)
        clf, class_names, model_id = _load_probe(req.pkl)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pil = Image.open(req.image_path).convert("RGB")

        # prediction
        # Handle both sklearn and PyTorch models
        if hasattr(clf, 'predict_proba') and not hasattr(clf, 'model'):
            # Sklearn model - expects features
            backbone, processor = load_hf_backbone(model_id)
            z = _embed_single(backbone, processor, pil, device)
            probs = clf.predict_proba(z.detach().cpu().numpy())[0]
            k = int(np.argmax(probs))
            pred_name = class_names[k]
            w_vec = torch.tensor(clf.coef_[k], dtype=torch.float32, device=device)
        elif hasattr(clf, 'model'):
            # PyTorch model wrapper - expects images
            pytorch_model = clf.model
            # Use model's processor
            from transformers import AutoImageProcessor
            from huggingface_hub import HfFolder
            hf_token = HfFolder.get_token() or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            processor = AutoImageProcessor.from_pretrained(model_id, token=hf_token)
            inputs = processor(images=pil, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = pytorch_model(**inputs)
                probs_tensor = torch.softmax(logits, dim=1)
                probs = probs_tensor[0].cpu().numpy()
            k = int(np.argmax(probs))
            pred_name = class_names[k]
            # Extract classifier head weights for gradient saliency
            classifier_layers = [m for m in pytorch_model.classifier.modules() if isinstance(m, nn.Linear)]
            if classifier_layers:
                last_layer = classifier_layers[-1]
                w_vec = last_layer.weight[k].detach().clone().to(device)
            else:
                raise ValueError("Could not extract classifier weights from PyTorch model")
            # Use model's backbone for explainability
            backbone = pytorch_model.backbone
        else:
            raise ValueError("Unknown classifier type")

        results = {"prediction": {"index": k, "name": pred_name, "prob": float(probs[k])}}
        results["probs"] = {class_names[i]: float(p) for i, p in enumerate(probs)}
        topk_idx = np.argsort(probs)[::-1][:5].tolist()
        results["topk"] = [{"name": class_names[i], "prob": float(probs[i])} for i in topk_idx]

        # attention rollout heatmap
        attn_png = None
        if req.do_attention:
            try:
                heat = _attention_rollout(backbone, processor, pil, device)
                attn_png = str(Path(req.out_dir) / "attn_rollout.png")
                _overlay_heatmap(pil, heat, attn_png, alpha=0.45)
            except Exception as e:
                attn_png = None
                results["attention_error"] = f"{type(e).__name__}: {e}"

        # gradient saliency heatmap
        grad_png = None
        if req.do_gradients:
            try:
                if hasattr(clf, 'model'):
                    # PyTorch model
                    g = _gradient_saliency(backbone, None, pil, pytorch_model=clf.model, class_idx=k)
                else:
                    # Sklearn model
                    g = _gradient_saliency(backbone, w_vec, pil)
                grad_png = str(Path(req.out_dir) / "grad_saliency.png")
                _overlay_heatmap(pil, g, grad_png, alpha=0.45)
            except Exception as e:
                grad_png = None
                results["grad_error"] = f"{type(e).__name__}: {e}"

        # DeepDream for predicted class
        dream_png = None
        if req.do_deepdream:
            try:
                img = deepdream_direction_octaves(
                    image_path=req.image_path,
                    backbone=backbone,
                    w_vec=w_vec,
                    target_suffix=req.layer_suffix,
                    steps_per_octave=req.steps,
                    lr=req.lr,
                    jitter=req.jitter,
                    tv_weight=req.tv_weight,
                    num_octaves=(req.num_octaves if req.octaves else 1),
                    octave_scale=req.octave_scale
                )
                dream_png = str(Path(req.out_dir) / "dream.png")
                img.save(dream_png)
            except Exception as e:
                dream_png = None
                results["dream_error"] = f"{type(e).__name__}: {e}"

        # Lucent class direction image
        lucent_png = None
        if req.do_lucent:
            try:
                lucent_png = str(Path(req.out_dir) / "lucent_class_dir.png")
                _ = _lucent_direction_for_class(backbone, model_id, w_vec, req.lucent_iters, lucent_png)
            except Exception as e:
                lucent_png = None
                results["lucent_error"] = f"{type(e).__name__}: {e}"

        # probs plot
        probs_png = None
        if req.do_probs:
            try:
                probs_png = str(Path(req.out_dir) / "probs.png")
                _save_prob_plot(class_names, np.array(probs), probs_png, topk=5)
            except Exception as e:
                probs_png = None
                results["probs_plot_error"] = f"{type(e).__name__}: {e}"

        # Occlusion map
        occlusion_png = None
        if req.do_occlusion:
            try:
                pytorch_model_for_occ = clf.model if hasattr(clf, 'model') else None
                occ_map = _occlusion_map(
                    backbone, clf, pil, processor, model_id,
                    patch_size=req.occlusion_patch_size,
                    stride=req.occlusion_stride,
                    pytorch_model=pytorch_model_for_occ
                )
                occlusion_png = str(Path(req.out_dir) / "occlusion_map.png")
                _overlay_heatmap(pil, occ_map, occlusion_png, alpha=0.45)
            except Exception as e:
                occlusion_png = None
                results["occlusion_error"] = f"{type(e).__name__}: {e}"

        results.update({
            "ok": True,
            "attn_rollout_png": attn_png,
            "grad_saliency_png": grad_png,
            "dream_png": dream_png,
            "lucent_png": lucent_png,
            "probs_png": probs_png,
            "occlusion_map_png": occlusion_png
        })
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"explain_image failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)

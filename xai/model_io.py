from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = PROJECT_ROOT / "artifacts" / "models" / "clf.pth"
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
IMG_SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def checkpoint_sha256(path: Path = DEFAULT_CKPT) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_trained_model(ckpt: Optional[Path] = None, device: Optional[torch.device] = None):
    from servers.pytorch_model_loader import load_pytorch_model

    ckpt = Path(ckpt or DEFAULT_CKPT)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, model_id = load_pytorch_model(str(ckpt), model_id=MODEL_ID)
    model.eval()
    # Inference loader freezes the backbone (requires_grad=False). Explanation
    # methods need a gradient graph. Enabling autograd does not change weight values.
    enable_explanation_autograd(model)
    info = {
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": checkpoint_sha256(ckpt),
        "model_id": model_id,
        "class_names": class_names,
        "device": str(device),
        "weights_modified": False,
        "explanation_autograd_enabled": True,
    }
    return model, info


def enable_explanation_autograd(model) -> None:
    """Allow gradients through a frozen backbone without changing parameters."""
    for p in model.parameters():
        p.requires_grad_(True)


def get_processor():
    from huggingface_hub import HfFolder
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(MODEL_ID, token=HfFolder.get_token())


def preprocess_pil(pil: Image.Image, processor=None, device="cpu") -> torch.Tensor:
    processor = processor or get_processor()
    return processor(images=pil.convert("RGB"), return_tensors="pt")["pixel_values"].to(device)


def predict(model, pixel_values: torch.Tensor) -> tuple[int, np.ndarray]:
    with torch.no_grad():
        logits = model(pixel_values=pixel_values)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return int(np.argmax(probs)), probs


def logits_close(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-3) -> bool:
    return bool(torch.max(torch.abs(a - b)).item() <= atol)


def compare_eager_vs_default(model, pixel_values: torch.Tensor) -> dict:
    """Confirm switching to eager attention does not materially change logits."""
    from xai.tokens import force_eager_attention

    model.eval()
    with torch.no_grad():
        logits_default = model(pixel_values=pixel_values)
    note = force_eager_attention(model.backbone)
    with torch.no_grad():
        logits_eager = model(pixel_values=pixel_values)
    max_abs = float(torch.max(torch.abs(logits_default - logits_eager)).item())
    return {
        "max_abs_logit_diff": max_abs,
        "materially_unchanged": max_abs <= 1e-3,
        "attention_backend_note": note,
    }

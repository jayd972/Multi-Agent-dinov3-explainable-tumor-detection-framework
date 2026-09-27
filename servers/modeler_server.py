from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from typing import List
import os, json, numpy as np, joblib, torch, torchvision as tv
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from utils.secrets import load_secrets
load_secrets()

app = FastAPI(title="modeler")

# ---------- Request Schemas ----------
class ExtractReq(BaseModel):
    data_root: str
    model_id: str
    size: int = 224
    out_train_npz: str
    out_test_npz: str

class TrainReq(BaseModel):
    train_npz: str
    out_pkl: str

class EvalReq(BaseModel):
    test_npz: str
    pkl: str
    out_json: str

class FinetuneReq(BaseModel):
    data_root: str
    model_id: str
    size: int = 224
    epochs: int = 3
    lr: float = 1e-5
    out_dir: str = "artifacts/models/finetuned"

# ---------- HF Backbone Loader ----------
def load_hf_backbone(model_id: str, train_mode=False):
    from transformers import AutoImageProcessor, AutoModel
    from huggingface_hub import HfFolder
    hf_token = HfFolder.get_token() or os.environ.get("HUGGINGFACE_HUB_TOKEN")
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

# ---------- Dataset ----------
class PathImageFolder(tv.datasets.ImageFolder):
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target, path

def make_loader(root_dir: str, size: int, batch: int = 64, shuffle: bool = False, num_workers: int = 0):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tfm = tv.transforms.Compose([
        tv.transforms.Resize((size, size), antialias=True),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(MEAN, STD),
    ])
    ds = PathImageFolder(root_dir, transform=tfm)
    ld = DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=num_workers, pin_memory=False)
    return ld, ds

# ---------- Feature extraction ----------
@torch.no_grad()
def embed_paths(backbone, processor, paths: List[str], device: str):
    pil_batch = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=pil_batch, return_tensors="pt").to(device)
    out = backbone(**inputs)
    z = getattr(out, "pooler_output", None)
    if z is None:
        z = out.last_hidden_state[:, 0]  # CLS token
    return z

@torch.no_grad()
def embed_loader(backbone, processor, loader, device: str):
    feats, labels, paths_all = [], [], []
    for _xb, yb, pb in tqdm(loader, desc="Embedding"):
        z = embed_paths(backbone, processor, list(pb), device)
        feats.append(z.detach().cpu().numpy())
        labels.append(yb.numpy())
        paths_all += list(pb)
    return np.concatenate(feats), np.concatenate(labels), paths_all

# ---------- Routes ----------
@app.post("/extract_features")
def extract_features(req: ExtractReq):
    data_root = Path(req.data_root)
    train_dir = data_root / "Training"
    test_dir  = data_root / "Testing"
    assert train_dir.is_dir() and test_dir.is_dir(), "Expected Training/ and Testing/ folders"

    backbone, processor = load_hf_backbone(req.model_id)
    device = next(backbone.parameters()).device.type

    tr_ld, tr_ds = make_loader(str(train_dir), size=req.size, shuffle=False)
    te_ld, te_ds = make_loader(str(test_dir),  size=req.size, shuffle=False)

    idx_to_class = {v: k for k, v in tr_ds.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    X_tr, y_tr, tr_paths = embed_loader(backbone, processor, tr_ld, device)
    X_te, y_te, te_paths = embed_loader(backbone, processor, te_ld, device)

    Path(req.out_train_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(req.out_train_npz,
             X=X_tr.astype("float32"),
             y=y_tr.astype("int64"),
             paths=np.array(tr_paths, dtype="U"),
             class_names=np.array(class_names, dtype="U"),
             model_id=np.array([req.model_id], dtype="U"))

    Path(req.out_test_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(req.out_test_npz,
             X=X_te.astype("float32"),
             y=y_te.astype("int64"),
             paths=np.array(te_paths, dtype="U"),
             class_names=np.array(class_names, dtype="U"),
             model_id=np.array([req.model_id], dtype="U"))

    return {"ok": True, "train": str(req.out_train_npz), "test": str(req.out_test_npz), "classes": class_names}

@app.post("/train_probe")
def train_probe(req: TrainReq):
    d = np.load(req.train_npz, allow_pickle=True)
    X_tr, y_tr = d["X"], d["y"]
    class_names = d["class_names"].tolist()
    model_id = d["model_id"].tolist()[0]

    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             solver="lbfgs", multi_class="multinomial", n_jobs=-1)
    clf.fit(X_tr, y_tr)

    Path(req.out_pkl).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "class_names": class_names, "model_id": model_id}, req.out_pkl)
    return {"ok": True, "pkl": str(req.out_pkl), "classes": class_names}

@app.post("/eval_probe")
def eval_probe(req: EvalReq):
    try:
        d = np.load(req.test_npz, allow_pickle=True)
        X_te, y_te = d["X"], d["y"]
        paths = d["paths"].tolist() if "paths" in d else None

        pkl_path = Path(req.pkl)

        # Check if it's a PyTorch model (.pth)
        if pkl_path.suffix == '.pth' or 'pth' in str(pkl_path).lower():
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from servers.pytorch_model_loader import load_pytorch_model
            import torch
            from PIL import Image

            # Get model_id and num_classes from test data
            test_model_id = d["model_id"].tolist()[0] if "model_id" in d else "facebook/dinov3-vitb16-pretrain-lvd1689m"
            test_class_names = d["class_names"].tolist() if "class_names" in d else ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
            num_classes = len(test_class_names)

            model, class_names, model_id = load_pytorch_model(str(pkl_path), model_id=test_model_id, num_classes=num_classes)
            device = next(model.parameters()).device

            # Load processor
            from transformers import AutoImageProcessor
            from huggingface_hub import HfFolder
            import os
            hf_token = HfFolder.get_token() or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            processor = AutoImageProcessor.from_pretrained(model_id, token=hf_token)

            # Predict on test images
            y_hat = []
            if paths:
                # Use image paths
                for path in tqdm(paths, desc="Evaluating"):
                    try:
                        img = Image.open(path).convert('RGB')
                        inputs = processor(images=img, return_tensors="pt")
                        # Move inputs to device
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        with torch.no_grad():
                            logits = model(**inputs)  # Unpack inputs dict
                        pred = logits.argmax(dim=1).cpu().item()
                        y_hat.append(pred)
                    except Exception as e:
                        print(f"Error processing {path}: {e}")
                        raise
            else:
                # Fallback: would need to reconstruct images from features (not ideal)
                raise ValueError("PyTorch model requires image paths, but paths not found in test.npz")

            y_hat = np.array(y_hat)
        else:
            # Original sklearn format
            meta = joblib.load(req.pkl)
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

        Path(req.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(req.out_json).write_text(json.dumps(metrics, indent=2))
        return {"ok": True, "metrics": str(req.out_json), "summary": {"acc": acc, "f1_macro": f1_m}}
    except Exception as e:
        import traceback
        error_msg = f"Error in eval_probe: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=error_msg)

# ---------- Fine-tuning Route ----------
@app.post("/finetune_model")
def finetune_model(req: FinetuneReq):
    data_root = Path(req.data_root)
    train_dir = data_root / "Training"
    test_dir = data_root / "Testing"
    assert train_dir.is_dir() and test_dir.is_dir(), "Expected Training/ and Testing/ folders"

    model, processor = load_hf_backbone(req.model_id, train_mode=True)
    device = next(model.parameters()).device

    tr_ld, tr_ds = make_loader(str(train_dir), size=req.size, shuffle=True)
    te_ld, te_ds = make_loader(str(test_dir), size=req.size, shuffle=False)

    num_classes = len(tr_ds.classes)
    head = torch.nn.Linear(model.config.hidden_size, num_classes).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=req.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(req.epochs):
        model.train()
        total_loss = 0
        for xb, yb, _ in tqdm(tr_ld, desc=f"Fine-tuning epoch {epoch+1}/{req.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            out = model(pixel_values=xb)
            if hasattr(out, "pooler_output"):
                z = out.pooler_output
            else:
                z = out.last_hidden_state[:, 0]
            logits = head(z)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(tr_ld):.4f}")

    Path(req.out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(req.out_dir)
    torch.save(head.state_dict(), Path(req.out_dir) / "classifier_head.pt")

    return {"ok": True, "message": f"Fine-tuned model saved at {req.out_dir}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

# servers/reporter_server.py
import os, base64, json
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="reporter")

# Load OpenRouter config directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
api_cfg_path = PROJECT_ROOT / "config" / "api_config.json"

if api_cfg_path.exists():
    try:
        cfg = json.loads(api_cfg_path.read_text(encoding="utf-8"))
        or_cfg = cfg.get("api_keys", {}).get("openrouter", {})
        DEFAULT_KEY = or_cfg.get("key", "")
        DEFAULT_MODEL = or_cfg.get("model", "qwen/qwen3.8-27b")
        DEFAULT_BASE = or_cfg.get("base_url", "https://openrouter.ai/api/v1")
    except Exception:
        DEFAULT_KEY = ""
        DEFAULT_MODEL = "qwen/qwen3.8-27b"
        DEFAULT_BASE = "https://openrouter.ai/api/v1"
else:
    DEFAULT_KEY = ""
    DEFAULT_MODEL = "qwen/qwen3.8-27b"
    DEFAULT_BASE = "https://openrouter.ai/api/v1"

class ExplainReq(BaseModel):
    images: List[Optional[str]]  # file paths, can contain None
    prediction: dict             # {"index": int, "name": str, "prob": float}
    topk: List[dict]             # [{"name": str, "prob": float}, ...]
    out_txt: Optional[str] = None
    model: str = DEFAULT_MODEL
    language: str = "en"

def _file_to_data_url(path: str) -> str:
    if path is None or not Path(path).exists(): 
        return ""
    b = Path(path).read_bytes()
    enc = base64.b64encode(b).decode("utf-8")
    ext = Path(path).suffix.lower().lstrip(".") or "png"
    return f"data:image/{ext};base64,{enc}"

@app.post("/explain_results")
def explain_results(req: ExplainReq):
    try:
        api_key = os.getenv("OPENROUTER_API_KEY", DEFAULT_KEY)
        if not api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY / OpenRouter key in api_config.json")
            
        client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE)

        imgs = [p for p in req.images if p]
        content = [{"type": "text", "text":
            "You are a cautious assistant for tumor image analysis. "
            "Use plain language. This is a medical diagnosis. "
            "Describe what each visualization suggests about the model reasoning. "
            f"Predicted class: {req.prediction.get('name')} with probability {req.prediction.get('prob'):.3f}. "
            f"Top scores: " + ", ".join([f"{t['name']}: {t['prob']:.3f}" for t in req.topk])
        }]

        labels = ["Attention rollout", "Gradient saliency", "DeepDream", "Probabilities", "Lucent class direction"]
        for label, p in zip(labels, req.images):
            if p and Path(p).exists():
                content.append({"type": "text", "text": f"{label}:"})
                content.append({"type": "image_url", "image_url": {"url": _file_to_data_url(p)}})

        resp = client.chat.completions.create(
            model=req.model,
            temperature=0.2,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": "Be accurate and clear. Avoid speculation. Add a caution that this is for research explanation and not a clinical diagnosis."},
                {"role": "user", "content": content}
            ],
            extra_headers={
                "HTTP-Referer": "https://github.com/google-deepmind/antigravity",
                "X-Title": "Antigravity Explainable Tumor Detection Framework",
            }
        )
        text = resp.choices[0].message.content.strip()
        if req.out_txt:
            Path(req.out_txt).parent.mkdir(parents=True, exist_ok=True)
            Path(req.out_txt).write_text(text, encoding="utf-8")
        return {"ok": True, "text": text, "out_txt": req.out_txt}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"explain_results failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)


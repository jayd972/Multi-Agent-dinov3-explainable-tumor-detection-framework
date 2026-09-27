import os, json
from pathlib import Path

def load_secrets():
    cfg_path = Path("config/api_config.json")
    if not cfg_path.exists():
        return {}

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    api = cfg.get("api_keys", {})

    openai = api.get("openai", {}).get("key")
    hf = api.get("huggingface", {}).get("token")

    if openai:
        os.environ["OPENAI_API_KEY"] = openai
    if hf:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf

    return {"openai": openai, "huggingface": hf}

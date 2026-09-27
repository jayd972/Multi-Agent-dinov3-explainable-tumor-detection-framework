# agents/modeler_client.py
import requests

def _post(base: str, route: str, **kwargs):
    url = f"{base.rstrip('/')}/{route.lstrip('/')}"
    resp = requests.post(url, json=kwargs, timeout=600)
    if not resp.ok:
        print(f"Server error response ({resp.status_code}):")
        print(f"  URL: {url}")
        print(f"  Response text: {resp.text[:1000]}")  # First 1000 chars
        try:
            error_json = resp.json()
            print(f"  Response JSON: {error_json}")
            if "detail" in error_json:
                print(f"  Detail: {error_json['detail']}")
        except:
            pass
    resp.raise_for_status()
    return resp.json()

def extract_features(base: str, **kwargs):
    return _post(base, "/extract_features", **kwargs)

def train_probe(base: str, **kwargs):
    return _post(base, "/train_probe", **kwargs)

def eval_probe(base: str, **kwargs):
    return _post(base, "/eval_probe", **kwargs)

def finetune_model(base: str, **kwargs):
    return _post(base, "/finetune_model", **kwargs)

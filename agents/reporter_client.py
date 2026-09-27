# agents/reporter_client.py
import requests

def _post(base: str, route: str, **kwargs):
    url = f"{base.rstrip('/')}/{route.lstrip('/')}"
    resp = requests.post(url, json=kwargs, timeout=600)
    resp.raise_for_status()
    return resp.json()

def explain_results(base: str, **kwargs):
    return _post(base, "/explain_results", **kwargs)

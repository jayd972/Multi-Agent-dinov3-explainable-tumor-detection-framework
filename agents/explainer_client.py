# agents/explainer_client.py
import os
import requests

_DEFAULT_TIMEOUT = os.getenv("MCP_HTTP_TIMEOUT")
DEFAULT_TIMEOUT = None if _DEFAULT_TIMEOUT is None else float(_DEFAULT_TIMEOUT)

def _post(base: str, route: str, timeout=DEFAULT_TIMEOUT, **kwargs):
    url = f"{base.rstrip('/')}/{route.lstrip('/')}"
    resp = requests.post(url, json=kwargs, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def lucent_prototypes(base: str, **kwargs):
    return _post(base, "/lucent_prototypes", **kwargs)

def o2_grid(base: str, **kwargs):
    return _post(base, "/o2_grid", **kwargs)

def deepdream(base: str, **kwargs):
    return _post(base, "/deepdream", **kwargs)

def explain_image(base: str, **kwargs):
    return _post(base, "/explain_image", **kwargs)

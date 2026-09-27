# utils/io.py
import json
from pathlib import Path
def read_json(p): return json.loads(Path(p).read_text())
def write_json(p, obj): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(obj, indent=2))

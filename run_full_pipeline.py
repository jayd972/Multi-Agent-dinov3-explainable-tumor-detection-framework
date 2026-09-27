"""
Run the full pipeline: start servers, wait for them, then run the pipeline.
"""
import subprocess
import sys
import time
import requests
import os
from pathlib import Path

def check_server(port, name):
    """Check if a server is running."""
    try:
        url = f"http://127.0.0.1:{port}/docs" if port == 8001 else f"http://127.0.0.1:{port}"
        resp = requests.get(url, timeout=2)
        return True
    except:
        return False

def start_servers(hf_token=None):
    """Start all servers in background."""
    print("🚀 Starting servers...")
    
    servers = [
        ("servers/modeler_server.py", 8001, "Modeler"),
        ("servers/explainer_server.py", 8002, "Explainer"),
        ("servers/reporter_server.py", 8003, "Reporter"),
    ]
    
    processes = []
    env = os.environ.copy()
    if hf_token:
        env["HUGGINGFACE_HUB_TOKEN"] = hf_token
        print(f"   Using Hugging Face token: {hf_token[:10]}...")
    
    for script, port, name in servers:
        print(f"   Starting {name} on port {port}...")
        if sys.platform == "win32":
            proc = subprocess.Popen(
                [sys.executable, script],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=Path(__file__).parent,
                env=env
            )
        else:
            proc = subprocess.Popen(
                [sys.executable, script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent,
                env=env
            )
        processes.append((proc, name, port))
        time.sleep(2)
    
    # Wait for servers to be ready
    print("\n⏳ Waiting for servers to initialize...")
    max_wait = 30
    waited = 0
    while waited < max_wait:
        all_ready = all(check_server(port, name) for _, name, port in processes)
        if all_ready:
            print("✅ All servers are ready!")
            return processes
        time.sleep(1)
        waited += 1
        if waited % 5 == 0:
            print(f"   Still waiting... ({waited}s)")
    
    print("⚠️  Some servers may not be ready, but continuing...")
    return processes

def main():
    """Main function."""
    print("="*60)
    print("🧠 Brain Tumor Classification Pipeline")
    print("="*60)
    
    # Get HF token from environment or use default
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        print(f"✅ Using Hugging Face token from environment")
    else:
        print("⚠️  No HUGGINGFACE_HUB_TOKEN found in environment")
    
    # Start servers
    processes = start_servers(hf_token=hf_token)
    
    # Run pipeline
    print("\n" + "="*60)
    print("🔄 Running Pipeline")
    print("="*60)
    
    from agents.orchestrator import run_pipeline
    from utils.io import read_json
    
    cfg = read_json("config/task_card.json")
    
    try:
        result = run_pipeline(cfg)
        print("\n" + "="*60)
        print("✅ Pipeline Completed!")
        print("="*60)
        print(f"Results saved to:")
        print(f"  - Metrics: artifacts/metrics/metrics.json")
        print(f"  - Lucent Prototypes: artifacts/explain/o1/")
        print(f"  - Single Image Explanations: artifacts/explain/by_image/")
        if result.get("single_image"):
            img_dir = Path(result["single_image"].get("out_dir", ""))
            if img_dir.exists():
                print(f"\nGenerated files:")
                for f in img_dir.glob("*.png"):
                    print(f"  - {f.name}")
    except Exception as e:
        print("\n" + "="*60)
        print("❌ Pipeline Failed")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n⚠️  Servers are still running. Close the server windows to stop them.")
    return 0

if __name__ == "__main__":
    exit(main())


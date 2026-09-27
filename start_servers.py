"""
Start all the FastAPI servers for the multi-agent system.
"""
import subprocess
import sys
import time
import os
from pathlib import Path

def start_server(script_path, port, name):
    """Start a server process."""
    print(f"🚀 Starting {name} on port {port}...")
    # Use pythonw on Windows to run in background, or python with nohup on Linux
    if sys.platform == "win32":
        proc = subprocess.Popen(
            [sys.executable, script_path],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=Path(__file__).parent
        )
    else:
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent
        )
    return proc

def main():
    """Start all servers."""
    print("="*60)
    print("Starting Multi-Agent System Servers")
    print("="*60)
    
    servers = [
        ("servers/modeler_server.py", 8001, "Modeler Agent"),
        ("servers/explainer_server.py", 8002, "Explainer Agent"),
        ("servers/reporter_server.py", 8003, "Reporter Agent"),
    ]
    
    processes = []
    for script, port, name in servers:
        proc = start_server(script, port, name)
        processes.append((proc, name, port))
        time.sleep(2)  # Give each server time to start
    
    print("\n" + "="*60)
    print("✅ All servers started!")
    print("="*60)
    print("\nServers running:")
    for proc, name, port in processes:
        print(f"  {name}: http://127.0.0.1:{port}")
    
    print("\n⚠️  Keep this window open to keep servers running")
    print("   Press Ctrl+C to stop all servers")
    
    try:
        # Wait for all processes
        for proc, name, port in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping all servers...")
        for proc, name, port in processes:
            proc.terminate()
            print(f"   Stopped {name}")
        print("✅ All servers stopped")

if __name__ == "__main__":
    main()


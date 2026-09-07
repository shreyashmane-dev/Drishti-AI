"""
DRISHTI AI - Main Application Launcher.
Launches the Web GUI Studio with one click.
"""

from pathlib import Path
import sys

# Ensure root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai.server import start_server

if __name__ == "__main__":
    start_server(host="127.0.0.1", port=8000, open_browser=True)

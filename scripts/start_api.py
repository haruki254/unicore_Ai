"""
scripts/start_api.py

Starts the FastAPI trading intelligence server.

Usage:
    python scripts/start_api.py
    python scripts/start_api.py --port 8080 --reload
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from config.settings import settings


def start(host: str = None, port: int = None, reload: bool = False):
    h = host or settings.api_host
    p = port or settings.api_port

    print("=" * 55)
    print("  Trading Intelligence API")
    print(f"  http://{h}:{p}")
    print(f"  Docs → http://{h}:{p}/docs")
    print(f"  Health → http://{h}:{p}/health")
    print("=" * 55)

    uvicorn.run(
        "api.main:app",
        host    = h,
        port    = p,
        reload  = reload,
        log_level = settings.log_level.lower(),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host",   default=None)
    p.add_argument("--port",   type=int, default=None)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()
    start(args.host, args.port, args.reload)

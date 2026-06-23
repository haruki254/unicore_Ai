#!/usr/bin/env python3
"""
run.py  —  Trading Intelligence System master launcher
Place this file inside your trading_intelligence folder, then run:

  First time / full reset:
      python run.py --fresh

  Daily startup (just start the server):
      python run.py

  Retrain models then start server:
      python run.py --retrain

  Train only, don't start server:
      python run.py --train-only
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# ── Colours (work on Windows 10+ and all Mac/Linux terminals) ─────────────────
os.system("")  # enable ANSI on Windows
G = "\033[92m"   # green
Y = "\033[93m"   # yellow
R = "\033[91m"   # red
C = "\033[96m"   # cyan
B = "\033[1m"    # bold
X = "\033[0m"    # reset

ROOT = Path(__file__).parent.resolve()


# ── Helpers ───────────────────────────────────────────────────────────────────

def banner():
    print(f"""
{C}{B}╔══════════════════════════════════════════════╗
║   Trading Intelligence System  v1.0          ║
║   Dual-AI Signal Filter for MetaTrader 5     ║
╚══════════════════════════════════════════════╝{X}
""")


def section(n, total, title):
    print(f"\n{B}── Step {n}/{total}: {title}{X}")
    print("  " + "─" * 48)


def ok(msg):    print(f"  {G}✓  {msg}{X}")
def warn(msg):  print(f"  {Y}⚠  {msg}{X}")
def info(msg):  print(f"     {msg}")
def err(msg):   print(f"  {R}✗  {msg}{X}")


def run(script, *flags):
    """Run a Python script from the ROOT folder. Returns True on success."""
    cmd = [sys.executable, str(ROOT / script)] + list(flags)
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode == 0


# ── Checks ────────────────────────────────────────────────────────────────────

def check_env():
    env = ROOT / ".env"
    if env.exists():
        ok(".env file found")
        return True

    warn(".env file is missing!")
    info("Create a file called  .env  inside your trading_intelligence folder")
    info("with this content (change the password to anything you like):")
    print()
    print(f"  {C}API_SECRET_KEY=my-trading-password-123")
    print(f"  API_DEBUG=true{X}")
    print()
    ans = input("  Continue without it? (y/n): ").strip().lower()
    if ans != "y":
        print(f"\n  {Y}Stopped. Create your .env file and run again.{X}\n")
        sys.exit(1)
    return False


def data_ready():
    """Returns True if synthetic training data already exists."""
    d = ROOT / "data"
    return d.exists() and any(d.glob("*.pkl"))


def models_ready():
    """Returns True if trained model files already exist."""
    m = ROOT / "models_saved"
    return m.exists() and any(m.glob("*.pkl"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Trading Intelligence master launcher"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Regenerate training data + retrain models + start server",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain models (skip data generation) then start server",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Generate data + train models, then exit (don't start server)",
    )
    args = parser.parse_args()

    banner()

    # ── Decide what to do ─────────────────────────────────────────────────────
    do_data   = args.fresh or (not data_ready() and not args.retrain)
    do_train  = args.fresh or args.retrain or not models_ready()
    do_server = not args.train_only

    # Count total steps
    total = 1 + int(do_data) + int(do_train) + int(do_server)
    cur = 1

    # ── Step 1: .env check ────────────────────────────────────────────────────
    section(cur, total, "Checking configuration")
    check_env()
    cur += 1

    # ── Step 2: Generate training data ────────────────────────────────────────
    if do_data:
        section(cur, total, "Generating synthetic training data")
        info("Creating 2,000 example trades for the AI to learn from ...")
        if run("scripts/generate_sample_data.py"):
            ok("Training data generated  →  data/sample_trades.pkl")
        else:
            err("Data generation failed. Check the output above for details.")
            sys.exit(1)
        cur += 1
    else:
        info("(Training data already exists — skipping generation)")

    # ── Step 3: Train models ──────────────────────────────────────────────────
    if do_train:
        section(cur, total, "Training Trader AI + Risk Manager AI")
        info("This usually takes 2–5 minutes ...")
        if run("scripts/train_models.py", "--force"):
            ok("Models trained and saved  →  models_saved/")
        else:
            err("Training failed. Check the output above for details.")
            sys.exit(1)
        cur += 1
    else:
        info("(Trained models already exist — skipping training)")

    # ── Step 4: Start API server ──────────────────────────────────────────────
    if do_server:
        section(cur, total, "Starting the AI server")
        print()
        print(f"  {G}{B}Server is starting — keep this window open while trading!{X}")
        print()
        info(f"Local URL  :  {C}http://localhost:8000{X}")
        info(f"Health     :  {C}http://localhost:8000/health{X}")
        info(f"API docs   :  {C}http://localhost:8000/docs{X}")
        print()
        info(f"Configure MT5 EA with:")
        info(f"  API_URL  =  http://127.0.0.1:8000")
        info(f"  API_KEY  =  (@Youtube2017)")
        print()
        info(f"{Y}Press Ctrl+C to stop the server.{X}")
        print()
        run("scripts/start_api.py")
    else:
        print()
        ok(f"Done! Training complete.")
        info("Run  python run.py  to start the server whenever you want to trade.")
        print()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
run.py  —  Trading Intelligence System master launcher
Place this file inside your trading_intelligence folder, then run:

      python run.py

Every run always does all three steps in order:
  1. Generate training data   (scripts/generate_sample_data.py)
  2. Train Trader AI + Risk Manager AI   (scripts/train_models.py --force)
  3. Start the API server   (scripts/start_api.py)

  Train only, don't start server:
      python run.py --train-only

--fresh and --retrain are accepted but no longer change anything — data
generation and training now always run regardless of these flags, so they're
kept only so old commands/scripts that still pass them don't error out.
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
    """
    Returns True if synthetic training data already exists.
    NOTE: no longer called from main() — do_data is now unconditionally
    True on every run. Left in place in case you want conditional
    skipping back later.
    """
    d = ROOT / "data"
    return d.exists() and any(d.glob("*.pkl"))


def models_ready():
    """
    Returns True if trained model files already exist.
    NOTE: no longer called from main() — do_train is now unconditionally
    True on every run. Left in place in case you want conditional
    skipping back later.

    Also, pre-existing bug worth knowing about if you ever revive this:
    it globs for "*.pkl", but train_models.py actually saves
    trader_ai.joblib / risk_manager.joblib (.joblib, not .pkl) — so this
    would always return False even with fully trained models sitting in
    models_saved/. Not fixed here since the function isn't in use.
    """
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
        help="(no-op — data generation and training now always run every time)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="(no-op — data generation and training now always run every time)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Generate data + train models, then exit (don't start server)",
    )
    args = parser.parse_args()

    banner()

    # ── Decide what to do ─────────────────────────────────────────────────────
    # Always generate data and train, every single run — no skipping based on
    # whether data/models already exist. --fresh and --retrain are accepted
    # (see argparse above) but don't affect this anymore since it's now
    # unconditional either way.
    do_data   = True
    do_train  = True
    do_server = not args.train_only

    # Count total steps
    total = 1 + int(do_data) + int(do_train) + int(do_server)
    cur = 1

    # ── Step 1: .env check ────────────────────────────────────────────────────
    section(cur, total, "Checking configuration")
    check_env()
    cur += 1

    # ── Step 2: Generate training data ────────────────────────────────────────
    section(cur, total, "Generating training data")
    info("Pulling real closed trades from trade_history for the AI to learn from ...")
    if run("scripts/generate_sample_data.py"):
        ok("Training data generated  →  data/sample_trades.pkl")
    else:
        err("Data generation failed. Check the output above for details.")
        sys.exit(1)
    cur += 1

    # ── Step 3: Train models ──────────────────────────────────────────────────
    section(cur, total, "Training Trader AI + Risk Manager AI")
    info("This usually takes under a minute for small datasets ...")
    if run("scripts/train_models.py", "--force"):
        ok("Models trained and saved  →  models_saved/")
    else:
        err("Training failed. Check the output above for details.")
        sys.exit(1)
    cur += 1

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
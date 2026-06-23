"""
scripts/train_models.py

Trains Trader AI and Risk Manager AI on saved trade data.
Runs walk-forward validation, selects best algorithm per model,
saves models to ./models_saved/

Usage:
    python scripts/train_models.py                  # uses ./data/sample_trades.pkl
    python scripts/train_models.py --force          # retrain even if recently trained
"""

import sys, os, argparse, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from core.models.trader_ai       import TraderAI
from core.models.risk_manager_ai import RiskManagerAI
from core.learning.pipeline      import LearningPipeline
from config.settings             import settings


DATA_FILE = Path("./data/sample_trades.pkl")


def load_trades() -> list:
    if not DATA_FILE.exists():
        print(f"❌  {DATA_FILE} not found.")
        print("    Run first:  python scripts/generate_sample_data.py")
        sys.exit(1)
    with open(DATA_FILE, "rb") as f:
        trades = pickle.load(f)
    print(f"✅  Loaded {len(trades)} trades from {DATA_FILE}")
    return trades


def train(force: bool = False):
    trades = load_trades()

    trader_ai    = TraderAI()
    risk_manager = RiskManagerAI()

    # ── Trader AI ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("  TRAINING TRADER AI")
    print("="*55)
    t0 = time.perf_counter()
    X_trader, df_trader = trader_ai.build_training_data(trades, None)
    if not X_trader.empty:
        metrics = trader_ai.train(X_trader, df_trader)
        best    = trader_ai.get_best_metrics()
        print(f"\n  Best algorithm : {trader_ai.best_algorithm}")
        print(f"  ROC-AUC (WF)   : {best.roc_auc:.4f}" if best else "  No metrics")
        print(f"  Elapsed        : {time.perf_counter()-t0:.1f}s")
    else:
        print("  ⚠️  No training data available for Trader AI")

    # ── Risk Manager AI ───────────────────────────────────────
    print("\n" + "="*55)
    print("  TRAINING RISK MANAGER AI")
    print("="*55)
    t1 = time.perf_counter()
    X_risk, df_risk = risk_manager.build_training_data(trades)
    if not X_risk.empty:
        metrics = risk_manager.train(X_risk, df_risk)
        best    = risk_manager.get_best_metrics()
        print(f"\n  Best algorithm : {risk_manager.best_algorithm}")
        print(f"  ROC-AUC (WF)   : {best.roc_auc:.4f}" if best else "  No metrics")
        print(f"  Elapsed        : {time.perf_counter()-t1:.1f}s")
    else:
        print("  ⚠️  No training data available for Risk Manager AI")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  TRAINING COMPLETE")
    print(f"  Trader AI     → {settings.model_save_path}/trader_ai.joblib")
    print(f"  Risk Manager  → {settings.model_save_path}/risk_manager.joblib")
    print("="*55)
    print("\nNext: python scripts/start_api.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    train(force=args.force)

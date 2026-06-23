"""
scripts/backtest.py

Runs the full AI pipeline against a historical OHLCV CSV.

Expected CSV columns (case-insensitive):
  time, open, high, low, close, volume, ea_signal

ea_signal column should contain BUY or SELL for each bar
(from your original strategy/indicator).

Usage:
    python scripts/backtest.py --csv path/to/data.csv
    python scripts/backtest.py --csv data.csv --tp 20 --sl 10
    python scripts/backtest.py --demo           # generates 1000-bar demo
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

from backtesting.engine import BacktestEngine
from monitoring.logger  import backtest_logger


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if "ea_signal" not in df.columns:
        # Default: alternating BUY/SELL for demo
        df["ea_signal"] = ["BUY" if i % 2 == 0 else "SELL" for i in range(len(df))]
        backtest_logger.warning("No ea_signal column found — using alternating BUY/SELL")
    if "time" not in df.columns and "date" in df.columns:
        df["time"] = df["date"]
    return df


def generate_demo_data(n: int = 1000) -> pd.DataFrame:
    """Generate a realistic demo dataset."""
    np.random.seed(42)
    closes = 1.0850 + np.cumsum(np.random.randn(n) * 0.0003)
    highs  = closes + np.abs(np.random.randn(n) * 0.0005)
    lows   = closes - np.abs(np.random.randn(n) * 0.0005)
    opens  = np.roll(closes, 1); opens[0] = closes[0]
    vols   = np.random.randint(500, 5000, n).astype(float)

    # Add next-bar high/low so P&L simulation can work
    nxt_h  = np.roll(highs, -1); nxt_h[-1] = highs[-1]
    nxt_l  = np.roll(lows,  -1); nxt_l[-1] = lows[-1]

    df = pd.DataFrame({
        "time":       pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open":       opens,
        "high":       highs,
        "low":        lows,
        "close":      closes,
        "volume":     vols,
        "next_high":  nxt_h,
        "next_low":   nxt_l,
        "ea_signal":  np.random.choice(["BUY","SELL"], n),
    })
    print(f"Demo data generated: {n} bars")
    return df


def run(args):
    # ── Load data ─────────────────────────────────────────────
    if args.demo or not args.csv:
        df = generate_demo_data(args.bars)
    else:
        if not Path(args.csv).exists():
            print(f"❌ File not found: {args.csv}")
            sys.exit(1)
        df = load_csv(args.csv)
        print(f"Loaded {len(df)} rows from {args.csv}")

    # ── Run backtest ──────────────────────────────────────────
    engine = BacktestEngine(
        tp_pips     = args.tp,
        sl_pips     = args.sl,
        spread_pips = args.spread,
    )

    print(f"\nRunning backtest (TP={args.tp}p SL={args.sl}p Spread={args.spread}p)...")
    metrics = engine.run(df, walk_forward=not args.no_wf)

    # ── Save results ──────────────────────────────────────────
    out = args.output or "./data/backtest_results.json"
    engine.save_results(metrics, out)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Trading Intelligence Backtester")
    p.add_argument("--csv",     default=None,  help="Path to OHLCV CSV file")
    p.add_argument("--demo",    action="store_true", help="Use generated demo data")
    p.add_argument("--bars",    type=int, default=1000, help="Demo bars (default 1000)")
    p.add_argument("--tp",      type=float, default=20.0, help="Take profit pips")
    p.add_argument("--sl",      type=float, default=10.0, help="Stop loss pips")
    p.add_argument("--spread",  type=float, default=1.5,  help="Spread pips")
    p.add_argument("--no-wf",   action="store_true", help="Disable walk-forward split")
    p.add_argument("--output",  default=None, help="Output JSON path")
    args = p.parse_args()
    run(args)

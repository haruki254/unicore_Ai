"""
scripts/generate_sample_data.py

Pulls real trade history from Supabase and converts it into labeled
training data for the ML models.

If fewer than MIN_REAL_TRADES exist in the DB, it falls back to
synthetic augmentation so early bootstrapping still works.

Usage:
    python scripts/generate_sample_data.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ── Supabase ──────────────────────────────────────────────────
try:
    from supabase import create_client, Client
except ImportError:
    raise SystemExit(
        "supabase-py not installed.\n"
        "Run:  pip install supabase"
    )

from core.features.feature_pipeline import FeaturePipeline
from core.memory.trade_memory       import TradeMemoryEngine
from config.settings                import settings, ALL_FEATURE_NAMES, MARKET_REGIMES, SESSIONS
from monitoring.logger              import model_logger

np.random.seed(2024)

OUT_PATH         = Path("./data/sample_trades.pkl")
MIN_REAL_TRADES  = 0           # always train on real data only — no synthetic fallback
AUGMENT_TARGET   = 0           # unused


# ── Supabase credentials ──────────────────────────────────────
SUPABASE_URL = "https://dwitvurcslwyhiwybzki.supabase.co"
SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR3aXR2dXJjc2x3eWhpd3liemtpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjE3MzMxMTksImV4cCI6MjA3NzMwOTExOX0"
    ".lqXY9j8tAS9Zd_PhA-Mm73YHP5e1bH8fCSvqKv4WC6g"
)

def _get_supabase() -> "Client":
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ── Fetch all rows from trade_history ────────────────────────
def fetch_trades(client: "Client") -> list[dict]:
    """
    Page through trade_history and return every closed trade.
    Supabase caps single responses at 1,000 rows, so we paginate.
    Joins snapshot features where snapshot_id is set so _row_to_trade
    can use real market conditions instead of fabricated ones.
    """
    rows, page_size, offset = [], 1000, 0
    while True:
        resp = (
            client.table("trade_history")
            .select("*")
            .not_.is_("outcome", "null")          # only labeled trades
            .not_.is_("closed_at", "null")
            .neq("regime", "UNKNOWN")             # exclude unclassified rows
            .order("closed_at", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    print(f"  Fetched {len(rows)} trades from Supabase.")

    # ── Enrich with snapshot features where available ─────────────
    # Collect unique snapshot IDs that are non-null
    snap_ids = list({r["snapshot_id"] for r in rows if r.get("snapshot_id")})
    snap_features: dict[str, dict] = {}

    if snap_ids:
        print(f"  Fetching features for {len(snap_ids)} linked snapshots...")
        # Batch fetch in chunks of 100 to avoid URL length limits
        for i in range(0, len(snap_ids), 100):
            chunk = snap_ids[i:i+100]
            try:
                resp = (
                    client.table("market_snapshots")
                    .select("id,ea_id,features,candles_m5,candles_m15,candles_h1,candles_h4,candles_d1,captured_at,spread_pips,close_price")
                    .in_("id", chunk)
                    .execute()
                )
                for s in (resp.data or []):
                    snap_features[s["id"]] = s
            except Exception as e:
                print(f"  Warning: snapshot batch {i//100+1} failed: {e}")

        # Attach snapshot data to each trade row
        enriched = 0
        for row in rows:
            sid = row.get("snapshot_id")
            if sid and sid in snap_features:
                snap = snap_features[sid]
                row["_snapshot"] = snap
                # Use snapshot ea_id if trade row has none / default
                if not row.get("ea_id") or row["ea_id"] == "default":
                    row["ea_id"] = snap.get("ea_id", "default")
                enriched += 1
        print(f"  Enriched {enriched}/{len(rows)} trades with real snapshot features.")
    else:
        print("  No snapshot_id links found — using reconstructed features (consider running backfill).")

    return rows


# ── Infer regime from DB row ──────────────────────────────────
def _infer_regime(row: dict) -> str:
    """
    Best-effort regime inference from the columns we have.
    Extend this once you log regime to the DB.
    """
    if "regime" in row and row["regime"]:
        return row["regime"]

    pnl   = row.get("pnl_pips", 0) or 0
    direction = (row.get("direction") or "BUY").upper()

    if pnl > 15 and direction == "BUY":
        return "strong_bull_trend"
    if pnl > 15 and direction == "SELL":
        return "strong_bear_trend"
    if pnl > 5 and direction == "BUY":
        return "weak_bull_trend"
    if pnl > 5 and direction == "SELL":
        return "weak_bear_trend"
    if abs(pnl) <= 3:
        return "sideways_range"
    if row.get("max_drawdown_pips", 0) > 20:
        return "high_volatility"
    return "low_volatility"


# ── Convert one DB row → training record ─────────────────────
def _row_to_trade(row: dict, pipeline: FeaturePipeline) -> dict | None:
    """
    Map a trade_history row to the dict format expected by train_models.py.
    Features are derived from the price action we have; anything
    missing is filled with neutral defaults.
    """
    try:
        opened_at = (
            datetime.fromisoformat(row["opened_at"].replace("Z", "+00:00"))
            if row.get("opened_at")
            else datetime.now(timezone.utc)
        )
    except Exception:
        opened_at = datetime.now(timezone.utc)

    regime    = _infer_regime(row)
    session   = row.get("session") or "london"
    direction = (row.get("direction") or "BUY").upper()
    outcome   = (row.get("outcome") or "LOSS").upper()
    pnl       = float(row.get("pnl_pips") or 0)
    max_dd    = float(row.get("max_drawdown_pips") or abs(pnl))
    entry     = float(row.get("entry_price") or 1.0)
    exit_p    = float(row.get("exit_price") or entry)
    spread    = float(row.get("spread_pips") or 1.0) if "spread_pips" in row else 1.0

    # ── Use real snapshot features when available, else reconstruct ──
    snap = row.get("_snapshot")  # attached by fetch_trades() if snapshot_id linked

    if snap and snap.get("features"):
        # Real features from the market snapshot captured at trade entry time
        raw_features = snap["features"]
        if isinstance(raw_features, str):
            import json
            raw_features = json.loads(raw_features)
        features = {k: raw_features.get(k, 0.0) for k in ALL_FEATURE_NAMES}

        # Use real candles from snapshot if available for feature recompute
        if snap.get("candles_m5"):
            try:
                snapshot = {
                    "symbol":      row.get("symbol", "XAUUSD"),
                    "timestamp":   opened_at,
                    "price":       float(snap.get("close_price") or exit_p),
                    "spread_pips": float(snap.get("spread_pips") or spread),
                    "candles_m5":  snap["candles_m5"],
                    "candles_m15": snap.get("candles_m15") or snap["candles_m5"][::3],
                    "candles_h1":  snap.get("candles_h1")  or snap["candles_m5"][::12],
                    "candles_h4":  snap.get("candles_h4")  or snap["candles_m5"][::48],
                    "candles_d1":  snap.get("candles_d1")  or snap["candles_m5"][::288],
                }
                recomputed = pipeline.compute(snapshot)
                # Merge: prefer recomputed but fill gaps from stored features
                for k in ALL_FEATURE_NAMES:
                    if recomputed.get(k) is not None:
                        features[k] = recomputed[k]
            except Exception as e:
                warnings.warn(f"Snapshot recompute failed for ticket {row.get('mt5_ticket')}: {e}")
    else:
        # Fallback: reconstruct a plausible price series from entry → exit
        # This path is used for legacy trades with no snapshot_id
        n = 110
        price_move = (exit_p - entry) / max(n, 1)
        closes = entry + np.cumsum(np.random.randn(n) * abs(price_move) * 0.5) + np.linspace(0, exit_p - entry, n)
        highs  = closes + np.abs(np.random.randn(n) * abs(price_move) * 0.3)
        lows   = closes - np.abs(np.random.randn(n) * abs(price_move) * 0.3)
        opens  = np.roll(closes, 1); opens[0] = closes[0]
        vols   = np.random.randint(200, 2000, n).astype(float)

        df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})
        candles = df.to_dict("records")

        snapshot = {
            "symbol":      row.get("symbol", "XAUUSD"),
            "timestamp":   opened_at,
            "price":       exit_p,
            "spread_pips": spread,
            "candles_m5":  candles,
            "candles_m15": candles[::3],
            "candles_h1":  candles[::12],
            "candles_h4":  candles[::48],
            "candles_d1":  candles[::288],
        }

        try:
            features = pipeline.compute(snapshot)
        except Exception as e:
            warnings.warn(f"Feature computation failed for ticket {row.get('mt5_ticket')}: {e}")
            features = {k: 0.0 for k in ALL_FEATURE_NAMES}

    # win_prob back-calculated from outcome + a little noise
    win_prob = 0.65 if outcome == "WIN" else 0.35
    win_prob = float(np.clip(win_prob + np.random.uniform(-0.05, 0.05), 0.25, 0.85))

    conditions = {k: features.get(k, 0.0) for k in ALL_FEATURE_NAMES}

    # ea_id: prefer from snapshot (authoritative), then from row
    snap = row.get("_snapshot")
    ea_id = (
        (snap.get("ea_id") if snap else None)
        or row.get("ea_id")
        or "default"
    )

    return {
        "id":          str(row.get("id") or uuid.uuid4()),
        "mt5_ticket":  row.get("mt5_ticket"),
        "snapshot_id": row.get("snapshot_id"),
        "ea_id":       ea_id,
        "symbol":      row.get("symbol", "XAUUSD"),
        "direction":   direction,
        "outcome":     outcome,
        "pnl_pips":    round(pnl, 2),
        "regime":      regime,
        "session":     session,
        "opened_at":   opened_at.isoformat(),
        "conditions": conditions,
        "risk_context": {
            "account_drawdown_pct": float(row.get("account_drawdown_pct") or np.random.uniform(0, 0.03)),
            "recent_loss_streak":   int(row.get("recent_loss_streak") or np.random.randint(0, 4)),
            "recent_win_streak":    int(row.get("recent_win_streak") or np.random.randint(0, 5)),
            "trades_today":         int(row.get("trades_today") or np.random.randint(0, 8)),
            "session_quality":      float(features.get("session_quality", 0.5)),
            "spread_pips":          spread,
            "similar_win_rate":     win_prob,
            "similar_avg_pnl":      pnl * 0.9,
            "similar_count":        np.random.randint(5, 30),
        },
        "prediction": {
            "trader_buy_prob":   0.65 if direction == "BUY" else 0.35,
            "trader_sell_prob":  0.35 if direction == "BUY" else 0.65,
            "trader_confidence": abs(0.65 - 0.35),
        },
    }


# ── Synthetic fallback (kept from original, used only when DB is thin) ──
REGIME_WIN_RATE = {
    "strong_bull_trend": 0.68, "weak_bull_trend":  0.58,
    "strong_bear_trend": 0.65, "weak_bear_trend":  0.56,
    "sideways_range":    0.47, "high_volatility":  0.41,
    "low_volatility":    0.51, "news_volatility":  0.32,
    "liquidity_grab":    0.62,
}
SESSION_WIN_BOOST = {
    "overlap_london_ny": +0.06, "london":              +0.04,
    "new_york":          +0.02, "asian":               -0.03,
    "overlap_asian_london": -0.01, "off_hours":        -0.10,
}

def _make_candles(n=110, start_price=1.0):
    closes = start_price + np.cumsum(np.random.randn(n) * 0.0002)
    highs  = closes + np.abs(np.random.randn(n) * 0.0003)
    lows   = closes - np.abs(np.random.randn(n) * 0.0003)
    opens  = np.roll(closes, 1); opens[0] = closes[0]
    vols   = np.random.randint(200, 2000, n).astype(float)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})

def _synthetic_trade(i: int, t_start: datetime, pipeline: FeaturePipeline) -> dict | None:
    regime  = np.random.choice(MARKET_REGIMES, p=[0.15,0.12,0.14,0.11,0.12,0.08,0.10,0.05,0.13])
    session = np.random.choice(
        ["overlap_london_ny","london","new_york","asian","overlap_asian_london","off_hours"],
        p=[0.18, 0.25, 0.22, 0.15, 0.10, 0.10],
    )
    ts   = t_start + timedelta(hours=i * 3 + np.random.randint(0, 3))
    base = np.random.uniform(2300.0, 2700.0)   # realistic XAUUSD price range
    df   = _make_candles(110, base)
    if "bull" in regime:
        df["close"] += np.linspace(0, 5.0, len(df))   # ~5 pip gold move
    elif "bear" in regime:
        df["close"] -= np.linspace(0, 5.0, len(df))
    candles  = df.to_dict("records")
    snapshot = {
        "symbol":      "XAUUSD", "timestamp": ts,
        "price":       float(df["close"].iloc[-1]),
        "spread_pips": np.random.uniform(0.5, 2.5),
        "candles_m5":  candles, "candles_m15": candles[::3],
        "candles_h1":  candles[::12], "candles_h4": candles[::48],
        "candles_d1":  candles[::288],
    }
    try:
        features = pipeline.compute(snapshot)
    except Exception:
        return None

    base_wr  = REGIME_WIN_RATE.get(regime, 0.50)
    boost    = SESSION_WIN_BOOST.get(session, 0.0)
    if features.get("adx_14", 20) > 25:                    boost += 0.04
    if abs(features.get("trend_alignment_score", 0)) > 0.5: boost += 0.05
    win_prob  = float(np.clip(base_wr + boost, 0.25, 0.85))
    outcome   = "WIN" if np.random.rand() < win_prob else "LOSS"
    direction = ("BUY"  if regime in ("strong_bull_trend","weak_bull_trend") else
                 "SELL" if regime in ("strong_bear_trend","weak_bear_trend") else
                 np.random.choice(["BUY","SELL"]))
    pnl    = np.random.uniform(8, 40) if outcome == "WIN" else -np.random.uniform(5, 20)
    max_dd = abs(pnl) * np.random.uniform(0.2, 0.6) if outcome == "WIN" else abs(pnl)

    conditions = {k: features.get(k, 0.0) for k in ALL_FEATURE_NAMES}
    return {
        "id":         str(uuid.uuid4()),
        "mt5_ticket": 1_000_000 + i,
        "symbol":     "XAUUSD",              # ← fixed: was hardcoded EURUSD
        "direction":  direction,
        "outcome":    outcome,
        "pnl_pips":   round(pnl, 2),
        "regime":     regime,
        "session":    session,
        "opened_at":  ts.isoformat(),
        "conditions": conditions,
        "risk_context": {
            "account_drawdown_pct": np.random.uniform(0, 0.03),
            "recent_loss_streak":   np.random.randint(0, 4),
            "recent_win_streak":    np.random.randint(0, 5),
            "trades_today":         np.random.randint(0, 8),
            "session_quality":      features.get("session_quality", 0.5),
            "spread_pips":          snapshot["spread_pips"],
            "similar_win_rate":     win_prob,
            "similar_avg_pnl":      pnl * 0.9,
            "similar_count":        np.random.randint(5, 30),
        },
        "prediction": {
            "trader_buy_prob":   0.65 if direction == "BUY" else 0.35,
            "trader_sell_prob":  0.35 if direction == "BUY" else 0.65,
            "trader_confidence": abs(0.65 - 0.35),
        },
        "_synthetic": True,
    }


# ── Main ──────────────────────────────────────────────────────
def generate():
    client   = _get_supabase()
    pipeline = FeaturePipeline()
    memory   = TradeMemoryEngine(save_dir="./data")

    # ── 1. Pull real trades ───────────────────────────────────
    raw_rows = fetch_trades(client)

    trades: list[dict] = []
    print(f"Converting {len(raw_rows)} DB rows → training records...")
    for idx, row in enumerate(raw_rows):
        record = _row_to_trade(row, pipeline)
        if record is None:
            continue
        trades.append(record)

        pnl    = float(row.get("pnl_pips") or 0)
        max_dd = float(row.get("max_drawdown_pips") or abs(pnl))
        memory.add(
            record_id    = record["id"],
            features     = record["conditions"],
            outcome      = record["outcome"],
            pnl_pips     = pnl,
            max_drawdown = max_dd,
            regime       = record["regime"],
            session      = record["session"],
            direction    = record["direction"],
        )
        if (idx + 1) % 200 == 0:
            print(f"  {idx+1}/{len(raw_rows)} converted...")

    real_count = len(trades)

    # ── 2. Augment with synthetics if DB is thin ──────────────
    if real_count < MIN_REAL_TRADES:
        needed = AUGMENT_TARGET - real_count
        print(f"\n⚠️  Only {real_count} real trades found (min={MIN_REAL_TRADES}).")
        print(f"   Generating {needed} synthetic trades to reach {AUGMENT_TARGET} total...")

        t_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        synth_added = 0
        for i in range(needed * 2):          # over-generate to account for failures
            if synth_added >= needed:
                break
            record = _synthetic_trade(i, t_start, pipeline)
            if record is None:
                continue
            trades.append(record)
            pnl    = record["pnl_pips"]
            max_dd = abs(pnl)
            memory.add(
                record_id    = record["id"],
                features     = record["conditions"],
                outcome      = record["outcome"],
                pnl_pips     = pnl,
                max_drawdown = max_dd,
                regime       = record["regime"],
                session      = record["session"],
                direction    = record["direction"],
            )
            synth_added += 1
            if (synth_added) % 200 == 0:
                print(f"  {synth_added}/{needed} synthetic trades generated...")
    else:
        print(f"\n✅ {real_count} real trades — no synthetic augmentation needed.")

    # ── 3. Save ───────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(trades, f)
    memory.save()

    wins      = sum(1 for t in trades if t["outcome"] == "WIN")
    synth     = sum(1 for t in trades if t.get("_synthetic"))
    print(f"\n{'='*55}")
    print(f"  Total records : {len(trades)}")
    print(f"  Real trades   : {real_count}")
    print(f"  Synthetic     : {synth}")
    print(f"  Win rate      : {wins/len(trades):.1%}")
    print(f"  Memory size   : {memory.size()}")
    print(f"  Saved to      : {OUT_PATH}")
    print(f"{'='*55}")
    print(f"\nNext: python scripts/train_models.py")
    return trades


if __name__ == "__main__":
    generate()
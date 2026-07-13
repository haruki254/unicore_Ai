"""
database/client.py  —  v3 (generate_sample_data pipeline + empty-string JSONB fix)

Roles after the new generate_sample_data.py pipeline:

  WRITE  →  save_snapshot, save_prediction, save_trade,
             save_trade_to_history, update_trade_outcome,
             save_model_result

  READ   →  analytics only (regime, session, equity curve, etc.)

  TRAIN  →  fetch_completed_trades() delegates to
             generate_sample_data.generate() — single source of truth
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.settings import settings
from monitoring.logger import db_logger

try:
    from supabase import create_client
    _LIB = True
except ImportError:
    _LIB = False


class DatabaseClient:

    def __init__(self):
        self._client    = None
        self._connected = False
        self._connect()

    # ── Connection ────────────────────────────────────────────

    def _connect(self):
        if not _LIB:
            db_logger.warning("supabase-py not installed — offline mode")
            return
        url = settings.supabase_url
        key = settings.supabase_service_key or settings.supabase_key
        if not url or not key:
            db_logger.warning("Supabase credentials missing — offline mode")
            return
        try:
            self._client    = create_client(url, key)
            self._connected = True
            db_logger.info("Supabase connected")
        except Exception as e:
            db_logger.error("Supabase connection failed: {}", e)

    @property
    def is_connected(self):
        return self._connected

    def _tbl(self, name):
        return self._client.table(name)

    def _ins(self, table: str, data: dict) -> bool:
        if not self._connected:
            db_logger.warning(
                "insert {} SKIPPED — client not connected (check SUPABASE_URL / "
                "SUPABASE_SERVICE_KEY)", table
            )
            return False
        # Sanitize empty strings → None (helps with NOT NULL + JSONB columns)
        clean = {k: (None if isinstance(v, str) and v.strip() == "" else v)
                 for k, v in data.items()}
        try:
            self._tbl(table).insert(clean).execute()
            return True
        except Exception as e:
            db_logger.error("insert {} failed: {} | payload={}", table, e, clean)
            return False

    # =========================================================
    # WRITE — called by the API on every signal / trade close
    # =========================================================

    def save_snapshot(self, s: dict) -> Optional[str]:
        sid = s.get("id") or str(uuid.uuid4())
        self._ins("market_snapshots", {
            "id":           sid,
            "ea_id":        s.get("ea_id", "default"),
            "captured_at":  _ts(s.get("timestamp")),
            "symbol":       s.get("symbol", "?"),
            "timeframe":    s.get("timeframe", "M5"),
            "open_price":   s.get("open",  0),
            "high_price":   s.get("high",  0),
            "low_price":    s.get("low",   0),
            "close_price":  s.get("close", s.get("price", 0)),
            "spread_pips":  s.get("spread_pips", 0),
            "candles_m5":   json.dumps(s.get("candles_m5",  [])),
            "candles_m15":  json.dumps(s.get("candles_m15", [])),
            "candles_h1":   json.dumps(s.get("candles_h1",  [])),
            "candles_h4":   json.dumps(s.get("candles_h4",  [])),
            "candles_d1":   json.dumps(s.get("candles_d1",  [])),
            "features":     json.dumps(s.get("features", {})),
        })
        return sid

    def save_prediction(self, snapshot_id: str, r: dict, prediction_id: Optional[str] = None) -> Optional[str]:
        pid = prediction_id or str(uuid.uuid4())
        self._ins("predictions", {
            "id":                    pid,
            "snapshot_id":           snapshot_id,
            "ea_id":                 r.get("ea_id", "default"),
            "ea_signal":             r.get("ea_signal"),
            "trader_buy_prob":       r.get("trader_buy_prob"),
            "trader_sell_prob":      r.get("trader_sell_prob"),
            "trader_direction":      r.get("trader_direction"),
            "trader_confidence":     r.get("trader_confidence"),
            "trader_model_used":     r.get("trader_model_used"),
            "risk_allow_prob":       r.get("risk_quality_score"),
            "risk_block_prob":       1 - (r.get("risk_quality_score") or 0),
            "risk_quality_score":    r.get("risk_quality_score"),
            "risk_model_used":       r.get("risk_model_used"),
            "risk_block_reasons":    json.dumps(r.get("risk_block_reasons", [])),
            "final_decision":        r.get("final_decision"),
            "is_flip":               r.get("is_flip",    False),
            "is_blocked":            r.get("is_blocked", False),
            "similar_setups_count":  r.get("similar_count",    0),
            "similar_setups_win_rate": r.get("similar_win_rate"),
            "similar_setups_avg_pnl":  r.get("similar_avg_pnl"),
            "inference_ms":          r.get("inference_ms", 0),
        })
        return pid

    def save_trade(self, t: dict) -> Optional[str]:
        tid = t.get("id") or str(uuid.uuid4())
        self._ins("trades", {
            "id":              tid,
            "ea_id":           t.get("ea_id", "default"),
            "prediction_id":   t.get("prediction_id"),
            "snapshot_id":     t.get("snapshot_id"),
            "mt5_ticket":      t.get("mt5_ticket"),
            "symbol":          t.get("symbol", "?"),
            "direction":       t.get("direction", "BUY"),
            "was_flipped":     t.get("was_flipped", False),
            "original_signal": t.get("original_signal"),
            "entry_price":     t.get("entry_price", 0),
            "exit_price":      t.get("exit_price"),
            "stop_loss":       t.get("stop_loss"),
            "take_profit":     t.get("take_profit"),
            "lot_size":        t.get("lot_size"),
            "opened_at":       _ts(t.get("opened_at")),
            "closed_at":       _ts(t.get("closed_at")),
            "pnl_pips":        t.get("pnl_pips"),
            "pnl_usd":         t.get("pnl_usd"),
            "outcome":         t.get("outcome", "PENDING"),
            "session":         t.get("session"),
            "regime":          t.get("regime"),
        })
        return tid

    def save_trade_to_history(self, trade: dict) -> bool:
        return self._ins("trade_history", {
            "ea_id":             trade.get("ea_id", "default"),
            "mt5_ticket":        trade.get("mt5_ticket"),
            "prediction_id":     trade.get("prediction_id"),
            "snapshot_id":       trade.get("snapshot_id"),
            "symbol":            trade.get("symbol",    "XAUUSD"),
            "direction":         trade.get("direction", "BUY"),
            "was_flipped":       trade.get("was_flipped", False),
            "original_signal":   trade.get("original_signal"),
            "entry_price":       trade.get("entry_price",  0),
            "exit_price":        trade.get("exit_price",   0),
            "pnl_pips":          trade.get("pnl_pips",     0),
            "pnl_usd":           trade.get("pnl_usd",      0),
            "outcome":           trade.get("outcome",   "LOSS"),
            "lot_size":          trade.get("lot_size",   0.01),
            "session":           trade.get("session"),
            "max_drawdown_pips": trade.get("max_drawdown_pips", 0),
            "opened_at":         _ts(trade.get("opened_at")),
            "closed_at":         _ts(trade.get("closed_at") or datetime.utcnow()),
            "source":            "live_trade",
        })

    def update_trade_outcome(self, mt5_ticket, outcome, pnl_pips,
                             pnl_usd=0.0, exit_price=0.0,
                             closed_at=None, direction="BUY",
                             entry_price=0.0, lot_size=0.01,
                             session=None, ea_id="default",
                             prediction_id=None, snapshot_id=None,
                             was_flipped=False, original_signal=None,
                             regime=None, max_drawdown_pips=0.0,
                             symbol="XAUUSD", opened_at=None):
        if not self._connected:
            db_logger.warning(
                "update_trade_outcome SKIPPED for ticket {} — client not connected",
                mt5_ticket,
            )
            return False
        try:
            self._tbl("trades").update({
                "ea_id":      ea_id,
                "outcome":    outcome,
                "pnl_pips":   pnl_pips,
                "pnl_usd":    pnl_usd,
                "exit_price": exit_price,
                "closed_at":  _ts(closed_at or datetime.utcnow()),
            }).eq("mt5_ticket", mt5_ticket).execute()
        except Exception as e:
            db_logger.error("update_trade_outcome (trades): {}", e)

        history_ok = self.save_trade_to_history({
            "mt5_ticket":   mt5_ticket,
            "ea_id":        ea_id,
            "prediction_id": prediction_id,
            "snapshot_id":  snapshot_id,
            "symbol":       symbol,
            "direction":    direction,
            "was_flipped":  was_flipped,
            "original_signal": original_signal,
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "pnl_pips":     pnl_pips,
            "pnl_usd":      pnl_usd,
            "outcome":      outcome,
            "lot_size":     lot_size,
            "session":      session,
            "regime":       regime,
            "max_drawdown_pips": max_drawdown_pips,
            "opened_at":    opened_at,
            "closed_at":    closed_at or datetime.utcnow(),
        })
        if not history_ok:
            db_logger.error(
                "trade_history insert FAILED for mt5_ticket={}", mt5_ticket,
            )
        return history_ok

    def save_model_result(self, r: dict) -> bool:
        if not self._connected:
            return True
        try:
            self._tbl("model_results").update(
                {"is_active": False}
            ).eq("model_type", r["model_type"]).execute()
        except Exception:
            pass
        return self._ins("model_results", r)

    # =========================================================
    # EA PROFILES / ADAPTIVE STATS  ← FIXED
    # =========================================================

    def get_ea_profile(self, ea_id: str) -> Optional[Dict[str, Any]]:
        if not self._connected:
            return None
        try:
            rows = (
                self._tbl("ea_profiles")
                .select("*")
                .eq("ea_id", ea_id)
                .limit(1)
                .execute()
                .data or []
            )
            if not rows:
                return None
            profile = dict(rows[0])
            for key in (
                "regime_weights",
                "session_weights",
                "volatility_weights",
                "momentum_weights",
                "level_prox_weights",
                "sample_counts",
            ):
                profile[key] = _json_or(profile.get(key), {})
            return profile
        except Exception as e:
            db_logger.error("get_ea_profile({}): {}", ea_id, e)
            return None

    def save_ea_profile(self, ea_id: str, profile: Dict[str, Any]) -> bool:
        """Fixed: treat empty strings as missing for JSONB columns."""
        if not self._connected:
            return True
        try:
            payload = dict(profile)
            payload["ea_id"] = ea_id
            payload["updated_at"] = datetime.utcnow().isoformat()

            # Ensure JSONB columns are never null or empty-string
            for key, default in [
                ("regime_weights", {}),
                ("session_weights", {}),
                ("volatility_weights", {}),
                ("momentum_weights", {}),
                ("level_prox_weights", {}),
                ("sample_counts", {}),
            ]:
                val = payload.get(key)
                if val is None or (isinstance(val, str) and val.strip() == ""):
                    payload[key] = default

            self._tbl("ea_profiles").upsert(payload, on_conflict="ea_id").execute()
            return True
        except Exception as e:
            db_logger.error("save_ea_profile({}): {}", ea_id, e)
            return False

    def update_ea_profile(self, ea_id: str, profile: Dict[str, Any]) -> bool:
        return self.save_ea_profile(ea_id, profile)

    def get_flip_stats(self, ea_id: str) -> Dict[str, Any]:
        if not self._connected:
            return {}
        try:
            rows = (
                self._tbl("ea_flip_stats")
                .select("*")
                .eq("ea_id", ea_id)
                .limit(1)
                .execute()
                .data or []
            )
            if not rows:
                return {}
            stats = dict(rows[0])
            stats["recent_flip_outcomes"] = _json_or(
                stats.get("recent_flip_outcomes"), []
            )
            stats["recent_block_outcomes"] = _json_or(
                stats.get("recent_block_outcomes"), []
            )
            return stats
        except Exception as e:
            db_logger.error("get_flip_stats({}): {}", ea_id, e)
            return {}

    def update_flip_stats(self, ea_id: str, stats: Dict[str, Any]) -> bool:
        """Fixed: treat empty strings as missing for JSONB columns."""
        if not self._connected:
            return True
        try:
            payload = dict(stats)
            payload["ea_id"] = ea_id
            payload["updated_at"] = datetime.utcnow().isoformat()

            for key, default in [
                ("recent_flip_outcomes", []),
                ("recent_block_outcomes", []),
            ]:
                val = payload.get(key)
                if val is None or (isinstance(val, str) and val.strip() == ""):
                    payload[key] = default

            self._tbl("ea_flip_stats").upsert(payload, on_conflict="ea_id").execute()
            return True
        except Exception as e:
            db_logger.error("update_flip_stats({}): {}", ea_id, e)
            return False

    def get_snapshot_features(self, snapshot_id: str) -> Dict[str, Any]:
        if not self._connected or not snapshot_id:
            return {}
        try:
            rows = (
                self._tbl("market_snapshots")
                .select("features,symbol,close_price,spread_pips,captured_at")
                .eq("id", snapshot_id)
                .limit(1)
                .execute()
                .data or []
            )
            if not rows:
                return {}
            row = rows[0]
            features = _json_or(row.get("features"), {})
            if row.get("symbol") is not None:
                features.setdefault("symbol", row.get("symbol"))
            if row.get("close_price") is not None:
                features.setdefault("price", row.get("close_price"))
            if row.get("spread_pips") is not None:
                features.setdefault("spread_pips", row.get("spread_pips"))
            if row.get("captured_at") is not None:
                features.setdefault("timestamp", row.get("captured_at"))
            return features
        except Exception as e:
            db_logger.error("get_snapshot_features({}): {}", snapshot_id, e)
            return {}

    # Training methods remain the same
    def fetch_completed_trades(self, limit: int = 10_000, min_date=None) -> List[Dict]:
        try:
            from scripts.generate_sample_data import generate
            db_logger.info("fetch_completed_trades: delegating to generate_sample_data.generate()")
            trades = generate()
            if min_date:
                trades = [t for t in trades if t.get("opened_at", "") >= min_date.isoformat()]
            db_logger.info("fetch_completed_trades: {} records returned", len(trades))
            return trades
        except Exception as e:
            db_logger.error("fetch_completed_trades via generate() failed: {}", e)
            return self._load_pkl_fallback()

# ── Helpers ───────────────────────────────────────────────────

def _ts(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


def _json_or(v, default):
    if v is None:
        return default
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return default
    return v
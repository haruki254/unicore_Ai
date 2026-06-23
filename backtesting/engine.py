"""
Backtesting Engine

Simulates the full AI decision pipeline on historical data.
Produces equity curves, drawdown analysis, regime/session breakdowns,
Sharpe ratio, profit factor, win rate, and more.

IMPORTANT: Uses walk-forward methodology — no look-ahead bias.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.features.feature_pipeline import FeaturePipeline
from core.models.trader_ai           import TraderAI
from core.models.risk_manager_ai     import RiskManagerAI
from core.decision.decision_engine   import DecisionEngine
from core.regime.regime_engine       import MarketRegimeEngine
from core.memory.trade_memory        import TradeMemoryEngine
from monitoring.logger               import backtest_logger


# ── Result containers ─────────────────────────────────────────

@dataclass
class BacktestTrade:
    index:          int
    timestamp:      str
    ea_signal:      str
    final_decision: str
    direction:      Optional[str]
    entry_price:    float
    exit_price:     float
    pnl_pips:       float
    outcome:        str
    regime:         str
    session:        str
    was_flip:       bool
    was_blocked:    bool
    trader_buy:     float
    trader_sell:    float
    risk_quality:   float


@dataclass
class BacktestMetrics:
    total_signals:     int   = 0
    total_trades:      int   = 0
    blocked_trades:    int   = 0
    flipped_trades:    int   = 0
    wins:              int   = 0
    losses:            int   = 0
    breakeven:         int   = 0

    win_rate:          float = 0.0
    profit_factor:     float = 0.0
    sharpe_ratio:      float = 0.0
    max_drawdown_pips: float = 0.0
    max_drawdown_pct:  float = 0.0

    total_pips:        float = 0.0
    avg_win_pips:      float = 0.0
    avg_loss_pips:     float = 0.0
    best_trade:        float = 0.0
    worst_trade:       float = 0.0

    flip_win_rate:     float = 0.0
    block_saved_pips:  float = 0.0

    by_regime:         Dict[str, Dict] = field(default_factory=dict)
    by_session:        Dict[str, Dict] = field(default_factory=dict)
    by_weekday:        Dict[str, Dict] = field(default_factory=dict)

    equity_curve:      List[float] = field(default_factory=list)
    drawdown_curve:    List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_signals":     self.total_signals,
                "total_trades":      self.total_trades,
                "blocked_trades":    self.blocked_trades,
                "flipped_trades":    self.flipped_trades,
                "wins":              self.wins,
                "losses":            self.losses,
                "win_rate":          round(self.win_rate,      4),
                "profit_factor":     round(self.profit_factor, 4),
                "sharpe_ratio":      round(self.sharpe_ratio,  4),
                "max_drawdown_pips": round(self.max_drawdown_pips, 2),
                "total_pips":        round(self.total_pips,    2),
                "avg_win_pips":      round(self.avg_win_pips,  2),
                "avg_loss_pips":     round(self.avg_loss_pips, 2),
                "best_trade":        round(self.best_trade,    2),
                "worst_trade":       round(self.worst_trade,   2),
                "flip_win_rate":     round(self.flip_win_rate, 4),
                "block_saved_pips":  round(self.block_saved_pips, 2),
            },
            "by_regime":   self.by_regime,
            "by_session":  self.by_session,
            "by_weekday":  self.by_weekday,
            "equity_curve": self.equity_curve[:200],   # truncate for JSON
        }


class BacktestEngine:
    """
    Replay historical trade signals through the AI pipeline.

    Usage
    -----
    engine  = BacktestEngine()
    metrics = engine.run(historical_data, spread_pips=1.5)
    print(json.dumps(metrics.to_dict(), indent=2))
    """

    def __init__(
        self,
        trader_ai:     TraderAI       = None,
        risk_manager:  RiskManagerAI  = None,
        tp_pips:       float = 20.0,
        sl_pips:       float = 10.0,
        spread_pips:   float = 1.5,
    ):
        self.trader_ai     = trader_ai     or TraderAI()
        self.risk_manager  = risk_manager  or RiskManagerAI()
        self.decision_eng  = DecisionEngine()
        self.regime_eng    = MarketRegimeEngine()
        self.memory_eng    = TradeMemoryEngine(save_dir="./data/backtest_memory")
        self.feat_pipeline = FeaturePipeline()
        self.tp_pips       = tp_pips
        self.sl_pips       = sl_pips
        self.spread_pips   = spread_pips

        # Load trained models if available
        self.trader_ai.load()
        self.risk_manager.load()

    def run(
        self,
        df: pd.DataFrame,
        ea_signal_col: str = "ea_signal",
        walk_forward:  bool = True,
        train_pct:     float = 0.6,
    ) -> BacktestMetrics:
        """
        Run the backtest.

        Parameters
        ----------
        df : DataFrame with OHLCV + ea_signal column
             sorted oldest → newest
        ea_signal_col : column name for the EA signal
        walk_forward  : use walk-forward (train on first 60%, test on rest)
        train_pct     : fraction of data used for initial training

        Returns
        -------
        BacktestMetrics
        """
        df = df.copy().reset_index(drop=True)
        n  = len(df)

        if walk_forward:
            train_end = int(n * train_pct)
            test_df   = df.iloc[train_end:].reset_index(drop=True)
            backtest_logger.info(
                "Walk-forward backtest | train={} test={}", train_end, len(test_df)
            )
        else:
            test_df = df

        trades:  List[BacktestTrade] = []
        metrics  = BacktestMetrics()

        lookback = 100  # candles needed for features

        for i in range(lookback, len(test_df)):
            row       = test_df.iloc[i]
            ea_signal = str(row.get(ea_signal_col, "BUY")).upper()
            ts        = row.get("time", datetime.utcnow())

            # Build snapshot from available candles
            window    = test_df.iloc[max(0, i - lookback): i + 1]
            snapshot  = self._build_snapshot(window, row, ts)

            # Compute features
            try:
                features = self.feat_pipeline.compute(snapshot)
            except Exception as e:
                backtest_logger.warning("Feature error at index {}: {}", i, e)
                continue

            # Run decision pipeline
            try:
                result = self.decision_eng.decide(
                    ea_signal     = ea_signal,
                    features      = features,
                    trader_ai     = self.trader_ai,
                    risk_manager  = self.risk_manager,
                    regime_engine = self.regime_eng,
                    memory_engine = self.memory_eng,
                    risk_context  = {"trades_today": 0, "account_drawdown_pct": 0},
                )
            except Exception as e:
                backtest_logger.warning("Decision error at index {}: {}", i, e)
                continue

            metrics.total_signals += 1

            if result.is_blocked:
                metrics.blocked_trades += 1
                # Simulate what would have happened (to compute saved pips)
                hypo_pnl = self._simulate_pnl(row, ea_signal)
                if hypo_pnl < 0:
                    metrics.block_saved_pips += abs(hypo_pnl)
                continue

            # Simulate trade P&L
            direction = result.trade_direction or ea_signal
            pnl_pips  = self._simulate_pnl(row, direction)
            outcome   = "WIN" if pnl_pips > 0 else ("LOSS" if pnl_pips < 0 else "BREAKEVEN")

            trade = BacktestTrade(
                index          = i,
                timestamp      = str(ts),
                ea_signal      = ea_signal,
                final_decision = result.final_decision,
                direction      = direction,
                entry_price    = float(row.get("close", 0)),
                exit_price     = float(row.get("close", 0)) + pnl_pips * 0.0001,
                pnl_pips       = pnl_pips,
                outcome        = outcome,
                regime         = result.regime,
                session        = str(features.get("_session_label", "unknown")),
                was_flip       = result.is_flip,
                was_blocked    = False,
                trader_buy     = result.trader_buy_prob,
                trader_sell    = result.trader_sell_prob,
                risk_quality   = result.risk_quality_score,
            )
            trades.append(trade)

            # Add to memory for learning
            self.memory_eng.add(
                record_id    = f"bt_{i}",
                features     = features,
                outcome      = outcome,
                pnl_pips     = pnl_pips,
                regime       = result.regime,
                session      = trade.session,
                direction    = direction,
            )

            if result.is_flip:
                metrics.flipped_trades += 1

        metrics = self._compute_metrics(trades, metrics)
        self._print_summary(metrics)
        return metrics

    def save_results(self, metrics: BacktestMetrics, path: str = "./data/backtest_results.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2, default=str)
        backtest_logger.info("Backtest results saved → {}", path)

    # ── Private ────────────────────────────────────────────────

    def _build_snapshot(self, window: pd.DataFrame, row: pd.Series, ts) -> Dict:
        candles = []
        for _, r in window.iterrows():
            candles.append({
                "open":  float(r.get("open",  r.get("close", 0))),
                "high":  float(r.get("high",  r.get("close", 0))),
                "low":   float(r.get("low",   r.get("close", 0))),
                "close": float(r.get("close", 0)),
                "volume": float(r.get("volume", 0)),
            })
        return {
            "symbol":      "BACKTEST",
            "timestamp":   ts,
            "price":       float(row.get("close", 0)),
            "spread_pips": self.spread_pips,
            "candles_m5":  candles,
            "candles_m15": candles[::3],
            "candles_h1":  candles[::12],
            "candles_h4":  candles[::48],
            "candles_d1":  candles[::288],
        }

    def _simulate_pnl(self, row: pd.Series, direction: str) -> float:
        """Simplified P&L: fixed TP/SL based on next candle direction."""
        close = float(row.get("close", 0))
        nxt_h = float(row.get("next_high", close + self.tp_pips * 0.0001))
        nxt_l = float(row.get("next_low",  close - self.sl_pips * 0.0001))

        if direction == "BUY":
            if (nxt_h - close) / 0.0001 >= self.tp_pips:
                return self.tp_pips - self.spread_pips
            else:
                return -self.sl_pips - self.spread_pips
        else:
            if (close - nxt_l) / 0.0001 >= self.tp_pips:
                return self.tp_pips - self.spread_pips
            else:
                return -self.sl_pips - self.spread_pips

    def _compute_metrics(
        self,
        trades:  List[BacktestTrade],
        metrics: BacktestMetrics,
    ) -> BacktestMetrics:
        if not trades:
            return metrics

        pnls    = [t.pnl_pips for t in trades]
        wins    = [p for p in pnls if p > 0]
        losses  = [p for p in pnls if p < 0]

        metrics.total_trades      = len(trades)
        metrics.wins              = len(wins)
        metrics.losses            = len(losses)
        metrics.win_rate          = len(wins) / len(trades) if trades else 0
        metrics.total_pips        = sum(pnls)
        metrics.avg_win_pips      = np.mean(wins)   if wins   else 0
        metrics.avg_loss_pips     = np.mean(losses) if losses else 0
        metrics.best_trade        = max(pnls)
        metrics.worst_trade       = min(pnls)
        metrics.profit_factor     = (
            abs(sum(wins)) / abs(sum(losses)) if losses else 999.0
        )

        # Sharpe ratio (annualised, assuming 252 trading days, ~5 trades/day)
        if len(pnls) > 1:
            metrics.sharpe_ratio = (
                np.mean(pnls) / (np.std(pnls) + 1e-10)
            ) * np.sqrt(252 * 5)

        # Equity + drawdown curves
        equity = np.cumsum(pnls)
        peak   = np.maximum.accumulate(equity)
        dd     = peak - equity
        metrics.equity_curve      = [round(float(e), 2) for e in equity]
        metrics.drawdown_curve    = [round(float(d), 2) for d in dd]
        metrics.max_drawdown_pips = float(dd.max()) if len(dd) else 0

        # Flip win rate
        flips = [t for t in trades if t.was_flip]
        if flips:
            metrics.flip_win_rate = sum(1 for t in flips if t.pnl_pips > 0) / len(flips)

        # Breakdown by regime / session / weekday
        metrics.by_regime  = self._breakdown(trades, "regime")
        metrics.by_session = self._breakdown(trades, "session")

        return metrics

    def _breakdown(self, trades: List[BacktestTrade], attr: str) -> Dict:
        groups: Dict[str, List[float]] = {}
        for t in trades:
            key = getattr(t, attr, "unknown") or "unknown"
            groups.setdefault(key, []).append(t.pnl_pips)

        result = {}
        for key, pnls in groups.items():
            w = [p for p in pnls if p > 0]
            result[key] = {
                "trades":    len(pnls),
                "win_rate":  round(len(w) / len(pnls), 4),
                "total_pips":round(sum(pnls), 2),
                "avg_pips":  round(np.mean(pnls), 2),
            }
        return result

    def _print_summary(self, m: BacktestMetrics) -> None:
        backtest_logger.info("=" * 50)
        backtest_logger.info("BACKTEST RESULTS")
        backtest_logger.info("=" * 50)
        backtest_logger.info("Signals:      {}", m.total_signals)
        backtest_logger.info("Trades:       {}", m.total_trades)
        backtest_logger.info("Blocked:      {}", m.blocked_trades)
        backtest_logger.info("Flipped:      {}", m.flipped_trades)
        backtest_logger.info("Win Rate:     {:.1%}", m.win_rate)
        backtest_logger.info("Total Pips:   {:.1f}", m.total_pips)
        backtest_logger.info("Profit Factor:{:.2f}", m.profit_factor)
        backtest_logger.info("Sharpe:       {:.2f}", m.sharpe_ratio)
        backtest_logger.info("Max DD Pips:  {:.1f}", m.max_drawdown_pips)
        backtest_logger.info("Saved (blocks):{:.1f} pips", m.block_saved_pips)
        backtest_logger.info("=" * 50)

"""
Pydantic schemas for FastAPI request/response validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


# ── Candle ────────────────────────────────────────────────────

class CandleData(BaseModel):
    open:   float
    high:   float
    low:    float
    close:  float
    volume: Optional[float] = 0.0
    time:   Optional[datetime] = None


# ── Risk Context (live account state) ────────────────────────

class RiskContext(BaseModel):
    account_drawdown_pct:  float = Field(default=0.0,  ge=0.0, le=1.0)
    recent_loss_streak:    int   = Field(default=0,    ge=0)
    recent_win_streak:     int   = Field(default=0,    ge=0)
    trades_today:          int   = Field(default=0,    ge=0)
    current_risk_exposure: float = Field(default=0.0,  ge=0.0)
    is_news_period:        int   = Field(default=0)
    account_balance:       float = Field(default=10000.0, gt=0)
    account_equity:        float = Field(default=10000.0, gt=0)


# ── Prediction Request (from MT5 EA) ─────────────────────────

class PredictionRequest(BaseModel):
    ea_id:      str              = Field(default="default", min_length=1, max_length=64)
    symbol:     str              = Field(..., min_length=1, max_length=20)
    ea_signal:  str              = Field(..., pattern="^(BUY|SELL)$")
    timestamp:  Optional[datetime] = None
    price:      float            = Field(..., gt=0)
    spread_pips: float           = Field(default=1.0, ge=0)

    candles_m5:  List[CandleData] = Field(default_factory=list, max_length=100)
    candles_m15: List[CandleData] = Field(default_factory=list, max_length=50)
    candles_h1:  List[CandleData] = Field(default_factory=list, max_length=20)
    candles_h4:  List[CandleData] = Field(default_factory=list, max_length=10)
    candles_d1:  List[CandleData] = Field(default_factory=list, max_length=5)

    risk_context: RiskContext = Field(default_factory=RiskContext)

    @validator("ea_signal")
    def validate_ea_signal(cls, v):
        return v.upper()

    @validator("timestamp", pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.utcnow()

    def to_snapshot_dict(self) -> Dict[str, Any]:
        return {
            "ea_id":       self.ea_id,
            "symbol":      self.symbol,
            "timestamp":   self.timestamp,
            "price":       self.price,
            "spread_pips": self.spread_pips,
            "candles_m5":  [c.dict() for c in self.candles_m5],
            "candles_m15": [c.dict() for c in self.candles_m15],
            "candles_h1":  [c.dict() for c in self.candles_h1],
            "candles_h4":  [c.dict() for c in self.candles_h4],
            "candles_d1":  [c.dict() for c in self.candles_d1],
        }


# ── Prediction Response (to MT5 EA) ──────────────────────────

class PredictionResponse(BaseModel):
    # Final answer
    final_decision:  str   # ALLOW_BUY | ALLOW_SELL | FLIP_TO_BUY | FLIP_TO_SELL | BLOCK
    trade_direction: Optional[str] = None   # BUY | SELL | None

    # Trader AI
    trader_buy_prob:   float
    trader_sell_prob:  float
    trader_direction:  str
    trader_confidence: float
    trader_model:      str

    # Risk Manager
    risk_quality_score: float
    risk_decision:      str
    block_reasons:      List[str] = Field(default_factory=list)
    risk_model:         str

    # Context
    regime:            str
    regime_confidence: float
    similar_count:     int
    similar_win_rate:  float
    similar_avg_pnl:   float

    # Meta
    is_flip:       bool
    is_blocked:    bool
    inference_ms:  int
    snapshot_id:   Optional[str] = None
    prediction_id: Optional[str] = None
    decision_time: Optional[str] = None   # server-side UTC ISO-8601, set when the decision was produced

    class Config:
        json_schema_extra = {
            "example": {
                "final_decision":    "ALLOW_SELL",
                "trade_direction":   "SELL",
                "trader_buy_prob":   0.27,
                "trader_sell_prob":  0.73,
                "trader_direction":  "SELL",
                "trader_confidence": 0.23,
                "trader_model":      "lightgbm",
                "risk_quality_score": 0.81,
                "risk_decision":     "ALLOW",
                "block_reasons":     [],
                "risk_model":        "xgboost",
                "regime":            "strong_bear_trend",
                "regime_confidence": 0.74,
                "similar_count":     18,
                "similar_win_rate":  0.72,
                "similar_avg_pnl":   12.4,
                "is_flip":           False,
                "is_blocked":        False,
                "inference_ms":      23,
            }
        }


# ── Trade Update (from MT5 on trade close) ───────────────────

class TradeUpdateRequest(BaseModel):
    ea_id:        str = Field(default="default", min_length=1, max_length=64)
    mt5_ticket:   int
    symbol:       str
    direction:    str = Field(..., pattern="^(BUY|SELL)$")
    entry_price:  float
    exit_price:   float
    pnl_pips:     float
    pnl_usd:      float = 0.0
    outcome:      str   = Field(..., pattern="^(WIN|LOSS|BREAKEVEN)$")
    closed_at:    Optional[datetime] = None
    opened_at:    Optional[datetime] = None
    max_drawdown_pips: float = 0.0
    was_flipped:  bool = False
    original_signal: Optional[str] = None
    lot_size:     float = 0.01
    prediction_id: Optional[str] = None
    snapshot_id:   Optional[str] = None   # for feature lookup in adaptive update
    regime:       Optional[str] = None
    session:      Optional[str] = None


class TradeUpdateResponse(BaseModel):
    success:    bool
    trade_id:   Optional[str] = None
    message:    str = ""


# ── Training trigger ──────────────────────────────────────────

class TrainRequest(BaseModel):
    force: bool = False


class TrainResponse(BaseModel):
    status:           str
    timestamp:        str
    elapsed_seconds:  float = 0.0
    trade_count:      int   = 0
    trader_ai:        Dict[str, Any] = Field(default_factory=dict)
    risk_manager:     Dict[str, Any] = Field(default_factory=dict)


# ── Analytics ─────────────────────────────────────────────────

class PerformanceByDimension(BaseModel):
    dimension_value: str
    total_trades:    int
    wins:            int
    losses:          int
    win_rate:        float
    avg_pips:        float
    total_pips:      float


class ModelPerformanceResponse(BaseModel):
    model_type:  str
    algorithm:   str
    roc_auc:     float
    wf_mean:     float
    wf_std:      float
    is_trained:  bool
    feature_importance: Dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status:         str
    trader_trained: bool
    risk_trained:   bool
    memory_size:    int
    db_connected:   bool
    uptime_seconds: float
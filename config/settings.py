"""
Central configuration — pure stdlib + python-dotenv.
No pydantic-settings required.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=False)

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

def _envf(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default

def _envi(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default

def _envb(key: str, default: bool) -> bool:
    val = os.environ.get(key, str(default)).lower()
    return val in ("1", "true", "yes")


class Settings:
    # ── Supabase ──────────────────────────────────────────────
    supabase_url:        str   = _env("SUPABASE_URL")
    supabase_key:        str   = _env("SUPABASE_KEY")
    supabase_service_key:str   = _env("SUPABASE_SERVICE_KEY")
    database_url:        str   = _env("DATABASE_URL")

    # ── API ───────────────────────────────────────────────────
    api_host:            str   = _env("API_HOST", "0.0.0.0")
    api_port:            int   = _envi("API_PORT", 8000)
    api_secret_key:      str   = _env("API_SECRET_KEY", "change-me")
    api_debug:           bool  = _envb("API_DEBUG", True)

    # ── Models ────────────────────────────────────────────────
    model_retrain_interval_hours: int  = _envi("MODEL_RETRAIN_INTERVAL_HOURS", 24)
    model_min_training_samples:   int  = _envi("MODEL_MIN_TRAINING_SAMPLES", 200)
    model_save_path:              str  = _env("MODEL_SAVE_PATH", "./models_saved")
    walk_forward_splits:          int  = _envi("WALK_FORWARD_SPLITS", 5)

    # ── Trading ───────────────────────────────────────────────
    min_trader_confidence: float = _envf("MIN_TRADER_CONFIDENCE", 0.55)
    min_risk_quality:      float = _envf("MIN_RISK_QUALITY", 0.55)
    flip_threshold:        float = _envf("FLIP_THRESHOLD", 0.65)
    max_spread_pips:       float = _envf("MAX_SPREAD_PIPS", 3.0)
    max_daily_trades:      int   = _envi("MAX_DAILY_TRADES", 10)
    max_drawdown_pct:      float = _envf("MAX_DRAWDOWN_PCT", 0.05)

    # ── Features ──────────────────────────────────────────────
    candle_history_short: int  = _envi("CANDLE_HISTORY_SHORT", 50)
    candle_history_long:  int  = _envi("CANDLE_HISTORY_LONG", 100)
    atr_period:           int  = _envi("ATR_PERIOD", 14)
    rsi_period:           int  = _envi("RSI_PERIOD", 14)
    adx_period:           int  = _envi("ADX_PERIOD", 14)
    macd_fast:            int  = _envi("MACD_FAST", 12)
    macd_slow:            int  = _envi("MACD_SLOW", 26)
    macd_signal:          int  = _envi("MACD_SIGNAL", 9)

    # ── Sessions (UTC hours) ──────────────────────────────────
    asian_session_start:  int  = _envi("ASIAN_SESSION_START", 0)
    asian_session_end:    int  = _envi("ASIAN_SESSION_END", 8)
    london_session_start: int  = _envi("LONDON_SESSION_START", 7)
    london_session_end:   int  = _envi("LONDON_SESSION_END", 16)
    ny_session_start:     int  = _envi("NY_SESSION_START", 13)
    ny_session_end:       int  = _envi("NY_SESSION_END", 22)

    # ── Logging ───────────────────────────────────────────────
    log_level:  str = _env("LOG_LEVEL", "INFO")
    log_file:   str = _env("LOG_FILE", "./logs/trading_intelligence.log")

    # ── Dashboard ─────────────────────────────────────────────
    dashboard_port: int = _envi("DASHBOARD_PORT", 8501)

    def ensure_dirs(self) -> None:
        for d in [self.model_save_path,
                  str(Path(self.log_file).parent),
                  "./data", "./logs"]:
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

# ── Constants ─────────────────────────────────────────────────
MARKET_REGIMES = [
    "strong_bull_trend", "weak_bull_trend",
    "strong_bear_trend", "weak_bear_trend",
    "sideways_range", "high_volatility", "low_volatility",
    "news_volatility", "liquidity_grab",
]

SESSIONS = [
    "asian", "london", "new_york",
    "off_hours", "overlap_london_ny", "overlap_asian_london",
]

TRADE_DIRECTIONS  = ["BUY", "SELL"]
FINAL_DECISIONS   = ["ALLOW_BUY", "ALLOW_SELL", "FLIP_TO_BUY", "FLIP_TO_SELL", "BLOCK"]
TIMEFRAMES        = ["M5", "M15", "H1", "H4", "D1"]
ML_MODELS         = ["random_forest", "xgboost", "lightgbm", "catboost", "logistic_regression"]

FEATURE_GROUPS = {
    "market_structure": [
        "hh_count","hl_count","lh_count","ll_count",
        "bos_bullish","bos_bearish","choch_bullish","choch_bearish",
        "structure_score",
    ],
    "price_action": [
        "body_size_avg_50","body_size_avg_100","wick_upper_avg","wick_lower_avg",
        "momentum_5","momentum_10","momentum_20","close_vs_open",
        "candle_direction_ratio","large_body_count",
    ],
    "trend": [
        "trend_m5","trend_m15","trend_h1","trend_h4","trend_d1",
        "trend_alignment_score","trend_strength",
    ],
    "volatility": [
        "atr_14","atr_normalized","std_dev_20","range_expansion",
        "range_contraction","volatility_regime",
    ],
    "indicators": [
        "rsi_14","rsi_overbought","rsi_oversold",
        "adx_14","adx_trending","macd_line","macd_signal_line","macd_histogram",
        "ma_20","ma_50","ma_200","price_vs_ma20","price_vs_ma50",
        "ma_cross_bullish","ma_cross_bearish",
    ],
    "liquidity": [
        "dist_to_support","dist_to_resistance","sr_ratio",
        "prev_day_high","prev_day_low","dist_to_pdh","dist_to_pdl",
        "weekly_high","weekly_low","dist_to_wh","dist_to_wl",
    ],
    "session": [
        "session_asian","session_london","session_ny",
        "session_overlap","hour_of_day","day_of_week",
    ],
}

ALL_FEATURE_NAMES = [f for group in FEATURE_GROUPS.values() for f in group]

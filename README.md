# Trading Intelligence System

Institutional-grade dual-AI trading system for MetaTrader 5.

```
EA Signal → Python API → Trader AI + Risk Manager AI → ALLOW / FLIP / BLOCK → MT5
```

---

## Architecture

```
trading_intelligence/
├── config/
│   ├── settings.py              # All config (reads .env)
│   └── supabase_schema.sql      # Full DB schema
├── core/
│   ├── features/                # Feature engineering (7 modules)
│   ├── regime/                  # Market regime engine (9 states)
│   ├── memory/                  # FAISS vector similarity search
│   ├── models/                  # Trader AI + Risk Manager AI
│   ├── decision/                # Final decision engine
│   └── learning/                # Retraining pipeline
├── database/                    # Supabase client
├── api/                         # FastAPI application
├── backtesting/                 # Walk-forward backtester
├── dashboard/                   # Streamlit analytics
├── mt5/                         # MQL5 Expert Advisor
└── scripts/                     # CLI tools
```

---

## Quick Start (5 steps)

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

Minimum viable (no Supabase, no heavy ML):
```bash
pip install scikit-learn numpy pandas fastapi uvicorn python-dotenv
```

Full install (adds XGBoost, LightGBM, CatBoost, FAISS):
```bash
pip install -r requirements.txt
```

---

### Step 2 — Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set API_SECRET_KEY
# Supabase fields are optional (system works offline without them)
```

Key `.env` settings:
```
API_SECRET_KEY=your-secret-here      # sent by MT5 EA as X-API-Key header
API_DEBUG=true                        # set false in production
SUPABASE_URL=https://xxx.supabase.co  # optional
SUPABASE_SERVICE_KEY=xxxx             # optional
```

---

### Step 3 — Generate training data & train models

```bash
# Generate 2,000 synthetic labeled trades (for bootstrapping)
python scripts/generate_sample_data.py

# Train Trader AI + Risk Manager AI
python scripts/train_models.py
```

Expected output:
```
TRAINING TRADER AI
  Best algorithm : random_forest
  ROC-AUC (WF)   : 0.5480

TRAINING RISK MANAGER AI
  Best algorithm : random_forest
  ROC-AUC (WF)   : 0.8200
```

> **Note:** Initial AUC will be modest (0.50–0.65) on synthetic data.
> The system improves automatically as real trades accumulate.

---

### Step 4 — Start the API

```bash
python scripts/start_api.py
```

```
http://0.0.0.0:8000
Docs → http://0.0.0.0:8000/docs
```

Verify:
```bash
curl http://localhost:8000/health
# → {"status":"healthy","trader_trained":true,...}
```

---

### Step 5 — Connect MT5 EA

1. Copy `mt5/TradingIntelligenceEA.mq5` into your MT5 `Experts` folder
2. Compile in MetaEditor (F7)
3. Attach to any chart with settings:
   - `API_URL = http://YOUR_PC_IP:8000`
   - `API_KEY = your-secret-here` (must match `.env`)
4. Enable **WebRequest** in MT5: Tools → Options → Expert Advisors → Allow WebRequest for listed URL → add your API URL

The EA calls `ProcessSignal("BUY")` or `ProcessSignal("SELL")` from your strategy's `OnTick()`.

---

### Optional — Launch Dashboard

```bash
streamlit run dashboard/app.py
# → http://localhost:8501
```

---

## API Reference

### `POST /predict` — Main prediction endpoint

```json
// Request (from MT5 EA)
{
  "symbol": "EURUSD",
  "ea_signal": "BUY",
  "price": 1.08550,
  "spread_pips": 1.2,
  "candles_m5":  [{"open":1.085,"high":1.086,"low":1.084,"close":1.0855}, ...],
  "candles_m15": [...],
  "candles_h1":  [...],
  "candles_h4":  [...],
  "candles_d1":  [...],
  "risk_context": {
    "account_drawdown_pct": 0.01,
    "recent_loss_streak": 0,
    "trades_today": 2
  }
}

// Response
{
  "final_decision": "ALLOW_SELL",    // ALLOW_BUY | ALLOW_SELL | FLIP_TO_BUY | FLIP_TO_SELL | BLOCK
  "trade_direction": "SELL",
  "trader_buy_prob": 0.27,
  "trader_sell_prob": 0.73,
  "risk_quality_score": 0.81,
  "regime": "strong_bear_trend",
  "similar_count": 18,
  "similar_win_rate": 0.72,
  "is_flip": true,
  "inference_ms": 23
}
```

### `POST /trade/update` — Report closed trade

```json
{
  "mt5_ticket": 123456,
  "symbol": "EURUSD",
  "direction": "SELL",
  "entry_price": 1.08550,
  "exit_price": 1.08350,
  "pnl_pips": 20.0,
  "outcome": "WIN"
}
```

### `POST /train` — Trigger retraining

```json
{"force": true}
```

### `GET /analytics/regime` — Regime performance
### `GET /analytics/session` — Session performance
### `GET /analytics/equity` — Equity + drawdown data
### `GET /analytics/models` — Model metrics + feature importance

---

## Backtesting

```bash
# With your own CSV (needs: time, open, high, low, close, volume, ea_signal)
python scripts/backtest.py --csv path/to/data.csv --tp 20 --sl 10

# Built-in demo (1000 bars, no CSV needed)
python scripts/backtest.py --demo
```

Results saved to `data/backtest_results.json`.

---

## Database (Supabase) — Optional

Without Supabase the system runs fully in-memory.
To persist data, add Supabase credentials to `.env` and run:

```bash
# Option A: Supabase SQL Editor
# Copy-paste config/supabase_schema.sql into Supabase → SQL Editor → Run

# Option B: direct psycopg2
pip install psycopg2-binary
python scripts/init_database.py
```

Tables created:
- `market_snapshots` — raw candle data per signal
- `predictions` — AI decision records
- `trades` — trade outcomes
- `trade_conditions` — feature values per trade
- `similarity_vectors` — vector embeddings for memory search
- `model_results` — trained model metadata
- `performance_metrics` — aggregated analytics

---

## Docker

```bash
cp .env.example .env   # fill in values
docker-compose up -d

# API  → http://localhost:8000
# Dash → http://localhost:8501
```

---

## Decision Logic

```
EA Signal (BUY or SELL)
        │
        ▼
  Feature Engineering (123 features)
  ├── Market Structure (HH/HL/LH/LL/BOS/CHoCH)
  ├── Price Action (bodies, wicks, momentum)
  ├── Trend (M5→D1 alignment)
  ├── Volatility (ATR, BB, range)
  ├── Indicators (RSI, ADX, MACD, MAs)
  ├── Liquidity (S/R, PDH/PDL, weekly levels)
  └── Session (Asian/London/NY/overlap)
        │
        ▼
  Market Regime Engine → one of 9 regimes
        │
        ▼
  Trade Memory → find 20 similar past setups → win rate, avg PnL
        │
        ├──▶  Trader AI ──────────────────────────────────┐
        │     BUY prob + SELL prob                         │
        │                                                  │
        └──▶  Risk Manager AI ───────────────────────────▶┤
              ALLOW / BLOCK + quality score (0–100%)       │
                                                           ▼
                                               Decision Engine
                                               ┌───────────────────────────┐
                                               │ Risk=BLOCK        → BLOCK  │
                                               │ Risk=ALLOW                  │
                                               │  dir == ea_signal → ALLOW   │
                                               │  dir != ea_signal           │
                                               │    prob > 65%    → FLIP     │
                                               │    else          → BLOCK    │
                                               └───────────────────────────┘
```

---

## Retraining Schedule

The system retrains automatically every 24 hours (configurable).
Manual trigger:

```bash
curl -X POST http://localhost:8000/train \
     -H "X-API-Key: your-key" \
     -H "Content-Type: application/json" \
     -d '{"force": true}'
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `config/settings.py` | All settings from `.env` |
| `config/supabase_schema.sql` | Full DB schema |
| `core/features/feature_pipeline.py` | Master feature orchestrator |
| `core/regime/regime_engine.py` | 9-state regime classifier |
| `core/memory/trade_memory.py` | Vector similarity search |
| `core/models/trader_ai.py` | Direction classifier |
| `core/models/risk_manager_ai.py` | Trade quality classifier |
| `core/decision/decision_engine.py` | ALLOW/FLIP/BLOCK logic |
| `core/learning/pipeline.py` | Retraining orchestrator |
| `api/main.py` | FastAPI application |
| `database/client.py` | Supabase wrapper |
| `backtesting/engine.py` | Walk-forward backtester |
| `dashboard/app.py` | Streamlit dashboard |
| `mt5/TradingIntelligenceEA.mq5` | MetaTrader 5 EA |
| `scripts/generate_sample_data.py` | Bootstrap training data |
| `scripts/train_models.py` | Train both AI models |
| `scripts/start_api.py` | Start FastAPI server |
| `scripts/backtest.py` | Run backtests |
| `scripts/init_database.py` | Apply Supabase schema |

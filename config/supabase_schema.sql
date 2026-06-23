-- ============================================================
-- TRADING INTELLIGENCE SYSTEM — SUPABASE SCHEMA
-- Run this entire file in the Supabase SQL editor
-- ============================================================

-- ── Enable Required Extensions ────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";          -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS "pg_trgm";         -- trigram similarity

-- ── ENUM Types ────────────────────────────────────────────────
DO $$ BEGIN
    CREATE TYPE trade_direction AS ENUM ('BUY', 'SELL');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE final_decision AS ENUM (
        'ALLOW_BUY', 'ALLOW_SELL', 'FLIP_TO_BUY', 'FLIP_TO_SELL', 'BLOCK'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE market_regime AS ENUM (
        'strong_bull_trend', 'weak_bull_trend',
        'strong_bear_trend', 'weak_bear_trend',
        'sideways_range', 'high_volatility', 'low_volatility',
        'news_volatility', 'liquidity_grab'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE session_type AS ENUM (
        'asian', 'london', 'new_york', 'off_hours',
        'overlap_london_ny', 'overlap_asian_london'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE trade_outcome AS ENUM ('WIN', 'LOSS', 'BREAKEVEN', 'PENDING');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ═══════════════════════════════════════════════════════════════
-- TABLE: market_snapshots
-- Stores the raw market state at the time of each EA signal
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS market_snapshots (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    captured_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(5) NOT NULL DEFAULT 'M5',

    -- Price data
    open_price      NUMERIC(18, 8) NOT NULL,
    high_price      NUMERIC(18, 8) NOT NULL,
    low_price       NUMERIC(18, 8) NOT NULL,
    close_price     NUMERIC(18, 8) NOT NULL,
    spread_pips     NUMERIC(8, 4),

    -- Session
    session         session_type,
    hour_utc        SMALLINT,
    day_of_week     SMALLINT,

    -- Raw candle data (JSONB array of OHLCV)
    candles_m5      JSONB,           -- last 100 M5 candles
    candles_m15     JSONB,           -- last 50 M15 candles
    candles_h1      JSONB,           -- last 20 H1 candles
    candles_h4      JSONB,           -- last 10 H4 candles
    candles_d1      JSONB,           -- last 5 D1 candles

    -- Computed features (stored for audit / replay)
    features        JSONB,

    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_time ON market_snapshots (symbol, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_session     ON market_snapshots (session);
CREATE INDEX IF NOT EXISTS idx_snapshots_features    ON market_snapshots USING gin(features);


-- ═══════════════════════════════════════════════════════════════
-- TABLE: market_regimes
-- Regime classification for each snapshot
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS market_regimes (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    snapshot_id     UUID REFERENCES market_snapshots(id) ON DELETE CASCADE,
    symbol          VARCHAR(20) NOT NULL,
    classified_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    regime          market_regime NOT NULL,
    confidence      NUMERIC(5, 4) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    regime_scores   JSONB,           -- full score vector for all regimes
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_regimes_symbol_time ON market_regimes (symbol, classified_at DESC);
CREATE INDEX IF NOT EXISTS idx_regimes_regime      ON market_regimes (regime);


-- ═══════════════════════════════════════════════════════════════
-- TABLE: predictions
-- Trader AI and Risk Manager AI outputs
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS predictions (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    snapshot_id             UUID REFERENCES market_snapshots(id) ON DELETE CASCADE,
    predicted_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- EA original signal
    ea_signal               trade_direction NOT NULL,

    -- Trader AI
    trader_buy_prob         NUMERIC(5, 4) NOT NULL,
    trader_sell_prob        NUMERIC(5, 4) NOT NULL,
    trader_direction        trade_direction NOT NULL,
    trader_confidence       NUMERIC(5, 4) NOT NULL,
    trader_model_used       VARCHAR(50),

    -- Risk Manager AI
    risk_allow_prob         NUMERIC(5, 4) NOT NULL,
    risk_block_prob         NUMERIC(5, 4) NOT NULL,
    risk_quality_score      NUMERIC(5, 4) NOT NULL,
    risk_model_used         VARCHAR(50),
    risk_block_reasons      JSONB,           -- array of reason strings

    -- Final Decision Engine
    final_decision          final_decision NOT NULL,
    is_flip                 BOOLEAN DEFAULT FALSE,
    is_blocked              BOOLEAN DEFAULT FALSE,

    -- Similar setups from memory
    similar_setups_count    INTEGER DEFAULT 0,
    similar_setups_win_rate NUMERIC(5, 4),
    similar_setups_avg_pnl  NUMERIC(10, 4),
    similar_setups_avg_dd   NUMERIC(10, 4),

    -- Latency
    inference_ms            INTEGER,

    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_snapshot    ON predictions (snapshot_id);
CREATE INDEX IF NOT EXISTS idx_predictions_time        ON predictions (predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_decision    ON predictions (final_decision);
CREATE INDEX IF NOT EXISTS idx_predictions_ea_signal   ON predictions (ea_signal);


-- ═══════════════════════════════════════════════════════════════
-- TABLE: trades
-- Completed trade records with outcomes
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS trades (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id   UUID REFERENCES predictions(id) ON DELETE SET NULL,
    snapshot_id     UUID REFERENCES market_snapshots(id) ON DELETE SET NULL,
    mt5_ticket      BIGINT UNIQUE,
    symbol          VARCHAR(20) NOT NULL,

    -- Direction
    direction       trade_direction NOT NULL,
    was_flipped     BOOLEAN DEFAULT FALSE,
    original_signal trade_direction,

    -- Prices
    entry_price     NUMERIC(18, 8) NOT NULL,
    exit_price      NUMERIC(18, 8),
    stop_loss       NUMERIC(18, 8),
    take_profit     NUMERIC(18, 8),
    lot_size        NUMERIC(8, 4),

    -- Time
    opened_at       TIMESTAMPTZ NOT NULL,
    closed_at       TIMESTAMPTZ,
    duration_minutes INTEGER,

    -- P&L
    pnl_pips        NUMERIC(10, 4),
    pnl_usd         NUMERIC(12, 4),
    max_drawdown_pips NUMERIC(10, 4),
    max_profit_pips   NUMERIC(10, 4),

    -- Outcome
    outcome         trade_outcome DEFAULT 'PENDING',

    -- Context
    session         session_type,
    regime          market_regime,
    day_of_week     SMALLINT,

    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_time  ON trades (symbol, opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_outcome      ON trades (outcome);
CREATE INDEX IF NOT EXISTS idx_trades_regime       ON trades (regime);
CREATE INDEX IF NOT EXISTS idx_trades_session      ON trades (session);
CREATE INDEX IF NOT EXISTS idx_trades_mt5_ticket   ON trades (mt5_ticket);


-- ═══════════════════════════════════════════════════════════════
-- TABLE: trade_conditions
-- Detailed market conditions recorded per trade (for ML training)
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS trade_conditions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_id            UUID REFERENCES trades(id) ON DELETE CASCADE,
    snapshot_id         UUID REFERENCES market_snapshots(id) ON DELETE SET NULL,

    -- Market Structure
    hh_count            SMALLINT,
    hl_count            SMALLINT,
    lh_count            SMALLINT,
    ll_count            SMALLINT,
    bos_bullish         BOOLEAN,
    bos_bearish         BOOLEAN,
    choch_bullish       BOOLEAN,
    choch_bearish       BOOLEAN,
    structure_score     NUMERIC(6, 4),

    -- Price Action
    body_size_avg_50    NUMERIC(10, 6),
    body_size_avg_100   NUMERIC(10, 6),
    wick_upper_avg      NUMERIC(10, 6),
    wick_lower_avg      NUMERIC(10, 6),
    momentum_5          NUMERIC(10, 6),
    momentum_10         NUMERIC(10, 6),
    momentum_20         NUMERIC(10, 6),

    -- Trend
    trend_m5            SMALLINT,   -- -1 bear, 0 neutral, 1 bull
    trend_m15           SMALLINT,
    trend_h1            SMALLINT,
    trend_h4            SMALLINT,
    trend_d1            SMALLINT,
    trend_alignment_score NUMERIC(5, 4),

    -- Volatility
    atr_14              NUMERIC(10, 6),
    atr_normalized      NUMERIC(8, 6),
    std_dev_20          NUMERIC(10, 6),
    range_expansion     BOOLEAN,
    range_contraction   BOOLEAN,

    -- Indicators
    rsi_14              NUMERIC(6, 4),
    adx_14              NUMERIC(6, 4),
    macd_histogram      NUMERIC(10, 8),
    price_vs_ma20       NUMERIC(8, 6),
    price_vs_ma50       NUMERIC(8, 6),

    -- Liquidity
    dist_to_support     NUMERIC(10, 6),
    dist_to_resistance  NUMERIC(10, 6),
    dist_to_pdh         NUMERIC(10, 6),
    dist_to_pdl         NUMERIC(10, 6),

    -- Session
    session             session_type,
    hour_utc            SMALLINT,

    -- Regime
    regime              market_regime,

    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conditions_trade   ON trade_conditions (trade_id);
CREATE INDEX IF NOT EXISTS idx_conditions_session ON trade_conditions (session);
CREATE INDEX IF NOT EXISTS idx_conditions_regime  ON trade_conditions (regime);


-- ═══════════════════════════════════════════════════════════════
-- TABLE: similarity_vectors
-- Vector embeddings for nearest-neighbor memory search
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS similarity_vectors (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_id        UUID REFERENCES trades(id) ON DELETE CASCADE,
    snapshot_id     UUID REFERENCES market_snapshots(id) ON DELETE CASCADE,
    embedding       vector(64),          -- 64-dim feature vector
    outcome         trade_outcome,
    pnl_pips        NUMERIC(10, 4),
    max_drawdown    NUMERIC(10, 4),
    regime          market_regime,
    session         session_type,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- IVFFlat index for fast approximate nearest neighbor
CREATE INDEX IF NOT EXISTS idx_vectors_embedding
    ON similarity_vectors USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_vectors_outcome  ON similarity_vectors (outcome);
CREATE INDEX IF NOT EXISTS idx_vectors_regime   ON similarity_vectors (regime);


-- ═══════════════════════════════════════════════════════════════
-- TABLE: training_datasets
-- Versioned snapshots of training data
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS training_datasets (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version         VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    sample_count    INTEGER,
    feature_count   INTEGER,
    positive_rate   NUMERIC(5, 4),
    date_range_start TIMESTAMPTZ,
    date_range_end   TIMESTAMPTZ,
    metadata        JSONB
);


-- ═══════════════════════════════════════════════════════════════
-- TABLE: model_results
-- Stores trained model metadata and validation metrics
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS model_results (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trained_at      TIMESTAMPTZ DEFAULT NOW(),
    model_type      VARCHAR(50) NOT NULL,    -- 'trader_ai' or 'risk_manager'
    algorithm       VARCHAR(50) NOT NULL,    -- 'random_forest', 'xgboost', etc.
    version         VARCHAR(20),
    dataset_id      UUID REFERENCES training_datasets(id),

    -- Performance metrics
    accuracy        NUMERIC(6, 4),
    precision_score NUMERIC(6, 4),
    recall_score    NUMERIC(6, 4),
    f1_score        NUMERIC(6, 4),
    roc_auc         NUMERIC(6, 4),
    log_loss        NUMERIC(8, 6),

    -- Walk-forward validation
    wf_mean_accuracy NUMERIC(6, 4),
    wf_std_accuracy  NUMERIC(6, 4),
    wf_min_accuracy  NUMERIC(6, 4),

    -- Hyperparameters
    hyperparameters JSONB,
    feature_importance JSONB,

    -- Is this the active model?
    is_active       BOOLEAN DEFAULT FALSE,
    file_path       VARCHAR(255),

    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_results_type    ON model_results (model_type, trained_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_results_active  ON model_results (is_active) WHERE is_active = TRUE;


-- ═══════════════════════════════════════════════════════════════
-- TABLE: performance_metrics
-- Aggregated trading performance by dimension
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS performance_metrics (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_date     DATE NOT NULL,
    symbol          VARCHAR(20),
    dimension       VARCHAR(50),     -- 'regime', 'session', 'weekday', etc.
    dimension_value VARCHAR(50),     -- 'strong_bull_trend', 'london', 'Monday', etc.

    total_trades    INTEGER DEFAULT 0,
    wins            INTEGER DEFAULT 0,
    losses          INTEGER DEFAULT 0,
    breakeven       INTEGER DEFAULT 0,
    blocked_trades  INTEGER DEFAULT 0,
    flipped_trades  INTEGER DEFAULT 0,
    flip_wins       INTEGER DEFAULT 0,

    win_rate        NUMERIC(5, 4),
    avg_pnl_pips    NUMERIC(10, 4),
    total_pnl_pips  NUMERIC(10, 4),
    max_drawdown    NUMERIC(10, 4),
    profit_factor   NUMERIC(8, 4),
    sharpe_ratio    NUMERIC(8, 4),

    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (metric_date, symbol, dimension, dimension_value)
);

CREATE INDEX IF NOT EXISTS idx_perf_date      ON performance_metrics (metric_date DESC);
CREATE INDEX IF NOT EXISTS idx_perf_dimension ON performance_metrics (dimension, dimension_value);


-- ═══════════════════════════════════════════════════════════════
-- VIEWS
-- ═══════════════════════════════════════════════════════════════

-- Daily summary view
CREATE OR REPLACE VIEW v_daily_summary AS
SELECT
    DATE(t.opened_at) AS trade_date,
    t.symbol,
    COUNT(*) AS total_trades,
    COUNT(*) FILTER (WHERE t.outcome = 'WIN') AS wins,
    COUNT(*) FILTER (WHERE t.outcome = 'LOSS') AS losses,
    ROUND(
        COUNT(*) FILTER (WHERE t.outcome = 'WIN')::NUMERIC /
        NULLIF(COUNT(*) FILTER (WHERE t.outcome IN ('WIN','LOSS')), 0), 4
    ) AS win_rate,
    SUM(t.pnl_pips) AS total_pips,
    AVG(t.pnl_pips) AS avg_pips,
    COUNT(*) FILTER (WHERE t.was_flipped) AS flipped,
    t.regime
FROM trades t
WHERE t.outcome != 'PENDING'
GROUP BY DATE(t.opened_at), t.symbol, t.regime;

-- Regime performance view
CREATE OR REPLACE VIEW v_regime_performance AS
SELECT
    t.regime,
    COUNT(*) AS total_trades,
    COUNT(*) FILTER (WHERE t.outcome = 'WIN') AS wins,
    ROUND(
        COUNT(*) FILTER (WHERE t.outcome = 'WIN')::NUMERIC /
        NULLIF(COUNT(*) FILTER (WHERE t.outcome IN ('WIN','LOSS')), 0), 4
    ) AS win_rate,
    ROUND(AVG(t.pnl_pips)::NUMERIC, 2) AS avg_pips,
    ROUND(SUM(t.pnl_pips)::NUMERIC, 2) AS total_pips
FROM trades t
WHERE t.outcome != 'PENDING'
GROUP BY t.regime
ORDER BY win_rate DESC;

-- Session performance view
CREATE OR REPLACE VIEW v_session_performance AS
SELECT
    t.session,
    COUNT(*) AS total_trades,
    ROUND(
        COUNT(*) FILTER (WHERE t.outcome = 'WIN')::NUMERIC /
        NULLIF(COUNT(*) FILTER (WHERE t.outcome IN ('WIN','LOSS')), 0), 4
    ) AS win_rate,
    ROUND(AVG(t.pnl_pips)::NUMERIC, 2) AS avg_pips
FROM trades t
WHERE t.outcome != 'PENDING'
GROUP BY t.session;

-- Blocked trades analysis view
CREATE OR REPLACE VIEW v_blocked_analysis AS
SELECT
    p.final_decision,
    p.risk_block_reasons,
    COUNT(*) AS count,
    AVG(p.trader_confidence) AS avg_trader_confidence,
    AVG(p.risk_quality_score) AS avg_risk_quality
FROM predictions p
GROUP BY p.final_decision, p.risk_block_reasons;

-- ── Triggers ──────────────────────────────────────────────────

-- Auto-update trades.updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trades_updated_at
    BEFORE UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ── Row Level Security (optional, for multi-tenant) ───────────
ALTER TABLE market_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Service role bypass (your Python backend uses service key)
CREATE POLICY "service_role_all" ON market_snapshots FOR ALL TO service_role USING (true);
CREATE POLICY "service_role_all" ON trades FOR ALL TO service_role USING (true);
CREATE POLICY "service_role_all" ON predictions FOR ALL TO service_role USING (true);

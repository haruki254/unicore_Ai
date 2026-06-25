"""
FastAPI Main Application

Endpoints:
  POST /predict          — Main prediction endpoint (called by MT5 EA)
  POST /trade/update     — Update trade outcome when closed
  POST /train            — Trigger model retraining
  GET  /health           — System health check
  GET  /analytics/*      — Dashboard data endpoints
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.schemas.market import (
    PredictionRequest, PredictionResponse,
    TradeUpdateRequest, TradeUpdateResponse,
    TrainRequest, TrainResponse,
    HealthResponse,
)
from core.features.feature_pipeline import FeaturePipeline
from core.models.trader_ai           import TraderAI
from core.models.risk_manager_ai     import RiskManagerAI
from core.decision.decision_engine   import DecisionEngine
from core.regime.regime_engine       import MarketRegimeEngine
from core.memory.trade_memory        import TradeMemoryEngine
from core.learning.pipeline          import LearningPipeline
from core.learning.adaptive_updater  import AdaptiveUpdater
from core.profiles                   import EAProfile
from database.client                 import DatabaseClient
from config.settings                 import settings
from monitoring.logger               import api_logger

# ── Global singletons ─────────────────────────────────────────
_start_time = time.time()

feature_pipeline = FeaturePipeline()
trader_ai        = TraderAI()
risk_manager     = RiskManagerAI()
decision_engine  = DecisionEngine()
regime_engine    = MarketRegimeEngine()
memory_engine    = TradeMemoryEngine()
db               = DatabaseClient()
adaptive_updater = AdaptiveUpdater()
learning_pipeline = LearningPipeline(
    trader_ai       = trader_ai,
    risk_manager    = risk_manager,
    db_client       = db,
    feature_pipeline= feature_pipeline,
)


# ── Lifespan ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load persisted models and memory on startup."""
    api_logger.info("Starting Trading Intelligence System...")

    loaded_trader = trader_ai.load()
    loaded_risk   = risk_manager.load()

    if not loaded_trader:
        api_logger.warning("Trader AI not found — awaiting training")
    if not loaded_risk:
        api_logger.warning("Risk Manager not found — awaiting training")

    api_logger.info(
        "System ready | trader={} risk={} memory={}",
        "✓" if loaded_trader else "✗",
        "✓" if loaded_risk   else "✗",
        memory_engine.size(),
    )
    yield
    # Shutdown
    memory_engine.save()
    api_logger.info("System shutdown — memory saved")


# ── App ───────────────────────────────────────────────────────

app = FastAPI(
    title       = "Trading Intelligence System",
    description = "Dual-AI institutional trading intelligence platform",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ── Auth dependency ───────────────────────────────────────────

async def verify_api_key(x_api_key: str = Header(default="")):
    if settings.api_debug:
        return  # Skip auth in debug mode
    if x_api_key != settings.api_secret_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _load_ea_profile(ea_id: str) -> tuple[EAProfile, bool]:
    profile_data = db.get_ea_profile(ea_id)
    if profile_data:
        return EAProfile.from_dict(profile_data), True
    return EAProfile(ea_id=ea_id), False


# ══════════════════════════════════════════════════════════════
# MAIN PREDICTION ENDPOINT
# ══════════════════════════════════════════════════════════════

@app.post(
    "/predict",
    response_model = PredictionResponse,
    tags           = ["Core"],
    summary        = "Get AI trade decision for an EA signal",
)
async def predict(
    req: PredictionRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(verify_api_key),
):
    """
    Main endpoint called by the MT5 Expert Advisor.

    1. Captures market snapshot
    2. Computes all features
    3. Runs Trader AI → BUY/SELL probabilities
    4. Runs Risk Manager AI → ALLOW/BLOCK
    5. Produces final decision
    6. Returns decision to EA

    Returns ALLOW_BUY | ALLOW_SELL | FLIP_TO_BUY | FLIP_TO_SELL | BLOCK
    """
    t0 = time.perf_counter()

    # ── Build snapshot dict ───────────────────────────────────
    snapshot = req.to_snapshot_dict()
    snapshot["symbol"] = req.symbol
    snapshot["ea_id"] = req.ea_id

    # ── Compute features ──────────────────────────────────────
    try:
        features = feature_pipeline.compute(snapshot)
        features["symbol"] = req.symbol
        features["ea_id"] = req.ea_id
    except Exception as e:
        api_logger.error("Feature computation error: {}", e)
        raise HTTPException(status_code=500, detail=f"Feature error: {e}")

    # ── Build risk context ────────────────────────────────────
    risk_ctx = req.risk_context.dict()
    risk_ctx["session_quality"] = features.get("session_quality", 0.5)
    ea_profile, _ = _load_ea_profile(req.ea_id)

    # ── Run decision pipeline ─────────────────────────────────
    try:
        result = decision_engine.decide(
            ea_signal      = req.ea_signal,
            features       = features,
            trader_ai      = trader_ai,
            risk_manager   = risk_manager,
            regime_engine  = regime_engine,
            memory_engine  = memory_engine,
            risk_context   = risk_ctx,
            ea_id          = req.ea_id,
            ea_profile     = ea_profile,
        )
    except Exception as e:
        api_logger.error("Decision pipeline error: {}", e)
        raise HTTPException(status_code=500, detail=f"Decision error: {e}")

    # ── Persist snapshot + prediction in background ───────────
    snapshot_id  = str(uuid.uuid4())
    prediction_id = None

    async def _persist():
        nonlocal prediction_id
        try:
            sid = db.save_snapshot({**snapshot, "id": snapshot_id, "features": features})
            pid = db.save_prediction(sid or snapshot_id, result.to_dict())
            if pid:
                prediction_id = pid
        except Exception as e:
            api_logger.error("Persist error: {}", e)

    background_tasks.add_task(_persist)

    api_logger.log_prediction(
        symbol        = req.symbol,
        ea_signal     = req.ea_signal,
        trader_buy    = result.trader_buy_prob,
        trader_sell   = result.trader_sell_prob,
        risk_quality  = result.risk_quality_score,
        final_decision= result.final_decision,
        inference_ms  = result.inference_ms,
    )

    print(DecisionEngine.format_summary(result))

    return PredictionResponse(
        final_decision    = result.final_decision,
        trade_direction   = result.trade_direction,
        trader_buy_prob   = result.trader_buy_prob,
        trader_sell_prob  = result.trader_sell_prob,
        trader_direction  = result.trader_direction,
        trader_confidence = result.trader_confidence,
        trader_model      = result.trader_model_used or "not_trained",
        risk_quality_score= result.risk_quality_score,
        risk_decision     = result.risk_decision,
        block_reasons     = result.risk_block_reasons,
        risk_model        = result.risk_model_used or "not_trained",
        regime            = result.regime,
        regime_confidence = result.regime_confidence,
        similar_count     = result.similar_count,
        similar_win_rate  = result.similar_win_rate,
        similar_avg_pnl   = result.similar_avg_pnl,
        is_flip           = result.is_flip,
        is_blocked        = result.is_blocked,
        inference_ms      = result.inference_ms,
        snapshot_id       = snapshot_id,
        prediction_id     = prediction_id,
    )


# ══════════════════════════════════════════════════════════════
# TRADE OUTCOME UPDATE
# ══════════════════════════════════════════════════════════════

@app.post(
    "/trade/update",
    response_model = TradeUpdateResponse,
    tags           = ["Core"],
    summary        = "Record trade outcome when MT5 closes a trade",
)
async def update_trade(
    req: TradeUpdateRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(verify_api_key),
):
    """
    Called by MT5 EA when a trade is closed.
    Updates database and adds to memory engine for future learning.
    """
    async def _process():
        try:
            # Update DB
            db.update_trade_outcome(
                mt5_ticket = req.mt5_ticket,
                outcome    = req.outcome,
                pnl_pips   = req.pnl_pips,
                pnl_usd    = req.pnl_usd,
                exit_price = req.exit_price,
                closed_at  = req.closed_at or datetime.utcnow(),
            )

            api_logger.log_trade_close(
                ticket      = req.mt5_ticket,
                symbol      = req.symbol,
                pnl_pips    = req.pnl_pips,
                outcome     = req.outcome,
                duration_min= 0,
            )

            # ── Adaptive update ───────────────────────────────
            try:
                profile_data = db.get_ea_profile(req.ea_id)
                flip_stats   = db.get_flip_stats(req.ea_id) or {}
                snapshot_features = {}

                if req.snapshot_id:
                    snap = db.get_snapshot_features(req.snapshot_id)
                    if snap:
                        snapshot_features = snap
                        snapshot_features["direction"] = req.outcome
                        snapshot_features["regime"]    = snap.get("regime") or req.regime
                        snapshot_features["session"]   = snap.get("session") or req.session

                if profile_data:
                    ea_profile    = EAProfile.from_dict(profile_data)
                    update_result = adaptive_updater.update(
                        ea_id             = req.ea_id,
                        outcome           = req.outcome,
                        snapshot_features = snapshot_features,
                        profile           = ea_profile,
                        flip_stats        = flip_stats,
                        was_flipped       = req.was_flipped,
                    )
                    db.save_ea_profile(req.ea_id, update_result.updated_profile.to_dict())
                    db.update_flip_stats(req.ea_id, update_result.updated_flip_stats)
                    api_logger.info("Adaptive update: {}", update_result.summary())
            except Exception as e:
                api_logger.error("Adaptive update failed for {}: {}", req.ea_id, e)
            # ── End adaptive update ───────────────────────────

            # Add to memory engine
            memory_engine.add(
                record_id    = str(req.mt5_ticket),
                features     = {},   # would be filled from saved snapshot
                outcome      = req.outcome,
                pnl_pips     = req.pnl_pips,
                max_drawdown = req.max_drawdown_pips,
                regime       = req.regime or "unknown",
                session      = req.session or "unknown",
                direction    = req.direction,
            )

        except Exception as e:
            api_logger.error("Trade update error: {}", e)

    background_tasks.add_task(_process)

    return TradeUpdateResponse(
        success  = True,
        trade_id = str(req.mt5_ticket),
        message  = f"Trade {req.mt5_ticket} recorded as {req.outcome}",
    )


# ══════════════════════════════════════════════════════════════
# TRAINING ENDPOINT
# ══════════════════════════════════════════════════════════════

@app.post(
    "/train",
    response_model = TrainResponse,
    tags           = ["Training"],
    summary        = "Trigger model retraining",
)
async def trigger_training(
    req: TrainRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(verify_api_key),
):
    """
    Trigger a full retraining cycle for both AI models.
    Runs in background — returns immediately.
    """
    async def _run_training():
        try:
            learning_pipeline.run(force=req.force)
        except Exception as e:
            api_logger.error("Training error: {}", e)

    background_tasks.add_task(_run_training)

    return TrainResponse(
        status          = "training_started",
        timestamp       = datetime.utcnow().isoformat(),
        elapsed_seconds = 0.0,
        trade_count     = 0,
        trader_ai       = {"status": "queued"},
        risk_manager    = {"status": "queued"},
    )


# ══════════════════════════════════════════════════════════════
# EA PROFILE ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get(
    "/ea-profile/{ea_id}",
    tags    = ["EA Profiles"],
    summary = "Return the current EA profile for inspection",
)
async def get_ea_profile_endpoint(
    ea_id: str,
    _: None = Depends(verify_api_key),
):
    """Return the current EA profile for inspection."""
    data = db.get_ea_profile(ea_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"No profile found for {ea_id}")
    return data


@app.get(
    "/ea-profiles",
    tags    = ["EA Profiles"],
    summary = "List all EA profile summaries",
)
async def list_ea_profiles(
    _: None = Depends(verify_api_key),
):
    """List all EA profile summaries."""
    try:
        resp = db.client.table("ea_profiles").select(
            "ea_id, total_trades, wins, losses, win_rate, "
            "flip_threshold, block_threshold, updated_at"
        ).execute()
        return resp.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/ea-profile/{ea_id}/rebuild",
    tags    = ["EA Profiles"],
    summary = "Force a profile rebuild from trade history for this EA",
)
async def rebuild_ea_profile(
    ea_id: str,
    _: None = Depends(verify_api_key),
):
    """Force a profile rebuild from trade history for this EA."""
    from core.profiles import EAProfileBuilder
    trades = db.fetch_completed_trades_for_ea(ea_id)
    if not trades:
        raise HTTPException(status_code=404, detail=f"No trades found for {ea_id}")
    builder  = EAProfileBuilder()
    profiles = builder.build_from_trades(trades)
    if ea_id not in profiles:
        raise HTTPException(status_code=422, detail="Could not build profile")
    db.save_ea_profile(ea_id, profiles[ea_id].to_dict())
    return {"status": "rebuilt", "ea_id": ea_id, "trades_used": len(trades)}


# ══════════════════════════════════════════════════════════════
# ANALYTICS ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/analytics/regime", tags=["Analytics"])
async def get_regime_performance():
    """Win rate and P&L breakdown by market regime."""
    data = db.get_regime_performance()
    return _aggregate_by(data, "regime")


@app.get("/analytics/session", tags=["Analytics"])
async def get_session_performance():
    """Win rate and P&L breakdown by trading session."""
    data = db.get_session_performance()
    return _aggregate_by(data, "session")


@app.get("/analytics/weekday", tags=["Analytics"])
async def get_weekday_performance():
    """Win rate and P&L breakdown by day of week."""
    data = db.get_performance_by_weekday()
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    agg  = _aggregate_by(data, "day_of_week")
    # Replace int keys with day names
    named = {}
    for k, v in agg.items():
        try:
            named[days[int(k)]] = v
        except (ValueError, IndexError):
            named[k] = v
    return named


@app.get("/analytics/equity", tags=["Analytics"])
async def get_equity_curve(symbol: str = None):
    """Equity curve and drawdown data."""
    data = db.get_equity_curve(symbol)
    equity   = []
    running  = 0.0
    peak     = 0.0
    drawdowns = []

    for row in data:
        pips = row.get("pnl_pips") or 0.0
        running += pips
        equity.append({
            "time":    row.get("opened_at"),
            "equity":  round(running, 2),
            "regime":  row.get("regime"),
            "session": row.get("session"),
        })
        if running > peak:
            peak = running
        dd = peak - running
        drawdowns.append(round(dd, 2))

    return {"equity_curve": equity, "drawdowns": drawdowns}


@app.get("/analytics/blocked", tags=["Analytics"])
async def get_blocked_analysis():
    """Analysis of blocked trades."""
    data = db.get_blocked_trades_analysis()
    return {"blocked_trades": data, "count": len(data)}


@app.get("/analytics/models", tags=["Analytics"])
async def get_model_performance():
    """Current model performance metrics."""
    trader_metrics = trader_ai.get_best_metrics()
    risk_metrics   = risk_manager.get_best_metrics()

    return {
        "trader_ai": {
            "is_trained":         trader_ai.is_trained,
            "algorithm":          trader_ai.best_algorithm,
            "metrics":            trader_metrics.to_dict() if trader_metrics else {},
            "feature_importance": trader_ai.get_feature_importance(),
        },
        "risk_manager": {
            "is_trained":         risk_manager.is_trained,
            "algorithm":          risk_manager.best_algorithm,
            "metrics":            risk_metrics.to_dict() if risk_metrics else {},
            "feature_importance": risk_manager.get_feature_importance(),
        },
    }


@app.get("/analytics/predictions/recent", tags=["Analytics"])
async def get_recent_predictions(limit: int = 50):
    """Most recent prediction records."""
    data = db.get_recent_predictions(limit)
    return {"predictions": data, "count": len(data)}


# ══════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """System health check."""
    return HealthResponse(
        status         = "healthy",
        trader_trained = trader_ai.is_trained,
        risk_trained   = risk_manager.is_trained,
        memory_size    = memory_engine.size(),
        db_connected   = db.is_connected,
        uptime_seconds = round(time.time() - _start_time, 1),
    )


@app.get("/", tags=["System"])
async def root():
    return {
        "name":    "Trading Intelligence System",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
    }


# ── Helper ────────────────────────────────────────────────────

def _aggregate_by(data: list, key: str) -> Dict[str, Any]:
    """Group trade records by a dimension key and compute stats."""
    groups: Dict[str, Dict] = {}
    for row in data:
        dim  = str(row.get(key) or "unknown")
        grp  = groups.setdefault(dim, {
            "total": 0, "wins": 0, "losses": 0, "pips": 0.0
        })
        grp["total"] += 1
        outcome = row.get("outcome", "")
        if outcome == "WIN":
            grp["wins"] += 1
        elif outcome == "LOSS":
            grp["losses"] += 1
        grp["pips"] += float(row.get("pnl_pips") or 0.0)

    result = {}
    for dim, grp in groups.items():
        denom = grp["wins"] + grp["losses"] or 1
        result[dim] = {
            "total_trades": grp["total"],
            "wins":         grp["wins"],
            "losses":       grp["losses"],
            "win_rate":     round(grp["wins"] / denom, 4),
            "total_pips":   round(grp["pips"], 2),
            "avg_pips":     round(grp["pips"] / (grp["total"] or 1), 2),
        }
    return result

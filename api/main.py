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

    # ── Generate IDs synchronously, BEFORE the response is built ──────
    snapshot_id   = str(uuid.uuid4())
    prediction_id = str(uuid.uuid4())
    decision_time = datetime.utcnow().isoformat()

    # ── Persist snapshot + prediction in background ───────────
    async def _persist():
        try:
            sid = db.save_snapshot({**snapshot, "id": snapshot_id, "features": features})
            db.save_prediction(sid or snapshot_id, result.to_dict(), prediction_id=prediction_id)
        except Exception as e:
            api_logger.error("Persist error: {}", e)

    background_tasks.add_task(_persist)

    # Throttled decision logging (only updates on new decisions)
    api_logger.log_prediction(
        symbol        = req.symbol,
        ea_signal     = req.ea_signal,
        trader_buy    = result.trader_buy_prob,
        trader_sell   = result.trader_sell_prob,
        risk_quality  = result.risk_quality_score,
        final_decision= result.final_decision,
        inference_ms  = result.inference_ms,
    )

    # Optional: keep for console (can be commented out)
    # print(DecisionEngine.format_summary(result))

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
        decision_time     = decision_time,
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
    """
    async def _process():
        try:
            history_ok = db.update_trade_outcome(
                mt5_ticket      = req.mt5_ticket,
                outcome         = req.outcome,
                pnl_pips        = req.pnl_pips,
                pnl_usd         = req.pnl_usd,
                exit_price      = req.exit_price,
                closed_at       = req.closed_at or datetime.utcnow(),
                symbol            = req.symbol,
                direction         = req.direction,
                entry_price       = req.entry_price,
                lot_size          = req.lot_size,
                session           = req.session,
                ea_id             = req.ea_id,
                prediction_id     = req.prediction_id,
                snapshot_id       = req.snapshot_id,
                was_flipped       = req.was_flipped,
                original_signal   = req.original_signal,
                regime            = req.regime,
                max_drawdown_pips = req.max_drawdown_pips,
                opened_at         = req.opened_at,
            )

            api_logger.log_trade_close(
                ticket      = req.mt5_ticket,
                symbol      = req.symbol,
                pnl_pips    = req.pnl_pips,
                outcome     = req.outcome,
                duration_min= 0,
            )

            # Adaptive update (reduced noise)
            try:
                profile_data = db.get_ea_profile(req.ea_id)
                flip_stats   = db.get_flip_stats(req.ea_id) or {}
                snapshot_features = {}
                snapshot_ok       = False

                if req.snapshot_id:
                    snap = db.get_snapshot_features(req.snapshot_id)
                    if snap:
                        snapshot_features = snap
                        snapshot_features["direction"] = req.outcome
                        snapshot_features["regime"]    = snap.get("regime") or req.regime
                        snapshot_features["session"]   = snap.get("session") or req.session
                        snapshot_ok = True

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

                    if update_result.n_changes > 0:
                        api_logger.info("Adaptive update: {}", update_result.summary())
                    # else: silent (no spam)

            except Exception as e:
                api_logger.error("Adaptive update failed for {}: {}", req.ea_id, e)

            # Memory add
            if not snapshot_ok:
                api_logger.debug("Memory record for trade {} uses empty features", req.mt5_ticket)

            memory_engine.add(
                record_id    = str(req.mt5_ticket),
                features     = snapshot_features,
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
# TRAINING, ANALYTICS, HEALTH, etc. (unchanged)
# ══════════════════════════════════════════════════════════════

@app.post("/train", response_model=TrainResponse, tags=["Training"])
async def trigger_training(req: TrainRequest, background_tasks: BackgroundTasks, _: None = Depends(verify_api_key)):
    async def _run_training():
        try:
            learning_pipeline.run(force=req.force)
        except Exception as e:
            api_logger.error("Training error: {}", e)
    background_tasks.add_task(_run_training)
    return TrainResponse(status="training_started", timestamp=datetime.utcnow().isoformat())


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status         = "healthy",
        trader_trained = trader_ai.is_trained,
        risk_trained   = risk_manager.is_trained,
        memory_size    = memory_engine.size(),
        db_connected   = db.is_connected,
        uptime_seconds = round(time.time() - _start_time, 1),
    )


# ... (all other endpoints like /analytics/*, EA profiles, etc. remain exactly as in your original file) ...
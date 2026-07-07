from fastapi import APIRouter, Depends
from typing import List, Dict, Any
import uuid
from datetime import datetime
from database.client import db  # your Supabase client

router = APIRouter(prefix="/ai", tags=["ai_executor"])

@router.get("/decisions")
async def get_ai_decisions(limit: int = 20):
    """Returns pending AI decisions for the Executor EA"""
    # Get recent unprocessed decisions
    decisions = await db.fetch_completed_predictions(limit=limit)  # your helper
    return {
        "status": "success",
        "count": len(decisions),
        "decisions": decisions
    }


@router.post("/decisions/add")
async def add_ai_decision(payload: Dict[str, Any]):
    """Called from /predict"""
    record = {
        "id": str(uuid.uuid4()),
        "final_decision": payload["final_decision"],
        "ea_id": payload.get("ea_id", "default"),
        "snapshot_id": payload.get("snapshot_id"),
        "prediction_id": payload.get("prediction_id"),
        "regime": payload.get("regime"),
        "timestamp": datetime.utcnow().isoformat(),
        "processed": False
    }
    
    # Save to a new table or use existing predictions
    await db.insert("ai_decisions", record)
    return {"status": "queued", "id": record["id"]}
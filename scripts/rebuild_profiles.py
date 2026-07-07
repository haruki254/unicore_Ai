import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import defaultdict
from supabase import create_client
from config.settings import settings
from database.client import DatabaseClient

def rebuild():
    print("Rebuilding profiles from trade_history...")

    db = DatabaseClient()
    if not db.is_connected:
        print("Supabase not connected")
        return

    # Fetch trades
    trades = db._client.table("trade_history").select("*").neq("outcome", "PENDING").execute().data or []

    print(f"Found {len(trades)} trades")

    by_ea = defaultdict(list)
    for t in trades:
        ea_id = str(t.get("ea_id") or "default")
        by_ea[ea_id].append(t)

    for ea_id, group in by_ea.items():
        total = len(group)
        wins = sum(1 for t in group if t.get("outcome") == "WIN")
        win_rate = wins / total if total > 0 else 0.0

        flips = [t for t in group if t.get("was_flipped") is True]
        flip_wins = sum(1 for t in flips if t.get("outcome") == "WIN")
        flip_win_rate = flip_wins / len(flips) if flips else 0.0

        profile = {
            "ea_id": ea_id,
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": round(win_rate, 4),
            "flip_threshold": 0.72,
            "block_threshold": 0.45,
            "regime_weights": {},
            "session_weights": {},
            "volatility_weights": {},
            "momentum_weights": {},
            "level_prox_weights": {}
        }

        flip_stats = {
            "ea_id": ea_id,
            "total_flips": len(flips),
            "flip_wins": flip_wins,
            "flip_losses": len(flips) - flip_wins,
            "flip_win_rate": round(flip_win_rate, 4),
            "flip_threshold": 0.72
        }

        db.save_ea_profile(ea_id, profile)
        db.update_flip_stats(ea_id, flip_stats)

        print(f"✓ {ea_id}: {total} trades, WR={win_rate:.1%}, Flips={len(flips)}")

    print("\n✅ Rebuild finished!")

if __name__ == "__main__":
    rebuild()
"""
Session Feature Engineering

Detects and encodes:
  - Asian, London, New York sessions
  - Session overlaps
  - Session quality scores
  - Hour-of-day and day-of-week features
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

from config.settings import settings


SESSION_HOURS = {
    "asian":   (settings.asian_session_start,  settings.asian_session_end),
    "london":  (settings.london_session_start, settings.london_session_end),
    "new_york":(settings.ny_session_start,     settings.ny_session_end),
}

# Typical pip volatility / quality score per session (normalised)
SESSION_QUALITY = {
    "asian":              0.45,
    "london":             0.85,
    "new_york":           0.80,
    "overlap_london_ny":  0.95,
    "overlap_asian_london":0.65,
    "off_hours":          0.20,
}


def get_session(dt: datetime) -> str:
    """
    Classify a UTC datetime into a trading session.

    Returns one of:
      'asian', 'london', 'new_york',
      'overlap_london_ny', 'overlap_asian_london', 'off_hours'
    """
    hour = dt.hour
    in_asian  = SESSION_HOURS["asian"][0]  <= hour < SESSION_HOURS["asian"][1]
    in_london = SESSION_HOURS["london"][0] <= hour < SESSION_HOURS["london"][1]
    in_ny     = SESSION_HOURS["new_york"][0] <= hour < SESSION_HOURS["new_york"][1]

    if in_london and in_ny:
        return "overlap_london_ny"
    elif in_asian and in_london:
        return "overlap_asian_london"
    elif in_london:
        return "london"
    elif in_ny:
        return "new_york"
    elif in_asian:
        return "asian"
    else:
        return "off_hours"


def compute_sessions(dt: datetime):
    """
    Compute all session-related features for a given UTC datetime.

    Parameters
    ----------
    dt : datetime
        UTC timestamp of the signal.

    Returns
    -------
    dict of feature_name -> value
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    session = get_session(dt)
    hour    = dt.hour
    dow     = dt.weekday()  # 0=Monday, 6=Sunday

    features: Dict[str, float] = {}

    # ── One-hot encode session ─────────────────────────────────
    all_sessions = ["asian", "london", "new_york",
                    "overlap_london_ny", "overlap_asian_london", "off_hours"]
    for s in all_sessions:
        features[f"session_{s}"] = float(session == s)

    # Convenience booleans
    features["session_asian"]   = float(session in ("asian", "overlap_asian_london"))
    features["session_london"]  = float(session in ("london", "overlap_london_ny", "overlap_asian_london"))
    features["session_ny"]      = float(session in ("new_york", "overlap_london_ny"))
    features["session_overlap"] = float(session in ("overlap_london_ny", "overlap_asian_london"))

    # ── Quality Score ─────────────────────────────────────────
    features["session_quality"] = SESSION_QUALITY.get(session, 0.20)

    # ── Temporal encodings ────────────────────────────────────
    features["hour_of_day"]  = float(hour)
    features["day_of_week"]  = float(dow)

    # Cyclical encoding (avoid discontinuity at hour 0 and day 0)
    import math
    features["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
    features["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
    features["dow_sin"]  = math.sin(2 * math.pi * dow  / 7.0)
    features["dow_cos"]  = math.cos(2 * math.pi * dow  / 7.0)

    # ── Peak hours flags ──────────────────────────────────────
    # London open (7-10 UTC): highest volatility window
    features["london_open_window"] = float(7 <= hour < 10)
    # NY open (13-16 UTC): second highest
    features["ny_open_window"]     = float(13 <= hour < 16)
    # Dead hours (22-1 UTC): avoid
    features["dead_zone"]          = float(hour >= 22 or hour < 1)

    # ── Day flags ─────────────────────────────────────────────
    features["is_monday"]  = float(dow == 0)
    features["is_friday"]  = float(dow == 4)
    features["is_weekend"] = float(dow >= 5)

    return features, session

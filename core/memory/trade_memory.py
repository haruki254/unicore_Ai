"""
Trade Memory Engine

Stores past market conditions as vector embeddings and performs
nearest-neighbor retrieval to find historically similar setups.

Uses FAISS for fast approximate nearest-neighbor search (fallback
to NumPy cosine similarity if FAISS not available).
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import settings
from monitoring.logger import memory_logger

# ── Try to import FAISS, fall back gracefully ─────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    memory_logger.warning("FAISS not found — using NumPy cosine similarity (slower)")


VECTOR_DIM     = 64
MIN_SETUPS     = 10     # minimum setups before similarity search is useful
MAX_MEMORY     = 10_000 # cap in-memory index size

EMBEDDING_FEATURE_KEYS = [
    # Structure
    "structure_score", "bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish",
    "hh_count", "hl_count", "lh_count", "ll_count",
    # Trend
    "trend_m5", "trend_m15", "trend_h1", "trend_h4", "trend_d1",
    "trend_alignment_score", "trend_strength",
    # Volatility
    "atr_rank", "volatility_regime", "range_expansion", "range_contraction",
    "bb_pct", "bb_width",
    # Indicators
    "rsi_14", "adx_14", "macd_histogram",
    "price_vs_ma20", "price_vs_ma50",
    # Liquidity
    "dist_to_support", "dist_to_resistance", "sr_ratio",
    "dist_to_pdh", "dist_to_pdl",
    "near_buy_side_liquidity", "near_sell_side_liquidity",
    # Session
    "session_london", "session_ny", "session_asian",
    "session_quality", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # Momentum
    "momentum_5", "momentum_10", "momentum_20",
    # Price action
    "candle_direction_ratio", "body_ratio_avg_50",
    "bullish_pin_bar", "bearish_pin_bar",
    "bullish_engulfing", "bearish_engulfing",
    # MA relationships
    "price_above_ma20", "price_above_ma50", "ma20_above_ma50",
    # Misc
    "spread_pips",
]

# Pad or truncate to VECTOR_DIM
assert len(EMBEDDING_FEATURE_KEYS) <= VECTOR_DIM, \
    f"Too many embedding keys ({len(EMBEDDING_FEATURE_KEYS)} > {VECTOR_DIM})"


@dataclass
class MemoryRecord:
    """A single memory entry."""
    record_id:    str
    embedding:    np.ndarray          # shape (VECTOR_DIM,)
    outcome:      str                 # WIN / LOSS / BREAKEVEN
    pnl_pips:     float
    max_drawdown: float
    regime:       str
    session:      str
    direction:    str                 # BUY or SELL
    features:     Dict[str, float]   # full feature dict for audit


@dataclass
class SimilarityResult:
    """Aggregated result from nearest-neighbor search."""
    count:        int
    win_rate:     float
    avg_pnl:      float
    avg_drawdown: float
    best_pnl:     float
    worst_pnl:    float
    regime_counts: Dict[str, int] = field(default_factory=dict)
    session_counts: Dict[str, int] = field(default_factory=dict)


class TradeMemoryEngine:
    """
    In-memory + optional persistent vector store for historical trade setups.

    Architecture
    ------------
    1. feature_vector → L2-normalise → 64-dim embedding
    2. FAISS IVF index (or NumPy fallback)
    3. On query: find k nearest neighbors → aggregate outcomes
    """

    def __init__(self, save_dir: str = None):
        self._records:    List[MemoryRecord] = []
        self._embeddings: Optional[np.ndarray] = None  # (N, VECTOR_DIM) cache
        self._index       = None                       # FAISS index
        self._dirty       = False                      # needs rebuild

        self._save_dir  = Path(save_dir or settings.model_save_path)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._memory_file = self._save_dir / "trade_memory.pkl"

        self._load()

    # ── Public API ─────────────────────────────────────────────

    def add(
        self,
        record_id:    str,
        features:     Dict[str, float],
        outcome:      str,
        pnl_pips:     float,
        max_drawdown: float = 0.0,
        regime:       str   = "unknown",
        session:      str   = "unknown",
        direction:    str   = "BUY",
    ) -> None:
        """Add a completed trade to memory."""
        embedding = self._build_embedding(features)

        record = MemoryRecord(
            record_id    = record_id,
            embedding    = embedding,
            outcome      = outcome,
            pnl_pips     = pnl_pips,
            max_drawdown = max_drawdown,
            regime       = regime,
            session      = session,
            direction    = direction,
            features     = {k: features.get(k, 0.0) for k in EMBEDDING_FEATURE_KEYS},
        )
        self._records.append(record)
        self._dirty = True

        # Cap memory size
        if len(self._records) > MAX_MEMORY:
            self._records = self._records[-MAX_MEMORY:]

        memory_logger.debug("Memory: {} records", len(self._records))

    def query(
        self,
        features:  Dict[str, float],
        k:         int   = 20,
        min_count: int   = MIN_SETUPS,
    ) -> Optional[SimilarityResult]:
        """
        Find k most similar historical setups and return aggregated stats.

        Returns None if not enough data.
        """
        if len(self._records) < min_count:
            return None

        if self._dirty:
            self._rebuild_index()

        embedding = self._build_embedding(features).reshape(1, -1)
        indices   = self._search(embedding, k=min(k, len(self._records)))

        neighbors = [self._records[i] for i in indices]

        # Aggregate outcomes
        wins    = sum(1 for r in neighbors if r.outcome == "WIN")
        losses  = sum(1 for r in neighbors if r.outcome == "LOSS")
        total   = wins + losses

        win_rate    = wins / total if total > 0 else 0.5
        avg_pnl     = float(np.mean([r.pnl_pips     for r in neighbors]))
        avg_dd      = float(np.mean([r.max_drawdown  for r in neighbors]))
        best_pnl    = float(max(r.pnl_pips for r in neighbors))
        worst_pnl   = float(min(r.pnl_pips for r in neighbors))

        # Frequency counts
        regime_counts  = {}
        session_counts = {}
        for r in neighbors:
            regime_counts[r.regime]   = regime_counts.get(r.regime, 0) + 1
            session_counts[r.session] = session_counts.get(r.session, 0) + 1

        result = SimilarityResult(
            count         = len(neighbors),
            win_rate      = win_rate,
            avg_pnl       = avg_pnl,
            avg_drawdown  = avg_dd,
            best_pnl      = best_pnl,
            worst_pnl     = worst_pnl,
            regime_counts  = regime_counts,
            session_counts = session_counts,
        )

        memory_logger.log_similarity(result.count, result.win_rate, result.avg_pnl)
        return result

    def size(self) -> int:
        return len(self._records)

    def save(self) -> None:
        """Persist memory to disk."""
        with open(self._memory_file, "wb") as f:
            pickle.dump(self._records, f)
        memory_logger.info("Memory saved: {} records → {}", len(self._records), self._memory_file)

    # ── Private ────────────────────────────────────────────────

    def _load(self) -> None:
        if self._memory_file.exists():
            try:
                with open(self._memory_file, "rb") as f:
                    self._records = pickle.load(f)
                memory_logger.info("Memory loaded: {} records", len(self._records))
                self._dirty = True
            except Exception as e:
                memory_logger.warning("Could not load memory: {} — starting fresh", e)
                self._records = []

    def _build_embedding(self, features: Dict[str, float]) -> np.ndarray:
        """Extract fixed-size embedding from feature dict."""
        vec = np.zeros(VECTOR_DIM, dtype=np.float32)
        for i, key in enumerate(EMBEDDING_FEATURE_KEYS):
            vec[i] = float(features.get(key, 0.0))

        # Clip extreme values
        vec = np.clip(vec, -10.0, 10.0)

        # L2 normalise
        norm = np.linalg.norm(vec) + 1e-10
        return (vec / norm).astype(np.float32)

    def _rebuild_index(self) -> None:
        """Rebuild the FAISS (or NumPy) search index."""
        if len(self._records) == 0:
            return

        embeddings = np.vstack([r.embedding for r in self._records])
        self._embeddings = embeddings.astype(np.float32)

        if FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(VECTOR_DIM)  # Inner product (cosine after L2 norm)
            index.add(self._embeddings)
            self._index = index
        # else: use self._embeddings directly for numpy search

        self._dirty = False
        memory_logger.debug("Index rebuilt: {} vectors", len(self._records))

    def _search(self, query: np.ndarray, k: int) -> List[int]:
        """Return indices of k nearest neighbors."""
        if FAISS_AVAILABLE and self._index is not None:
            _, indices = self._index.search(query, k)
            return indices[0].tolist()
        else:
            # NumPy cosine similarity (already L2-normalised → dot product)
            sims    = self._embeddings @ query.T
            sims    = sims.flatten()
            indices = np.argsort(-sims)[:k]
            return indices.tolist()

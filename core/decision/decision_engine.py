"""
Final Decision Engine

Combines outputs from:
  - Trader AI          → BUY/SELL probabilities
  - Risk Manager AI    → ALLOW/BLOCK + quality score
  - Trade Memory       → historical similarity context
  - Market Regime      → regime classification

Produces one of:
  ALLOW_BUY   — take the BUY as signalled or predicted
  ALLOW_SELL  — take the SELL as signalled or predicted
  FLIP_TO_BUY — EA said SELL but evidence says BUY
  FLIP_TO_SELL— EA said BUY but evidence says SELL
  BLOCK       — do not trade at all
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from config.settings        import settings
from core.profiles          import EAProfile
from core.scoring           import DirectionComparison, WeightedScorer
from monitoring.logger      import decision_logger


@dataclass
class DecisionResult:
    """Full output package returned to the MT5 EA."""

    # ── Trader AI ──────────────────────────────────────────────
    trader_buy_prob:    float = 0.5
    trader_sell_prob:   float = 0.5
    trader_direction:   str   = "BUY"
    trader_confidence:  float = 0.0
    trader_model_used:  str   = ""

    # ── Risk Manager ───────────────────────────────────────────
    risk_quality_score: float = 0.0
    risk_decision:      str   = "BLOCK"
    risk_block_reasons: List[str] = field(default_factory=list)
    risk_model_used:    str   = ""

    # ── Regime ────────────────────────────────────────────────
    regime:             str   = "unknown"
    regime_confidence:  float = 0.0

    # ── Memory ────────────────────────────────────────────────
    similar_count:      int   = 0
    similar_win_rate:   float = 0.5
    similar_avg_pnl:    float = 0.0
    similar_avg_dd:     float = 0.0
    profile_buy_confidence:  Optional[float] = None
    profile_sell_confidence: Optional[float] = None
    profile_favoured_direction: Optional[str] = None
    profile_confidence_gap: Optional[float] = None

    # ── Final ─────────────────────────────────────────────────
    ea_id:              str   = "default"
    ea_signal:          str   = "BUY"
    final_decision:     str   = "BLOCK"
    is_flip:            bool  = False
    is_blocked:         bool  = False
    inference_ms:       int   = 0

    # ── EA weighted scorer outputs ────────────────────────────
    ea_buy_score:        Optional[float] = None
    ea_sell_score:       Optional[float] = None
    ea_buy_confidence:   Optional[float] = None
    ea_sell_confidence:  Optional[float] = None
    scoring_method:      str             = "ml_only"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trader_buy_prob":    round(self.trader_buy_prob,   4),
            "trader_sell_prob":   round(self.trader_sell_prob,  4),
            "trader_direction":   self.trader_direction,
            "trader_confidence":  round(self.trader_confidence, 4),
            "trader_model_used":  self.trader_model_used,
            "risk_quality_score": round(self.risk_quality_score, 4),
            "risk_decision":      self.risk_decision,
            "risk_block_reasons": self.risk_block_reasons,
            "risk_model_used":    self.risk_model_used,
            "regime":             self.regime,
            "regime_confidence":  round(self.regime_confidence, 4),
            "similar_count":      self.similar_count,
            "similar_win_rate":   round(self.similar_win_rate,  4),
            "similar_avg_pnl":    round(self.similar_avg_pnl,   4),
            "similar_avg_dd":     round(self.similar_avg_dd,    4),
            "profile_buy_confidence":  self.profile_buy_confidence,
            "profile_sell_confidence": self.profile_sell_confidence,
            "profile_favoured_direction": self.profile_favoured_direction,
            "profile_confidence_gap": self.profile_confidence_gap,
            "ea_id":              self.ea_id,
            "ea_signal":          self.ea_signal,
            "final_decision":     self.final_decision,
            "is_flip":            self.is_flip,
            "is_blocked":         self.is_blocked,
            "inference_ms":       self.inference_ms,
            "ea_buy_score":       round(self.ea_buy_score,       4) if self.ea_buy_score       is not None else None,
            "ea_sell_score":      round(self.ea_sell_score,      4) if self.ea_sell_score      is not None else None,
            "ea_buy_confidence":  round(self.ea_buy_confidence,  4) if self.ea_buy_confidence  is not None else None,
            "ea_sell_confidence": round(self.ea_sell_confidence, 4) if self.ea_sell_confidence is not None else None,
            "scoring_method":     self.scoring_method,
        }

    @property
    def should_trade(self) -> bool:
        return not self.is_blocked

    @property
    def trade_direction(self) -> Optional[str]:
        if self.is_blocked:
            return None
        if "BUY" in self.final_decision:
            return "BUY"
        if "SELL" in self.final_decision:
            return "SELL"
        return None


class DecisionEngine:
    """
    Combines all AI outputs into a single actionable decision.

    Usage
    -----
    engine = DecisionEngine()
    result = engine.decide(
        ea_signal      = "BUY",
        features       = {...},
        trader_ai      = trader_ai_instance,
        risk_manager   = risk_manager_instance,
        regime_engine  = regime_engine_instance,
        memory_engine  = memory_engine_instance,
        risk_context   = {...},
    )
    """

    FLIP_THRESHOLD     = settings.flip_threshold
    MIN_TRADER_CONF    = settings.min_trader_confidence

    def __init__(self, weighted_scorer: WeightedScorer = None):
        self.weighted_scorer = weighted_scorer or WeightedScorer()

    def decide(
        self,
        ea_signal:      str,
        features:       Dict[str, Any],
        trader_ai,
        risk_manager,
        regime_engine,
        memory_engine,
        risk_context:   Dict[str, Any] = None,
        ea_id:          str = "default",
        ea_profile:     Optional[EAProfile] = None,
    ) -> DecisionResult:
        """
        Run the full decision pipeline.

        Returns
        -------
        DecisionResult with final_decision in:
          ALLOW_BUY | ALLOW_SELL | FLIP_TO_BUY | FLIP_TO_SELL | BLOCK
        """
        t_start = time.perf_counter()
        result  = DecisionResult(ea_id=ea_id, ea_signal=ea_signal)
        ctx     = risk_context or {}

        # ── 1. Market Regime ─────────────────────────────────
        regime, regime_conf, regime_scores = regime_engine.classify(features)
        features["_regime"]      = regime
        features["regime"]       = regime
        if "_session_label" in features:
            features["session"] = features["_session_label"]
        result.regime            = regime
        result.regime_confidence = regime_conf

        # Regime risk multiplier into context
        ctx["regime_risk_mult"] = regime_engine.regime_to_risk_multiplier(regime)

        # ── 2. Trade Memory ──────────────────────────────────
        similar = memory_engine.query(features)
        if similar:
            result.similar_count    = similar.count
            result.similar_win_rate = similar.win_rate
            result.similar_avg_pnl  = similar.avg_pnl
            result.similar_avg_dd   = similar.avg_drawdown
            ctx["similar_win_rate"] = similar.win_rate
            ctx["similar_avg_pnl"]  = similar.avg_pnl
            ctx["similar_count"]    = float(similar.count)

        # ── 3. Trader AI ─────────────────────────────────────
        buy_prob, sell_prob, direction, trader_conf = trader_ai.predict(
            features, ea_signal
        )
        result.trader_buy_prob   = buy_prob
        result.trader_sell_prob  = sell_prob
        result.trader_direction  = direction
        result.trader_confidence = trader_conf
        result.trader_model_used = trader_ai.best_algorithm or "heuristic"

        # ── 4. Risk Manager AI ───────────────────────────────
        quality, risk_dec, reasons = risk_manager.predict(
            features        = features,
            trader_buy_prob = buy_prob,
            trader_sell_prob= sell_prob,
            trader_confidence = trader_conf,
            risk_context    = ctx,
            similar_result  = similar,
        )
        result.risk_quality_score = quality
        result.risk_decision      = risk_dec
        result.risk_block_reasons = reasons
        result.risk_model_used    = risk_manager.best_algorithm or "heuristic"

        profile_cmp = self._score_profile(
            features=features,
            ea_profile=ea_profile,
            ea_id=ea_id,
            ea_signal=ea_signal,
            result=result,
        )

        # ── Store EA scorer outputs on result ─────────────────
        if profile_cmp is not None:
            result.ea_buy_score       = profile_cmp.buy.score
            result.ea_sell_score      = profile_cmp.sell.score
            result.ea_buy_confidence  = profile_cmp.buy.confidence
            result.ea_sell_confidence = profile_cmp.sell.confidence
            result.scoring_method     = "ea_profile"

        # ── 5. Final Decision Logic ───────────────────────────
        final = self._resolve_decision(
            ea_signal   = ea_signal,
            direction   = direction,
            buy_prob    = buy_prob,
            sell_prob   = sell_prob,
            trader_conf = trader_conf,
            risk_dec    = risk_dec,
            quality     = quality,
            regime      = regime,
            profile_cmp = profile_cmp,
            ea_profile  = ea_profile,
        )

        result.final_decision = final
        result.is_blocked     = (final == "BLOCK")
        result.is_flip        = ("FLIP" in final)
        result.inference_ms   = int((time.perf_counter() - t_start) * 1000)

        decision_logger.log_prediction(
            symbol        = str(features.get("symbol", "?")),
            ea_signal     = ea_signal,
            trader_buy    = buy_prob,
            trader_sell   = sell_prob,
            risk_quality  = quality,
            final_decision= final,
            inference_ms  = result.inference_ms,
        )

        return result

    def _score_profile(
        self,
        features: Dict[str, Any],
        ea_profile: Optional[EAProfile],
        ea_id: str,
        ea_signal: str,
        result: DecisionResult,
    ) -> Optional[DirectionComparison]:
        if ea_profile is None:
            return None
        try:
            cmp = self.weighted_scorer.compare_directions(features, ea_profile)
            result.profile_buy_confidence = cmp.buy.confidence
            result.profile_sell_confidence = cmp.sell.confidence
            result.profile_favoured_direction = cmp.favoured_direction
            result.profile_confidence_gap = cmp.confidence_gap
            decision_logger.info(
                self.weighted_scorer.format_breakdown_log(cmp, ea_id, ea_signal)
            )
            return cmp
        except Exception as e:
            decision_logger.error("EA profile scoring failed for {}: {}", ea_id, e)
            return None

    # ── Decision Resolution Logic ─────────────────────────────

    def _resolve_decision(
        self,
        ea_signal:   str,
        direction:   str,
        buy_prob:    float,
        sell_prob:   float,
        trader_conf: float,
        risk_dec:    str,
        quality:     float,
        regime:      str,
        profile_cmp: Optional[DirectionComparison] = None,
        ea_profile:  Optional[EAProfile] = None,
    ) -> str:
        """
        Core resolution table:

        Risk=BLOCK                                 → BLOCK
        Risk=ALLOW + direction == ea_signal        → ALLOW_{direction}
        Risk=ALLOW + direction != ea_signal        →
            if sell_prob > FLIP_THRESHOLD          → FLIP_TO_SELL
            if buy_prob  > FLIP_THRESHOLD          → FLIP_TO_BUY
            else                                   → BLOCK (ambiguous)
        """
        # Hard block
        if risk_dec == "BLOCK":
            return "BLOCK"

        # Regime block
        if regime in ("news_volatility",):
            return "BLOCK"

        block_threshold = (
            ea_profile.block_threshold if ea_profile else settings.min_risk_quality
        )
        if quality < block_threshold:
            return "BLOCK"

        # Trader confidence too low
        if trader_conf < self.MIN_TRADER_CONF - 0.5:
            return "BLOCK"

        # Direction agrees with EA signal
        if direction == ea_signal:
            if direction == "BUY":
                return "ALLOW_BUY"
            else:
                return "ALLOW_SELL"

        # Direction DISAGREES with EA — possible FLIP
        if profile_cmp is None:
            return "BLOCK"

        flip_high = (
            ea_profile.flip_threshold if ea_profile else self.FLIP_THRESHOLD
        )
        flip_low = max(0.0, min(1.0, 1.0 - flip_high))

        if ea_signal == "BUY" and direction == "SELL":
            if (
                profile_cmp.buy.confidence < flip_low and
                profile_cmp.sell.confidence > flip_high
            ):
                return "FLIP_TO_SELL"
            return "BLOCK"

        if ea_signal == "SELL" and direction == "BUY":
            if (
                profile_cmp.sell.confidence < flip_low and
                profile_cmp.buy.confidence > flip_high
            ):
                return "FLIP_TO_BUY"
            return "BLOCK"

        return "BLOCK"

    # ── Decision summary for logging ──────────────────────────

    @staticmethod
    def format_summary(result: DecisionResult) -> str:
        lines = [
            "=" * 55,
            f"  EA Signal     : {result.ea_signal}",
            f"  Trader AI     : BUY={result.trader_buy_prob:.0%}  "
            f"SELL={result.trader_sell_prob:.0%}  "
            f"→ {result.trader_direction}",
            f"  Risk Quality  : {result.risk_quality_score:.0%}  "
            f"→ {result.risk_decision}",
            f"  Regime        : {result.regime} ({result.regime_confidence:.0%})",
            f"  Memory        : {result.similar_count} setups  "
            f"WR={result.similar_win_rate:.0%}  "
            f"avgPnL={result.similar_avg_pnl:+.1f}pips",
        ]
        if result.risk_block_reasons:
            lines.append(f"  Block Reasons : {', '.join(result.risk_block_reasons)}")
        if result.profile_buy_confidence is not None:
            lines.append(
                f"  EA Profile    : BUY={result.profile_buy_confidence:.0%}  "
                f"SELL={result.profile_sell_confidence:.0%}  "
                f"favours {result.profile_favoured_direction}"
            )
        lines.append(f"  FINAL         : ▶  {result.final_decision}")
        lines.append(f"  Latency       : {result.inference_ms}ms")
        lines.append("=" * 55)
        return "\n".join(lines)

"""
Centralized logging — uses loguru when installed, else stdlib logging.
"""
import logging
import sys
from pathlib import Path

try:
    from loguru import logger as _loguru
    _USE_LOGURU = True
except ImportError:
    _USE_LOGURU = False

if _USE_LOGURU:
    # Without this, loguru only prints to stderr — nothing ever reaches
    # logs/trading_intelligence.log, so db_logger.error(...) calls (e.g.
    # a failed trade_history insert) were invisible unless you had the
    # console output open at the exact moment they fired.
    Path("./logs").mkdir(exist_ok=True)
    _loguru.add(
        "./logs/trading_intelligence.log",
        level="DEBUG",
        rotation="10 MB",
        retention="14 days",
        enqueue=True,
    )

# ── Bootstrap stdlib fallback ─────────────────────────────────
def _setup_stdlib():
    Path("./logs").mkdir(exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("./logs/trading_intelligence.log", encoding="utf-8"),
        ],
    )

if not _USE_LOGURU:
    _setup_stdlib()


class _Logger:
    """Thin wrapper so the rest of the codebase uses one API."""

    def __init__(self, name: str):
        self._name = name
        if _USE_LOGURU:
            self._l = _loguru.bind(component=name)
        else:
            self._l = logging.getLogger(name)

    def _msg(self, msg: str, **kw) -> str:
        if kw:
            try:
                return msg.format(**kw)
            except Exception:
                return msg + " " + str(kw)
        return msg

    def debug(self, msg: str, *args, **kw):
        m = self._msg(msg, **kw) if kw else (msg.format(*args) if args else msg)
        if _USE_LOGURU: self._l.debug(m)
        else:           self._l.debug(m)

    def info(self, msg: str, *args, **kw):
        m = self._msg(msg, **kw) if kw else (msg.format(*args) if args else msg)
        if _USE_LOGURU: self._l.info(m)
        else:           self._l.info(m)

    def warning(self, msg: str, *args, **kw):
        m = self._msg(msg, **kw) if kw else (msg.format(*args) if args else msg)
        if _USE_LOGURU: self._l.warning(m)
        else:           self._l.warning(m)

    def error(self, msg: str, *args, **kw):
        m = self._msg(msg, **kw) if kw else (msg.format(*args) if args else msg)
        if _USE_LOGURU: self._l.error(m)
        else:           self._l.error(m)

    def critical(self, msg: str, *args, **kw):
        m = self._msg(msg, **kw) if kw else (msg.format(*args) if args else msg)
        if _USE_LOGURU: self._l.critical(m)
        else:           self._l.critical(m)

    # ── Domain helpers ─────────────────────────────────────────
    def log_prediction(self, symbol, ea_signal, trader_buy, trader_sell,
                       risk_quality, final_decision, inference_ms):
        self.info(
            f"PREDICT | {symbol} | EA:{ea_signal} | "
            f"Buy:{trader_buy:.0%} Sell:{trader_sell:.0%} | "
            f"Risk:{risk_quality:.0%} | → {final_decision} | {inference_ms}ms"
        )

    def log_trade_close(self, ticket, symbol, pnl_pips, outcome, duration_min):
        emoji = "✅" if outcome == "WIN" else "❌" if outcome == "LOSS" else "➖"
        self.info(f"CLOSE #{ticket} | {symbol} | {emoji} {outcome} | {pnl_pips:+.1f} pips | {duration_min}min")

    def log_model_train(self, model_type, algorithm, accuracy, roc_auc, samples):
        self.info(f"TRAIN | {model_type} | {algorithm} | acc={accuracy:.2%} auc={roc_auc:.4f} | n={samples}")

    def log_regime(self, symbol, regime, confidence):
        self.debug(f"REGIME | {symbol} → {regime} ({confidence:.0%})")

    def log_similarity(self, count, win_rate, avg_pnl):
        self.debug(f"MEMORY | {count} similar | WR={win_rate:.0%} | avgPnL={avg_pnl:+.1f} pips")

    def log_block(self, symbol, reasons):
        self.warning(f"BLOCKED | {symbol} | {reasons}")


api_logger      = _Logger("api")
model_logger    = _Logger("models")
feature_logger  = _Logger("features")
regime_logger   = _Logger("regime")
memory_logger   = _Logger("memory")
decision_logger = _Logger("decision")
db_logger       = _Logger("database")
backtest_logger = _Logger("backtest")
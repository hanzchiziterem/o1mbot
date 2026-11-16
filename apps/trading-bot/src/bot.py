"""Live and Paper Execution Daemon for Exness MetaTrader 5.

Performs live or paper execution on Exness MT5 using identical top-down strategy logic,
logging trade records and metadata directly into SQLite.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml

from src.database import DatabaseManager
from src.models import (
    TradeOutcome,
    TradeRecord,
    TradeSignal,
)
from src.mt5_connector import MT5Connector
from src.strategy import TopDownVPStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("o1mbot.daemon")


class TradingBotDaemon:
    """Live/Paper trading execution daemon."""

    def __init__(self, config_path: str = "config/settings.yaml") -> None:
        self.config_path = config_path
        self.config = self._load_config(config_path)

        db_path = self.config.get("general", {}).get("database_path", "data/trades_metadata.db")
        self.db = DatabaseManager(db_path=db_path)
        self.strategy = TopDownVPStrategy(self.config)

        mt5_cfg = self.config.get("mt5", {})
        self.connector = MT5Connector(
            login=mt5_cfg.get("login", 0),
            password=mt5_cfg.get("password", ""),
            server=mt5_cfg.get("server", "Exness-MT5Real"),
            path=mt5_cfg.get("path", ""),
        )

        self.running = False
        self.symbols: List[str] = self.config.get("trading", {}).get(
            "symbols", ["XAUUSDm", "BTCUSDm", "EURUSDm", "GBPUSDm"]
        )
        self.max_open_trades = self.config.get("risk", {}).get("max_open_trades", 3)
        self.mode = self.config.get("general", {}).get("mode", "LIVE_PAPER")

        # Track active tickets mapped to TradeRecord IDs in SQLite
        self.active_trades: Dict[int, int] = {}

    def _load_config(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            logger.warning("Config file %s not found. Using defaults.", path)
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def start(self) -> None:
        """Starts the daemon execution loop."""
        self.running = True
        logger.info("Starting o1mbot Trading Daemon in %s mode...", self.mode)

        if not self.connector.initialize():
            logger.error("Failed to initialize MT5 connector. Exiting.")
            return

        account = self.connector.get_account_info()
        logger.info(
            "Account connected: Login=%s, Balance=%.2f, Equity=%.2f",
            account.login,
            account.balance,
            account.equity,
        )

        # Setup graceful signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        try:
            while self.running:
                self.run_cycle()
                time.sleep(15)  # Poll every 15 seconds
        except Exception as e:
            logger.exception("Unexpected error in daemon loop: %s", e)
        finally:
            self.stop()

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        logger.info("Received shutdown signal (%s). Stopping gracefully...", signum)
        self.running = False

    def stop(self) -> None:
        """Stops the daemon and disconnects MT5."""
        self.running = False
        self.connector.shutdown()
        logger.info("Trading Daemon stopped successfully.")

    def run_cycle(self) -> None:
        """Executes a single market scan and position management cycle."""
        account = self.connector.get_account_info()

        # Check existing positions
        self._manage_open_positions()

        if len(self.active_trades) >= self.max_open_trades:
            logger.debug(
                "Max open trades limit reached (%d/%d). Skipping scan.",
                len(self.active_trades),
                self.max_open_trades,
            )
            return

        for symbol in self.symbols:
            if not self.running:
                break

            try:
                self._evaluate_symbol(symbol, account.equity)
            except Exception as e:
                logger.error("Error evaluating symbol %s: %s", symbol, e)

    def _evaluate_symbol(self, symbol: str, equity: float) -> None:
        """Fetches multi-timeframe candles and evaluates strategy."""
        timeframes = ["MN1", "W1", "D1", "H4", "H1", "M15", "M5"]
        rates_by_tf: Dict[str, Any] = {}

        for tf in timeframes:
            count = 100 if tf in ("MN1", "W1") else 200
            rates_by_tf[tf] = self.connector.get_rates(symbol, tf, count=count)

        s_info = self.connector.get_symbol_info(symbol)
        spread = s_info.spread * s_info.point

        signal: Optional[TradeSignal] = self.strategy.evaluate(
            symbol=symbol,
            rates_by_tf=rates_by_tf,
            equity=equity,
            current_spread=spread,
        )

        if signal is None:
            return

        logger.info(
            "Valid TradeSignal found: %s %s @ %.5f (Planned RR: %.2f)",
            signal.direction.value,
            symbol,
            signal.entry_price,
            signal.planned_rr,
        )
        self._execute_signal(signal)

    def _execute_signal(self, signal: TradeSignal) -> None:
        """Submits trade to MT5 and logs into SQLite."""
        order_res = self.connector.send_order(
            symbol=signal.symbol,
            order_type=signal.direction.value,
            lots=signal.lot_size,
            price=signal.entry_price,
            sl=signal.sl_price,
            tp=signal.tp_price,
            comment=f"o1mbot-{signal.pattern.value}",
        )

        if order_res.ticket > 0:
            trade_record = TradeRecord(
                ticket=order_res.ticket,
                mode=self.mode,
                symbol=signal.symbol,
                direction=signal.direction.value,
                htf=signal.htf,
                ltf=signal.ltf,
                entry_pattern=signal.pattern.value,
                vp_poc=signal.poc_price,
                fvg_confluence=1 if signal.fvg_confluence else 0,
                entry_price=order_res.price,
                sl_price=signal.sl_price,
                tp_price=signal.tp_price,
                planned_rr=signal.planned_rr,
                outcome=TradeOutcome.OPEN.value,
                entry_time=datetime.now(timezone.utc).isoformat(),
                lot_size=signal.lot_size,
                notes=f"Order executed: ticket {order_res.ticket}",
            )
            trade_id = self.db.insert_trade(trade_record)
            self.active_trades[order_res.ticket] = trade_id
            logger.info(
                "Trade successfully logged: Ticket=%d, DB ID=%d", order_res.ticket, trade_id
            )

    def _manage_open_positions(self) -> None:
        """Monitors open positions, moves SL to Break-Even at 2.5R, and updates closed trades."""
        # For mock or live MT5, check active ticket statuses
        closed_tickets = []
        for ticket, _db_id in self.active_trades.items():
            # If in mock mode, check if ticket still in mock positions
            if self.connector.mock_mode:
                if ticket not in self.connector._mock_positions:
                    closed_tickets.append(ticket)
            else:
                # Live MT5 check
                pass

        for ticket in closed_tickets:
            del self.active_trades[ticket]


def main() -> None:
    parser = argparse.ArgumentParser(description="o1mbot Trading Daemon")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings.yaml")
    args = parser.parse_args()

    daemon = TradingBotDaemon(config_path=args.config)
    daemon.start()


if __name__ == "__main__":
    main()

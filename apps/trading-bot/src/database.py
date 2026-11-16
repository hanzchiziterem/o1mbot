"""SQLite metadata storage and KPI reporting module.

Logs trades and performance statistics for both BACKTEST and LIVE_PAPER modes.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import pandas as pd

from src.models import TradeRecord


class DatabaseManager:
    """Manages SQLite trades metadata database."""

    def __init__(self, db_path: str = "data/trades_metadata.db") -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initializes tables if they do not exist."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER,
                    mode TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    htf TEXT NOT NULL,
                    ltf TEXT NOT NULL,
                    entry_pattern TEXT NOT NULL,
                    vp_poc REAL,
                    fvg_confluence INTEGER,
                    entry_price REAL NOT NULL,
                    sl_price REAL NOT NULL,
                    tp_price REAL NOT NULL,
                    close_price REAL,
                    planned_rr REAL NOT NULL,
                    realized_rr REAL,
                    outcome TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    close_time TEXT,
                    lot_size REAL NOT NULL,
                    profit REAL,
                    notes TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trades_mode_symbol
                ON trades(mode, symbol)
                """
            )
            conn.commit()

    def insert_trade(self, trade: TradeRecord) -> int:
        """Inserts a new trade record into the database."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO trades (
                    ticket, mode, symbol, direction, htf, ltf, entry_pattern,
                    vp_poc, fvg_confluence, entry_price, sl_price, tp_price,
                    close_price, planned_rr, realized_rr, outcome, entry_time,
                    close_time, lot_size, profit, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.ticket,
                    trade.mode,
                    trade.symbol,
                    trade.direction,
                    trade.htf,
                    trade.ltf,
                    trade.entry_pattern,
                    trade.vp_poc,
                    trade.fvg_confluence,
                    trade.entry_price,
                    trade.sl_price,
                    trade.tp_price,
                    trade.close_price,
                    trade.planned_rr,
                    trade.realized_rr,
                    trade.outcome,
                    trade.entry_time,
                    trade.close_time,
                    trade.lot_size,
                    trade.profit,
                    trade.notes,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)

    def update_trade(self, trade: TradeRecord) -> None:
        """Updates an existing trade record."""
        if trade.id is None:
            raise ValueError("Trade ID must be specified for update")

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE trades SET
                    ticket = ?, mode = ?, symbol = ?, direction = ?, htf = ?, ltf = ?,
                    entry_pattern = ?, vp_poc = ?, fvg_confluence = ?, entry_price = ?,
                    sl_price = ?, tp_price = ?, close_price = ?, planned_rr = ?,
                    realized_rr = ?, outcome = ?, entry_time = ?, close_time = ?,
                    lot_size = ?, profit = ?, notes = ?
                WHERE id = ?
                """,
                (
                    trade.ticket,
                    trade.mode,
                    trade.symbol,
                    trade.direction,
                    trade.htf,
                    trade.ltf,
                    trade.entry_pattern,
                    trade.vp_poc,
                    trade.fvg_confluence,
                    trade.entry_price,
                    trade.sl_price,
                    trade.tp_price,
                    trade.close_price,
                    trade.planned_rr,
                    trade.realized_rr,
                    trade.outcome,
                    trade.entry_time,
                    trade.close_time,
                    trade.lot_size,
                    trade.profit,
                    trade.notes,
                    trade.id,
                ),
            )
            conn.commit()

    def get_trades(
        self,
        mode: Optional[str] = None,
        symbol: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> List[TradeRecord]:
        """Retrieves trades with optional filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params: List[Any] = []

        if mode and mode != "ALL":
            query += " AND mode = ?"
            params.append(mode)
        if symbol and symbol != "All":
            query += " AND symbol = ?"
            params.append(symbol)
        if outcome and outcome != "All":
            query += " AND outcome = ?"
            params.append(outcome)

        query += " ORDER BY id DESC"

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            trades: List[TradeRecord] = []
            for r in rows:
                trades.append(
                    TradeRecord(
                        id=r["id"],
                        ticket=r["ticket"],
                        mode=r["mode"],
                        symbol=r["symbol"],
                        direction=r["direction"],
                        htf=r["htf"],
                        ltf=r["ltf"],
                        entry_pattern=r["entry_pattern"],
                        vp_poc=r["vp_poc"],
                        fvg_confluence=r["fvg_confluence"],
                        entry_price=r["entry_price"],
                        sl_price=r["sl_price"],
                        tp_price=r["tp_price"],
                        close_price=r["close_price"],
                        planned_rr=r["planned_rr"],
                        realized_rr=r["realized_rr"],
                        outcome=r["outcome"],
                        entry_time=r["entry_time"],
                        close_time=r["close_time"],
                        lot_size=r["lot_size"],
                        profit=r["profit"],
                        notes=r["notes"],
                    )
                )
            return trades

    def get_kpi_summary(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Calculates key performance indicators."""
        trades = self.get_trades(mode=mode)
        closed_trades = [t for t in trades if t.outcome in ("WIN", "LOSS", "BE")]

        total_trades = len(closed_trades)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_count": 0,
                "loss_count": 0,
                "be_count": 0,
                "win_rate_pct": 0.0,
                "loss_rate_pct": 0.0,
                "avg_planned_rr": 0.0,
                "avg_realized_rr": 0.0,
                "cumulative_r": 0.0,
            }

        wins = [t for t in closed_trades if t.outcome == "WIN"]
        losses = [t for t in closed_trades if t.outcome == "LOSS"]
        bes = [t for t in closed_trades if t.outcome == "BE"]

        win_rate = (len(wins) / total_trades) * 100.0
        loss_rate = (len(losses) / total_trades) * 100.0

        avg_planned_rr = sum(t.planned_rr for t in closed_trades) / total_trades

        realized_rrs = [t.realized_rr for t in closed_trades if t.realized_rr is not None]
        avg_realized_rr = (sum(realized_rrs) / len(realized_rrs)) if realized_rrs else 0.0
        cumulative_r = sum(realized_rrs) if realized_rrs else 0.0

        return {
            "total_trades": total_trades,
            "win_count": len(wins),
            "loss_count": len(losses),
            "be_count": len(bes),
            "win_rate_pct": round(win_rate, 2),
            "loss_rate_pct": round(loss_rate, 2),
            "avg_planned_rr": round(avg_planned_rr, 2),
            "avg_realized_rr": round(avg_realized_rr, 2),
            "cumulative_r": round(cumulative_r, 2),
        }

    def get_trades_dataframe(self, mode: Optional[str] = None) -> pd.DataFrame:
        """Returns trades as a pandas DataFrame."""
        with self._connection() as conn:
            query = "SELECT * FROM trades"
            params: List[Any] = []
            if mode and mode != "ALL":
                query += " WHERE mode = ?"
                params.append(mode)
            query += " ORDER BY id ASC"
            return pd.read_sql_query(query, conn, params=params)

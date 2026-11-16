"""Tests for SQLite database manager and KPI calculations."""

from __future__ import annotations

import os
import tempfile
from typing import Generator

import pytest

from src.database import DatabaseManager
from src.models import TradeRecord


@pytest.fixture
def temp_db() -> Generator[DatabaseManager, None, None]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_path=path)
    yield db
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def test_insert_and_get_trades(temp_db: DatabaseManager) -> None:
    trade = TradeRecord(
        ticket=1001,
        mode="BACKTEST",
        symbol="XAUUSDm",
        direction="SELL",
        htf="H4",
        ltf="M15",
        entry_pattern="QMR",
        vp_poc=2665.0,
        fvg_confluence=1,
        entry_price=2665.0,
        sl_price=2672.0,
        tp_price=2620.0,
        close_price=2620.0,
        planned_rr=6.43,
        realized_rr=6.43,
        outcome="WIN",
        entry_time="2026-06-01T10:00:00",
        close_time="2026-06-01T15:00:00",
        lot_size=0.14,
        profit=630.0,
    )

    trade_id = temp_db.insert_trade(trade)
    assert trade_id > 0

    trades = temp_db.get_trades(mode="BACKTEST")
    assert len(trades) == 1
    assert trades[0].ticket == 1001
    assert trades[0].outcome == "WIN"


def test_kpi_summary(temp_db: DatabaseManager) -> None:
    # Insert 1 win (6R) and 1 loss (-1R)
    t1 = TradeRecord(
        mode="BACKTEST",
        symbol="BTCUSDm",
        direction="BUY",
        entry_price=90000.0,
        sl_price=89000.0,
        tp_price=96000.0,
        planned_rr=6.0,
        realized_rr=6.0,
        outcome="WIN",
        entry_time="2026-06-01",
    )
    t2 = TradeRecord(
        mode="BACKTEST",
        symbol="BTCUSDm",
        direction="BUY",
        entry_price=92000.0,
        sl_price=91000.0,
        tp_price=98000.0,
        planned_rr=6.0,
        realized_rr=-1.0,
        outcome="LOSS",
        entry_time="2026-06-02",
    )
    temp_db.insert_trade(t1)
    temp_db.insert_trade(t2)

    kpis = temp_db.get_kpi_summary(mode="BACKTEST")
    assert kpis["total_trades"] == 2
    assert kpis["win_rate_pct"] == 50.0
    assert kpis["loss_rate_pct"] == 50.0
    assert kpis["cumulative_r"] == 5.0

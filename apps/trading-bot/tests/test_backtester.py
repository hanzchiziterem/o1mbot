"""Tests for Historical Backtester."""

from __future__ import annotations

import os
import tempfile
from typing import Dict

import pandas as pd

from src.backtester import HistoricalBacktester
from src.database import DatabaseManager
from src.strategy import TopDownVPStrategy


def test_backtester_run(multi_tf_data: Dict[str, pd.DataFrame]) -> None:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    db = DatabaseManager(db_path=path)
    strategy = TopDownVPStrategy()
    backtester = HistoricalBacktester(strategy=strategy, database=db, initial_balance=10000.0)

    result = backtester.run("XAUUSDm", multi_tf_data, log_to_db=True)
    assert hasattr(result, "final_balance")
    assert hasattr(result, "total_trades")
    assert hasattr(result, "win_rate_pct")
    assert hasattr(result, "profit_factor")

    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

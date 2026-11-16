"""Tests for end-to-end Strategy pipeline."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from src.models import TradeSignal
from src.strategy import TopDownVPStrategy


def test_strategy_orchestrator(multi_tf_data: Dict[str, pd.DataFrame]) -> None:
    strategy = TopDownVPStrategy()
    assert strategy is not None

    signal = strategy.evaluate(
        symbol="XAUUSDm",
        rates_by_tf=multi_tf_data,
        equity=10000.0,
    )

    # If all criteria align, signal is TradeSignal; otherwise None
    if signal is not None:
        assert isinstance(signal, TradeSignal)
        assert signal.planned_rr >= 5.0

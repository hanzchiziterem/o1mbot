"""Tests for MT5 Connector and Mock Offline Adapter."""

from __future__ import annotations

from src.mt5_connector import MT5Connector


def test_mock_connector_lifecycle() -> None:
    connector = MT5Connector(mock_mode=True)
    assert connector.initialize() is True

    account = connector.get_account_info()
    assert account.balance >= 1000.0
    assert account.equity >= 1000.0

    s_info = connector.get_symbol_info("XAUUSDm")
    assert s_info.contract_size == 100.0
    assert s_info.bid > 0

    rates = connector.get_rates("XAUUSDm", "H1", count=50)
    assert len(rates) == 50
    assert "close" in rates.columns
    assert "tick_volume" in rates.columns

    # Order simulation
    res = connector.send_order("XAUUSDm", "SELL", lots=0.1, price=2650.0, sl=2660.0, tp=2600.0)
    assert res.ticket > 0
    assert connector.close_order(res.ticket) is True

    connector.shutdown()

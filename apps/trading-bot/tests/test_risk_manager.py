"""Tests for Risk Management, Lot Sizing, and Minimum 1:5 RR Rejection Rule."""

from __future__ import annotations

from datetime import datetime, timezone

from src.models import (
    EntryPattern,
    LTFEntryResult,
    MarketBias,
    OrderType,
    QuarterlyIPDALevels,
)
from src.risk_manager import RiskManager


def test_reject_when_planned_rr_below_5() -> None:
    rm = RiskManager(min_rr=5.0)

    # Risk = 10, Reward = 40 -> RR = 4.0 (< 5.0) -> Must be rejected!
    entry_result = LTFEntryResult(
        triggered=True,
        pattern=EntryPattern.QMR,
        entry_price=2660.0,
        sl_price=2670.0,  # Risk = 10 points
        trigger_time=datetime.now(timezone.utc),
    )

    ipda = QuarterlyIPDALevels(
        anchor_date=datetime.now(timezone.utc),
        high_20d=2700.0,
        low_20d=2620.0,
        high_40d=2720.0,
        low_40d=2610.0,
        high_60d=2750.0,
        low_60d=2620.0,  # Reward = 40 points -> RR = 4.0
        dol_price=2620.0,
        bias=MarketBias.BEARISH,
    )

    signal = rm.evaluate_and_filter(
        symbol="XAUUSDm",
        direction=OrderType.SELL,
        ltf_entry=entry_result,
        ipda=ipda,
    )

    # Strictly rejected!
    assert signal is None


def test_accept_when_planned_rr_at_or_above_5() -> None:
    rm = RiskManager(min_rr=5.0)

    # Risk = 10, Reward = 60 -> RR = 6.0 (>= 5.0) -> Accepted!
    entry_result = LTFEntryResult(
        triggered=True,
        pattern=EntryPattern.QMR,
        entry_price=2660.0,
        sl_price=2669.8,  # With buffer approx 10 points risk
        trigger_time=datetime.now(timezone.utc),
    )

    ipda = QuarterlyIPDALevels(
        anchor_date=datetime.now(timezone.utc),
        high_20d=2700.0,
        low_20d=2600.0,
        high_40d=2720.0,
        low_40d=2590.0,
        high_60d=2750.0,
        low_60d=2600.0,  # Reward = 60 points -> RR = 6.0
        dol_price=2600.0,
        bias=MarketBias.BEARISH,
    )

    signal = rm.evaluate_and_filter(
        symbol="XAUUSDm",
        direction=OrderType.SELL,
        ltf_entry=entry_result,
        ipda=ipda,
    )

    assert signal is not None
    assert signal.planned_rr >= 5.0
    assert signal.symbol == "XAUUSDm"
    assert signal.direction == OrderType.SELL


def test_lot_size_calculations() -> None:
    rm = RiskManager(risk_per_trade_pct=1.0)  # 1% of $10,000 = $100 risk

    # 1. Gold (XAUUSDm): Contract size = 100 oz
    # Risk distance = $10.0 -> Loss per 1.0 lot = $10.0 * 100 = $1,000
    # Expected lots = $100 / $1000 = 0.10 lots
    lots_gold = rm.calculate_lot_size(
        "XAUUSDm", equity=10000.0, entry_price=2660.0, sl_price=2670.0
    )
    assert lots_gold == 0.10

    # 2. Bitcoin (BTCUSDm): Contract size = 1 BTC
    # Risk distance = $1,000 -> Loss per 1.0 lot = $1,000
    # Expected lots = $100 / $1000 = 0.10 lots
    lots_btc = rm.calculate_lot_size(
        "BTCUSDm", equity=10000.0, entry_price=92000.0, sl_price=93000.0
    )
    assert lots_btc == 0.10

    # 3. Forex (EURUSDm): Contract size = 100,000
    # Risk distance = 0.0020 (20 pips) -> Loss per 1.0 lot = 0.0020 * 100,000 = $200
    # Expected lots = $100 / $200 = 0.50 lots
    lots_fx = rm.calculate_lot_size("EURUSDm", equity=10000.0, entry_price=1.0850, sl_price=1.0870)
    assert lots_fx == 0.50

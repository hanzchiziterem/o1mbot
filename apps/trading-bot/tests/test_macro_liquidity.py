"""Tests for Macro Liquidity, Quarterly IPDA Levels, and Inducement Sweeps."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.macro_liquidity import MacroLiquidityAnalyzer
from src.models import FairValueGap, MarketBias


def test_determine_macro_bias(sample_daily_data: pd.DataFrame) -> None:
    analyzer = MacroLiquidityAnalyzer()

    # Downward trending data should register bearish
    bias = analyzer.determine_macro_bias(sample_daily_data.iloc[:30], sample_daily_data)
    assert bias in (MarketBias.BEARISH, MarketBias.NEUTRAL)


def test_quarterly_anchor_dates() -> None:
    analyzer = MacroLiquidityAnalyzer()
    dates = analyzer.get_quarter_anchor_dates(2026)
    assert len(dates) == 9
    assert dates[0] == datetime(2025, 12, 1)
    assert dates[2] == datetime(2026, 3, 1)
    assert dates[4] == datetime(2026, 6, 1)
    assert dates[6] == datetime(2026, 9, 1)


def test_ipda_levels_calculation(sample_daily_data: pd.DataFrame) -> None:
    analyzer = MacroLiquidityAnalyzer()
    levels = analyzer.calculate_ipda_levels(sample_daily_data, bias=MarketBias.BEARISH)

    assert levels.high_20d > levels.low_20d
    assert levels.high_40d >= levels.high_20d
    assert levels.high_60d >= levels.high_40d
    assert levels.low_60d <= levels.low_40d
    # Unswept 60-day low is primary DOL for bearish bias
    assert levels.dol_price == levels.low_60d


def test_check_inducement_sweep_bearish(sample_daily_data: pd.DataFrame) -> None:
    analyzer = MacroLiquidityAnalyzer()
    levels = analyzer.calculate_ipda_levels(sample_daily_data, bias=MarketBias.BEARISH)

    # Artificially spike recent daily candle above 20d high
    modified_daily = sample_daily_data.copy()
    swept_price = levels.high_20d + 5.0
    modified_daily.loc[modified_daily.index[-1], "high"] = swept_price

    fvg = FairValueGap(
        top=swept_price + 2.0,
        bottom=swept_price - 2.0,
        is_bullish=False,
        start_time=datetime.now(timezone.utc),
    )
    swept, price = analyzer.check_inducement_sweep(modified_daily, levels, [fvg], [])

    assert swept is True
    assert price is not None
    assert price >= levels.high_20d

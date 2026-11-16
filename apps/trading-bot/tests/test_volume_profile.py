"""Tests for Fixed Range Volume Profile and PD Array confluence."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.models import FairValueGap, OrderBlock
from src.volume_profile import FixedRangeVolumeProfile


def test_volume_profile_poc_and_value_area() -> None:
    # 50 candles with high volume concentrated around 2650.0
    prices = np.linspace(2600.0, 2700.0, 50)
    highs = prices + 2.0
    lows = prices - 2.0
    # High volume near 2650.0
    volumes = np.array([5000 if 2645 <= p <= 2655 else 100 for p in prices])

    df = pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
            "tick_volume": volumes,
        }
    )

    vrp = FixedRangeVolumeProfile(bin_count=80, value_area_pct=0.70)
    res = vrp.compute_profile(df, 0, len(df) - 1)

    assert 2640.0 <= res.poc <= 2660.0
    assert res.val <= res.poc <= res.vah
    assert len(res.bins) == 80


def test_pd_array_confluence_fvg() -> None:
    vrp = FixedRangeVolumeProfile()
    poc = 2650.0
    fvg = FairValueGap(
        top=2652.0, bottom=2648.0, is_bullish=False, start_time=datetime.now(timezone.utc)
    )

    confluent, desc = vrp.check_pd_array_confluence(poc, fvg=fvg)
    assert confluent is True
    assert desc is not None
    assert "aligned with FVG" in desc


def test_pd_array_confluence_breaker() -> None:
    vrp = FixedRangeVolumeProfile()
    poc = 2650.0
    breaker = OrderBlock(
        top=2651.0,
        bottom=2649.0,
        is_bullish=False,
        candle_time=datetime.now(timezone.utc),
        is_breaker=True,
    )

    confluent, desc = vrp.check_pd_array_confluence(poc, breaker=breaker)
    assert confluent is True
    assert desc is not None
    assert "aligned with Breaker" in desc

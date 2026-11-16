"""Pytest fixtures and synthetic market data generators."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_daily_data() -> pd.DataFrame:
    """Generates 90 days of synthetic daily candles."""
    np.random.seed(123)
    count = 90
    now = datetime(2026, 6, 1)
    times = [now - timedelta(days=count - i) for i in range(count)]

    # Trend downwards from 2700 to 2600
    base = 2700.0
    drift = np.linspace(0, -100, count)
    noise = np.random.normal(0, 5, count)
    close_prices = base + drift + noise

    highs = close_prices + np.abs(np.random.normal(5, 2, count))
    lows = close_prices - np.abs(np.random.normal(5, 2, count))
    opens = (highs + lows) / 2.0
    volumes = np.random.randint(1000, 50000, count)

    return pd.DataFrame(
        {
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": close_prices,
            "volume": volumes,
            "tick_volume": volumes,
        }
    )


@pytest.fixture
def sample_htf_data() -> pd.DataFrame:
    """Generates 100 H4 candles with clear swing pivots and an FVG."""
    np.random.seed(456)
    count = 100
    now = datetime(2026, 6, 1)
    times = [now - timedelta(hours=4 * (count - i)) for i in range(count)]

    # Create explicit swings
    prices = np.sin(np.linspace(0, 3 * np.pi, count)) * 30 + 2650.0
    highs = prices + 5.0
    lows = prices - 5.0
    opens = prices - 1.0
    closes = prices + 1.0

    # Prominent swing high and low
    highs[20] = 2720.0
    lows[50] = 2580.0

    # Introduce a prominent bearish FVG between candle 60 and 62
    # Bearish FVG: high[62] < low[60]
    lows[60] = 2670.0
    highs[60] = 2675.0
    highs[62] = 2660.0
    lows[62] = 2655.0

    volumes = np.random.randint(500, 10000, count)
    return pd.DataFrame(
        {
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "tick_volume": volumes,
        }
    )


@pytest.fixture
def sample_ltf_data() -> pd.DataFrame:
    """Generates 60 M15 candles with clear QMR structure."""
    np.random.seed(789)
    count = 60
    now = datetime(2026, 6, 1, 12, 0)
    times = [now - timedelta(minutes=15 * (count - i)) for i in range(count)]

    base = 2660.0
    opens = np.full(count, base)
    highs = np.full(count, base + 2.0)
    lows = np.full(count, base - 2.0)
    closes = np.full(count, base)

    # Left Shoulder at index 35 (high = 2668.0)
    highs[35] = 2668.0
    closes[35] = 2666.0

    # Intermediate Neckline low at index 40 (low = 2658.0)
    lows[40] = 2658.0
    closes[40] = 2659.0

    # Head at index 45 (high = 2675.0)
    highs[45] = 2675.0
    closes[45] = 2672.0

    # Breakout below neckline at index 50 (close = 2654.0)
    lows[50] = 2653.0
    closes[50] = 2654.0

    # Retracement to Left Shoulder (2668.0) at latest candle
    closes[-1] = 2667.5
    highs[-1] = 2668.0
    lows[-1] = 2665.0

    volumes = np.random.randint(100, 3000, count)
    return pd.DataFrame(
        {
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "tick_volume": volumes,
        }
    )


@pytest.fixture
def multi_tf_data(
    sample_daily_data: pd.DataFrame, sample_htf_data: pd.DataFrame, sample_ltf_data: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Provides dictionary of multi-timeframe candles."""
    return {
        "MN1": sample_daily_data.iloc[:30].copy(),
        "W1": sample_daily_data.iloc[:50].copy(),
        "D1": sample_daily_data.copy(),
        "H4": sample_htf_data.copy(),
        "H1": sample_htf_data.copy(),
        "M15": sample_ltf_data.copy(),
        "M5": sample_ltf_data.copy(),
    }

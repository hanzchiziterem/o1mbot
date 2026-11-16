"""Tests for HTF Structure, Swing Pivots, FVGs, and CHOCH detection."""

from __future__ import annotations

import pandas as pd

from src.htf_structure import HTFStructureAnalyzer
from src.models import MarketBias, SwingType


def test_find_swings(sample_htf_data: pd.DataFrame) -> None:
    analyzer = HTFStructureAnalyzer(swing_lookback=3)
    swings = analyzer.find_swings(sample_htf_data)

    assert len(swings) >= 2
    types = [s.type for s in swings]
    assert SwingType.SWING_HIGH in types
    assert SwingType.SWING_LOW in types


def test_find_fvgs(sample_htf_data: pd.DataFrame) -> None:
    analyzer = HTFStructureAnalyzer(min_fvg_size_pct=0.0001)
    fvgs = analyzer.find_fvgs(sample_htf_data)

    # In conftest, we engineered high[62] < low[60] (bearish FVG)
    bearish_fvgs = [f for f in fvgs if not f.is_bullish]
    assert len(bearish_fvgs) >= 1
    assert bearish_fvgs[0].top > bearish_fvgs[0].bottom


def test_detect_choch(sample_htf_data: pd.DataFrame) -> None:
    analyzer = HTFStructureAnalyzer(swing_lookback=3)
    choch = analyzer.detect_choch(sample_htf_data, macro_bias=MarketBias.BEARISH)

    # Verify return object format
    assert hasattr(choch, "occurred")
    assert hasattr(choch, "breakout_price")

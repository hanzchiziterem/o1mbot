"""Tests for LTF precision entry detectors (QMR and AMD)."""

from __future__ import annotations

import pandas as pd

from src.ltf_entries import LTFEntryDetector
from src.models import EntryPattern, MarketBias


def test_detect_qmr_sell(sample_ltf_data: pd.DataFrame) -> None:
    detector = LTFEntryDetector(zone_tolerance_pct=0.01)
    htf_poc = 2674.0  # Close to the head at 2675.0

    res = detector.detect_qmr(sample_ltf_data, bias=MarketBias.BEARISH, htf_poc=htf_poc)
    assert res.triggered is True
    assert res.pattern == EntryPattern.QMR
    assert res.head_price == 2675.0
    assert res.left_shoulder_price == 2668.0
    assert res.sl_price == 2675.0


def test_detect_amd_sell() -> None:
    detector = LTFEntryDetector(amd_box_candles=8, amd_displacement_mult=1.0)
    # Construct 16 candles: 8 consolidation, 4 manipulation sweep, 4 distribution
    count = 16
    base = 2650.0

    highs = (
        [base + 1.0] * 8
        + [base + 5.0, base + 4.0, base + 2.0, base + 1.0]
        + [base - 2.0, base - 4.0, base - 5.0, base - 6.0]
    )
    lows = (
        [base - 1.0] * 8
        + [base, base, base - 1.0, base - 2.0]
        + [base - 4.0, base - 6.0, base - 7.0, base - 8.0]
    )
    closes = (
        [base] * 8
        + [base + 4.0, base + 3.0, base + 1.0, base - 1.0]
        + [base - 3.0, base - 5.0, base - 6.5, base - 7.5]
    )
    opens = [base] * count

    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})

    htf_poc = base + 5.0
    res = detector.detect_amd(df, bias=MarketBias.BEARISH, htf_poc=htf_poc)
    assert res.triggered is True
    assert res.pattern == EntryPattern.AMD
    assert res.sl_price >= base + 5.0

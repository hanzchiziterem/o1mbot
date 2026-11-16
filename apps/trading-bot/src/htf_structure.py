"""Higher Timeframe (HTF) Market Structure module.

Detects Swing Highs/Lows, Change of Character (CHOCH), Fair Value Gaps (FVGs),
and Order Blocks / Breakers on H4 and H1 timeframes.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from src.models import (
    CHOCHResult,
    FairValueGap,
    MarketBias,
    OrderBlock,
    SwingPoint,
    SwingType,
)


class HTFStructureAnalyzer:
    """Analyzes market structure, CHOCH events, FVGs, and Breaker blocks."""

    def __init__(self, swing_lookback: int = 5, min_fvg_size_pct: float = 0.0002) -> None:
        self.swing_lookback = swing_lookback
        self.min_fvg_size_pct = min_fvg_size_pct

    def find_swings(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Detects swing highs and swing lows using fractal pivot window."""
        swings: List[SwingPoint] = []
        if len(df) < self.swing_lookback * 2 + 1:
            return swings

        n = self.swing_lookback
        highs = df["high"].values
        lows = df["low"].values
        times = df["time"].values if "time" in df.columns else df.index.values

        for i in range(n, len(df) - n):
            is_swing_high = True
            is_swing_low = True

            curr_high = highs[i]
            curr_low = lows[i]

            for j in range(1, n + 1):
                if highs[i - j] >= curr_high or highs[i + j] >= curr_high:
                    is_swing_high = False
                if lows[i - j] <= curr_low or lows[i + j] <= curr_low:
                    is_swing_low = False

            if is_swing_high:
                swings.append(
                    SwingPoint(
                        index=i,
                        timestamp=pd.to_datetime(times[i]),
                        price=float(curr_high),
                        type=SwingType.SWING_HIGH,
                    )
                )
            if is_swing_low:
                swings.append(
                    SwingPoint(
                        index=i,
                        timestamp=pd.to_datetime(times[i]),
                        price=float(curr_low),
                        type=SwingType.SWING_LOW,
                    )
                )

        return swings

    def find_fvgs(self, df: pd.DataFrame, min_bars_ago: int = 0) -> List[FairValueGap]:
        """Detects 3-candle Fair Value Gaps (FVG) and tracks mitigation."""
        fvgs: List[FairValueGap] = []
        if len(df) < 3:
            return fvgs

        highs = df["high"].values
        lows = df["low"].values
        times = df["time"].values if "time" in df.columns else df.index.values

        for i in range(2, len(df) - min_bars_ago):
            # Bullish FVG: low[i] > high[i-2]
            if lows[i] > highs[i - 2]:
                gap_size = lows[i] - highs[i - 2]
                if gap_size / highs[i - 2] >= self.min_fvg_size_pct:
                    fvg = FairValueGap(
                        top=float(lows[i]),
                        bottom=float(highs[i - 2]),
                        is_bullish=True,
                        start_time=pd.to_datetime(times[i]),
                    )
                    # Check mitigation in subsequent candles
                    for k in range(i + 1, len(df)):
                        if lows[k] <= fvg.top:
                            fvg.mitigated = True
                            fvg.mitigation_time = pd.to_datetime(times[k])
                            break
                    fvgs.append(fvg)

            # Bearish FVG: high[i] < low[i-2]
            elif highs[i] < lows[i - 2]:
                gap_size = lows[i - 2] - highs[i]
                if gap_size / lows[i - 2] >= self.min_fvg_size_pct:
                    fvg = FairValueGap(
                        top=float(lows[i - 2]),
                        bottom=float(highs[i]),
                        is_bullish=False,
                        start_time=pd.to_datetime(times[i]),
                    )
                    # Check mitigation in subsequent candles
                    for k in range(i + 1, len(df)):
                        if highs[k] >= fvg.bottom:
                            fvg.mitigated = True
                            fvg.mitigation_time = pd.to_datetime(times[k])
                            break
                    fvgs.append(fvg)

        return fvgs

    def find_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """Detects Order Blocks (last opposing candle before displacement) and Breakers."""
        obs: List[OrderBlock] = []
        if len(df) < 4:
            return obs

        opens = df["open"].values
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        times = df["time"].values if "time" in df.columns else df.index.values

        for i in range(1, len(df) - 2):
            # Bullish OB: down candle followed by strong bullish displacement
            if closes[i] < opens[i]:
                displacement = (
                    (closes[i + 2] - highs[i]) if (i + 2 < len(df)) else (closes[i + 1] - highs[i])
                )
                candle_body = abs(opens[i] - closes[i])
                if displacement > candle_body:
                    ob = OrderBlock(
                        top=float(highs[i]),
                        bottom=float(lows[i]),
                        is_bullish=True,
                        candle_time=pd.to_datetime(times[i]),
                    )
                    # Check if violated -> becomes Bearish Breaker
                    for k in range(i + 1, len(df)):
                        if closes[k] < ob.bottom:
                            ob.is_breaker = True
                            break
                    obs.append(ob)

            # Bearish OB: up candle followed by strong bearish displacement
            elif closes[i] > opens[i]:
                displacement = (
                    (lows[i] - closes[i + 2]) if (i + 2 < len(df)) else (lows[i] - closes[i + 1])
                )
                candle_body = abs(opens[i] - closes[i])
                if displacement > candle_body:
                    ob = OrderBlock(
                        top=float(highs[i]),
                        bottom=float(lows[i]),
                        is_bullish=False,
                        candle_time=pd.to_datetime(times[i]),
                    )
                    # Check if violated -> becomes Bullish Breaker
                    for k in range(i + 1, len(df)):
                        if closes[k] > ob.top:
                            ob.is_breaker = True
                            break
                    obs.append(ob)

        return obs

    def detect_choch(
        self,
        df: pd.DataFrame,
        macro_bias: MarketBias,
    ) -> CHOCHResult:
        """Detects Change of Character (CHOCH) conforming to macro bias.

        - For SELL bias: Price sweeps a swing high, then closes below the higher low that produced that high.
        - For BUY bias: Price sweeps a swing low, then closes above the lower high that produced that low.
        """
        swings = self.find_swings(df)
        if len(swings) < 4:
            return CHOCHResult(
                occurred=False,
                direction=MarketBias.NEUTRAL,
                breakout_price=0.0,
                swing_high=0.0,
                swing_low=0.0,
                impulse_start_idx=0,
                impulse_end_idx=0,
            )

        closes = df["close"].values
        fvgs = self.find_fvgs(df)
        obs = self.find_order_blocks(df)

        if macro_bias == MarketBias.BEARISH:
            # Look for: Swing High -> Higher Low -> Breakout below Higher Low
            swing_highs = [s for s in swings if s.type == SwingType.SWING_HIGH]
            swing_lows = [s for s in swings if s.type == SwingType.SWING_LOW]

            if swing_highs and swing_lows:
                last_high = swing_highs[-1]
                # Find intermediate low before this high
                prior_lows = [s for s in swing_lows if s.index < last_high.index]
                if prior_lows:
                    higher_low = prior_lows[-1]
                    # Check if recent candles closed below this higher low
                    for k in range(last_high.index, len(df)):
                        if closes[k] < higher_low.price:
                            # Confluence FVG or Breaker formed during this leg
                            leg_fvgs = [f for f in fvgs if not f.is_bullish and not f.mitigated]
                            leg_breakers = [b for b in obs if b.is_breaker]
                            chosen_fvg = leg_fvgs[-1] if leg_fvgs else None
                            chosen_breaker = leg_breakers[-1] if leg_breakers else None

                            return CHOCHResult(
                                occurred=True,
                                direction=MarketBias.BEARISH,
                                breakout_price=float(higher_low.price),
                                swing_high=float(last_high.price),
                                swing_low=float(higher_low.price),
                                impulse_start_idx=higher_low.index,
                                impulse_end_idx=last_high.index,
                                fvg=chosen_fvg,
                                breaker=chosen_breaker,
                            )

        elif macro_bias == MarketBias.BULLISH:
            # Look for: Swing Low -> Lower High -> Breakout above Lower High
            swing_highs = [s for s in swings if s.type == SwingType.SWING_HIGH]
            swing_lows = [s for s in swings if s.type == SwingType.SWING_LOW]

            if swing_highs and swing_lows:
                last_low = swing_lows[-1]
                prior_highs = [s for s in swing_highs if s.index < last_low.index]
                if prior_highs:
                    lower_high = prior_highs[-1]
                    for k in range(last_low.index, len(df)):
                        if closes[k] > lower_high.price:
                            leg_fvgs = [f for f in fvgs if f.is_bullish and not f.mitigated]
                            leg_breakers = [b for b in obs if b.is_breaker]
                            chosen_fvg = leg_fvgs[-1] if leg_fvgs else None
                            chosen_breaker = leg_breakers[-1] if leg_breakers else None

                            return CHOCHResult(
                                occurred=True,
                                direction=MarketBias.BULLISH,
                                breakout_price=float(lower_high.price),
                                swing_high=float(lower_high.price),
                                swing_low=float(last_low.price),
                                impulse_start_idx=lower_high.index,
                                impulse_end_idx=last_low.index,
                                fvg=chosen_fvg,
                                breaker=chosen_breaker,
                            )

        return CHOCHResult(
            occurred=False,
            direction=MarketBias.NEUTRAL,
            breakout_price=0.0,
            swing_high=0.0,
            swing_low=0.0,
            impulse_start_idx=0,
            impulse_end_idx=0,
        )

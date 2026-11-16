"""Macro and Quarterly IPDA Liquidity Bias module.

Analyzes Monthly/Weekly market structure and 20/40/60-day IPDA quarterly levels
anchored from December (Dec 1, Mar 1, Jun 1, Sep 1 plus intermediate checkpoints)
to establish macro directional bias, inducement sweeps, and primary Draw on Liquidity (DOL).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models import FairValueGap, MarketBias, OrderBlock, QuarterlyIPDALevels


class MacroLiquidityAnalyzer:
    """Calculates quarterly IPDA liquidity cycles and macro structure bias."""

    def __init__(
        self,
        anchor_month: int = 12,
        anchor_day: int = 1,
        lookbacks: Tuple[int, int, int] = (20, 40, 60),
        primary_dol_lookback: int = 60,
    ) -> None:
        self.anchor_month = anchor_month
        self.anchor_day = anchor_day
        self.lookbacks = lookbacks
        self.primary_dol_lookback = primary_dol_lookback

    def determine_macro_bias(
        self,
        monthly_df: Optional[pd.DataFrame],
        weekly_df: Optional[pd.DataFrame],
    ) -> MarketBias:
        """Determines market bias across Monthly and Weekly timeframes.

        Rule:
        - Breaking highs and holding lows -> BULLISH
        - Breaking lows and resisting highs -> BEARISH
        """
        biases: List[MarketBias] = []

        for df in [monthly_df, weekly_df]:
            if df is None or len(df) < 5:
                continue

            # Analyze recent 3 swing legs / candles
            recent = df.iloc[-5:]
            higher_highs = (recent["high"].diff().dropna() > 0).sum()
            higher_lows = (recent["low"].diff().dropna() > 0).sum()
            lower_lows = (recent["low"].diff().dropna() < 0).sum()
            lower_highs = (recent["high"].diff().dropna() < 0).sum()

            latest_close = recent["close"].iloc[-1]
            prior_close = recent["close"].iloc[-2]

            if higher_highs >= higher_lows and latest_close >= prior_close:
                biases.append(MarketBias.BULLISH)
            elif lower_lows >= lower_highs and latest_close <= prior_close:
                biases.append(MarketBias.BEARISH)
            else:
                biases.append(MarketBias.NEUTRAL)

        if not biases:
            return MarketBias.NEUTRAL

        if MarketBias.BULLISH in biases and MarketBias.BEARISH not in biases:
            return MarketBias.BULLISH
        elif MarketBias.BEARISH in biases and MarketBias.BULLISH not in biases:
            return MarketBias.BEARISH
        else:
            # Fall back to latest weekly close relation to open
            if weekly_df is not None and len(weekly_df) >= 2:
                last_w = weekly_df.iloc[-1]
                return (
                    MarketBias.BULLISH if last_w["close"] >= last_w["open"] else MarketBias.BEARISH
                )
            return MarketBias.NEUTRAL

    def get_quarter_anchor_dates(self, current_year: int) -> List[datetime]:
        """Returns quarterly anchor dates starting Dec 1 of prior year through the current year.

        Anchors: Dec 1 (prior year), Mar 1, Jun 1, Sep 1, Dec 1, plus 2-month intermediate checkpoints.
        """
        anchors = [
            datetime(current_year - 1, 12, 1),
            datetime(current_year, 2, 1),  # 2-month intermediate
            datetime(current_year, 3, 1),  # Q2
            datetime(current_year, 5, 1),  # 2-month intermediate
            datetime(current_year, 6, 1),  # Q3
            datetime(current_year, 8, 1),  # 2-month intermediate
            datetime(current_year, 9, 1),  # Q4
            datetime(current_year, 11, 1),  # 2-month intermediate
            datetime(current_year, 12, 1),  # Q1 next
        ]
        return sorted(anchors)

    def calculate_ipda_levels(
        self,
        daily_df: pd.DataFrame,
        as_of_date: Optional[datetime] = None,
        bias: MarketBias = MarketBias.BEARISH,
    ) -> QuarterlyIPDALevels:
        """Calculates 20-day, 40-day, and 60-day highs and lows looking backward from reference date."""
        if daily_df.empty:
            raise ValueError("Daily DataFrame cannot be empty")

        df = daily_df.copy()
        if "time" in df.columns and not np.issubdtype(df["time"].dtype, np.datetime64):
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time")

        if as_of_date is not None and "time" in df.columns:
            df = df[df["time"] <= as_of_date]

        if len(df) < 60:
            # If less than 60 candles available, scale available bars gracefully
            l20 = min(len(df), 20)
            l40 = min(len(df), 40)
            l60 = len(df)
        else:
            l20, l40, l60 = self.lookbacks

        high_20 = float(df["high"].iloc[-l20:].max())
        low_20 = float(df["low"].iloc[-l20:].min())
        high_40 = float(df["high"].iloc[-l40:].max())
        low_40 = float(df["low"].iloc[-l40:].min())
        high_60 = float(df["high"].iloc[-l60:].max())
        low_60 = float(df["low"].iloc[-l60:].min())

        anchor = as_of_date if as_of_date is not None else datetime.now(timezone.utc)

        # Primary DOL: Unswept 60-day low for BEARISH, unswept 60-day high for BULLISH
        dol_price = low_60 if bias == MarketBias.BEARISH else high_60

        return QuarterlyIPDALevels(
            anchor_date=anchor,
            high_20d=high_20,
            low_20d=low_20,
            high_40d=high_40,
            low_40d=low_40,
            high_60d=high_60,
            low_60d=low_60,
            dol_price=dol_price,
            bias=bias,
        )

    def check_inducement_sweep(
        self,
        daily_df: pd.DataFrame,
        ipda: QuarterlyIPDALevels,
        htf_fvgs: List[FairValueGap],
        htf_obs: List[OrderBlock],
        lookback_bars: int = 5,
    ) -> Tuple[bool, Optional[float]]:
        """Verifies if price recently swept a 20d/40d/60d liquidity pool into an HTF FVG or Order Block.

        - For BEARISH: Price sweeps above a 20d, 40d, or 60d high into a Bearish FVG/OB.
        - For BULLISH: Price sweeps below a 20d, 40d, or 60d low into a Bullish FVG/OB.
        """
        if daily_df.empty:
            return False, None

        recent = daily_df.iloc[-lookback_bars:]

        if ipda.bias == MarketBias.BEARISH:
            # Check if any recent high breached 20d/40d/60d level
            high_levels = [ipda.high_20d, ipda.high_40d, ipda.high_60d]
            for _, row in recent.iterrows():
                high = float(row["high"])
                for level in high_levels:
                    # Tolerance: within 0.1% or swept through
                    if high >= level * 0.9995:
                        # Check if inside an HTF FVG or OB
                        for fvg in htf_fvgs:
                            if not fvg.is_bullish and (fvg.bottom <= high <= fvg.top * 1.002):
                                return True, high
                        for ob in htf_obs:
                            if not ob.is_bullish and (ob.bottom <= high <= ob.top * 1.002):
                                return True, high
                        # If no direct FVG recorded but level swept convincingly
                        if high > level:
                            return True, high

        elif ipda.bias == MarketBias.BULLISH:
            low_levels = [ipda.low_20d, ipda.low_40d, ipda.low_60d]
            for _, row in recent.iterrows():
                low = float(row["low"])
                for level in low_levels:
                    if low <= level * 1.0005:
                        for fvg in htf_fvgs:
                            if fvg.is_bullish and (fvg.bottom * 0.998 <= low <= fvg.top):
                                return True, low
                        for ob in htf_obs:
                            if ob.is_bullish and (ob.bottom * 0.998 <= low <= ob.top):
                                return True, low
                        if low < level:
                            return True, low

        return False, None

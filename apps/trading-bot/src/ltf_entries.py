"""Lower Timeframe (LTF) Precision Entry module.

Detects Quasimodo Reversal (QMR) and Accumulation-Manipulation-Distribution (AMD)
patterns on M15 and M5 inside the HTF POC confluence zone.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.models import (
    EntryPattern,
    LTFEntryResult,
    MarketBias,
)


class LTFEntryDetector:
    """Detects precision entry patterns (QMR and AMD) on lower timeframes."""

    def __init__(
        self,
        amd_box_candles: int = 12,
        amd_displacement_mult: float = 1.3,
        shoulder_tolerance_pct: float = 0.0015,
        zone_tolerance_pct: float = 0.0025,
    ) -> None:
        self.amd_box_candles = amd_box_candles
        self.amd_displacement_mult = amd_displacement_mult
        self.shoulder_tolerance_pct = shoulder_tolerance_pct
        self.zone_tolerance_pct = zone_tolerance_pct

    def detect_qmr(
        self,
        df: pd.DataFrame,
        bias: MarketBias,
        htf_poc: float,
    ) -> LTFEntryResult:
        """Detects Quasimodo Reversal (QMR) pattern.

        Sell QMR:
        - Left Shoulder (swing high)
        - Head (higher high sweeping into HTF POC zone)
        - Neckline (intermediate low between Shoulder and Head)
        - CHOCH / Break of Neckline
        - Return to Left Shoulder level for entry with SL above Head

        Buy QMR:
        - Left Shoulder (swing low)
        - Head (lower low sweeping into HTF POC zone)
        - Neckline (intermediate high)
        - CHOCH / Break of Neckline
        - Return to Left Shoulder level for entry with SL below Head
        """
        if len(df) < 15:
            return LTFEntryResult(
                triggered=False, pattern=EntryPattern.QMR, entry_price=0.0, sl_price=0.0
            )

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        times = df["time"].values if "time" in df.columns else df.index.values

        if bias == MarketBias.BEARISH:
            # Look backwards from recent candles
            # Find Head: highest high in recent 15 candles that touched HTF POC zone
            head_idx = int(np.argmax(highs[-15:])) + (len(df) - 15)
            head_price = float(highs[head_idx])

            # Head must touch or be close to HTF POC zone
            if (
                abs(head_price - htf_poc) / htf_poc > self.zone_tolerance_pct
                and head_price < htf_poc
            ):
                return LTFEntryResult(
                    triggered=False, pattern=EntryPattern.QMR, entry_price=0.0, sl_price=0.0
                )

            # Left Shoulder: prior swing high before head
            prior_highs = highs[max(0, head_idx - 15) : head_idx]
            if len(prior_highs) < 3:
                return LTFEntryResult(
                    triggered=False, pattern=EntryPattern.QMR, entry_price=0.0, sl_price=0.0
                )

            left_shoulder_local_idx = int(np.argmax(prior_highs))
            left_shoulder_idx = max(0, head_idx - 15) + left_shoulder_local_idx
            left_shoulder_price = float(highs[left_shoulder_idx])

            # Head must be higher than left shoulder
            if head_price <= left_shoulder_price:
                return LTFEntryResult(
                    triggered=False, pattern=EntryPattern.QMR, entry_price=0.0, sl_price=0.0
                )

            # Neckline: lowest low between left shoulder and head
            neckline_lows = lows[left_shoulder_idx : head_idx + 1]
            if len(neckline_lows) == 0:
                return LTFEntryResult(
                    triggered=False, pattern=EntryPattern.QMR, entry_price=0.0, sl_price=0.0
                )
            neckline_price = float(np.min(neckline_lows))

            # Check if after head, price broke below neckline
            broke_neckline = False
            break_idx = -1
            for k in range(head_idx + 1, len(df)):
                if closes[k] < neckline_price:
                    broke_neckline = True
                    break_idx = k
                    break

            if not broke_neckline or break_idx == -1:
                return LTFEntryResult(
                    triggered=False, pattern=EntryPattern.QMR, entry_price=0.0, sl_price=0.0
                )

            # Retracement to Left Shoulder
            latest_price = float(closes[-1])
            is_at_shoulder = (
                abs(latest_price - left_shoulder_price) / left_shoulder_price
                <= self.shoulder_tolerance_pct
            )

            # Alternatively, if price retested left shoulder and started rejecting
            retested = any(
                highs[k] >= left_shoulder_price * (1.0 - self.shoulder_tolerance_pct)
                for k in range(break_idx, len(df))
            )

            if is_at_shoulder or (retested and latest_price <= left_shoulder_price):
                trigger_time = pd.to_datetime(times[-1])
                return LTFEntryResult(
                    triggered=True,
                    pattern=EntryPattern.QMR,
                    entry_price=latest_price,
                    sl_price=head_price,
                    head_price=head_price,
                    left_shoulder_price=left_shoulder_price,
                    trigger_time=trigger_time,
                    details={
                        "head_price": head_price,
                        "left_shoulder": left_shoulder_price,
                        "neckline": neckline_price,
                    },
                )

        elif bias == MarketBias.BULLISH:
            head_idx = int(np.argmin(lows[-15:])) + (len(df) - 15)
            head_price = float(lows[head_idx])

            if (
                abs(head_price - htf_poc) / htf_poc > self.zone_tolerance_pct
                and head_price > htf_poc
            ):
                return LTFEntryResult(
                    triggered=False, pattern=EntryPattern.QMR, entry_price=0.0, sl_price=0.0
                )

            prior_lows = lows[max(0, head_idx - 15) : head_idx]
            if len(prior_lows) < 3:
                return LTFEntryResult(
                    triggered=False, pattern=EntryPattern.QMR, entry_price=0.0, sl_price=0.0
                )

            left_shoulder_local_idx = int(np.argmin(prior_lows))
            left_shoulder_idx = max(0, head_idx - 15) + left_shoulder_local_idx
            left_shoulder_price = float(lows[left_shoulder_idx])

            # Head must be lower than left shoulder
            if head_price >= left_shoulder_price:
                return LTFEntryResult(
                    triggered=False, pattern=EntryPattern.QMR, entry_price=0.0, sl_price=0.0
                )

            neckline_highs = highs[left_shoulder_idx : head_idx + 1]
            if len(neckline_highs) == 0:
                return LTFEntryResult(
                    triggered=False, pattern=EntryPattern.QMR, entry_price=0.0, sl_price=0.0
                )
            neckline_price = float(np.max(neckline_highs))

            broke_neckline = False
            break_idx = -1
            for k in range(head_idx + 1, len(df)):
                if closes[k] > neckline_price:
                    broke_neckline = True
                    break_idx = k
                    break

            if not broke_neckline or break_idx == -1:
                return LTFEntryResult(
                    triggered=False, pattern=EntryPattern.QMR, entry_price=0.0, sl_price=0.0
                )

            latest_price = float(closes[-1])
            is_at_shoulder = (
                abs(latest_price - left_shoulder_price) / left_shoulder_price
                <= self.shoulder_tolerance_pct
            )
            retested = any(
                lows[k] <= left_shoulder_price * (1.0 + self.shoulder_tolerance_pct)
                for k in range(break_idx, len(df))
            )

            if is_at_shoulder or (retested and latest_price >= left_shoulder_price):
                trigger_time = pd.to_datetime(times[-1])
                return LTFEntryResult(
                    triggered=True,
                    pattern=EntryPattern.QMR,
                    entry_price=latest_price,
                    sl_price=head_price,
                    head_price=head_price,
                    left_shoulder_price=left_shoulder_price,
                    trigger_time=trigger_time,
                    details={
                        "head_price": head_price,
                        "left_shoulder": left_shoulder_price,
                        "neckline": neckline_price,
                    },
                )

        return LTFEntryResult(
            triggered=False, pattern=EntryPattern.QMR, entry_price=0.0, sl_price=0.0
        )

    def detect_amd(
        self,
        df: pd.DataFrame,
        bias: MarketBias,
        htf_poc: float,
    ) -> LTFEntryResult:
        """Detects Accumulation, Manipulation, Distribution (AMD) pattern.

        - A (Accumulation): Range consolidation over N candles.
        - M (Manipulation): Liquidity sweep into HTF POC zone.
        - D (Distribution): Strong displacement candle closing through opposite side of the box.
        """
        if len(df) < self.amd_box_candles + 4:
            return LTFEntryResult(
                triggered=False, pattern=EntryPattern.AMD, entry_price=0.0, sl_price=0.0
            )

        # Accumulation box followed by Manipulation & Distribution action
        box_len = min(self.amd_box_candles, len(df) - 4)
        box_df = df.iloc[:box_len]
        action_df = df.iloc[box_len:]

        box_high = float(box_df["high"].max())
        box_low = float(box_df["low"].min())
        box_range = box_high - box_low

        if box_range == 0:
            return LTFEntryResult(
                triggered=False, pattern=EntryPattern.AMD, entry_price=0.0, sl_price=0.0
            )

        times = df["time"].values if "time" in df.columns else df.index.values

        if bias == MarketBias.BEARISH:
            # Manipulation: sweeps above box_high into HTF POC
            action_high = float(action_df["high"].max())
            swept_above = action_high > box_high

            poc_reached = (action_high >= htf_poc * (1.0 - self.zone_tolerance_pct)) or (
                action_high >= box_high * 1.0005
            )
            if not (swept_above and poc_reached):
                return LTFEntryResult(
                    triggered=False, pattern=EntryPattern.AMD, entry_price=0.0, sl_price=0.0
                )

            # Distribution: strong close back down through box_low or below box midpoint
            last_close = float(action_df["close"].iloc[-1])
            displacement_down = (action_high - last_close) >= box_range * self.amd_displacement_mult
            closed_below_box = last_close < (box_low + 0.3 * box_range)

            if displacement_down or closed_below_box:
                return LTFEntryResult(
                    triggered=True,
                    pattern=EntryPattern.AMD,
                    entry_price=last_close,
                    sl_price=action_high,
                    manipulation_extreme=action_high,
                    trigger_time=pd.to_datetime(times[-1]),
                    details={
                        "box_high": box_high,
                        "box_low": box_low,
                        "manipulation_extreme": action_high,
                    },
                )

        elif bias == MarketBias.BULLISH:
            action_low = float(action_df["low"].min())
            swept_below = action_low < box_low

            poc_reached = (action_low <= htf_poc * (1.0 + self.zone_tolerance_pct)) or (
                action_low <= box_low * 0.9995
            )
            if not (swept_below and poc_reached):
                return LTFEntryResult(
                    triggered=False, pattern=EntryPattern.AMD, entry_price=0.0, sl_price=0.0
                )

            last_close = float(action_df["close"].iloc[-1])
            displacement_up = (last_close - action_low) >= box_range * self.amd_displacement_mult
            closed_above_box = last_close > (box_high - 0.3 * box_range)

            if displacement_up or closed_above_box:
                return LTFEntryResult(
                    triggered=True,
                    pattern=EntryPattern.AMD,
                    entry_price=last_close,
                    sl_price=action_low,
                    manipulation_extreme=action_low,
                    trigger_time=pd.to_datetime(times[-1]),
                    details={
                        "box_high": box_high,
                        "box_low": box_low,
                        "manipulation_extreme": action_low,
                    },
                )

        return LTFEntryResult(
            triggered=False, pattern=EntryPattern.AMD, entry_price=0.0, sl_price=0.0
        )

    def evaluate_entry(
        self,
        df: pd.DataFrame,
        bias: MarketBias,
        htf_poc: float,
    ) -> Optional[LTFEntryResult]:
        """Evaluates both QMR and AMD, returning the valid pattern offering tighter risk."""
        qmr_res = self.detect_qmr(df, bias, htf_poc)
        amd_res = self.detect_amd(df, bias, htf_poc)

        if qmr_res.triggered and amd_res.triggered:
            qmr_risk = abs(qmr_res.entry_price - qmr_res.sl_price)
            amd_risk = abs(amd_res.entry_price - amd_res.sl_price)
            return qmr_res if qmr_risk <= amd_risk else amd_res
        elif qmr_res.triggered:
            return qmr_res
        elif amd_res.triggered:
            return amd_res

        return None

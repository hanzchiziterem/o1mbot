"""Fixed Range Volume Profile (FRVP) and PD Array confluence module.

Calculates Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL)
across the opposing impulse swing and verifies alignment with unmitigated FVGs or Breakers.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models import (
    FairValueGap,
    OrderBlock,
    VolumeProfileBin,
    VolumeProfileResult,
)


class FixedRangeVolumeProfile:
    """Calculates Fixed Range Volume Profile across specific impulse swings."""

    def __init__(self, bin_count: int = 80, value_area_pct: float = 0.70) -> None:
        self.bin_count = bin_count
        self.value_area_pct = value_area_pct

    def compute_profile(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
    ) -> VolumeProfileResult:
        """Computes volume profile between start_idx and end_idx (inclusive).

        Splits price span into 60-100 bins and allocates candle volume proportionally.
        """
        if df.empty or start_idx < 0 or end_idx >= len(df) or start_idx > end_idx:
            # Fall back to entire DataFrame if indices invalid
            sub_df = df.copy()
        else:
            sub_df = df.iloc[start_idx : end_idx + 1].copy()

        if sub_df.empty:
            return VolumeProfileResult(poc=0.0, vah=0.0, val=0.0, total_volume=0.0)

        min_price = float(sub_df["low"].min())
        max_price = float(sub_df["high"].max())

        if min_price == max_price:
            poc = min_price
            return VolumeProfileResult(poc=poc, vah=poc, val=poc, total_volume=1.0)

        price_step = (max_price - min_price) / self.bin_count
        bin_edges = np.linspace(min_price, max_price, self.bin_count + 1)
        bin_volumes = np.zeros(self.bin_count)

        vol_col = "tick_volume" if "tick_volume" in sub_df.columns else "volume"
        if vol_col not in sub_df.columns or (sub_df[vol_col] == 0).all():
            volumes = np.ones(len(sub_df))
        else:
            volumes = sub_df[vol_col].values

        highs = sub_df["high"].values
        lows = sub_df["low"].values

        # Distribute each candle's volume uniformly across intersected bins
        for c_idx in range(len(sub_df)):
            c_high = highs[c_idx]
            c_low = lows[c_idx]
            c_vol = volumes[c_idx]

            if c_high <= c_low:
                # Flat candle
                bin_i = min(int((c_low - min_price) / price_step), self.bin_count - 1)
                bin_volumes[bin_i] += c_vol
                continue

            # Find overlapping bins
            b_start = max(0, int((c_low - min_price) / price_step))
            b_end = min(self.bin_count - 1, int((c_high - min_price) / price_step))

            n_bins = max(1, b_end - b_start + 1)
            vol_per_bin = c_vol / n_bins
            for b in range(b_start, b_end + 1):
                bin_volumes[b] += vol_per_bin

        # Identify POC (bin with maximum volume)
        poc_bin_idx = int(np.argmax(bin_volumes))
        poc = float((bin_edges[poc_bin_idx] + bin_edges[poc_bin_idx + 1]) / 2.0)

        # Calculate Value Area (70% of total volume expanding outward from POC)
        total_vol = float(np.sum(bin_volumes))
        target_vol = total_vol * self.value_area_pct

        curr_vol = bin_volumes[poc_bin_idx]
        up_idx = poc_bin_idx
        down_idx = poc_bin_idx

        while curr_vol < target_vol and (up_idx < self.bin_count - 1 or down_idx > 0):
            next_up_vol = bin_volumes[up_idx + 1] if up_idx < self.bin_count - 1 else 0.0
            next_down_vol = bin_volumes[down_idx - 1] if down_idx > 0 else 0.0

            if next_up_vol >= next_down_vol and up_idx < self.bin_count - 1:
                up_idx += 1
                curr_vol += next_up_vol
            elif down_idx > 0:
                down_idx -= 1
                curr_vol += next_down_vol
            else:
                break

        val = float(bin_edges[down_idx])
        vah = float(bin_edges[up_idx + 1])

        # Construct bin representations
        bins: List[VolumeProfileBin] = []
        for i in range(self.bin_count):
            bins.append(
                VolumeProfileBin(
                    price_bottom=float(bin_edges[i]),
                    price_top=float(bin_edges[i + 1]),
                    price_mid=float((bin_edges[i] + bin_edges[i + 1]) / 2.0),
                    volume=float(bin_volumes[i]),
                )
            )

        return VolumeProfileResult(
            poc=poc,
            vah=vah,
            val=val,
            total_volume=total_vol,
            bins=bins,
        )

    def check_pd_array_confluence(
        self,
        poc: float,
        fvg: Optional[FairValueGap] = None,
        breaker: Optional[OrderBlock] = None,
        tolerance_pct: float = 0.0020,
    ) -> Tuple[bool, Optional[str]]:
        """Verifies if the POC aligns with an unmitigated FVG or Breaker.

        Returns (is_confluent, confluence_description).
        """
        if fvg is not None and not fvg.mitigated:
            # Check if POC is inside FVG or within tolerance
            min_boundary = min(fvg.bottom, fvg.top) * (1.0 - tolerance_pct)
            max_boundary = max(fvg.bottom, fvg.top) * (1.0 + tolerance_pct)
            if min_boundary <= poc <= max_boundary:
                return True, f"POC @ {poc:.5f} aligned with FVG [{fvg.bottom:.5f} - {fvg.top:.5f}]"

        if breaker is not None and not breaker.mitigated:
            min_boundary = min(breaker.bottom, breaker.top) * (1.0 - tolerance_pct)
            max_boundary = max(breaker.bottom, breaker.top) * (1.0 + tolerance_pct)
            if min_boundary <= poc <= max_boundary:
                return (
                    True,
                    f"POC @ {poc:.5f} aligned with Breaker [{breaker.bottom:.5f} - {breaker.top:.5f}]",
                )

        return False, None

"""Integrated Top-Down + Volume Profile + Price Action Trading Strategy.

Orchestrates:
1. Macro & Quarterly Liquidity Bias (D1 / W1 / MN1)
2. Intermediate Structure & FRVP (H4 / H1)
3. Precision LTF Entries (M15 / M5)
4. Strict Risk Filter (Planned RR >= 5.0)

Shared identically between Historical Backtester and Live/Paper Execution Daemon.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.htf_structure import HTFStructureAnalyzer
from src.ltf_entries import LTFEntryDetector
from src.macro_liquidity import MacroLiquidityAnalyzer
from src.models import (
    MarketBias,
    OrderType,
    TradeSignal,
)
from src.risk_manager import RiskManager
from src.volume_profile import FixedRangeVolumeProfile


class TopDownVPStrategy:
    """End-to-end trading strategy orchestrator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        strat_cfg = cfg.get("strategy", {})
        risk_cfg = cfg.get("risk", {})

        # 1. Macro Liquidity Analyzer
        macro_cfg = strat_cfg.get("macro", {})
        self.macro_analyzer = MacroLiquidityAnalyzer(
            anchor_month=macro_cfg.get("quarters_anchor_month", 12),
            anchor_day=macro_cfg.get("quarter_anchor_day", 1),
            lookbacks=tuple(macro_cfg.get("ipda_lookbacks", [20, 40, 60])),  # type: ignore
            primary_dol_lookback=macro_cfg.get("dol_primary_days", 60),
        )

        # 2. HTF Structure Analyzer
        struct_cfg = strat_cfg.get("structure", {})
        self.htf_analyzer = HTFStructureAnalyzer(
            swing_lookback=struct_cfg.get("swing_lookback", 5),
            min_fvg_size_pct=struct_cfg.get("fvg_min_size_pct", 0.0002),
        )

        # 3. Fixed Range Volume Profile
        vp_cfg = strat_cfg.get("volume_profile", {})
        self.vp_analyzer = FixedRangeVolumeProfile(
            bin_count=vp_cfg.get("bin_count", 80),
            value_area_pct=vp_cfg.get("value_area_pct", 0.70),
        )
        self.poc_confluence_tolerance = vp_cfg.get("poc_confluence_tolerance_pct", 0.0020)

        # 4. LTF Entry Detector
        entry_cfg = strat_cfg.get("entries", {})
        self.ltf_detector = LTFEntryDetector(
            amd_box_candles=entry_cfg.get("amd_box_candles", 12),
            amd_displacement_mult=entry_cfg.get("amd_displacement_threshold_mult", 1.3),
            shoulder_tolerance_pct=entry_cfg.get("qmr_left_shoulder_tolerance_pct", 0.0015),
        )

        # 5. Risk Manager
        self.risk_manager = RiskManager(
            min_rr=risk_cfg.get("min_rr", 5.0),
            risk_per_trade_pct=risk_cfg.get("risk_per_trade_pct", 1.0),
            spread_buffer_pips=risk_cfg.get("spread_buffer_pips", 2.0),
            exness_specs=cfg.get("exness_specs"),
        )

    def evaluate(
        self,
        symbol: str,
        rates_by_tf: Dict[str, pd.DataFrame],
        equity: float = 10000.0,
        current_spread: Optional[float] = None,
    ) -> Optional[TradeSignal]:
        """Evaluates top-down pipeline across multiple timeframes.

        rates_by_tf keys: "MN1", "W1", "D1", "H4", "H1", "M15", "M5"
        """
        # --- Step 1: Macro & Quarterly Liquidity Bias ---
        monthly_df = rates_by_tf.get("MN1")
        weekly_df = rates_by_tf.get("W1")
        daily_df = rates_by_tf.get("D1")

        if daily_df is None or daily_df.empty:
            return None

        macro_bias = self.macro_analyzer.determine_macro_bias(monthly_df, weekly_df)
        if macro_bias == MarketBias.NEUTRAL:
            # Fall back to daily trend
            macro_bias = (
                MarketBias.BEARISH
                if daily_df["close"].iloc[-1] < daily_df["open"].iloc[-1]
                else MarketBias.BULLISH
            )

        ipda_levels = self.macro_analyzer.calculate_ipda_levels(daily_df, bias=macro_bias)

        # Check Inducement Sweep on Daily into HTF FVG/OB
        d1_fvgs = self.htf_analyzer.find_fvgs(daily_df)
        d1_obs = self.htf_analyzer.find_order_blocks(daily_df)
        swept, _ = self.macro_analyzer.check_inducement_sweep(
            daily_df, ipda_levels, d1_fvgs, d1_obs
        )

        if not swept:
            # Inducement filter not met
            return None

        direction = OrderType.SELL if macro_bias == MarketBias.BEARISH else OrderType.BUY

        # --- Step 2: Intermediate Structure & FRVP (Dynamic H4 or H1) ---
        candidates: List[TradeSignal] = []

        # Pairs to evaluate: (HTF, LTF)
        tf_pairs = [("H4", "M15"), ("H1", "M5")]

        for htf_name, ltf_name in tf_pairs:
            htf_df = rates_by_tf.get(htf_name)
            ltf_df = rates_by_tf.get(ltf_name)

            if htf_df is None or htf_df.empty or ltf_df is None or ltf_df.empty:
                continue

            choch = self.htf_analyzer.detect_choch(htf_df, macro_bias)
            if not choch.occurred:
                continue

            # Pull FRVP strictly across opposing impulse swing
            start_i = min(choch.impulse_start_idx, choch.impulse_end_idx)
            end_i = max(choch.impulse_start_idx, choch.impulse_end_idx)
            vp_result = self.vp_analyzer.compute_profile(htf_df, start_i, end_i)

            # Check PD Array Confluence (POC aligned with unmitigated FVG or Breaker)
            confluent, _ = self.vp_analyzer.check_pd_array_confluence(
                poc=vp_result.poc,
                fvg=choch.fvg,
                breaker=choch.breaker,
                tolerance_pct=self.poc_confluence_tolerance,
            )

            if not confluent:
                continue

            # --- Step 3: Precision Lower-Timeframe Entry (M15 or M5) ---
            entry_res = self.ltf_detector.evaluate_entry(ltf_df, macro_bias, vp_result.poc)
            if entry_res is None or not entry_res.triggered:
                continue

            # --- Step 4: Strict Risk & Minimum 1:5 RR Filter ---
            signal = self.risk_manager.evaluate_and_filter(
                symbol=symbol,
                direction=direction,
                ltf_entry=entry_res,
                ipda=ipda_levels,
                equity=equity,
                current_spread=current_spread,
                htf=htf_name,
                ltf=ltf_name,
                poc_price=vp_result.poc,
                fvg_confluence=confluent,
            )

            if signal is not None:
                candidates.append(signal)

        if not candidates:
            return None

        # Return the setup offering the highest planned RR
        candidates.sort(key=lambda s: s.planned_rr, reverse=True)
        return candidates[0]

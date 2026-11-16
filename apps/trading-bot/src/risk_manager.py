"""Risk Management, Dynamic Position Sizing, and Minimum 1:5 RR Filter module.

Enforces:
1. Dynamic Stop-Loss placement beyond QMR Head / AMD Manipulation extreme + spread buffer.
2. Take-Profit targeting the opposing 60-day Draw on Liquidity (DOL).
3. Hard minimum 1:5 Risk-to-Reward ratio (planned_rr >= 5.0). Strictly rejects any trade below 5.0 RR.
4. Position sizing based on Exness contract specifications (XAUUSDm, BTCUSDm, Forex).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from src.models import (
    LTFEntryResult,
    OrderType,
    QuarterlyIPDALevels,
    TradeSignal,
)


class RiskManager:
    """Calculates stop-loss, take-profit, planned RR, lot size, and validates >= 5.0 RR."""

    def __init__(
        self,
        min_rr: float = 5.0,
        risk_per_trade_pct: float = 1.0,
        spread_buffer_pips: float = 2.0,
        exness_specs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.min_rr = min_rr
        self.risk_per_trade_pct = risk_per_trade_pct
        self.spread_buffer_pips = spread_buffer_pips
        self.exness_specs = exness_specs or self._default_specs()

    @staticmethod
    def _default_specs() -> Dict[str, Any]:
        return {
            "XAUUSDm": {
                "contract_size": 100,
                "lot_step": 0.01,
                "min_lot": 0.01,
                "max_lot": 100.0,
                "pip_size": 0.1,
            },
            "BTCUSDm": {
                "contract_size": 1,
                "lot_step": 0.01,
                "min_lot": 0.01,
                "max_lot": 50.0,
                "pip_size": 1.0,
            },
            "EURUSDm": {
                "contract_size": 100000,
                "lot_step": 0.01,
                "min_lot": 0.01,
                "max_lot": 200.0,
                "pip_size": 0.0001,
            },
            "GBPUSDm": {
                "contract_size": 100000,
                "lot_step": 0.01,
                "min_lot": 0.01,
                "max_lot": 200.0,
                "pip_size": 0.0001,
            },
        }

    def get_symbol_spec(self, symbol: str) -> Dict[str, Any]:
        """Returns specification for symbol with fallback."""
        if symbol in self.exness_specs:
            return self.exness_specs[symbol]
        # Fallback to standard Forex specs
        return {
            "contract_size": 100000,
            "lot_step": 0.01,
            "min_lot": 0.01,
            "max_lot": 100.0,
            "pip_size": 0.0001,
        }

    def calculate_sl_tp(
        self,
        symbol: str,
        direction: OrderType,
        entry_price: float,
        structural_extreme: float,
        dol_price: float,
        current_spread: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Calculates dynamic SL (beyond pattern extreme + spread buffer) and TP at DOL."""
        spec = self.get_symbol_spec(symbol)
        pip_size = float(spec.get("pip_size", 0.0001))
        spread_buffer = self.spread_buffer_pips * pip_size

        if current_spread is not None:
            spread_buffer = max(spread_buffer, current_spread * 1.5)

        if direction == OrderType.SELL:
            sl_price = structural_extreme + spread_buffer
            tp_price = dol_price
        else:
            sl_price = structural_extreme - spread_buffer
            tp_price = dol_price

        return sl_price, tp_price

    def calculate_planned_rr(self, entry_price: float, sl_price: float, tp_price: float) -> float:
        """Calculates planned risk-to-reward ratio: abs(entry - tp) / abs(sl - entry)."""
        risk = abs(sl_price - entry_price)
        if risk == 0.0:
            return 0.0
        reward = abs(tp_price - entry_price)
        return float(reward / risk)

    def calculate_lot_size(
        self,
        symbol: str,
        equity: float,
        entry_price: float,
        sl_price: float,
    ) -> float:
        """Calculates position lot size respecting Exness contract specs."""
        risk_distance = abs(entry_price - sl_price)
        if risk_distance == 0.0 or equity <= 0:
            return 0.01

        spec = self.get_symbol_spec(symbol)
        contract_size = float(spec.get("contract_size", 100000))
        min_lot = float(spec.get("min_lot", 0.01))
        max_lot = float(spec.get("max_lot", 100.0))
        lot_step = float(spec.get("lot_step", 0.01))

        risk_amount_usd = equity * (self.risk_per_trade_pct / 100.0)
        # Loss per 1.0 lot = risk_distance * contract_size
        loss_per_lot = risk_distance * contract_size
        if loss_per_lot == 0:
            return min_lot

        raw_lots = risk_amount_usd / loss_per_lot

        # Step quantization with floating point epsilon to prevent 0.49999999 precision truncation
        steps = math.floor((raw_lots / lot_step) + 1e-7)
        lots = steps * lot_step
        lots = max(min_lot, min(max_lot, lots))
        return round(lots, 2)

    def evaluate_and_filter(
        self,
        symbol: str,
        direction: OrderType,
        ltf_entry: LTFEntryResult,
        ipda: QuarterlyIPDALevels,
        equity: float = 10000.0,
        current_spread: Optional[float] = None,
        htf: str = "H4",
        ltf: str = "M15",
        poc_price: float = 0.0,
        fvg_confluence: bool = True,
    ) -> Optional[TradeSignal]:
        """Validates minimum 1:5 RR rule and generates TradeSignal if passed."""
        if not ltf_entry.triggered:
            return None

        entry_price = ltf_entry.entry_price
        extreme = ltf_entry.sl_price  # Head or Manipulation extreme

        sl_price, tp_price = self.calculate_sl_tp(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            structural_extreme=extreme,
            dol_price=ipda.dol_price,
            current_spread=current_spread,
        )

        planned_rr = self.calculate_planned_rr(entry_price, sl_price, tp_price)

        # HARD RULE: Strictly reject any trade where planned_rr < 5.0
        if planned_rr < self.min_rr:
            return None

        # Ensure correct directional alignment
        if direction == OrderType.SELL and (sl_price <= entry_price or tp_price >= entry_price):
            return None
        if direction == OrderType.BUY and (sl_price >= entry_price or tp_price <= entry_price):
            return None

        lot_size = self.calculate_lot_size(symbol, equity, entry_price, sl_price)

        return TradeSignal(
            symbol=symbol,
            direction=direction,
            pattern=ltf_entry.pattern,
            htf=htf,
            ltf=ltf,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            planned_rr=round(planned_rr, 2),
            poc_price=poc_price,
            dol_price=ipda.dol_price,
            fvg_confluence=fvg_confluence,
            signal_time=ltf_entry.trigger_time,
            lot_size=lot_size,
        )

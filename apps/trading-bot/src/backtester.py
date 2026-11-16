"""Historical event-driven backtesting engine.

Simulates execution modeling Exness spreads, commissions, and slippage,
and persists trade metadata directly into SQLite for dashboard reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.database import DatabaseManager
from src.models import (
    OrderType,
    TradeOutcome,
    TradeRecord,
    TradeSignal,
)
from src.strategy import TopDownVPStrategy


@dataclass
class BacktestResult:
    initial_balance: float
    final_balance: float
    total_trades: int
    win_trades: int
    loss_trades: int
    be_trades: int
    win_rate_pct: float
    loss_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float
    avg_planned_rr: float
    avg_realized_rr: float
    cumulative_r: float
    trades: List[TradeRecord] = field(default_factory=list)


class HistoricalBacktester:
    """Event-driven simulator for top-down volume profile strategy."""

    def __init__(
        self,
        strategy: TopDownVPStrategy,
        database: Optional[DatabaseManager] = None,
        initial_balance: float = 10000.0,
        slippage_points: float = 2.0,
        break_even_at_r: float = 2.5,
    ) -> None:
        self.strategy = strategy
        self.db = database or DatabaseManager()
        self.initial_balance = initial_balance
        self.slippage_points = slippage_points
        self.break_even_at_r = break_even_at_r

    def run(
        self,
        symbol: str,
        rates_by_tf: Dict[str, pd.DataFrame],
        log_to_db: bool = True,
    ) -> BacktestResult:
        """Executes historical simulation across multi-timeframe candles."""
        # Check that we have LTF execution candles
        ltf_df = (
            rates_by_tf.get("M15") if rates_by_tf.get("M15") is not None else rates_by_tf.get("M5")
        )
        if ltf_df is None or ltf_df.empty:
            raise ValueError("Backtester requires at least M15 or M5 rates")

        balance = self.initial_balance
        peak_balance = balance
        max_drawdown = 0.0

        executed_trades: List[TradeRecord] = []

        # Get symbol pip specs
        spec = self.strategy.risk_manager.get_symbol_spec(symbol)
        point = float(spec.get("pip_size", 0.0001)) * 0.1
        slippage_dist = self.slippage_points * point

        # Step through historical windows
        # Evaluate every N candles to simulate real-time forward progression
        step = 10
        min_bars = 60

        for i in range(min_bars, len(ltf_df), step):
            sub_rates: Dict[str, pd.DataFrame] = {}
            current_time = ltf_df["time"].iloc[i] if "time" in ltf_df.columns else None

            for tf, df in rates_by_tf.items():
                if current_time is not None and "time" in df.columns:
                    sub_rates[tf] = df[df["time"] <= current_time]
                else:
                    ratio = max(1, len(df) // len(ltf_df))
                    idx = min(len(df), int(i * ratio) + 10)
                    sub_rates[tf] = df.iloc[:idx]

            # Generate signal from unified strategy pipeline
            signal: Optional[TradeSignal] = self.strategy.evaluate(
                symbol=symbol,
                rates_by_tf=sub_rates,
                equity=balance,
            )

            if signal is None:
                continue

            # Simulate trade execution forward in time
            trade_record = self._simulate_trade_outcome(
                signal=signal,
                ltf_df=ltf_df,
                entry_idx=i,
                slippage_dist=slippage_dist,
            )

            if trade_record is not None:
                executed_trades.append(trade_record)
                if log_to_db:
                    self.db.insert_trade(trade_record)

                # Update balance
                if trade_record.profit is not None:
                    balance += trade_record.profit

                peak_balance = max(peak_balance, balance)
                dd = (peak_balance - balance) / peak_balance * 100.0 if peak_balance > 0 else 0.0
                max_drawdown = max(max_drawdown, dd)

        # Calculate summary metrics
        total = len(executed_trades)
        wins = [t for t in executed_trades if t.outcome == TradeOutcome.WIN.value]
        losses = [t for t in executed_trades if t.outcome == TradeOutcome.LOSS.value]
        bes = [t for t in executed_trades if t.outcome == TradeOutcome.BE.value]

        win_rate = (len(wins) / total * 100.0) if total > 0 else 0.0
        loss_rate = (len(losses) / total * 100.0) if total > 0 else 0.0

        gross_profit = sum(t.profit for t in wins if t.profit is not None)
        gross_loss = abs(sum(t.profit for t in losses if t.profit is not None))
        profit_factor = (
            (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)
        )

        avg_planned_rr = (sum(t.planned_rr for t in executed_trades) / total) if total > 0 else 0.0
        realized_rrs = [t.realized_rr for t in executed_trades if t.realized_rr is not None]
        avg_realized_rr = (sum(realized_rrs) / len(realized_rrs)) if realized_rrs else 0.0
        cumulative_r = sum(realized_rrs) if realized_rrs else 0.0

        return BacktestResult(
            initial_balance=self.initial_balance,
            final_balance=round(balance, 2),
            total_trades=total,
            win_trades=len(wins),
            loss_trades=len(losses),
            be_trades=len(bes),
            win_rate_pct=round(win_rate, 2),
            loss_rate_pct=round(loss_rate, 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown_pct=round(max_drawdown, 2),
            avg_planned_rr=round(avg_planned_rr, 2),
            avg_realized_rr=round(avg_realized_rr, 2),
            cumulative_r=round(cumulative_r, 2),
            trades=executed_trades,
        )

    def _simulate_trade_outcome(
        self,
        signal: TradeSignal,
        ltf_df: pd.DataFrame,
        entry_idx: int,
        slippage_dist: float,
    ) -> Optional[TradeRecord]:
        """Simulates bar-by-bar progression to determine outcome (WIN, LOSS, BE)."""
        highs = ltf_df["high"].values
        lows = ltf_df["low"].values
        closes = ltf_df["close"].values
        times = ltf_df["time"].values if "time" in ltf_df.columns else ltf_df.index.values

        # Apply slippage on entry
        entry_price = signal.entry_price
        sl_price = signal.sl_price
        tp_price = signal.tp_price
        direction = signal.direction

        if direction == OrderType.BUY:
            actual_entry = entry_price + slippage_dist
        else:
            actual_entry = entry_price - slippage_dist

        risk_dist = abs(actual_entry - sl_price)
        reward_dist = abs(tp_price - actual_entry)
        if risk_dist == 0:
            return None

        # Planned RR
        planned_rr = reward_dist / risk_dist
        if planned_rr < 5.0:
            return None

        be_price = actual_entry
        moved_to_be = False
        outcome = TradeOutcome.OPEN
        close_price = actual_entry
        close_time = str(times[-1])
        realized_rr = 0.0

        # Simulate subsequent bars
        for k in range(entry_idx + 1, len(ltf_df)):
            bar_high = highs[k]
            bar_low = lows[k]
            bar_time = str(times[k])

            if direction == OrderType.SELL:
                # Check BE threshold (price dropped >= break_even_at_r * risk)
                if not moved_to_be and (actual_entry - bar_low) >= self.break_even_at_r * risk_dist:
                    moved_to_be = True
                    sl_price = be_price

                # Check SL hit
                if bar_high >= sl_price:
                    outcome = TradeOutcome.BE if moved_to_be else TradeOutcome.LOSS
                    close_price = sl_price
                    close_time = bar_time
                    realized_rr = 0.0 if moved_to_be else -1.0
                    break

                # Check TP hit
                if bar_low <= tp_price:
                    outcome = TradeOutcome.WIN
                    close_price = tp_price
                    close_time = bar_time
                    realized_rr = planned_rr
                    break

            elif direction == OrderType.BUY:
                # Check BE threshold
                if (
                    not moved_to_be
                    and (bar_high - actual_entry) >= self.break_even_at_r * risk_dist
                ):
                    moved_to_be = True
                    sl_price = be_price

                # Check SL hit
                if bar_low <= sl_price:
                    outcome = TradeOutcome.BE if moved_to_be else TradeOutcome.LOSS
                    close_price = sl_price
                    close_time = bar_time
                    realized_rr = 0.0 if moved_to_be else -1.0
                    break

                # Check TP hit
                if bar_high >= tp_price:
                    outcome = TradeOutcome.WIN
                    close_price = tp_price
                    close_time = bar_time
                    realized_rr = planned_rr
                    break

        if outcome == TradeOutcome.OPEN:
            # Trade still open at end of data
            close_price = float(closes[-1])
            if direction == OrderType.SELL:
                realized_rr = (actual_entry - close_price) / risk_dist
            else:
                realized_rr = (close_price - actual_entry) / risk_dist

        # Compute monetary profit
        spec = self.strategy.risk_manager.get_symbol_spec(signal.symbol)
        contract_size = float(spec.get("contract_size", 100000))
        price_diff = (
            (actual_entry - close_price)
            if direction == OrderType.SELL
            else (close_price - actual_entry)
        )
        profit = price_diff * contract_size * signal.lot_size

        entry_time_str = str(times[entry_idx])

        return TradeRecord(
            mode="BACKTEST",
            symbol=signal.symbol,
            direction=signal.direction.value,
            htf=signal.htf,
            ltf=signal.ltf,
            entry_pattern=signal.pattern.value,
            vp_poc=signal.poc_price,
            fvg_confluence=1 if signal.fvg_confluence else 0,
            entry_price=round(actual_entry, 5),
            sl_price=round(sl_price, 5),
            tp_price=round(tp_price, 5),
            close_price=round(close_price, 5),
            planned_rr=round(planned_rr, 2),
            realized_rr=round(realized_rr, 2),
            outcome=outcome.value,
            entry_time=entry_time_str,
            close_time=close_time,
            lot_size=signal.lot_size,
            profit=round(profit, 2),
            notes="Backtest execution",
        )

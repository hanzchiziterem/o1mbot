"""MetaTrader 5 Connector and Mock Offline Adapter for Exness.

Provides seamless switching between live Windows MT5 terminal connection
and a clean offline/mock adapter for headless Linux/CI pipelines and backtesting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try importing live MetaTrader5
try:
    import MetaTrader5 as mt5  # type: ignore

    HAS_MT5 = True
except (ImportError, Exception):
    mt5 = None
    HAS_MT5 = False


@dataclass
class AccountInfo:
    login: int
    balance: float
    equity: float
    margin: float
    free_margin: float
    leverage: int
    currency: str = "USD"


@dataclass
class SymbolInfo:
    name: str
    bid: float
    ask: float
    spread: float
    point: float
    digits: int
    contract_size: float
    min_lot: float
    max_lot: float
    lot_step: float


@dataclass
class OrderResult:
    retcode: int
    ticket: int
    price: float
    volume: float
    comment: str


class MT5Connector:
    """Wrapper around MetaTrader5 library with mock fallback."""

    TIMEFRAME_MAP = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 16385,
        "H4": 16388,
        "D1": 16408,
        "W1": 32769,
        "MN1": 49153,
    }

    def __init__(
        self,
        login: int = 0,
        password: str = "",
        server: str = "Exness-MT5Real",
        path: str = "",
        mock_mode: bool = False,
    ) -> None:
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.mock_mode = mock_mode or not HAS_MT5
        self.connected = False
        self._mock_positions: Dict[int, Dict[str, Any]] = {}
        self._mock_ticket_counter = 100000

    def initialize(self) -> bool:
        """Initializes connection to MT5 terminal or sets up mock adapter."""
        if self.mock_mode:
            logger.info("Initializing MockMT5Connector (Offline mode)")
            self.connected = True
            return True

        if not HAS_MT5 or mt5 is None:
            logger.warning("MetaTrader5 package not available. Falling back to Mock mode.")
            self.mock_mode = True
            self.connected = True
            return True

        init_args: Dict[str, Any] = {}
        if self.path:
            init_args["path"] = self.path
        if self.login:
            init_args["login"] = self.login
        if self.password:
            init_args["password"] = self.password
        if self.server:
            init_args["server"] = self.server

        if not mt5.initialize(**init_args):
            logger.warning(
                "MT5 initialize failed (%s). Falling back to Mock mode.", mt5.last_error()
            )
            self.mock_mode = True
            self.connected = True
            return True

        self.connected = True
        logger.info("Connected to live MetaTrader 5 terminal: %s", self.server)
        return True

    def shutdown(self) -> None:
        """Shuts down MT5 connection."""
        if not self.mock_mode and HAS_MT5 and mt5 is not None:
            mt5.shutdown()
        self.connected = False
        logger.info("MT5 connection closed.")

    def get_account_info(self) -> AccountInfo:
        """Fetches account balance, equity, and margin info."""
        if self.mock_mode or not HAS_MT5 or mt5 is None:
            return AccountInfo(
                login=self.login or 12345678,
                balance=10000.0,
                equity=10000.0,
                margin=0.0,
                free_margin=10000.0,
                leverage=100,
            )

        info = mt5.account_info()
        if info is None:
            return AccountInfo(
                login=0, balance=0.0, equity=0.0, margin=0.0, free_margin=0.0, leverage=1
            )

        return AccountInfo(
            login=info.login,
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            leverage=info.leverage,
            currency=info.currency,
        )

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Fetches symbol bid/ask and tick specification."""
        if self.mock_mode or not HAS_MT5 or mt5 is None:
            # Deterministic synthetic prices
            base_prices = {
                "XAUUSDm": 2650.0,
                "BTCUSDm": 92000.0,
                "EURUSDm": 1.0850,
                "GBPUSDm": 1.2950,
            }
            base = base_prices.get(symbol, 100.0)
            digits = 2 if "XAU" in symbol or "BTC" in symbol else 5
            point = 0.01 if digits == 2 else 0.00001
            spread_points = 15 if digits == 2 else 10
            return SymbolInfo(
                name=symbol,
                bid=base,
                ask=base + (spread_points * point),
                spread=spread_points,
                point=point,
                digits=digits,
                contract_size=100.0 if "XAU" in symbol else (1.0 if "BTC" in symbol else 100000.0),
                min_lot=0.01,
                max_lot=100.0,
                lot_step=0.01,
            )

        s_info = mt5.symbol_info(symbol)
        if s_info is None:
            raise ValueError(f"Symbol {symbol} not found in MT5")

        return SymbolInfo(
            name=s_info.name,
            bid=s_info.bid,
            ask=s_info.ask,
            spread=float(s_info.spread),
            point=s_info.point,
            digits=s_info.digits,
            contract_size=s_info.trade_contract_size,
            min_lot=s_info.volume_min,
            max_lot=s_info.volume_max,
            lot_step=s_info.volume_step,
        )

    def get_rates(
        self,
        symbol: str,
        timeframe: str,
        count: int = 200,
        start_pos: int = 0,
    ) -> pd.DataFrame:
        """Fetches OHLCV candlestick data for given symbol and timeframe."""
        if self.mock_mode or not HAS_MT5 or mt5 is None:
            return self._generate_synthetic_rates(symbol, timeframe, count)

        tf_code = self.TIMEFRAME_MAP.get(timeframe, 16385)
        rates = mt5.copy_rates_from_pos(symbol, tf_code, start_pos, count)
        if rates is None or len(rates) == 0:
            logger.warning(
                "No rates returned for %s on %s; generating synthetic fallback", symbol, timeframe
            )
            return self._generate_synthetic_rates(symbol, timeframe, count)

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df[["time", "open", "high", "low", "close", "tick_volume"]]

    def send_order(
        self,
        symbol: str,
        order_type: str,
        lots: float,
        price: float,
        sl: float,
        tp: float,
        comment: str = "o1mbot",
        magic: int = 101088,
    ) -> OrderResult:
        """Sends a market or limit order to MT5."""
        if self.mock_mode or not HAS_MT5 or mt5 is None:
            self._mock_ticket_counter += 1
            ticket = self._mock_ticket_counter
            self._mock_positions[ticket] = {
                "ticket": ticket,
                "symbol": symbol,
                "type": order_type,
                "lots": lots,
                "price": price,
                "sl": sl,
                "tp": tp,
                "open_time": datetime.now(timezone.utc),
                "comment": comment,
            }
            return OrderResult(
                retcode=0, ticket=ticket, price=price, volume=lots, comment="Mock Done"
            )

        mt5_type = mt5.ORDER_TYPE_BUY if order_type.upper() == "BUY" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": mt5_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err_msg = result.comment if result else str(mt5.last_error())
            logger.error("Order failed for %s: %s", symbol, err_msg)
            return OrderResult(
                retcode=getattr(result, "retcode", -1),
                ticket=0,
                price=price,
                volume=lots,
                comment=err_msg,
            )

        return OrderResult(
            retcode=result.retcode,
            ticket=result.order,
            price=result.price,
            volume=result.volume,
            comment=result.comment,
        )

    def close_order(self, ticket: int) -> bool:
        """Closes an open position by ticket."""
        if self.mock_mode or not HAS_MT5 or mt5 is None:
            if ticket in self._mock_positions:
                del self._mock_positions[ticket]
                return True
            return False

        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False

        pos = position[0]
        s_info = mt5.symbol_info(pos.symbol)
        if s_info is None:
            return False

        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = s_info.bid if close_type == mt5.ORDER_TYPE_SELL else s_info.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "price": price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": "o1mbot close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(request)
        return res is not None and res.retcode == mt5.TRADE_RETCODE_DONE

    def modify_order(self, ticket: int, sl: float, tp: float) -> bool:
        """Modifies SL and TP of an open position."""
        if self.mock_mode or not HAS_MT5 or mt5 is None:
            if ticket in self._mock_positions:
                self._mock_positions[ticket]["sl"] = sl
                self._mock_positions[ticket]["tp"] = tp
                return True
            return False

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }
        res = mt5.order_send(request)
        return res is not None and res.retcode == mt5.TRADE_RETCODE_DONE

    def _generate_synthetic_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Generates realistic synthetic OHLCV data for offline development/testing."""
        base_prices = {"XAUUSDm": 2650.0, "BTCUSDm": 92000.0, "EURUSDm": 1.0850, "GBPUSDm": 1.2950}
        base = base_prices.get(symbol, 100.0)
        volatility = (
            0.002 if "USD" in symbol and "XAU" not in symbol and "BTC" not in symbol else 0.005
        )

        tf_minutes = {
            "M1": 1,
            "M5": 5,
            "M15": 15,
            "H1": 60,
            "H4": 240,
            "D1": 1440,
            "W1": 10080,
            "MN1": 43200,
        }
        delta_m = tf_minutes.get(timeframe, 60)

        now = datetime.now(timezone.utc)
        start_time = now - timedelta(minutes=delta_m * count)

        times = [start_time + timedelta(minutes=delta_m * i) for i in range(count)]
        np.random.seed(42 + len(symbol) + delta_m)

        returns = np.random.normal(0, volatility, count)
        price_series = base * np.cumprod(1 + returns)

        opens = price_series
        closes = price_series * (1 + np.random.normal(0, volatility * 0.5, count))
        highs = np.maximum(opens, closes) * (
            1 + np.abs(np.random.normal(0, volatility * 0.4, count))
        )
        lows = np.minimum(opens, closes) * (
            1 - np.abs(np.random.normal(0, volatility * 0.4, count))
        )
        volumes = np.random.randint(100, 5000, count)

        df = pd.DataFrame(
            {
                "time": times,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "tick_volume": volumes,
            }
        )
        return df

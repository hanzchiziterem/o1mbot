"""Data models and type definitions for the trading bot and backtester."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MarketBias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class OrderType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class EntryPattern(str, Enum):
    QMR = "QMR"
    AMD = "AMD"


class TradeOutcome(str, Enum):
    OPEN = "OPEN"
    WIN = "WIN"
    LOSS = "LOSS"
    BE = "BE"


class SwingType(str, Enum):
    SWING_HIGH = "SWING_HIGH"
    SWING_LOW = "SWING_LOW"


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class SwingPoint:
    index: int
    timestamp: datetime
    price: float
    type: SwingType


@dataclass
class FairValueGap:
    top: float
    bottom: float
    is_bullish: bool
    start_time: datetime
    mitigated: bool = False
    mitigation_time: Optional[datetime] = None

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def size(self) -> float:
        return abs(self.top - self.bottom)


@dataclass
class OrderBlock:
    top: float
    bottom: float
    is_bullish: bool
    candle_time: datetime
    is_breaker: bool = False
    mitigated: bool = False

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass
class QuarterlyIPDALevels:
    anchor_date: datetime
    high_20d: float
    low_20d: float
    high_40d: float
    low_40d: float
    high_60d: float
    low_60d: float
    dol_price: float
    bias: MarketBias
    inducement_swept: bool = False
    sweep_price: Optional[float] = None


@dataclass
class VolumeProfileBin:
    price_bottom: float
    price_top: float
    price_mid: float
    volume: float


@dataclass
class VolumeProfileResult:
    poc: float
    vah: float
    val: float
    total_volume: float
    bins: List[VolumeProfileBin] = field(default_factory=list)


@dataclass
class CHOCHResult:
    occurred: bool
    direction: MarketBias
    breakout_price: float
    swing_high: float
    swing_low: float
    impulse_start_idx: int
    impulse_end_idx: int
    fvg: Optional[FairValueGap] = None
    breaker: Optional[OrderBlock] = None


@dataclass
class LTFEntryResult:
    triggered: bool
    pattern: EntryPattern
    entry_price: float
    sl_price: float
    head_price: Optional[float] = None
    left_shoulder_price: Optional[float] = None
    manipulation_extreme: Optional[float] = None
    trigger_time: datetime = field(default_factory=utc_now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeSignal:
    symbol: str
    direction: OrderType
    pattern: EntryPattern
    htf: str
    ltf: str
    entry_price: float
    sl_price: float
    tp_price: float
    planned_rr: float
    poc_price: float
    dol_price: float
    fvg_confluence: bool
    signal_time: datetime
    lot_size: float = 0.01


@dataclass
class TradeRecord:
    id: Optional[int] = None
    ticket: Optional[int] = None
    mode: str = "BACKTEST"
    symbol: str = "XAUUSDm"
    direction: str = "SELL"
    htf: str = "H4"
    ltf: str = "M15"
    entry_pattern: str = "QMR"
    vp_poc: float = 0.0
    fvg_confluence: int = 1
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    close_price: Optional[float] = None
    planned_rr: float = 0.0
    realized_rr: Optional[float] = None
    outcome: str = "OPEN"
    entry_time: str = ""
    close_time: Optional[str] = None
    lot_size: float = 0.01
    profit: Optional[float] = None
    notes: Optional[str] = None

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class OptionType(Enum):
    CE = "CE"
    PE = "PE"

class MarketDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

@dataclass
class OptionGreeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

@dataclass
class CandidateState:
    symbol: str
    strike: float
    option_type: OptionType
    composite_score: float
    confirmation_count: int
    first_seen_time: float
    last_seen_time: float
    direction: MarketDirection

@dataclass
class NormalizedOptionTick:
    symbol: str
    instrument_key: str
    underlying: str
    underlying_spot: float
    expiry: str
    strike: float
    option_type: OptionType
    ltp: float
    bid: float
    ask: float
    spread: float
    bid_qty: int
    ask_qty: int
    volume: int
    oi: int
    previous_oi: Optional[int]
    volume_change: Optional[int]
    oi_change: Optional[int]
    price_change: Optional[float]
    price_change_pct: Optional[float]
    oi_change_pct: Optional[float]
    iv: float  # Decimal IV (0.18 = 18%)
    timestamp: float
    sequence_no: int
    tte: float
    data_source: str

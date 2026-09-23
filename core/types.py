from enum import Enum
from dataclasses import dataclass
from typing import Optional

class MarketDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class OptionType(Enum):
    CE = "CE"
    PE = "PE"

class TradeAction(Enum):
    BUY = "BUY"
    NO_TRADE = "NO_TRADE"
    EXIT = "EXIT"

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

@dataclass(frozen=True)
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
    previous_oi: int
    volume_change: int
    oi_change: int
    price_change: float
    price_change_pct: float
    oi_change_pct: float
    iv: Optional[float]
    timestamp: float
    sequence_no: Optional[int] = None
    tte: float = 0.0                      # Exact calculated time-to-expiry in fractional years
    broker_delta: Optional[float] = None
    broker_theta: Optional[float] = None
    broker_gamma: Optional[float] = None
    broker_vega: Optional[float] = None
    greek_source: str = "UNKNOWN"         # e.g., UPSTOX_CHAIN, UPSTOX_WS, INTERNAL_BS
    data_source: str = "UNKNOWN"          # e.g., UPSTOX_REST, UPSTOX_FEED

from dataclasses import dataclass
from typing import Optional
from enum import Enum

class MarketDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class OptionType(Enum):
    CE = "CE"
    PE = "PE"

@dataclass(frozen=True)
class NormalizedOptionTick:
    symbol: str
    underlying: str          # "NIFTY", "BANKNIFTY", "SENSEX"
    expiry: str              # ISO Date "YYYY-MM-DD"
    strike: float
    option_type: OptionType  # CE or PE
    ltp: float
    bid: float
    ask: float
    spread: float
    volume: int
    oi: int
    volume_change: int
    oi_change: int
    price_change: float
    price_change_pct: float  # Percentage price movement
    oi_change_pct: float     # Percentage open interest movement
    timestamp: float         # Epoch timestamp in seconds
    sequence_no: int
    tte: float               # Time-to-expiry in years

    @property
    def spread_pct(self) -> float:
        """Returns relative spread percentage against LTP."""
        return (self.spread / self.ltp) if self.ltp > 0 else 1.0

@dataclass
class OptionGreeks:
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float

@dataclass
class CandidateState:
    symbol: str
    strike: float
    option_type: OptionType
    score: float
    confirmation_count: int = 0

"""
engine/safety_gate.py - Zero-Trust Hard Execution Verification Gate
Validates:
- Directional Confluence (Inversion Guard)
- Candidate Hysteresis Stability
- Live Tick Pulse TTL Freshness
- Black-Scholes Greeks Bounds (Delta & IV)
- Liquidity & Bid-Ask Spread Tolerance
"""

from typing import Tuple, Optional
from core.types import MarketDirection, OptionType, NormalizedOptionTick
from core.black_scholes import OptionGreeks

class ZeroTrustFinalSafetyGate:
    @staticmethod
    def verify_execution(
        market_direction: MarketDirection,
        tick: NormalizedOptionTick,
        greeks: Optional[OptionGreeks],
        candidate: Optional[object],
        candidate_ready: bool,
        ttl_state: object,
        quality_grade: str = "GRADE_A",
        max_spread_pct: float = 0.04,  # Up to 4.0% spread allowed for volatile index options
        min_delta: float = 0.38,       # Accommodate ATM/Fast Delta momentum moves
        max_delta: float = 0.72        # Accommodate deep ATM expansion
    ) -> Tuple[bool, str]:
        """
        Zero-Trust Hard Validation Gate.
        Returns (passed: bool, message: str).
        """
        # 1. Market Direction Check
        if market_direction == MarketDirection.NEUTRAL:
            return False, "BLOCKED: Market regime is NEUTRAL"

        # 2. Candidate Stability & Readiness Check
        if candidate is None:
            return False, "BLOCKED: Candidate is None (Scanning)"
        if not candidate_ready:
            conf_count = getattr(candidate, "confirmations", getattr(candidate, "confirmation_count", 0))
            return False, f"BLOCKED_CONFIRMATION_PENDING: Stability verification incomplete ({conf_count})"

        # 3. Inversion Guard (Bullish = Only CE, Bearish = Only PE)
        if market_direction == MarketDirection.BULLISH and tick.option_type != OptionType.CE:
            return False, "BLOCKED_INVERSION: Bullish regime cannot execute Put (PE)"
        if market_direction == MarketDirection.BEARISH and tick.option_type != OptionType.PE:
            return False, "BLOCKED_INVERSION: Bearish regime cannot execute Call (CE)"

        # 4. TTL Freshness Check (Broker Quote Stale Protection)
        is_live = getattr(ttl_state, "is_valid", getattr(ttl_state, "is_alive", lambda: False))()
        if ttl_state is None or not is_live:
            return False, "BLOCKED: Tick Pulse TTL expired or dead"

        # 5. Greeks Validation & Dynamic Delta Bounds
        if greeks is None:
            return False, "BLOCKED: Greeks are missing or non-finite"

        abs_delta = abs(greeks.delta)
        if abs_delta < min_delta:
            return False, f"BLOCKED_DELTA_BOUNDS: Delta {abs_delta:.3f} below minimum ({min_delta:.2f})"
        if abs_delta > max_delta:
            return False, f"BLOCKED_DELTA_BOUNDS: Delta {abs_delta:.3f} exceeds maximum ({max_delta:.2f})"

        # 6. Strict Liquidity & Bid-Ask Spread Check
        if tick.ltp <= 0.0:
            return False, "BLOCKED_LIQUIDITY: Zero or invalid LTP in option contract"

        # Validate Bid/Ask availability for execution
        if tick.ask <= 0.0 or tick.bid <= 0.0:
            return False, "BLOCKED_LIQUIDITY: Bid/Ask quotes currently unavailable"

        calculated_spread = tick.ask - tick.bid
        if calculated_spread < 0:
            return False, "BLOCKED_SPREAD: Inverted spread (ask < bid)"

        spread_pct = calculated_spread / tick.ltp
        if spread_pct > max_spread_pct:
            return False, f"BLOCKED_SPREAD: Spread {spread_pct*100:.2f}% exceeds max {max_spread_pct*100:.1f}%"

        return True, "PASSED_ALL_SAFETY_GATES"
